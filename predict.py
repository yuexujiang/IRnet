import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import os
import re
import pandas as pd
import logging
from sklearn.model_selection import KFold
import sklearn
import random
from random import choices
import argparse
from scipy import stats
random.seed(996)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout,Input,Reshape,BatchNormalization,Layer
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import categorical_accuracy, binary_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.regularizers import L1, L2, l2
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform, Initializer
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras import initializers, constraints
from tensorflow.keras.regularizers import Regularizer
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

import spektral
from spektral.data import MixedLoader
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool, GATConv, DiffPool, GlobalAttentionPool
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from scipy.special import softmax





class MyDataset(Dataset):
    """
    """
    def __init__(self, datafromnpzfile, **kwargs):
        self.a=None
        self.file=datafromnpzfile
        self.x=datafromnpzfile['x']
        self.y = datafromnpzfile['y']
        self.info = datafromnpzfile['info']
        self.cols=datafromnpzfile['cols']
        self.n=self.x.shape[0]
        super().__init__(**kwargs)
    def read(self):
        self.a=sp.csr_matrix(self.file['pathway_a'])
        graph=[]
        for i in range(self.n):
            x=self.x[i]
            y=self.y[i]
            graph.append(Graph(x=x,y=y))
        # We must return a list of Graph objects
        return graph
    


# data_dir = os.path.dirname(__file__)
class GMT():
    # genes_cols : start reading genes from genes_col(default 1, it can be 2 e.g. if an information col is added after the pathway col)
    # pathway col is considered to be the first column (0)
    def load_data(self, filename, genes_col=1, pathway_col=0):

        data_dict_list = []
        with open(filename) as gmt:

            data_list = gmt.readlines()

            # print data_list[0]
            for row in data_list:
                genes = row.strip().split('\t')
                genes = [re.sub('_copy.*', '', g) for g in genes]
                genes = [re.sub('\\n.*', '', g) for g in genes]
                for gene in genes[genes_col:]:
                    pathway = genes[pathway_col]
                    dict = {'group': pathway, 'gene': gene}
                    data_dict_list.append(dict)

        df = pd.DataFrame(data_dict_list)
        # print df.head()

        return df

    def load_data_dict(self, filename):

        data_dict_list = []
        dict = {}
        with open(os.path.join(data_dir, filename)) as gmt:
            data_list = gmt.readlines()

            # print data_list[0]
            for row in data_list:
                genes = row.split('\t')
                dict[genes[0]] = genes[2:]

        return dict

    def write_dict_to_file(self, dict, filename):
        lines = []
        with open(filename, 'w') as gmt:
            for k in dict:
                str1 = '	'.join(str(e) for e in dict[k])
                line = str(k) + '	' + str1 + '\n'
                lines.append(line)
            gmt.writelines(lines)
        return

    def __init__(self):

        return



def get_KEGG_map(input_list, filename='c2.cp.kegg.v6.1.symbols.gmt', genes_col=1, shuffle_genes=False):
    '''
    :param input_list: list of inputs under consideration (e.g. genes)
    :param filename: a gmt formated file e.g. pathway1 gene1 gene2 gene3
#                                     pathway2 gene4 gene5 gene6
    :param genes_col: the start index of the gene columns
    :param shuffle_genes: {True, False}
    :return: dataframe with rows =genes and columns = pathways values = 1 or 0 based on the membership of certain gene in the corresponding pathway
    '''
    d = GMT()
    df = d.load_data(filename, genes_col)
    df['value'] = 1
    mapp = pd.pivot_table(df, values='value', index='gene', columns='group', aggfunc=np.sum)
    mapp = mapp.fillna(0)
    cols_df = pd.DataFrame(index=input_list)
    mapp = cols_df.merge(mapp, right_index=True, left_index=True, how='left')
    mapp = mapp.fillna(0)
    genes = mapp.index
    pathways = mapp.columns
    mapp = mapp.values

    if shuffle_genes:
        logging.info('shuffling')
        ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
        logging.info('ones_ratio {}'.format(ones_ratio))
        mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
        logging.info('random map ones_ratio {}'.format(ones_ratio))
    return mapp, genes, pathways



class SparseTF(Layer):
    def __init__(self, units, map=None, nonzero_ind=None, kernel_initializer='glorot_uniform', W_regularizer=None,
                 activation='tanh', use_bias=True,
                 bias_initializer='zeros', bias_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.map = map
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation_fn = activations.get(activation)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super(SparseTF, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        # random sparse constarints on the weights
        # if self.map is None:
        #     mapp = np.random.rand(input_dim, self.units)
        #     mapp = mapp > 0.9
        #     mapp = mapp.astype(np.float32)
        #     self.map = mapp
        # else:
        if not self.map is None:
            self.map = self.map.astype(np.float32)

        # can be initialized directly from (map) or using a loaded nonzero_ind (useful for cloning models or create from config)
        if self.nonzero_ind is None:
            nonzero_ind = np.array(np.nonzero(self.map)).T
            self.nonzero_ind = nonzero_ind

        self.kernel_shape = (input_dim, self.units)
        # sA = sparse.csr_matrix(self.map)
        # self.sA=sA.astype(np.float32)
        # self.kernel_sparse = tf.SparseTensor(self.nonzero_ind, sA.data, sA.shape)

        # self.kernel_shape = (input_dim, self.units)
        # sA = sparse.csr_matrix(self.map)
        # self.sA=sA.astype(np.float32)
        # self.kernel_sparse = tf.SparseTensor(self.nonzero_ind, sA.data, sA.shape)
        # self.kernel_dense = tf.Variable(self.map)

        nonzero_count = self.nonzero_ind.shape[0]

        # initializer = initializers.get('uniform')
        # print 'nonzero_count', nonzero_count
        # self.kernel_vector = K.variable(initializer((nonzero_count,)), dtype=K.floatx(), name='kernel' )

        self.kernel_vector = self.add_weight(name='kernel_vector',
                                             shape=(nonzero_count,),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True, constraint=self.kernel_constraint)
        # self.kernel = tf.scatter_nd(self.nonzero_ind, self.kernel_vector, self.kernel_shape, name='kernel')
        # --------
        # init = np.random.rand(input_shape[1], self.units).astype( np.float32)
        # sA = sparse.csr_matrix(init)
        # self.kernel = K.variable(sA, dtype=K.floatx(), name= 'kernel',)
        # self.kernel_vector = K.variable(init, dtype=K.floatx(), name= 'kernel',)

        # print self.kernel.values
        # ind = np.array(np.nonzero(init))
        # stf = tf.SparseTensor(ind.T, sA.data, sA.shape)
        # print stf.dtype
        # print init.shape
        # # self.kernel = stf
        # self.kernel = tf.keras.backend.variable(stf, dtype='SparseTensor', name='kernel')
        # print self.kernel.values

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(SparseTF, self).build(input_shape)  # Be sure to call this at the end
        # self.trainable_weights = [self.kernel_vector]

    def call(self, inputs):
        # print self.kernel_vector.shape, inputs.shape
        # print self.kernel_shape, self.kernel_vector
        # print self.nonzero_ind
        # kernel_sparse= tf.S parseTensor(self.nonzero_ind, self.kernel_vector, self.kernel_shape)
        # pr = cProfile.Profile()
        # pr.enable()

        # print self.kernel_vector
        # self.kernel_sparse._values = self.kernel_vector
        tt = tf.scatter_nd(self.nonzero_ind, self.kernel_vector, self.kernel_shape)
        # print tt
        # update  = self.kernel_vector
        # tt= tf.scatter_add(self.kernel_dense, self.nonzero_ind, update)
        # tt= self.kernel_dense
        # tt[self.nonzero_ind].assign( self.kernel_vector)
        # self.kernel_dense[self.nonzero_ind] = self.kernel_vector
        # tt= tf.sparse.transpose(self.kernel_sparse)
        # output = tf.sparse.matmul(tt, tf.transpose(inputs ))
        # output = tf.matmul(tt, inputs )
        output = K.dot(inputs, tt)
        # pr.disable()
        # pr.print_stats(sort="time")
        # return tf.transpose(output)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            # 'kernel_shape': self.kernel_shape,
            'use_bias': self.use_bias,
            'nonzero_ind': np.array(self.nonzero_ind),
            # 'kernel_initializer': initializers.serialize(self.kernel_initializer),
            # 'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),

            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'W_regularizer': regularizers.serialize(self.kernel_regularizer),

        }
        base_config = super(SparseTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def call(self, inputs):
    #     print self.kernel.shape, inputs.shape
    #     tt= tf.sparse.transpose(self.kernel)
    #     output = tf.sparse.matmul(tt, tf.transpose(inputs ))
    #     return tf.transpose(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    # def get_weights(self):
    #
    #     return [self.kernel_vector, self.bias]



def build_model(mapp, n_genes, n_pathways, mapping_l1_reg = 2.5e-3, gat1_l2_reg = 2.5e-3, gat2_l2_reg = 2.5e-3, pool_l1_reg = 2.5e-3, x_dropRate = 0.3, 
                gat1_dropRate = 0.3, gat2_dropRate = 0.3, gat1_channel = 3, gat1_nhead = 3, gat2_channel = 6, pool_channel = 12, dense_channel = 6):
    x_in = Input(shape=(n_genes,))
    x_drop1 = Dropout(x_dropRate)(x_in)
    mapping_layer = SparseTF(n_pathways, mapp, activation='elu', W_regularizer=L1(mapping_l1_reg),
                                name='mapping', kernel_initializer='glorot_uniform',
                                use_bias=True)
    layer2_output = mapping_layer(x_drop1)
    layer2_res=Reshape([n_pathways,1])(layer2_output)
    a_in = Input(shape=(n_pathways,),sparse=True)
    x_1 = GATConv(
        gat1_channel,
        attn_heads=gat1_nhead,
        concat_heads=False,
        activation="tanh",
        return_attn_coef=False,
        dropout_rate=gat1_dropRate,
        kernel_regularizer=l2(gat1_l2_reg),
        attn_kernel_regularizer=l2(gat1_l2_reg),
        bias_regularizer=l2(gat1_l2_reg),
        bias_initializer='glorot_uniform',
    )([layer2_res, a_in])
    x1bn = layers.BatchNormalization()(x_1)
    x_2,att = GATConv(
        gat2_channel,
        attn_heads=1,
        concat_heads=True,
        activation="tanh",
        return_attn_coef=True,
        dropout_rate=gat2_dropRate,
        kernel_regularizer=l2(gat2_l2_reg),
        attn_kernel_regularizer=l2(gat2_l2_reg),
        bias_regularizer=l2(gat2_l2_reg),
        bias_initializer='glorot_uniform',
    )([x1bn, a_in])
    x2bn = layers.BatchNormalization()(x_2)
    attpool=GlobalAttentionPool(pool_channel, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=L1(pool_l1_reg))(x2bn)
    x_fc1 = Dense(dense_channel, activation="elu")(attpool)
    output = Dense(2, activation="softmax")(x_fc1)  # MNIST has 10 classes
    model = Model(inputs=[x_in, a_in], outputs=[output, x2bn, att])
    #optimizer = Adam(lr=1e-1)
    return model

# cross study evaluation
def get_scores_ensemble(model, x, info, a, weight_folder,weight_presuffix,weight_metric,cutoff, outdir,foldnum):
            results=np.zeros([foldnum,x.shape[0],2])
            #for weight_metric in ['tr_f1','va_f1']:
            #for weight_metric in ['tr_f1','tr_f1']:
            for fold in range(0,foldnum):
                model.load_weights(weight_folder+weight_presuffix+str(fold)+'/_'+weight_metric)
                predictions = model([x,a],training=False)[0]
                results[fold,:,:]=predictions
            
            results_mean = np.mean(results, axis=0)
            #print(results_mean)
            results_boolean=results_mean[:,1]>cutoff
            f=open(outdir+"./prediction_results.txt",'w')
            f.write("Patient ID\tPredicted score\tPredicted ICI response\n")
            # print(results_mean)
            # print(results_boolean)
            for i in range(0,results_boolean.shape[0]):
                f.write(info[i]+"\t"+str(results_mean[i,1])+"\t"+str(results_boolean[i])+"\n")
            f.close()

# def get_pathway_relation(datafromnpz,model,weight_folder,weight_presuffix,weight_metric):
def get_pathway_relation(model, pathways, x, info, a, weight_folder,weight_presuffix,weight_metric, outdir, foldnum):
    results=np.zeros([5,x.shape[0],344,344])
    for fold in range(5):
        model.load_weights(weight_folder+weight_presuffix+str(fold)+'/_'+weight_metric)
        results[fold,:,:,:] = np.array(model([x,a],training=False)[2]).reshape(x.shape[0],344,344)
    results_mean = np.mean(results, axis=0)
    print(results_mean.shape)

    for i in range(info.shape[0]):
        pid = info[i]
        pfile_path = os.path.join(outdir, pid)
        if not os.path.exists(pfile_path):
            os.makedirs(pfile_path)
        p_path_relation = results_mean[i]
        df=pd.DataFrame(p_path_relation,columns=pathways, index=pathways)
        pfile = os.path.join(pfile_path, 'Pathway_relation.csv')
        df.to_csv(pfile)
        

    return results_mean



# def get_pathway_importance(datafromnpz,model,weight_folder,weight_presuffix,weight_metric):
def get_pathway_importance(model, pathways, x, info, a, weight_folder,weight_presuffix,weight_metric, outdir, foldnum):
    results=np.zeros([5,x.shape[0],344,4])
    weights=np.zeros([5,4,8])
    predictions=np.zeros([5,x.shape[0],2])
    for fold in range(5):
        model.load_weights(weight_folder+weight_presuffix+str(fold)+'/_'+weight_metric)
        results[fold,:,:,:] = np.array(model([x,a],training=False)[1])
        weights[fold,:,:]=np.array(model.weights[20])
        prediction = model([x,a],training=False)[0]
        predictions[fold,:,:]=prediction
    predictions_mean = np.mean(predictions, axis=0)
    results_mean = np.mean(results, axis=0)
    # print(results_mean.shape)
    weights_mean = np.mean(weights, axis=0)
    # print(weights_mean.shape)
    att=np.matmul(results_mean, weights_mean)  
    att_pathway=np.mean(att,axis=2)
    m=softmax(att_pathway,axis=1)
    print(m.shape)

    pfile = os.path.join(outdir, 'Pathway_weight.csv')
    df=pd.DataFrame(m,columns=pathways, index=info)
    df_transposed = df.T
    df_transposed.to_csv(pfile)

    # for i in range(info.shape[0]):
    #     pid = info[i]
    #     pfile_path = os.path.join(outdir, pid)
    #     if not os.path.exists(pfile_path):
    #         os.makedirs(pfile_path)
    #     p_pathway = m[i]
    #     df=pd.DataFrame(p_pathway,columns='Weight', index=pathways)
    #     pfile = os.path.join(pfile_path, 'Pathway_weight.csv')
    #     df.to_csv(pfile)

    return m,predictions_mean
        

    


def data_process(file_exp, predir):
    pathway_list=[]
    with open(predir+'./Kegg/human_KeggPathwayGene.gmt','r') as reader:
        line = reader.readline().strip()
        while line != '':  # The EOF char is an empty string
            pathway_list.append(line.split('\t')[0])
            line = reader.readline()
    
    pathway_a=np.zeros((len(pathway_list),len(pathway_list)))
    with open(predir+"./Kegg/human_KeggPathwayNet.txt") as keggnet:
        while True:
            line=keggnet.readline().strip()
            if not line:
                break
            pathway1=line.split("\t")[0]
            pathway2=line.split("\t")[1]
            if pathway1==pathway2:
                continue
            cond3 = True
            cond4 = True
            if cond3 and cond4:
                indx1 = pathway_list.index(pathway1)
                indx2 = pathway_list.index(pathway2)
                pathway_a[indx1][indx2] = 1
                pathway_a[indx2][indx1] = 1
    
    col_sym=[]
    with open(predir+'./genelist_8080.txt','r') as reader:
        line = reader.readline().strip()
        while line != '':  # The EOF char is an empty string
            col_sym.append(line)
            line = reader.readline().strip()
    
    pt2exp={}
    f=open(file_exp)
    exp_title = f.readline().strip().split("\t")
    lines=f.readlines()
    exp=[]
    for line in lines:
        line=line.split("\t")
        for i in range(len(line)):
            line[i]=line[i].strip()
            if line[i]=="":
                line[i]="0.0"
        exp.append(line)
    exp=np.array(exp)
    genes=exp[:,0]
    exp=exp[:,1:]
    exp_order=np.zeros((len(col_sym),int(exp.shape[1])))
    for i in range(len(genes)):
        if not genes[i] in col_sym:
            continue
        j=col_sym.index(genes[i])
        exp_order[j,:]=exp[i,:]
    if len(exp_title)>exp_order.shape[1]:
        exp_title=exp_title[1:]
    x=[]
    info=[]
    for i in range(len(exp_title)):
        pid=exp_title[i]
        x.append(exp_order[:,i])
        info.append(pid)
    x_test=np.array(x)
    x_test=stats.zscore(x_test, axis=1)
    x_test=np.array(x_test,dtype=np.float16)
    info_test=np.array(info)
    return [x_test, info_test, pathway_a]



def main():
    parser=argparse.ArgumentParser(description='IRnet: Immunotherapy response prediction using pathway knowledge-informed graph neural network',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input',  dest='inputfile', type=str, help='Gene expression matrix with values separated by Tab. Rows are named with gene symbols. Columns are named with patient IDs.', required=True)
    parser.add_argument('-output',  dest='outputdir', type=str, help='The name of the output directory.', required=True)
    parser.add_argument('-treatment',  dest='drug', type=str, help='Specify the immunotherapy treatment target, either \"anti-PD1\", \"anti-PDL1\", or \"anti-CTLA4\"', required=True)
    
    parser.set_defaults(feature=True)
    args = parser.parse_args()

    inputfile=args.inputfile
    outputdir=args.outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    drug=args.drug

    curdir=os.path.realpath(__file__)
    predir=os.path.dirname(curdir)+"/"
    genes=[]
    with open(predir+'./genelist_8080.txt','r') as reader:
        line = reader.readline().strip()
        while line != '':  # The EOF char is an empty string
            genes.append(line)
            line = reader.readline().strip()
    genes=np.array(genes)
    
    mapp, genes, pathways = get_KEGG_map(genes, filename=predir+'./Kegg/human_KeggPathwayGene.gmt')
    n_genes, n_pathways = mapp.shape

    ###  random parameter  ###
    mapping_l1_reg = 2.5e-3
    gat1_l2_reg = 2.5e-3
    gat2_l2_reg = 2.5e-3
    pool_l1_reg = 2.5e-3
    x_dropRate = 0.5
    gat1_dropRate = 0.4
    gat2_dropRate = 0.4
    gat1_channel = 4
    gat1_nhead = 4
    gat2_channel = 4
    pool_channel = 8
    dense_channel = 8
    batch_size = 10  # Batch size
    
    model=build_model(mapp, n_genes, n_pathways, mapping_l1_reg, gat1_l2_reg, gat2_l2_reg, pool_l1_reg, x_dropRate, 
                    gat1_dropRate, gat2_dropRate, gat1_channel, gat1_nhead, gat2_channel, pool_channel, dense_channel)
    
    [x_test, info_test, pathway_a]=data_process(inputfile,predir)
    weight_folder=predir+"./weights_counts_zscore_focaloss_val/"
    if drug=="anti-PD1":
        weight_presuffix="Liu_bsp_TCGAtransSKCMBLCASTADboostp1_bootstp3_2_steval5_"
    elif drug=="anti-PDL1":
        weight_presuffix="IMvigor_bsp_TCGAtransSKCMBLCASTADboostp1_bootstp3_2_steval5_"
    elif drug=="anti-CTLA4":
        weight_presuffix="Gide_bsp_TCGAtransSKCMBLCASTADboostp1_bootstp3_2_steval5_"
    get_scores_ensemble(model, x_test, info_test, pathway_a, weight_folder,weight_presuffix,"va_f1",0.5, outputdir,5)
    get_pathway_relation(model, pathways, x_test, info_test, pathway_a, weight_folder,weight_presuffix,"va_f1",outputdir,5)
    get_pathway_importance(model, pathways, x_test, info_test, pathway_a, weight_folder,weight_presuffix,"va_f1",outputdir,5)
    

if __name__ == "__main__":
    main()
    

