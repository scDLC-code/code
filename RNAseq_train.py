# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:36:27 2019

@author: yb
"""

import tensorflow as tf
#from RNN_RNAseq_1 import RNN_RNAseqClassifier_1
from RNN_RNAseq_test import RNN_RNAseqClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'frist', 'name of the model')
tf.flags.DEFINE_integer('num_classes', 2, 'num_classes')
tf.flags.DEFINE_integer('num_steps', 100, 'num_steps')
tf.flags.DEFINE_integer('batch_size', 11, 'name of size in one batch')
tf.flags.DEFINE_integer('level_size', 3026, 'level_size')
tf.flags.DEFINE_integer('lstm_size', 64, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('n_epoch', 30, 'n_epoch')
tf.flags.DEFINE_float('train_keep_prob', 0.3, 'dropout rate during training')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 64, 'size of embedding')


tf.reset_default_graph() 

def main(_):
                
#simulation/simulation-data GSE121364_100
    data = pd.read_csv("E:\\Machine learning\\neural network\\RNN\\RNA\\Deep-classifier\\result\\realdata\\data\\GSE123454_1500.csv",header=0,index_col=0)   #real data 
    n=data.shape[0]-1#基因个数
    FLAGS.num_steps=300
    data=data.T
    X = data.iloc[:,0:300]

    X = np.array(X)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    X = scaler.fit_transform(X)
#    FLAGS.level_size=len(np.unique(X))+1
    FLAGS.level_size=X.max().max()+1
    Y=data.iloc[:,n]-1
    FLAGS.num_classes=Y.max()+1
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)#random_state=20)
    
    test_data=list([X_test,y_test])
    
    model = RNN_RNAseqClassifier(num_classes=FLAGS.num_classes,
                                 num_steps=FLAGS.num_steps,
                                 batch_size=FLAGS.batch_size,             
                                 lstm_size=FLAGS.lstm_size,
                                 num_layers=FLAGS.num_layers,
                                 #level_size=FLAGS.level_size,#303、3026
                                 #use_embedding=FLAGS.use_embedding,
                                 #embedding_size=FLAGS.embedding_size,
                                 train_keep_prob=FLAGS.train_keep_prob)
        
    log=model.train(X=X_train, Y=y_train,val_data=test_data,n_epoch=FLAGS.n_epoch,exp_decay=True)
    return 1 - max(log['val_acc'])
    
    
if __name__ == '__main__':
    tf.app.run()
 
###################9924