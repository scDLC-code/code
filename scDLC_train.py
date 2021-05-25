# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:36:27 2019

@author: yb
"""

import rpy2.robjects as robjects
import tensorflow as tf
from scDLC_model import scDLC_scRNAseqClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

robjects.r.source('selectgene.r')

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('num_classes', None, 'num_classes')
tf.flags.DEFINE_integer('num_steps', 100, 'num_steps')
tf.flags.DEFINE_integer('batch_size', 11, 'name of size in one batch')
tf.flags.DEFINE_integer('lstm_size', 64, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('n_epoch', 30, 'n_epoch')
tf.flags.DEFINE_float('train_keep_prob', 0.3, 'dropout rate during training')


tf.reset_default_graph() 

def main(_):
                
    data = pd.read_csv("data.csv",header=0,index_col=0)
    n=data.shape[0]-1
    data=data.T
    X = data.iloc[:,0:FLAGS.num_steps]
    X = np.array(X)
    Y=data.iloc[:,n]-1
    FLAGS.num_classes=Y.max()+1
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
    
    test_data=list([X_test,y_test])
    
    model = scDLC_scRNAseqClassifier(num_classes=FLAGS.num_classes,
                                 num_steps=FLAGS.num_steps,
                                 batch_size=FLAGS.batch_size,             
                                 lstm_size=FLAGS.lstm_size,
                                 num_layers=FLAGS.num_layers,
                                 train_keep_prob=FLAGS.train_keep_prob)
        
    log=model.train(X=X_train, Y=y_train,val_data=test_data,n_epoch=FLAGS.n_epoch,exp_decay=True)
    return log
    
    
if __name__ == '__main__':
    tf.app.run()
 