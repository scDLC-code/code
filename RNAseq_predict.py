# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:11:56 2019

@author: yb
"""

import tensorflow as tf

from RNN_RNAseq import RNN_RNAseqClassifier
#from sklearn.model_selection import train_test_split
#import pandas as pd
#import numpy as np

FLAGS = tf.flags.FLAGS
import os


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('lstm_size', 256, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('level_size', 777, 'level_size')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('checkpoint_path', 'E:/Machine learning/neural network/RNN/RNA/', 'checkpoint path' )


def main(_):
    
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
             tf.train.latest_checkpoint(FLAGS.checkpoint_path)
             
    model = RNN_RNAseqClassifier(lstm_size=FLAGS.lstm_size,
                                 num_seqs=FLAGS.num_seqs, 
                                 level_size=FLAGS.level_size,
                                 num_layers=FLAGS.num_layers,
                                 use_embedding=FLAGS.use_embedding,
                                 embedding_size=FLAGS.embedding_size)
    
    model.load(FLAGS.checkpoint_path)
    
    model.predict(X_test)

    
if __name__ == '__main__':
    tf.app.run()