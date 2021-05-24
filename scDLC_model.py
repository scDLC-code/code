# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:48:09 2019

@author: yb
"""

import tensorflow as tf
import sklearn
import numpy as np
import math

########################
class scDLC_scRNAseqClassifier:
    def __init__(self, num_classes,num_steps, batch_size=11,
                lstm_size=64, num_layers=2, grad_clip=5, train_keep_prob =0.3):
        
        self.session = tf.Session()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.buildgraph()

        tf.reset_default_graph() 

  
    def buildgraph(self):  
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()
        

        
    def build_inputs(self):
             
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_steps,1], name='inputs')
            self.targets = tf.placeholder(tf.int64, shape=self.batch_size, name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')            
            self.lr = tf.placeholder(tf.float32)
            self.lstm_inputs = tf.layers.dense(self.inputs,units=2*self.num_steps,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    def build_lstm(self):
        
        def get_a_cell(lstm_size, keep_prob):  
            lstm = tf.nn.rnn_cell.GRUCell(num_units=lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop
        
        with tf.name_scope('lstm'):
             cell = tf.nn.rnn_cell.MultiRNNCell(
                     [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
             )

             self.initial_state = cell.zero_state(self.batch_size, tf.float32)
             self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell,self.lstm_inputs, initial_state=self.initial_state)
             self.logits = tf.layers.dense(self.final_state[1], self.num_classes)


    def build_loss(self):
        
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, axis=1),self.targets),dtype=tf.float32))


    def build_optimizer(self):
    
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars), self.grad_clip)
        train_op = tf.train.RMSPropOptimizer(self.lr)
        self.optimizer = train_op.apply_gradients(zip(grads,tvars))
        
        
    
    def train(self, X, Y, val_data=None,n_epoch=20,exp_decay=True,isshuffle=True):
        if val_data is None:
            print("Train %d samples" % len(X))
        else:
            print("Train %d samples|Test %d samples" % (len(X), len(val_data[0])))            
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[], 'val_pred':[]}
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            global_step = 0
            for epoch in range(n_epoch):            
                if isshuffle:
                    X, Y = sklearn.utils.shuffle(X,Y)
                for local_step, (X_batch, Y_batch) in enumerate(self.a_batch(X=X, batch_size=self.batch_size,Y=Y)):
                    lr = self.decrease_lr(exp_decay,global_step, n_epoch, len(X), self.batch_size)
                    _, loss, acc = sess.run([self.optimizer, self.loss, self.acc],
                                                 feed_dict={self.inputs:X_batch,
                                                            self.targets:Y_batch,
                                                            self.lr:lr,
                                                            self.keep_prob:self.train_keep_prob})
                    global_step += 1
                    if local_step % 20 ==0:
                        print("Epoch %d | Step %d | Train loss: %.4f | Train acc: %.4f | lr: %.4f" %(
                                epoch+1, local_step, loss, acc, lr
                                ))
                        log['loss'].append(loss)
                        log['acc'].append(acc)
                
                if val_data is not None:
                    val_loss_list, val_acc_list, val_pred_list = [],[],[]
                    for (X_test_batch, Y_test_batch) in self.a_batch(X=val_data[0], batch_size=self.batch_size,Y=val_data[1]):
                        v_loss, v_acc, v_pred= sess.run([self.loss, self.acc, self.logits],feed_dict={
                                self.inputs: X_test_batch, 
                                self.targets:Y_test_batch,
                                self.keep_prob: 1.0
                        })
                        val_loss_list.append(v_loss)
                        val_acc_list.append(v_acc)
                        val_pred_list.append(v_pred)
                    val_loss, val_acc = self.list_avg(val_loss_list), self.list_avg(val_acc_list)
                    log['val_loss'].append(val_loss)
                    log['val_acc'].append(val_acc)
                    log['val_pred']= val_pred_list
                    print("val_data loss: %.4f | val_data acc: %.4f" % (val_loss, val_acc))
            self.saver.save(sess,"E:/Machine learning/neural network/RNN/RNA/model.ckpt")
            return log



    def predict(self,X_test):
        with self.session as sess:
            batch_pred_list = []
            for X_test_batch in self.a_batch(X=X_test, batch_size=self.batch_size):
                batch_pred = sess.run(self.logits,feed_dict={
                        self.inputs: X_test_batch,
                        self.keep_prob: 1.0
                })
                batch_pred_list.append(batch_pred)
            return np.argmax(np.vstack(batch_pred_list), 1)

  
    
    def a_batch(self,X, batch_size, Y=None):
        n_batch = int(len(X)/batch_size)
        X = X[:n_batch*batch_size]
        for i in range(0, len(X), batch_size):
            if Y is not None:
                Y_batch = Y[i:i+batch_size]
            X_batch = X[i:i+batch_size]     
            X_batch = np.reshape(X_batch,(batch_size,self.num_steps,1))
            yield X_batch, Y_batch
            
            
    def list_avg(self,l):
        return sum(l)/len(l)
    
    def decrease_lr(self,exp_decay,global_step,n_epoch,len_x,batch_size):
        if exp_decay:
            max_lr = 0.005
            min_lr = 0.001
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len_x/batch_size)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr
    
    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
    
################################
        