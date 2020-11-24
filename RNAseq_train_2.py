# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:19:02 2020

@author: yb
"""


import tensorflow as tf
#from RNN_RNAseq_1 import RNN_RNAseqClassifier_1
from RNN_RNAseq_test_2 import RNN_RNAseqClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

tf.reset_default_graph() 


                
#simulation/simulation-data GSE121364_100
data = pd.read_csv("Deep-classifier//data//simulation_1.csv",header=0,index_col=0)   #real data 
n=data.shape[0]-1#基因个数
num_steps=n
data=data.T
X = data.iloc[:,0:n]

X = np.array(X)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    X = scaler.fit_transform(X)
#    FLAGS.level_size=len(np.unique(X))+1
level_size=X.max().max()+1
Y=data.iloc[:,n]-1
num_classes=Y.max()+1

result3454=[]
size=[10,20,40,80,160,320,640]

for i in range(len(size)):
    
    for c in range(1):
        
        print('3069:',size[i],c+1)

        result=[]
        X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=len(X)-size[i])#random_state=20)
        
        test_data=list([X_test,y_test])
        
        model = RNN_RNAseqClassifier(num_classes=num_classes,
                                     num_steps=100,
                                     batch_size=11,             
                                     lstm_size=64,
                                     num_layers=2,
                                     #level_size=FLAGS.level_size,#303、3026
                                     #use_embedding=FLAGS.use_embedding,
                                     #embedding_size=FLAGS.embedding_size,
                                     train_keep_prob=0.3)
            
        log=model.train(X=X_train, Y=y_train,val_data=test_data,n_epoch=40,exp_decay=True)
        pred = np.reshape(np.array(log['val_pred']),(-1,4))
        y_pred = np.argmax(pred,axis=1)
        y_pred = pd.Series(y_pred)
        y_test = y_test[0:len(pred)]
        
        ####################################
        fpr,tpr,threshold = roc_curve(y_test, y_pred) ###计算真正率和假正率
        roc_auc = auc(fpr,tpr)
        result.append(roc_auc)
        
    result = np.mean(result)
        ###计算auc的值
    result3454.append(result)

#################################################    

 