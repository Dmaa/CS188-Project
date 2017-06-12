#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:19:43 2017

@author: dharma naid and vikashsingh
"""
#CS 188 Medical Imaging Project 
import os
os.environ['KERAS_BACKEND']='theano'
import theano 
from keras.utils import np_utils
import pandas as pd 
import sklearn 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold 
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD



rows_to_train = 50000
data = pd.read_csv('cs188TD.csv', header=None) 

print("Data loaded New")

print(data.shape)
data=data.dropna()
print(data.shape)

X=data.iloc[0:rows_to_train,4:622] 
print ("PassedNew")

Y=data.iloc[0:rows_to_train, 622]
#convert to matrix to avoid errors 
X = X.as_matrix()
Y = Y.as_matrix()

AUC=[]
cvscores=[]

kfold = StratifiedKFold(y = Y, n_folds = 10, shuffle = True, random_state = 3) 

    
model = Sequential()
AUC=[]
globpred=[]
globy_test=[]

for i, (train, test) in enumerate(kfold):
    model = Sequential()
    model.add(Dense(10, input_dim=618,init='uniform', activation='sigmoid'))  
    model.add(Dense(5,init='uniform', activation='sigmoid'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    print("Model started")

#odel.compile(loss='mse',  optimizer='adam', metrics=['accuracy'])
#odel.compile(loss='mean_squared_error',  optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model compiled")
    #verbose=0
    model.fit(X[train],Y[train], epochs=10, batch_size=9, verbose=0) 
    
    predictionsproba = model.predict(X[test]) 
    #print(roc_auc_score(Y[test], predictionsproba))
    AUC.append(roc_auc_score(Y[test], predictionsproba)) 
    globpred += predictionsproba.tolist()
    globy_test += Y[test].tolist()
    
print "The AUC is"
print(roc_auc_score(globy_test, globpred)) 

false_positive_rate, true_positive_rate, thresholds=roc_curve(globy_test, globpred) 
roc_auc = auc(false_positive_rate, true_positive_rate) 
plt.title('Receiver Operating Characteristic Neural Network')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate') 
plt.show()