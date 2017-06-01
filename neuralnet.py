#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:19:43 2017

@author: vikashsingh
"""
#CS 188 Medical Imaging Project 
import os
os.environ['KERAS_BACKEND']='theano'
import theano 
from keras.utils import np_utils
import pandas as pd 
import sklearn 
import numpy as np 
import matplotlib as plt 
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
data = pd.read_csv('~/Desktop/cs188TD.csv', header=None) 

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

kfold = StratifiedKFold(y = Y, n_folds = 3, shuffle = True, random_state = 3) 

    
model = Sequential()

for i, (train, test) in enumerate(kfold):
    model = Sequential()
    model.add(Dense(60, input_dim=618,init='uniform', activation='relu'))  
    model.add(Dense(39,init='uniform', activation='sigmoid'))
    model.add(Dense(20,init='uniform', activation='sigmoid'))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    print("Model started") 

#odel.compile(loss='mse',  optimizer='adam', metrics=['accuracy'])
#odel.compile(loss='mean_squared_error',  optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model compiled")
    #verbose=0
    model.fit(X[train], Y[train], nb_epoch=100, batch_size=9, verbose=0)
#evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    predictions=model.predict(X[test])
    print("The AUC is: ")
    print(roc_auc_score(Y[test],predictions))
    AUC.append(roc_auc_score(Y[test],predictions))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) 
    cvscores.append(scores[1] * 100)

print(np.mean(AUC)) 

