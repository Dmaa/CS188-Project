#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:58:22 2017

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
data = pd.read_csv('~/Desktop/cs188TD.csv', header=None)  

print("Data loaded New") 

print(data.shape)
data=data.dropna()
print(data.shape)

X=data.iloc[0:rows_to_train,4:622] 
print ("Data Loaded")

Y=data.iloc[0:rows_to_train, 622] 
#convert to matrix to avoid errors 
X = X.as_matrix()
Y = Y.as_matrix()
AUC=[]
#Store unique patient values in a list 
uniquepatients=data[0].unique()
#Randomly sample six patients to be used for test set 
globpred=[]
globy_test=[]
for x in range(0,10):
    testlength=0
    y=6
    #Ensure that the test set has at least 5000 cases for each fold of validation
    while(testlength<5000): 
        
        testpatients=random.sample(uniquepatients, y) 
        #print(testpatients)
    
        testdata=data[data[0].isin(testpatients)] 
        testlength=testdata.shape[0]
        y+=1; 
    

    #testdata=data[data[0].isin(testpatients)]
    #print(testpatients)
    #print(testdata.shape[0]) 
    traindata=data[-(data[0].isin(testpatients))]  
    #print(testdata.shape)
    #print(traindata.shape)
    #X represents model input, Y represents binary labels 
    traindataX=traindata.iloc[:,4:622] 
    traindataY=traindata.iloc[:,622]
    testdataX=testdata.iloc[:,4:622]
    testdataY=testdata.iloc[:,622]

    #Cast dataframes into numpy arrays to allow for Keras processing 
    traindataX=np.array(traindataX)
    traindataY=np.array(traindataY)
    testdataX=np.array(testdataX)
    testdataY=np.array(testdataY) 
    #gnb=GaussianNB()
    #gnb.fit(traindataX, traindataY)
    
    model = Sequential()
    model.add(Dense(30, input_dim=618,init='uniform', activation='sigmoid')) 
    model.add(Dropout(.2))
    model.add(Dense(50, init='uniform', activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(10, init='uniform', activation='sigmoid'))
    model.add(Dense(20, init='uniform', activation='sigmoid'))
    model.add(Dense(5, init='uniform', activation='sigmoid'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))  
    print("Model started") 

#odel.compile(loss='mse',  optimizer='adam', metrics=['accuracy'])
#odel.compile(loss='mean_squared_error',  optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
    print("Model compiled")
    #verbose=0
    model.fit(traindataX,traindataY, nb_epoch=10, batch_size=9, verbose=0) 
    
    predictionsproba = model.predict(testdataX) 
    #print(roc_auc_score(testdataY, predictionsproba))
    AUC.append(roc_auc_score(testdataY, predictionsproba)) 
    globpred+=predictionsproba.tolist()
    globy_test+=testdataY.tolist()
print "The AUC is"
print(roc_auc_score(globy_test, globpred)) 
#print np.mean(AUC)
false_positive_rate, true_positive_rate, thresholds=roc_curve(globy_test, globpred) 
roc_auc = auc(false_positive_rate, true_positive_rate) 
plt.title('Receiver Operating Characteristic NN')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate') 
plt.show()
#plt.savefig("AllRELUNN.pdf") 