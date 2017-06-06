#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:58:22 2017

@author: dharma naidu and vikashsingh
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
data = pd.read_csv('~/Desktop/cs188TD.csv', header=None)  #Load the data from the csv file using pandas, and store it in "data"

print("Data loaded New")

print(data.shape)
data=data.dropna() #Drop empty rows just in case data isn't formatted perfectly
print(data.shape)

X=data.iloc[0:rows_to_train,4:622] #Store first 622 column, or our training parameters, in "X"

Y=data.iloc[0:rows_to_train, 622] #store last column, or result column, in y

#convert to matrix to avoid errors 
X = X.as_matrix()
Y= Y.as_matrix()

AUC=[] #empty auc array to store AUCs of each k fold

#empty global prediction array to store prediction probability of each k fold (Y value in AUC graph)
globpred=[]

#empty global prediction array to store test probability of each k fold (X value in AUC graph)
globy_test=[]

#Store unique patient values in a list 
uniquepatients=data[0].unique()
#Randomly sample six patients to be used for test set 
for x in range(0,10): #10 means 10 k folds
    testpatients=random.sample(uniquepatients, 6)
    print(testpatients)
    
    testdata=data[data[0].isin(testpatients)]
    traindata=data[-(data[0].isin(testpatients))]
    #print(testdata.shape)
    #print(traindata.shape)
    #X represents model input, Y represents binary labels 
    traindataX=traindata.iloc[:,4:622] 
    traindataY=traindata.iloc[:,622]
    testdataX=testdata.iloc[:,4:622]
    testdataY=testdata.iloc[:,622] 
    #gnb=GaussianNB()
    #gnb.fit(traindataX, traindataY)
    
    #create the layers of the neural network
    model = Sequential()
    model.add(Dense(60, input_dim=618,init='uniform', activation='relu'))  
    model.add(Dense(39,init='uniform', activation='sigmoid'))
    model.add(Dense(20,init='uniform', activation='sigmoid'))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    print("Model started") 

#model.compile(loss='mse',  optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mean_squared_error',  optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model compiled")
    #verbose=0
    #fit the model
    model.fit(np.array(traindataX), np.array(traindataY), nb_epoch=100, batch_size=9, verbose=0)
    
    #test and evaluate the model, create AUC
    predictionsproba = model.predict_proba(testdataX)[:,1]
    print(roc_auc_score(testdataY, predictionsproba))
    AUC.append(roc_auc_score(testdataY, predictionsproba))
    globpred+=predictionsproba.tolist() #add predictproba info to globred (for use in AUC graph)
    globy_test+=testdataY.tolist() #add Y test info to globytest(for use in AUC graph)
    
print "The AUC is"
print(roc_auc_score(globy_test, globpred))
#print np.mean(AUC)
false_positive_rate, true_positive_rate, thresholds=roc_curve(globy_test, globpred)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic GNB')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate') 
plt.show()
#plt.savefig("savedFigs/gnb") 