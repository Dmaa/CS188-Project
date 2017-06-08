#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:19:43 2017

@author: dharma naidu and vikashsingh
"""
#CS 188 Medical Imaging Project 
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

#make a kfold data structure with 10 folds, a shuffle, and a random seed of 5
kfold = StratifiedKFold(y = Y, n_folds = 10, shuffle = True, random_state=5) 

#empty global prediction array to store prediction probability of each k fold (Y value in AUC graph)
globpred=[]

#empty global prediction array to store test probability of each k fold (X value in AUC graph)
globy_test=[]



for i, (train, test) in enumerate(kfold):  #for each fold in the kfold  
    print(i)
    gnb = GaussianNB() #create a new Gaussian Naive Bayes model
    #print(test);
    gnb.fit(X[train], Y[train]) #train the model using data from X and Y
    predictionsproba = gnb.predict_proba(X[test])[:,1]#store prediction probability in predictproba

    
    AUC.append(roc_auc_score(Y[test], predictionsproba)) #append predict proba to our auc array
    globpred += predictionsproba.tolist() #add predictproba info to globred (for use in AUC graph)
    globy_test += Y[test].tolist() #add Y test info to globytest(for use in AUC graph)

print(np.mean(AUC)) #print the mean AUC of all our k folds

#graph the AUC graph
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
