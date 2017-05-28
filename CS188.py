#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:19:43 2017

@author: vikashsingh
"""
#CS 188 Medical Imaging Project 
import pandas as pd 
import sklearn 
import numpy as np 
import matplotlib as plt 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
#from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model, datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split 

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
Y= Y.as_matrix()

AUC=[]

kfold = StratifiedKFold(y = Y, n_folds = 10, shuffle = True, random_state=5)

for i, (train, test) in enumerate(kfold):
    logreg = linear_model.LogisticRegression(C = 1e5)
    logreg.fit(X[train], Y[train])
    predictionsproba=logreg.predict_proba(X[test])[:,1]
    AUC.append(roc_auc_score(Y[test], predictionsproba)) 
    


print np.mean(AUC) 


