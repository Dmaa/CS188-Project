#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:19:43 2017

@author: vikashsingh
"""
#CS 188 Medical Imaging Project 
import pandas as pd 
import sklearn 
import numpy 
import matplotlib as plt 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model, datasets

rows_to_train = 50000
data = pd.read_csv('image_data.csv', header=None) 

print("Data loaded New")

print(data.shape)
data=data.dropna()
print(data.shape)

X=data.iloc[0:rows_to_train,4:622] 
print ("PassedNew")

Y=data.iloc[0:rows_to_train, 622]
X = X.as_matrix()
Y= Y.as_matrix()

kfold = StratifiedKFold(y = Y, n_folds = 2, shuffle = True, random_state = 5)

for i, (train, test) in enumerate(kfold):
    logreg = linear_model.LogisticRegression(C = 1e5)
    logreg.fit(X[train], Y[train])
    predictions = logreg.predict_proba(X[test])
    
    
print("Gittest")

print("Passed2")