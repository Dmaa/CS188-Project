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

data = pd.read_csv('~/Desktop/cs188TD.csv') 

print("Data loaded")

print data.shape
data=data.dropna()
print data.shape

X=data.iloc[0:49999,0:622] 
print ("Passed")

Y=data.iloc[0:500, 620:622]
#rint("X shape")

#Y=data['Outcome']

#print(X.shape)
#print(Y.shape)
#X=X.dropna()
#Y=Y.dropna()


#X=X.as_matrix()
#Y=Y.as_matrix()


