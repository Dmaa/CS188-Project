 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 12:11:53 2017
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
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
kfold = StratifiedKFold(y = Y, n_folds = 2, shuffle = True, random_state=5) 

#empty global prediction array to store prediction probability of each k fold (Y value in AUC graph)
globpred=[]

#empty global prediction array to store test probability of each k fold (X value in AUC graph)
globy_test=[]

for i, (train, test) in enumerate(kfold):    
    forest = RandomForestClassifier(n_estimators = 100) #make a random forest with 100 estimators
    forest.fit(X[train], Y[train]) #fit the data using X and Y
    predictionsproba = forest.predict_proba(X[test])[:,1] #make predictions for the model
    
    
    AUC.append(roc_auc_score(Y[test], predictionsproba)) #append AUC scores to AUC
    globpred += predictionsproba.tolist() #add predictproba info to globred (for use in AUC graph)
    globy_test += Y[test].tolist() #add Y test info to globytest(for use in AUC graph)
    



print(np.mean(AUC)) #print the mean AUC of all our k folds

#graph the AUC graph
false_positive_rate, true_positive_rate, thresholds=roc_curve(globy_test, globpred)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate') 
plt.show()
plt.savefig("savedFigs/randomForest")
