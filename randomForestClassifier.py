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

kfold = StratifiedKFold(y = Y, n_folds = 10, shuffle = True, random_state = 3)
AUC=[]
globpred=[]
globy_test=[]

for i, (train, test) in enumerate(kfold):    
    forest = RandomForestClassifier(n_estimators = 100)
    forest.fit(X[train], Y[train])
    predictionsproba = forest.predict_proba(X[test])[:,1]
    
    
    AUC.append(roc_auc_score(Y[test], predictionsproba))
    globpred += predictionsproba.tolist()
    globy_test += Y[test].tolist()
    


print(np.mean(AUC))
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
