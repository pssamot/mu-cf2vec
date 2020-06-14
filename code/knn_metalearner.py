from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import pandas as pd
import os

print('OLA MESTRE')
with open('Failed.txt', 'a') as f:
   f.write('vamos comecar CDAE with normalization')
#read in the data using pandas
metadataset = pd.read_csv('metadataset_k_20.csv')

#---- SET INPUTS -----
#remove the original_id and the target (first_place)

scaler = StandardScaler()
#Compute the mean and std to be used for later scaling.
scaler.fit(metadataset.drop(columns=['first_place','original_id']))
# Perform standardization by centering and scaling
inputs_transform = scaler.transform(metadataset.drop(columns=['first_place','original_id']))
inputs = pd.DataFrame(inputs_transform)

#---- SET TARGET -----
target = metadataset['first_place'].values

#create new a knn model
knn2 = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1, 201)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

start_train = time.time()
knn_gscv.fit(inputs, target)
with open('Failed.txt', 'a') as f:
   f.write("Training took", time.time() - start_train)
   f.write(" BEST : ", knn_gscv.best_params_)

print('FIM MESTRE')
