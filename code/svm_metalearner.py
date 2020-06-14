from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm
import pandas as pd
import numpy as np
import sys
import os


is_vae = True
# Just choose the name of the dataset directory
DATA_DIR = '/Users/tomas/Documents/FEUP/Tese/data/ml-20m/processed_70_10_20'
if is_vae:
    PARSE_DATA_DIR = os.path.join(DATA_DIR, 'embeddings/vae')
else:
    PARSE_DATA_DIR = os.path.join(DATA_DIR, 'embeddings/cdae')

file = 'metadataset_k_20.csv'
#read in the data using pandas
print('ler')
metadataset = pd.read_csv(os.path.join(PARSE_DATA_DIR, file ))


#als:0
#bpr:1
#lmf:2
#most_pop_3
#zeros:4
print('encode')
target_pre = metadataset['first_place'].values 
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target_pre)

print('normalize')
normalize = False
if normalize:
  #---- SET INPUTS -----
  scaler = StandardScaler()
  #Compute the mean and std to be used for later scaling.
  scaler.fit(metadataset.drop(columns=['first_place','original_id']))
  # Perform standardization by centering and scaling
  inputs_transform = scaler.transform(metadataset.drop(columns=['first_place','original_id']))
  inputs = pd.DataFrame(inputs_transform)
  inputs.head()
else:
  inputs = metadataset.drop(columns=['first_place','original_id'])

param_grid = {'C': [0.1, 1, 10, 500],  
              'gamma': [1, 0.1, 0.01], 
              'kernel': ['rbf']}

print('fit')         
grid = GridSearchCV(svm.SVC(), param_grid, cv=2, verbose = 10, n_jobs=-1) 
grid.fit(inputs, target) 

print(grid.best_params_)


