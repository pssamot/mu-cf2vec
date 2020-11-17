from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm
import pandas as pd
import numpy as np
import sys
import os

is_vae = False
# Just choose the name of the dataset directory
DATA_DIR = '/Users/tomas/Documents/FEUP/Tese/data/ml-20m/processed_70_10_20'
if is_vae:
	print('--- IS VAE --- ')
	PARSE_DATA_DIR = os.path.join(DATA_DIR, 'embeddings/vae')
else:
	PARSE_DATA_DIR = os.path.join(DATA_DIR, 'embeddings/cdae')

file = '200_fac_metadataset_k_20.csv'
metadataset = pd.read_csv(os.path.join(PARSE_DATA_DIR, file ))

#als:0
#bpr:1
#lmf:2
#most_pop_3
#zeros:4
target_pre = metadataset['first_place'].values 
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target_pre)

normalize = False
if normalize:
	#---- SET INPUTS -----
	scaler = StandardScaler()
	#Compute the mean and std to be used for later scaling.
	scaler.fit(metadataset.drop(columns=['first_place','original_id']))
	# Perform standardization by centering and scaling
	inputs_transform = scaler.transform(metadataset.drop(columns=['first_place','original_id']))
	inputs = pd.DataFrame(inputs_transform)
	print('--- IS NORMALIZED --- ')
	inputs.head()
else:
	inputs = metadataset.drop(columns=['first_place','original_id'])

param_grid = {
    'bootstrap': [True],
    'max_depth': [10 ,75, 150],
    'max_features': ['auto', 'log2'],
    'n_estimators': [100, 250, 1000],
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 2, n_jobs = -1, verbose = 10)
grid_search.fit(inputs, target) 

print(grid_search.best_params_)