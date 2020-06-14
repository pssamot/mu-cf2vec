import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt

# sklearn tools for model training and assesment
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import (roc_curve, auc, accuracy_score)

#read in the data using pandas
metadataset = pd.read_csv('vae/metadataset_k_20.csv')


#PARSE TARGET
#als:0
#bpr:1
#lmf:2
#most_pop_3
#zeros:4
from sklearn.preprocessing import LabelEncoder
target_pre = metadataset['first_place'].values
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target_pre)
np.unique(target)

#PARSE INPUTS
normalize = True
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


# params
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': {'multi_logloss', 'auc'},
    'importance_type': 'gain',
    'num_class': 5,
    'max_bin': 255,
    'max_depth':  -1,
    'is_unbalance': False,
}

#params to search
gridParams = {
    'learning_rate': np.arange(0.1, 0.2, 0.01),
    'num_leaves': np.arange(10, 100, 10),
    'colsample_bytree': np.arange(0.1, 0.5, 0.05),
    'reg_alpha': np.arange(0.05, 0.3, 0.05),
    'reg_lambda': np.arange(0.05, 0.3, 0.05),
}


#model
mdl = lgb.LGBMClassifier(
    objective= 'multiclass',
    metric= 'multi_logloss',
    importance_type= 'gain',
    num_class= 5,
    max_bin= 255,
    max_depth= -1,
    is_unbalance= False, 
    n_jobs = -1)

#fit
grid = GridSearchCV(mdl, gridParams, verbose=2, cv=5, n_jobs=-1)
grid.fit(inputs, target)
print(grid.best_params_)




