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


def main():
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

  param_grid = {'C': [0.1, 1, 10, 500],  
                'gamma': [1, 0.1, 0.01], 
                'kernel': ['rbf']}
  is_smote = False
  kf = KFold(n_splits=5)
  kf.get_n_splits()
  print(kf)
  i = 1 
  reports = []
  for train_index, test_index in kf.split(inputs):
      print('iteration: ', i)
      #get data fold
      X_train, X_test = inputs.iloc[train_index], inputs.iloc[test_index]
      y_train, y_test = target[train_index], target[test_index]
      
      #start model 
      print('fit')
      svm.SVC()
      clf = svm.SVC(
          kernel='linear',
          C=params['C'],
          gamma=params['gamma'],
          kerner=params['kernel'],
          verbose=True) # Linear Kernel
      
      if is_smote:
          print('dataset shape %s' % Counter(y_train))
          sm = SMOTE(random_state=42)
          X_train_re, y_train_re = sm.fit_resample(X_train, y_train)
          print('Resampled dataset shape %s' % Counter(y_train_re))

          clf.fit(X_train_re, y_train_re)
          print('predict')
      else:
          clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      
      report = classification_report(y_test, 
                                 y_pred, 
                                 target_names=np.unique(metadataset['first_place'].values),
                                output_dict=True)
      reports.append(report)
      print('end: ', i)
      i+=1
  avg_reports = report_average(reports)
  print_report(avg_reports)
  pass

def print_report(avg_reports):
    from prettytable import PrettyTable
    x = PrettyTable()

    x.field_names = ["Algorithm", "Precision", "Recall", "F1"]

    for label in avg_reports.keys():
        if label in 'accuracy':
            x.add_row(['---','---','---','---'])
            continue
        x.add_row([label, 
                   avg_reports[label]['precision'], 
                   avg_reports[label]['recall'], 
                   avg_reports[label]['f1-score']])


    print(x)
    print('Accuracy: ', avg_reports['accuracy'])
    
def report_average(reports):
    mean_dict = dict()
    for label in reports[0].keys():
        dictionary = dict()

        if label in 'accuracy':
            mean_dict[label] = sum(d[label] for d in reports) / len(reports)
            continue

        for key in reports[0][label].keys():
            dictionary[key] = sum(d[label][key] for d in reports) / len(reports)
        mean_dict[label] = dictionary

    return mean_dict


















