import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#PARSE TARGET
#als:0
#bpr:1
#lmf:2
#most_pop_3
#zeros:4
def main():
    # true if vae
    if True :
        metadataset = pd.read_csv('vae/metadataset_k_20.csv')
    else:
        metadataset = pd.read_csv('cdae/200_fac_metadataset_k_20.csv')
    target_pre = metadataset['first_place'].values
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target_pre)
    keys = label_encoder.classes_
    values = label_encoder.transform(keys)
    labels = dict(zip(values,keys))
    #PARSE INPUTS
    normalize =False
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
    params = {
            'colsample_bytree': 0.25, 
            'learning_rate': 0.01, 
            'num_leaves': 100, 
            'reg_alpha': 0.25, 
            'reg_lambda': 0.25
            }
    cvParams = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 5,
            'is_unbalance': False,
            'importance_type': 'gain',
            'learning_rate': params['learning_rate'],
            'max_bin': 40,
            'num_leaves': params['num_leaves'], 
            'max_depth': -1,
            'colsample_bytree': params['colsample_bytree'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
            'random_state': 42,
            }
    kf = KFold(n_splits=5)
    kf.get_n_splits()
    np.set_printoptions(precision=4)
    i = 1 
    reports = []
    matrix = []

    for train_index, test_index in kf.split(inputs):
        print('\n ---iteration: ', i)
        
        #get data fold
        print('get data')
        X_train, X_test = inputs.iloc[train_index], inputs.iloc[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
        #build dataset
        print('build dataset')
        metadataset_lgb = lgb.Dataset(data = X_train, label = y_train)
        #train model
        print('train')
        model = lgb.train(cvParams, metadataset_lgb, num_boost_round=1000, verbose_eval=5)
        
        #make predictions
        print('predict')
        preds = model.predict(X_test)
        predictions = []
        for x in preds:
            predictions.append(np.argmax(x))
        
        report = classification_report(y_test, predictions,target_names=np.unique(metadataset['first_place'].values),output_dict=True)
        reports.append(report)
        confusion = confusion_matrix(y_test,predictions)
        matrix.append(confusion)
        np.set_printoptions(suppress=True)
        i+=1
    avg_reports = report_average(reports)
    print_report(avg_reports)
    print_confusion(np.mean( np.array(matrix),axis=0 ),labels)



def print_confusion(values, classes):
    from prettytable import PrettyTable
    x = PrettyTable()
    print(classes)
    
    names = []
    names.append('algorithm')
    names = names + list(classes.values())
    x.field_names = names   
    
    i = 0
    for row in values:
        #row = np.array(row)
        row = [classes[i], row[0],row[1],row[2],row[3],row[4]]
        #r.append(classes[i])
        #r = r + row
        #row = np.insert(row,0,'als') 
        x.add_row(row)
        #r  = np.concatenate(csses[i],row[])
        i +=1
    print(x)


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

if __name__ == "__main__":
    # execute only if run as a script
    main()







