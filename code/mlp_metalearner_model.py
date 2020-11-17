from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import sys
import os

def main():

	is_vae = True


	if is_vae:
	    metadataset = pd.read_csv('vae/metadataset_k_20.csv')
	else:
	     metadataset = pd.read_csv('cdae/metadataset_k_20.csv')

	#als:0
	#bpr:1
	#lmf:2
	#most_pop_3
	#zeros:4
	target_pre = metadataset['first_place'].values 
	label_encoder = LabelEncoder()
	target = label_encoder.fit_transform(target_pre)


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


	kf = KFold(n_splits=5)
	kf.get_n_splits()
	print(kf)


	params = {
	    'activation': 'relu',
	    'alpha': 0.05,
	    'hidden_layer_sizes': (50, 50, 50),
	    'learning_rate': 'constant',
	    'solver': 'sgd'
	}


	i = 1 
	reports = []
	base_impact_with_zeroes = []
	base_impact_without_zeroes_most = []
	base_impact_without_zeroes_best = []
	matrix = []
	for train_index, test_index in kf.split(inputs):
	    print('iteration: ', i)
	    #get data fold
	    X_train, X_test = inputs.iloc[train_index], inputs.iloc[test_index]
	    y_train, y_test = target[train_index], target[test_index]
	    
	    #start model 
	    print('fit')
	    clf = MLPClassifier(random_state=0, 
	                        max_iter=300,
	                       activation=params['activation'],
	                       alpha=params['alpha'],
	                       hidden_layer_sizes=params['hidden_layer_sizes'],
	                       learning_rate=params['learning_rate'],
	                       solver=params['solver'],
	                       verbose=5)
	        


	    
	    clf.fit(X_train, y_train)
	    y_pred = clf.predict(X_test)
	    
	    report = classification_report(y_test, 
	                                   y_pred, 
	                                   target_names=np.unique(metadataset['first_place'].values),
	                                  output_dict=True)
	    
	    bl_zeroes, bl_no_zeroes_most, bl_no_zeroes_best = base_level_eval(metadataset.iloc[test_index]['original_id'].values,
	             list(label_encoder.inverse_transform(y_pred)))

	    base_impact_with_zeroes.append(bl_zeroes)
	    base_impact_without_zeroes_most.append(bl_no_zeroes_most)
	    base_impact_without_zeroes_best.append(bl_no_zeroes_best)

	    confusion = confusion_matrix(y_test,y_pred)
	    matrix.append(confusion)
	    np.set_printoptions(suppress=True)    
	    reports.append(report)
	    
	    print('end: ', i)
	    i+=1


	print('base level impact zeroes', np.mean(base_impact_with_zeroes))
	print('base level impact  whithout zeroes, replaced most_pop', np.mean(base_impact_without_zeroes_most))
	print('base level impact  whithout zeroes, replaced best', np.mean(base_impact_without_zeroes_best))
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
        r = []
        r.append(classes[i])
        row = r + list(row)
        #r.append(classes[i])
        #r = r + row
        #row = np.insert(row,0,'als')
        x.add_row(row)
        #r  = np.concatenate(csses[i],row[])
        i +=1
    print(x)
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
def base_level_eval(users, predictions):
    """Uses the predctions to return the average of the ndcg impact at base level.
    Args:
        users: list of users ids
        predictions:predictions for users. PREDS HAVE TO be the same index ahas the users list
    Returns:
        average of ndcg
    """
    print('starting base_level_eval')
    results_algo = pd.read_csv(os.path.join(DATA_DIR, 'results_metadataset.csv'))
    base_impact = []
    base_impact_zeroes_most = []
    base_impact_zeroes_best = []
    for user_uid, pred in zip(users, predictions):


        val = results_algo.loc[ results_algo['original_id'] == user_uid, pred ]
        if pred == 'zeroes':
            val_zeroes = results_algo.loc[ results_algo['original_id'] == user_uid, 'most_popular_ndcg']
            best = results_algo.loc[ results_algo['original_id'] == user_uid]

            base_impact.append(val.values[0])
            base_impact_zeroes_most.append(val_zeroes.values[0])
            base_impact_zeroes_best.append(best.drop('original_id', 1).max(axis=1).values[0])
        else:
            base_impact.append(val.values[0])
            base_impact_zeroes_most.append(val.values[0])
            base_impact_zeroes_best.append(val.values[0])


        if len(val.values) > 1:
            raise Exception("More than one case")

    return np.mean(base_impact), np.mean(base_impact_zeroes_most), np.mean(base_impact_zeroes_best)

if __name__ == '__main__':
	main()





