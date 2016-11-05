import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def read_csv(file_name):
	data = pd.read_csv(file_name, dtype = 'float64')
	total_columns = len(data.columns.tolist())
	y = data.iloc[:,total_columns-1]
	X = data.drop(data.columns[[total_columns-1]], axis=1)
	return X.as_matrix(), np.array(y)

def get_train_test_indexes(y, train_proportion = 0.8):
		y = np.array(y)
		train_idx = np.zeros(len(y), dtype = bool)
		test_idx = np.zeros(len(y), dtype = bool)
		values = np.unique(y)
	 
		for label in values:
			label_idx = np.nonzero(y == label)[0]
			np.random.seed(4)
			np.random.shuffle(label_idx)
			threshold = int(train_proportion * len(label_idx))
			train_idx[label_idx[:threshold]] = True
			test_idx[label_idx[threshold:]] = True
	  
		return train_idx, test_idx

def fitAdaBoost(X, y):
	param_grid = {  "base_estimator__criterion" : ["gini", "entropy"],
						"base_estimator__splitter" :   ["best", "random"]}
	clf_dt = DecisionTreeClassifier(random_state = 4, max_features = "auto", class_weight = "balanced", max_depth = None)
	clf_abc = AdaBoostClassifier(base_estimator = clf_dt)
	clf = GridSearchCV(clf_abc, param_grid = param_grid, cv = 5, scoring = 'accuracy')
	clf.fit(X, y)
	best_clf_dt = DecisionTreeClassifier(random_state = 4, max_features = "auto", class_weight = "balanced", max_depth = None,
										criterion = clf.best_params_['base_estimator__criterion'], splitter = clf.best_params_['base_estimator__splitter'])
	best_clf = AdaBoostClassifier(base_estimator = best_clf_dt)
	return best_clf

def fitRandomForest(X, y):

	param_grid = { "n_estimators" : [10, 20], "criterion" : ["gini", "entropy"], "max_features" : ["auto", "log2", None]}
	clf_rf = RandomForestClassifier(random_state = 4)
	clf = GridSearchCV(clf_rf, param_grid, cv = 5, scoring = 'accuracy')
	clf.fit(X,y)
	best_clf = RandomForestClassifier(criterion = clf.best_params_['criterion'], max_features = clf.best_params_['max_features'],
									  n_estimators = clf.best_params_['n_estimators'])
	return best_clf

def fitLR(X, y):

	param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	clf = GridSearchCV(LogisticRegression(), param_grid, cv = 5, scoring = 'accuracy')
	clf.fit(X,y)
	best_clf = LogisticRegression(C = clf.best_params_['C'])
	return best_clf 

train_file_name = sys.argv[1]
num_splits = 5
split_ratio = [0.10, 0.25, 0.50, 0.75, 1.00]

dt_errors = np.zeros((num_splits, len(split_ratio)))
rf_errors = np.zeros((num_splits, len(split_ratio)))
lr_errors = np.zeros((num_splits, len(split_ratio)))

X, y = read_csv(train_file_name)

best_clf_adaboost, ada_max_accuracy = None, 0
best_clf_rf, rf_max_accuracy = None, 0
best_clf_lr, lr_max_accuracy = None, 0

for i in range(num_splits):
	print "Running iteration %d " % (i+1)
	stratified_split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = i+1)
	for train_index, test_index in stratified_split.split(X, y):
		X_test, y_test = X[test_index], y[test_index]
		for j in range(len(split_ratio)):
			required_size = int(split_ratio[j] * train_index.shape[0])
			required_index = train_index[:required_size]
			X_train, y_train = X[required_index], y[required_index]
			
			clf = fitAdaBoost(X_train, y_train)
			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			dt_errors[i][j] = np.mean(y_pred == y_test)
			if 1.00 == split_ratio[j] and np.mean(y_pred == y_test) > ada_max_accuracy:
				best_clf_adaboost, ada_max_accuracy = clf, np.mean(y_pred == y_test)
			
			clf_rf = fitRandomForest(X_train, y_train)
			clf_rf.fit(X_train, y_train)
			y_pred = clf_rf.predict(X_test)
			rf_errors[i][j] = np.mean(y_pred == y_test)
			if 1.00 == split_ratio[j] and np.mean(y_pred == y_test) > rf_max_accuracy:
				best_clf_rf, rf_max_accuracy = clf, np.mean(y_pred == y_test)
			
			clf_lr = fitLR(X_train, y_train)
			clf_lr.fit(X_train, y_train)
			y_pred = clf_lr.predict(X_test)
			lr_errors[i][j] = np.mean(y_pred == y_test)
			if 1.00 == split_ratio[j] and np.mean(y_pred == y_test) > lr_max_accuracy:
				best_clf_lr, lr_max_accuracy = clf_lr, np.mean(y_pred == y_test)
			
dt_mean = np.mean(dt_errors, axis = 0)
dt_std = np.std(dt_errors, axis = 0)

rf_mean = np.mean(rf_errors, axis = 0)
rf_std = np.std(rf_errors, axis = 0)

lr_mean = np.mean(lr_errors, axis = 0)
lr_std = np.std(lr_errors, axis = 0)

print dt_mean
print dt_std
print " ==== "
print rf_mean
print rf_std
print " ==== "
print lr_mean
print lr_std
print " ==== "

train_index, test_index = get_train_test_indexes(y)
X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]

best_clf_adaboost.fit(X_train, y_train)
y_pred_adaboost = best_clf_adaboost.predict(X_test)
print "Using AdaBoost Classifier ... "
print confusion_matrix(y_test, y_pred_adaboost,[0, 1])
print f1_score(y_test, y_pred_adaboost, [0, 1], average = None)

best_clf_rf.fit(X_train, y_train)
y_pred_rf = best_clf_rf.predict(X_test)
print "Using RandomForestClassifier Classifier ... "
print confusion_matrix(y_test, y_pred_rf,[0, 1])
print f1_score(y_test, y_pred_rf, [0, 1], average = None)

best_clf_lr.fit(X_train, y_train)
y_pred_lr = best_clf_lr.predict(X_test)
print "Using LogisticRegression Classifier ... "
print confusion_matrix(y_test, y_pred_lr,[0, 1])
print f1_score(y_test, y_pred_lr, [0, 1], average = None)