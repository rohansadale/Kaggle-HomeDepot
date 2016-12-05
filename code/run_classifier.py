# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, f1_score 

def plot_learning_curve(ada_error_mean, ada_error_std,
						rf_error_mean, rf_error_std,
						lr_error_mean, lr_error_std,
						split_ratio, file_name):
	plt.figure()
	plt.title("Comparsion of different classifiers for predicting product's relevance to search term.")

	plt.xlabel("Training examples")
	plt.ylabel("Accuracy")
	plt.grid()

	plt.fill_between(split_ratio, ada_error_mean - ada_error_std, ada_error_mean + ada_error_std, alpha=0.1, color="r")
	plt.fill_between(split_ratio, rf_error_mean - ada_error_std, rf_error_mean + rf_error_std, alpha=0.1, color="g")
	plt.fill_between(split_ratio, lr_error_mean - lr_error_std, lr_error_mean + lr_error_std, alpha=0.1, color="b")
	plt.plot(split_ratio, ada_error_mean, 'o-', color="r", label="AdaBoostClassifier")
	plt.plot(split_ratio, rf_error_mean, 'o-', color="g", label="Random Forest")
	plt.plot(split_ratio, lr_error_mean, 'o-', color="b", label="Logistic Regression")
	plt.legend(loc="best")
	plt.savefig(file_name)

def plot_learning_curve(tr_errors, te_errors, split_ratio, file_name):
	
	plt.figure()
	plt.title("Learning Curve")
	plt.xlabel("Training examples")
	plt.ylabel("Error")
	plt.grid()

	plt.plot(split_ratio, tr_errors, 'o-', color="g", label="Training Error")
	plt.plot(split_ratio, te_errors, 'o-', color="r", label="Test Error")
	plt.legend(loc="best")
	plt.savefig(file_name)

def plot_feature_importances(importance_list, name_list, feature_importance_file_name):
	plt.barh(range(len(name_list)),importance_list,align='center')
	plt.yticks(range(len(name_list)),name_list, rotation = 45)
	plt.xlabel('Relative Importance in the Random Forest')
	plt.ylabel('Features')
	plt.title('Relative importance of Each Feature')
	plt.savefig(feature_importance_file_name)

def read_csv(file_name):
	data = pd.read_csv(file_name, dtype = 'float64')
	total_columns = len(data.columns.tolist())
	cols = [total_columns-3, total_columns-2, total_columns-1]
	y, y_ternary, score = data.iloc[:,total_columns-3], data.iloc[:, total_columns-2], data.iloc[:, total_columns-1]
	X = data.drop(data.columns[cols], axis=1)
	return X.as_matrix(), np.array(y), np.array(y_ternary), np.array(score), data.columns.values[:-3]

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
	param_grid = {"base_estimator__criterion" : ["gini", "entropy"]}
	clf_dt = DecisionTreeClassifier(random_state = 4, class_weight = "balanced", max_features = None, max_depth = 5)
	clf_abc = AdaBoostClassifier(base_estimator = clf_dt)
	clf = GridSearchCV(clf_abc, param_grid = param_grid, cv = 5, scoring = 'accuracy', n_jobs=4)
	clf.fit(X, y)
	best_clf_dt = DecisionTreeClassifier(random_state = 4, max_features = None, class_weight = "balanced", max_depth = 5,
										criterion = clf.best_params_['base_estimator__criterion'])
	best_clf = AdaBoostClassifier(base_estimator = best_clf_dt)
	return best_clf

def fitRandomForest(X, y):

	param_grid = { "n_estimators" : [10, 20], "criterion" : ["gini", "entropy"], "max_features" : ["auto", "log2", None]}
	clf_rf = RandomForestClassifier(random_state = 4)
	clf = GridSearchCV(clf_rf, param_grid, cv = 5, scoring = 'accuracy', n_jobs=4)
	clf.fit(X,y)
	best_clf = RandomForestClassifier(criterion = clf.best_params_['criterion'], max_features = clf.best_params_['max_features'],
									  n_estimators = clf.best_params_['n_estimators'])
	return best_clf

def fitLR(X, y):

	param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	clf = GridSearchCV(LogisticRegression(), param_grid, cv = 5, scoring = 'accuracy', n_jobs=4)
	clf.fit(X,y)
	best_clf = LogisticRegression(C = clf.best_params_['C'])
	return best_clf 

def fitRegression(X, y):
	gb_learning_grid = [0.1, 0.05, 0.02, 0.01]
	param_grid = {'learning_rate':gb_learning_grid} #, 'n_estimators':gb_estimators_grid, 'min_samples_leaf':gb_minleaf_grid}
	clf = GridSearchCV(GradientBoostingRegressor(), param_grid=param_grid, n_jobs=4, cv=5)
	clf.fit(X, y)
	best_clf = GradientBoostingRegressor(learning_rate = clf.best_params_['learning_rate'],
									#n_estimators = clf.best_params_['n_estimators'],
									#min_samples_leaf = clf.best_params_['min_samples_leaf']
									)
	return best_clf

def doClassification(X, y, split_ratio, file_name, msg):
	print msg
	tr_errors = np.zeros(len(split_ratio))
	te_errors = np.zeros(len(split_ratio))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
	for i, ratio in enumerate(split_ratio):
		X_iter, y_iter = X_train[:int(ratio * X_train.shape[0])], y_train[:int(ratio * X_train.shape[0])]
		clf = fitRandomForest(X_train, y_train)
		clf.fit(X_train, y_train)
		tr_errors[i] = 1.0 - np.mean(clf.predict(X_iter) == y_iter)
		te_errors[i] = 1.0 - np.mean(clf.predict(X_test) == y_test)
		print "Ran iteration %d with %d training data points with %f training error and %f test error" %  (i+1, X_iter.shape[0],
			tr_errors[i], te_errors[i])

	plot_learning_curve(tr_errors, te_errors, split_ratio, '../plots/' + file_name)
	print " ============================================================ "

def doRegression(X, y, split_ratio, file_name, msg):
	print msg
	tr_errors = np.zeros(len(split_ratio))
	te_errors = np.zeros(len(split_ratio))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
	for i, ratio in enumerate(split_ratio):
		X_iter, y_iter = X_train[:int(ratio * X_train.shape[0])], y_train[:int(ratio * X_train.shape[0])]
		clf = fitRegression(X_train, y_train)
		clf.fit(X_train, y_train)
		tr_errors[i] = mean_squared_error(clf.predict(X_iter), y_iter)
		te_errors[i] = mean_squared_error(clf.predict(X_test), y_test)
		print "Ran iteration %d with %d training data points with %f training RMSE and %f test RMSE" %  (i+1, X_iter.shape[0],
			tr_errors[i], te_errors[i])

	plot_learning_curve(tr_errors, te_errors, split_ratio, '../plots/' + file_name)
	print " ============================================================ "

train_file_name = sys.argv[1]
num_splits = 5
split_ratio = np.arange(0.05, 1.0, 0.05)

ada_errors = np.zeros((num_splits, len(split_ratio)))
rf_errors = np.zeros((num_splits, len(split_ratio)))
lr_errors = np.zeros((num_splits, len(split_ratio)))
tr_errors = np.zeros(len(split_ratio))
te_errors = np.zeros(len(split_ratio))

X, y, y_ternary, score, column_names = read_csv(train_file_name)
print "Class distribution for binary labels ..... "
print np.unique(y, return_counts = True)

print "Class distribution for ternary labels ..... "
print np.unique(y_ternary, return_counts = True)

doRegression(X, score, split_ratio, 'learning_curve_regression.png', 'Doing Regression .....')
doClassification(X, y_ternary, split_ratio, 'learning_curve_ternary.png', 'Doing ternary classification .....')
doClassification(X, y, split_ratio, 'learning_curve_binary.png', 'Doing binary classification .....')

"""
Learning Curve
"""
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
			ada_errors[i][j] = np.mean(y_pred == y_test)
			
			clf_rf = fitRandomForest(X_train, y_train)
			clf_rf.fit(X_train, y_train)
			y_pred = clf_rf.predict(X_test)
			rf_errors[i][j] = np.mean(y_pred == y_test)
			
			clf_lr = fitLR(X_train, y_train)
			clf_lr.fit(X_train, y_train)
			y_pred = clf_lr.predict(X_test)
			lr_errors[i][j] = np.mean(y_pred == y_test)

ada_mean = np.mean(ada_errors, axis = 0)
ada_std = np.std(ada_errors, axis = 0)

rf_mean = np.mean(rf_errors, axis = 0)
rf_std = np.std(rf_errors, axis = 0)

lr_mean = np.mean(lr_errors, axis = 0)
lr_std = np.std(lr_errors, axis = 0)

print "Average accuracy using AdaBoostClassifier .... "
print ada_mean
print ada_std
print " =========================================== "

print "Average accuracy using RandomForestClassifier .... "
print rf_mean
print rf_std
print " =========================================== "

print "Average accuracy using LogisticRegression .... "
print lr_mean
print lr_std
print " ========================================= "

plot_learning_curve(ada_mean, ada_std, rf_mean, rf_std, lr_mean, lr_std, split_ratio, '../plots/learning_curve.png')

"""
Feature Importance & F1 - scores
"""
train_index, test_index = get_train_test_indexes(y)
X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]

clf_rf = fitRandomForest(X_train, y_train)
clf_rf.fit(X_train, y_train)
y_pred = clf_rf.predict(X_test)
print "Using RandomForestClassifier Classifier ... "
print confusion_matrix(y_test, y_pred,[0, 1])
print f1_score(y_test, y_pred, [0, 1], average = None)

print "Feature importance .... "
print clf_rf.feature_importances_
importance_list = clf_rf.feature_importances_
importance_list, name_list = zip(*sorted(zip(map(lambda x: round(x, 4), clf_rf.feature_importances_), column_names), reverse=True))
importance_list, name_list = importance_list[:5], name_list[:5]
print importance_list, name_list
plot_feature_importances(importance_list, name_list, '../plots/feature_importance.png')

clf_adaboost = fitAdaBoost(X_train, y_train)
clf_adaboost.fit(X_train, y_train)
y_pred = clf_adaboost.predict(X_test)
print "Using AdaBoost Classifier ... "
print confusion_matrix(y_test, y_pred,[0, 1])
print f1_score(y_test, y_pred, [0, 1], average = None)

clf_lr = fitLR(X_train, y_train)
clf_lr.fit(X_train, y_train)
y_pred = clf_lr.predict(X_test)
print "Using LogisticRegression Classifier ... "
print confusion_matrix(y_test, y_pred,[0, 1])
print f1_score(y_test, y_pred, [0, 1], average = None)