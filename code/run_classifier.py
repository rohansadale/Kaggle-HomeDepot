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

def plot_learning_curve(tr_errors, te_errors, split_ratio, title, file_name):
	
	plt.figure()
	plt.title(title)
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
	clf = GridSearchCV(clf_abc, param_grid = param_grid, cv = 5, scoring = 'accuracy', n_jobs=-1)
	clf.fit(X, y)
	best_clf_dt = DecisionTreeClassifier(random_state = 4, max_features = None, class_weight = "balanced", max_depth = 5,
										criterion = clf.best_params_['base_estimator__criterion'])
	best_clf = AdaBoostClassifier(base_estimator = best_clf_dt)
	return best_clf

def fitRandomForest(X, y):

	param_grid = { "n_estimators" : [10, 20], "criterion" : ["gini", "entropy"], "max_features" : ["auto", "log2", None]}
	clf_rf = RandomForestClassifier(random_state = 4)
	clf = GridSearchCV(clf_rf, param_grid, cv = 5, scoring = 'accuracy', n_jobs=-1)
	clf.fit(X,y)
	best_clf = RandomForestClassifier(criterion = clf.best_params_['criterion'], max_features = clf.best_params_['max_features'],
									  n_estimators = clf.best_params_['n_estimators'])
	return best_clf

def fitLR(X, y):

	param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	clf = GridSearchCV(LogisticRegression(), param_grid, cv = 5, scoring = 'accuracy', n_jobs=-1)
	clf.fit(X,y)
	best_clf = LogisticRegression(C = clf.best_params_['C'])
	return best_clf 

def fitRegression(X, y):
	gb_learning_grid = [0.1, 0.05, 0.02, 0.01]
	param_grid = {'learning_rate':gb_learning_grid}
	clf = GridSearchCV(GradientBoostingRegressor(), param_grid=param_grid, n_jobs=-1, cv=5)
	clf.fit(X, y)
	best_clf = GradientBoostingRegressor(learning_rate = clf.best_params_['learning_rate'])
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

	title = "Learning Curve for " + msg
	plot_learning_curve(tr_errors, te_errors, split_ratio, title, '../plots/' + file_name)
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

	title = "Learning Curve for " + msg
	plot_learning_curve(tr_errors, te_errors, split_ratio, title, '../plots/' + file_name)
	print " ============================================================ "

train_file_name = sys.argv[1]
X, y, y_ternary, score, column_names = read_csv(train_file_name)
print "Class distribution for binary labels ..... "
print np.unique(y, return_counts = True)

print "Class distribution for ternary labels ..... "
print np.unique(y_ternary, return_counts = True)

"""
Learning Curve with RandomForestClassifier and GradientBoostingRegressor
"""

print "Learning Curve with RandomForestClassifier and GradientBoostingRegressor ...."
doRegression(X, score, np.arange(0.1, 1.01, 0.1), 'learning_curve_regression.png', 'Regression .....')
doClassification(X, y_ternary, np.arange(0.1, 1.01, 0.1), 'learning_curve_ternary.png', 'ternary classification .....')
doClassification(X, y, np.arange(0.1, 1.01, 0.1), 'learning_curve_binary.png', 'binary classification .....')

train_index, test_index = get_train_test_indexes(y)
X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]

"""
10 Fold Cross Validation....
"""

print " ============================================================================= "
clf_rf = fitRandomForest(X, y)
scores = cross_val_score(clf_rf, X, y, cv = 10)
print "10 fold cv for binary classification using RandomForests had accuracy %.3f and std. dev %.3f " % (np.mean(scores), np.std(scores))

clf_adaboost = fitAdaBoost(X, y)
scores = cross_val_score(clf_adaboost, X, y, cv = 10)
print "10 fold cv for binary classification using AdaBoost had accuracy %.3f and std. dev %.3f " % (np.mean(scores), np.std(scores))

clf_rf = fitRandomForest(X, y_ternary)
scores = cross_val_score(clf_rf, X, y_ternary, cv = 10)
print "10 fold cv for ternary classification using RandomForests had accuracy %.3f and std. dev %.3f " % (np.mean(scores), np.std(scores))

clf_adaboost = fitAdaBoost(X, y_ternary)
scores = cross_val_score(clf_adaboost, X, y_ternary, cv = 10)
print "10 fold cv for ternary classification using AdaBoost had accuracy %.3f and std. dev %.3f " % (np.mean(scores), np.std(scores))


clf_reg = fitRegression(X, score)
scores = cross_val_score(clf_reg, X, score, cv = 10)
print "10 fold cv for regression using GradientBoostingRegressor had RMSE %.3f and std. dev %.3f " % (np.mean(scores), np.std(scores))

train_index, test_index = get_train_test_indexes(y)
X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]

print " ============================================================================= "

"""
Feature Importance & F1 - scores
"""

print "Getting Feature importances and F1 scores for binary classification ..... "
clf_rf = fitRandomForest(X_train, y_train)
clf_rf.fit(X_train, y_train)
y_pred = clf_rf.predict(X_test)
print "F1 scores and confusion-matrix using RandomForestClassifier Classifier ... "
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
print "F1 scores and confusion-matrix using AdaBoost Classifier ... "
print confusion_matrix(y_test, y_pred,[0, 1])
print f1_score(y_test, y_pred, [0, 1], average = None)

train_index, test_index = get_train_test_indexes(y_ternary)
X_train, y_train, X_test, y_test = X[train_index], y_ternary[train_index], X[test_index], y_ternary[test_index]

print "Getting Feature importances and F1 scores for ternary classification ..... "
clf_rf = fitRandomForest(X_train, y_train)
clf_rf.fit(X_train, y_train)
y_pred = clf_rf.predict(X_test)
print "F1 scores and confusion-matrix using RandomForestClassifier Classifier ... "
print confusion_matrix(y_test, y_pred,[1, 2, 3])
print f1_score(y_test, y_pred, [1, 2, 3], average = None)

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
print "F1 scores and confusion-matrix using AdaBoost Classifier ... "
print confusion_matrix(y_test, y_pred,[1, 2, 3])
print f1_score(y_test, y_pred, [1, 2 , 3], average = None)