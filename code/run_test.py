import sys
import pandas as pd
import numpy as np

from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV

def read_csv_train(file_name):
	data = pd.read_csv(file_name, dtype = 'float64')
	total_columns = len(data.columns.tolist())
	cols = [total_columns-3, total_columns-2, total_columns-1]
	y, y_ternary, score = data.iloc[:,total_columns-3], data.iloc[:, total_columns-2], data.iloc[:, total_columns-1]
	X = data.drop(data.columns[cols], axis=1)
	return X.as_matrix(), np.array(y), np.array(y_ternary), np.array(score), data.columns.values[:-3]

def read_csv_test(file_name):
	data = pd.read_csv(file_name, dtype = 'float64')
	return data.as_matrix(), data.columns.values

def fitxgBoostRegressor(X, y):
	param_grid = {"n_estimators" : [100, 150, 200], "max_depth": [3, 5, 7, 9]}
	clf_xgb = XGBRegressor()
	clf = GridSearchCV(clf_xgb, param_grid, cv = 5, scoring = 'neg_mean_squared_error', n_jobs=-1)
	clf.fit(X,y)
	return XGBRegressor(n_estimators = clf.best_params_['n_estimators'], max_depth = clf.best_params_['max_depth'])

train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
original_test_file = sys.argv[3]

X_train, y_train, y_ternary, score, column_names = read_csv_train(train_file_name)
X_test, column_names = read_csv_test(test_file_name)
test_df = pd.read_csv(original_test_file)

clf_reg = fitxgBoostRegressor(X_train, score)
clf_reg.fit(X_train, score)
predicted_score = clf_reg.predict(X_test)
predicted_score = map(lambda x: min(x, 3.0), predicted_score)
predicted_score = map(lambda x: max(x, 1.0), predicted_score)
result = zip(test_df['id'], predicted_score)

with open('../data/reg_score.csv', 'w') as f:
	for result in result:
		f.write(str(result[0]) + "," + str(result[1]) + "\n")