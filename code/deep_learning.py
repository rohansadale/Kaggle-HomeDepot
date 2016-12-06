import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from sklearn.model_selection import StratifiedShuffleSplit

def plot_learning_curve(tr_errors, te_errors, num_epoch, file_name):
	
	plt.figure()
	plt.title("Change in accuracy as number of epochs increase")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.grid()

	plt.plot(num_epoch, tr_errors, 'o-', color="g", label="Training Accuracy")
	plt.plot(num_epoch, te_errors, 'o-', color="r", label="Validation Accuracy")
	plt.legend(loc="best")
	plt.savefig(file_name)

def read_csv(file_name):
	data = pd.read_csv(file_name, dtype = 'float64')
	total_columns = len(data.columns.tolist())
	cols = [total_columns-3, total_columns-2, total_columns-1]
	y, y_ternary, score = data.iloc[:,total_columns-3], data.iloc[:, total_columns-2], data.iloc[:, total_columns-1]
	X = data.drop(data.columns[cols], axis=1)
	return X.as_matrix(), np.array(y), np.array(y_ternary), np.array(score), data.columns.values[:-3]

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.tr_accuracy = []
		self.val_accuracy = []

	def on_epoch_end(self, batch, logs={}):
		self.tr_accuracy.append(logs.get('acc'))
		self.val_accuracy.append(logs.get('val_acc'))

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

train_file_name = sys.argv[1]
X, y, y_ternary, score, column_names = read_csv(train_file_name)
row, columns = X.shape
print "Class distribution for binary labels ..... "
print np.unique(y, return_counts = True)

print "Class distribution for ternary labels ..... "
print np.unique(y_ternary, return_counts = True)

model = Sequential()
model.add(Dense(int(1.5 *columns), input_dim = columns, init = 'uniform', activation = 'relu')) #1st hidden layer
model.add(Dense(columns , init = 'normal', activation = 'relu')) #2nd hidden layer
model.add(Dense(1, activation = 'sigmoid')) #output layer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_splits = 5
num_epoch = 500
batch_size = 1000
tr_accuracy = np.zeros((num_splits, num_epoch))
te_accuracy = np.zeros((num_splits, num_epoch))
i = 0

stratified_split = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=4)
for train_index, test_index in stratified_split.split(X, y):
	
	history = LossHistory()
	X_train, y_train = X[train_index], y[train_index]
	X_test, y_test = X[test_index], y[test_index]
	model.fit(X_train, y_train, nb_epoch=num_epoch, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[history],
					verbose = 0)
	tr_accuracy[i,:] = history.tr_accuracy
	te_accuracy[i,:] = history.val_accuracy
	print "Running iteration %d with training accuracy %.3f and testing accuracy %.3f ... " % (i+1, tr_accuracy[i][num_epoch-1],
		te_accuracy[i][num_epoch-1])
	i = i +1

plot_learning_curve(np.mean(tr_accuracy, axis = 0), np.mean(te_accuracy, axis = 0), np.arange(1, num_epoch+1),
	'../plots/NN_learning.png')