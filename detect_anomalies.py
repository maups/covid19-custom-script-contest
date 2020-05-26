import matplotlib
matplotlib.use("Agg")
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')

from pandas import read_csv
import math
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
import sys

dataframe = read_csv(sys.argv[1], usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

def create_dataset(dataset, look_back=1):
	X, Y = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		X.append(a)
		Y.append(dataset[i + look_back, 0])
	return np.array(X), np.array(Y)

look_back = 1
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

nepochs=100
nbatch_size=32
history = model.fit(X_train, Y_train, epochs=nepochs, batch_size=nbatch_size, validation_data=(X_test, Y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=50)], verbose=1, shuffle=False)

model.summary()

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))

anomaliesindex = np.arange(1,len(Y_test[0])+1,1)
anomaliesPlot = []
threshold = 0.20

min_value_Y=math.floor(min(Y_test[0]))
max_value_Y=math.floor(max(Y_test[0]))
min_value_Ypred=math.floor(min(test_predict[:,0]))
max_value_Ypred=math.floor(max(test_predict[:,0]))

interval_Y = (max_value_Y - min_value_Y)
interval_Yp = (max_value_Ypred - min_value_Ypred)

previous = test_predict[:,0][0]
for elem1,elem2 in zip(Y_test[0],test_predict[:,0]):
	difference = abs(elem2 - previous)
	ratio = difference/max(interval_Y,interval_Yp)
	if ratio > threshold:
		anomaliesPlot.append(True)
	else:   
		anomaliesPlot.append(False)
	previous = elem2  

Ytr = np.concatenate((Y_train[0], Y_test[0]), axis=0)
Ypr = np.concatenate((train_predict[:,0], test_predict[:,0]), axis=0)

aa=[x for x in range(len(Ytr))]
plt.figure(figsize=(16,4))
plt.plot(aa, Ytr, label="Signal")
plt.plot(aa, Ypr, 'r', label="Prediction")
plt.axvline(x=len(Y_train[0]), label='Train/Test', c='k')
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2)
plt.ylabel('Count', size=15)
plt.xlabel('Time', size=15)
plt.legend(fontsize=15)
plt.savefig('result.png')

positions = []
labels = []
step = 20
for x in range(0,len(Y_test[0]),step):
	positions.insert(x,x)
	labels.insert(x,str(x + len(Y_train[0])))

dsize=len(Y_test[0])
bb=[x for x in range(dsize)]
plt.figure(figsize=(16,4))
plt.xticks(positions, labels)
plt.plot(bb, Y_test[0][:dsize], label="Signal")
plt.plot(bb, test_predict[:,0][:dsize], 'r', label="Prediction")
plt.scatter(anomaliesindex[anomaliesPlot], Y_test[0][anomaliesPlot], label='Anomalies', c='r')
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2)
plt.ylabel('Count', size=15)
plt.xlabel('Time', size=15)
plt.legend(fontsize=15)
plt.savefig('result_zoom.png')

