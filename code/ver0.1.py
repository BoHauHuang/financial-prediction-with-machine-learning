import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

data = pd.read_csv('wti_daily.csv')
#print(data)
data = data.drop(['Date'],1)
n = data.shape[0]
p = data.shape[1]
#print(n)
data = data.values
#pyplot.axis([])
pyplot.plot(data)

#######################

train_start = 0
train_end = int(np.floor*(0.8*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

########################

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

x_train = data_train[:, 1:]
y_train = data_train[:, 0]
x_test = data_test[:, 1:]
y_test = data_test[:, 0]

########################

import tensorflow as tf
x = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
y = tf.placeholder(dtype=tf.float32, shape=[None])

# Model architecture parameters
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Layer 1
w_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
