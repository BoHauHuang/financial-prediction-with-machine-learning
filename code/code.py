import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, LSTM, RepeatVector
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

numbers = []

path = 'wti.csv'

data = pd.read_csv(path, keep_default_na=False)
#print(data['date'])
#print(data.head())

data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['date'] = data['Date'].dt.day

#print(data)

dataset = data.drop(['Date'], axis=1)
#print(dataset)
dataset = dataset.dropna()
print(dataset)
#print(len(dataset['Price']))
#dataset = dataset.astype('float64')
#dataset = dataset.apply(lambda x: (x-np.mean(x)) /(np.max(x) - np.min(x)))

#x_scaler = MinMaxScaler()
#y_scaler = MinMaxScaler()

#print(dataset)


look_back = 1

def build_train(dataset, past,look_back):
    dataX, dataY = [], []
    for i in range(dataset.shape[0]-past-look_back):
        dataX.append(np.array(dataset.iloc[i:i+past]))
        dataY.append(np.array(dataset.iloc[i+past: i+past+look_back]['Price']))
    return np.array(dataX), np.array(dataY)

def splitData(x,y,rate):
    train_x = x[:-int(x.shape[0]*rate):]
    train_y = y[:-int(y.shape[0]*rate):]
    test_x = x[-int(x.shape[0]*rate):]
    test_y = y[-int(y.shape[0]*rate):]
    return train_x, train_y, test_x, test_y


train_x, train_y = build_train(dataset, 5, look_back)


#train_y = train_y[:,:,np.newaxis]
#print(train_x.shape)
#print(train_y.shape)

#print(train_x)
#print(train_y)

train_x, train_y, test_x, test_y = splitData(train_x, train_y, 0.1)
#print(train_x.shape)
#print(train_y.shape)
#print(test_x.shape)
#print(test_y.shape)
#print(train_y)
#print(test_y)

#train_y = train_y[:,:]
#print(train_y)
#print(train_y.shape)
#train_y = np.array(train_y, (train_y.shape[0],1))

#print(train_y)
#print(train_x)
#train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
#print(train_x)
#test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

model = Sequential()
#model.add(LSTM(8, input_shape=(1, look_back)))
model.add(LSTM(8, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=False, activation = 'linear')) 
# 100: selu 1.76
# 150: linear 0.97
# 200: softplus 1.15
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

callback = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')
model.fit(train_x, train_y, epochs=300, batch_size=128, verbose=0, validation_data = (test_x, test_y))


train_predict = model.predict(train_x)
#print(train_predict[:,0])
#print(train_predict.shape)
#print(len(train_predict))
test_predict = model.predict(test_x)
#print(test_predict[0])
#print(test_predict.shape)
print(test_predict[:,0])
#print(dataset['Price'].values)


#print(train_y[:,0])


#print(len(dataset['Price'])-1-len(train_predict)-(look_back*2)-1)




#train_predict = x_scaler.inverse_transform(train_predict[0])
#print(train_predict)
#train_y = y_scaler.inverse_transform([train_y])
#test_predict = x_scaler.inverse_transform(test_predict[0])
#test_y = y_scaler.inverse_transform([test_y])

train_Score = math.sqrt(mean_squared_error(train_y[:,0], train_predict[:,0]))
print('Train Score: %.2f Error' %(train_Score))
testScore = math.sqrt(mean_squared_error(test_y[:,0], test_predict[:,0]))
print('Test Score: %.2f Error' % (testScore))

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back] = train_predict
#print(trainPredictPlot)

testPredictPlot = np.empty_like(dataset['Price'])
#print(testPredictPlot)
testPredictPlot[:] = np.nan
#print(testPredictPlot)
testPredictPlot[len(train_predict)+(look_back*2)+1:len(dataset['Price'])-3] = test_predict[:,0]
#print(testPredictPlot)

plt.plot(dataset['Price'])
#plt.plot(train_predict)
#plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot,'r')
#plt.plot(testPredictPlot,'b')
plt.show()

