#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sarvandani
"""
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=6000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2
#data1 = yf.download('AAPL')
#print(data1)
# some days, stock market is off, right? ;)
data = yf.download('AAPL', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
# include date in dataset
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", 
             "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
data.tail()
#------------------------------------------------------
#features
x = data[["Open", "High", "Low", "Volume"]]

#lable
y = data["Close"]
#Convert the DataFrame to a NumPy array.
x = x.to_numpy()
y = y.to_numpy()
#Now trying to reshape with (-1, 1) . We have provided column as 1 but rows as unknown .
y = y.reshape(-1, 1)

#------------------------------------------------------
#splitting data set
#test size= should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=7)
#_____________________
#Neural network: define model
# If you want to use stacked layers of LSTMs then use return_sequences=True before passing input to the next LSTM layer.
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(32))
model.add(Dense(1))
model.summary()
#______________________________________________________________
#Fitting the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
fitting_info= model.fit(xtrain, ytrain, validation_data = (xtest,ytest), batch_size=1, epochs=40)
#plotting the info of fitting model
plt.plot(fitting_info.history['loss']) 
plt.plot(fitting_info.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()
#------------------------------------------------------------------------
# model prediction
#features = [Open, High, Low, Volume]
# we can change number of row in data
features = data.iloc[4133, [1,2,3,6]]
features = features.to_numpy()
type(features)
np.ndarray
features = features.reshape(-1, 1)
features= np.transpose(features)
features = np.array(features).astype('float64')
pridicted_price= model.predict(features)
print(pridicted_price)
#--------------------------------------------------------------------------




