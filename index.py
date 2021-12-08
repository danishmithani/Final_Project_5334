import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas_datareader as data

start = '2010-01-01'
end = '2021-12-01'


st.title('Stock Price Predictor - The PREDICTIT Version 1')
user_input = st.text_input('Enter the ticket', 'DLF.BO')
df = data.DataReader(user_input, 'yahoo', start, end)
st.write(df.describe())

st.subheader('Closing Time VS Time Chart with Moving Averages')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig1 = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.xlabel('days')
plt.ylabel('Share Price - INR')
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.legend(['Closing Price', 'MA100', 'MA200'], loc = 'lower right')
st.pyplot(fig1)

data = df.filter(['Close'])
data.head()
dataset = data.values
training_length_dataset = math.ceil(len(dataset)*0.8)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range=(0,1))
scaled_dataset = scale.fit_transform(dataset)

train_data = scaled_dataset[0:training_length_dataset, :]
# print(len(train_data))

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] ,1))
st.write("Number of days We will consider : ", x_train.shape[0])

# from keras.layers import Dense, Dropout, LSTM
# from keras.models import Sequential

# model = Sequential()
# model.add(LSTM(units=50, activation='relu', return_sequences=True,
#                input_shape=(x_train.shape[1], 1)))                    #Input shape = (100, 1) 1 because we are only predicting Close column

# model.add(LSTM(units=50, activation='relu', return_sequences=False))                    #Input shape = (100, 1) 1 because we are only predicting Close column
# model.add(Dense(25))
# model.add(Dense(1))
# st.write(model.summary())

from keras.models import load_model
# model.save('keras_model_v1.h5')
model = load_model('keras_model_v1.h5')


test_data = scaled_dataset[training_length_dataset-100: , :]
# print(test_data)

x_test = []
y_test = dataset[training_length_dataset:, :]
for i in range(100, len(test_data)):
    x_test.append(test_data[i-100:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scale.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions-y_test)**2)
st.write("Root Mean Square Error : ", rmse)

train = pd.DataFrame(dataset[:training_length_dataset], columns=['Close'])
valid = pd.DataFrame(dataset[training_length_dataset:], columns=['Close'])

valid['Predictions'] = predictions
# train.tail()

valid['trial'] = range(training_length_dataset, len(dataset))

valid = valid.set_index('trial')


fig2 = plt.figure(figsize=(16,10))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.plot(train['Close'])
# plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
st.pyplot(fig2)


data_copy = df.filter(['Close'])
# data_copy.head()
last_60_days = data_copy[-100:].values
scaled_last_60_days = scale.fit_transform(last_60_days)

x_test = []
x_test.append(scaled_last_60_days)
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pred_price = model.predict(x_test)

pred_price = scale.inverse_transform(pred_price)
st.write("Predited Price for Tomorrow: ", pred_price[0][0])
risk = (rmse/(pred_price[0][0]))*100
st.write("Risk of Variation in the predicted Price: ", risk, " %")
