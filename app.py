import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Trend Prediction")

stock = st.text_input("Enter Stock Symbol","AAPL")

data = yf.download(stock, start='2015-01-01', end='2024-01-01')

st.subheader("Stock Data")
st.write(data.tail())

st.subheader("Closing Price Chart")
fig = plt.figure(figsize=(10,4))
plt.plot(data.Close)
st.pyplot(fig)

# moving averages
ma100 = data.Close.rolling(100).mean()

st.subheader("Closing Price vs MA100")
fig = plt.figure(figsize=(10,4))
plt.plot(data.Close)
plt.plot(ma100)
st.pyplot(fig)

# load model
model = load_model("keras_model.h5")

# data preprocessing
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.7)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.7):])

scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)

final_df = pd.concat([past_100_days,data_test],ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)

y_predicted = model.predict(x_test)

scale = 1/scaler.scale_

y_predicted = y_predicted * scale
y_test = y_test * scale

st.subheader("Predictions vs Original")

fig2 = plt.figure(figsize=(10,4))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)