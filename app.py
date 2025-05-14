import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the model
from keras.models import load_model
model = load_model('C:/Users/mitta/OneDrive/Desktop/stock/Stock Predictions Model.keras', compile=False)

# Streamlit header
st.header('Stock Market Predictor')

# Input for stock symbol
stock = st.text_input("Enter Stock Symbol", "GOOG")

# Define the date range
start = '2012-01-01'
end = '2022-12-31'

# Download stock data
data = yf.download(stock, start, end)

# Display raw stock data
st.subheader('Raw Stock Data')
st.write(data)

# Train and test data split
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))

# Using the last 100 days of the training data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Prepare x and y
x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)
predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

# Calculate moving averages
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

# Plotting
# Plot Price vs Ma50
st.subheader('Price vs Ma50')
fig2 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'r', label='50-Day Moving Average')
plt.plot(data.Close, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)

# Plot Price vs Ma50 vs Ma100
st.subheader('Price vs Ma50 vs Ma100')
fig3 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'r', label='50-Day Moving Average')
plt.plot(ma_100_days, 'b', label='100-Day Moving Average')
plt.plot(data.Close, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig3)

# Plot Price vs Ma100 vs Ma200
st.subheader('Price vs Ma100 vs Ma200')
fig4 = plt.figure(figsize=(10, 8))
plt.plot(ma_100_days, 'b', label='100-Day Moving Average')
plt.plot(ma_200_days, 'purple', label='200-Day Moving Average')
plt.plot(data.Close, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig1 = plt.figure(figsize=(10, 8))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig1)