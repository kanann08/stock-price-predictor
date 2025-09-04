"""
This is a web application that allows users to enter a stock ID and view the predicted stock price for the next day.
This uses Keras and LSTM.
"""

import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model  # type: ignore
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Stock Price Predictor")

# User input
stock = st.text_input("Enter the Stock ID", "GOOG")

# Date range
end = datetime.now()
start_date = st.date_input("Start Date", datetime(end.year - 20, end.month, end.day))
end_date = st.date_input("End Date", datetime.now())
google_data = yf.download(stock, start=start_date, end=end_date)

if google_data.empty:
    st.error("No data found for this stock. Check the stock ID or date range.")
    st.stop()

st.subheader("Stock Data")
st.write(google_data)

# Load LSTM model
model = load_model("Latest_stock_price_model.keras")
st.text(model.summary())

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data.Close, 'b')
    if extra_data and extra_dataset is not None:
        plt.plot(extra_dataset)
    return fig

# Moving averages plots
for days in [250, 200, 100]:
    ma_col = f'MA_for_{days}_days'
    google_data[ma_col] = google_data.Close.rolling(days).mean()
    st.subheader(f'Original Close Price and MA for {days} days')
    st.pyplot(plot_graph((15,6), google_data[ma_col], google_data, 0))

# MA 100 vs 250
st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Prepare data for prediction
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

if len(x_test) < 100:
    st.warning("Not enough data to generate predictions. Try a longer date range or different stock.")
else:
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(x_test)

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    st.text(f"x_data shape: {x_data.shape}")
    st.text(f"y_data shape: {y_data.shape}")

    predictions = model.predict(x_data)
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data.reshape(-1,1))

    # Prepare dataframe for plotting
    ploting_data = pd.DataFrame(
        {
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        },
        index = google_data.index[splitting_len+100:]
    )

    st.subheader("Original values vs Predicted values")
    st.write(ploting_data)

    st.subheader('Original Close Price vs Predicted Close price')
    fig = plt.figure(figsize=(15,6))
    plt.plot(pd.concat([google_data.Close[:splitting_len+100], ploting_data['predictions']], axis=0))
    plt.legend(["Data - not used", "Predicted Test data"])
    st.pyplot(fig)
