# 📈 Stock Price Predictor using Keras and LSTM

## 🔍 Overview
This project is a **Stock Price Prediction Web App** that forecasts future stock prices of publicly traded companies using a **Keras LSTM (Long Short-Term Memory) model**. Users can enter stock ticker symbols (e.g., AAPL, MSFT, GOOG) to view historical stock data, visualize trends, and see predicted prices for upcoming days.

The app is built using **Python** and **Streamlit**, providing a simple and interactive interface for stock price prediction.


## 📚 Libraries & Frameworks Used

- **Keras** – Deep learning (LSTM model)  
- **Streamlit** – Web app interface  
- **Pandas & NumPy** – Data handling and numerical computations  
- **Matplotlib** – Data visualization  
- **yfinance** – Fetch historical stock data from Yahoo Finance  
- **scikit-learn** – Data preprocessing (MinMaxScaler)  



## ⚙️ How It Works

### 1. Data Collection
- Users input a stock ticker symbol (e.g., GOOG).  
- Historical stock data is fetched from Yahoo Finance using the `yfinance` API.  

### 2. Data Preprocessing
- Handle missing values if present.  
- Create sliding windows for the LSTM model using the **last 100 days** of stock prices.  
- Scale the data with `MinMaxScaler` to prepare it for the LSTM model.  

### 3. LSTM Model
- **Input shape:** `(100, 1)`  
- **Architecture:**  
  - LSTM layer with 50 units  
  - Dense layer with 1 unit (linear activation)  
- **Loss function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  

### 4. Predictions & Visualization
- Predicts future stock prices based on the trained LSTM model.  
- Visualize the **original vs predicted stock prices**.  
- Plot **moving averages** for 100, 200, and 250 days.

📈 Example Use Cases

-Forecast stock prices for investment decisions.
-Compare predicted vs historical stock trends.
-Educational tool for learning time series forecasting with LSTM models.

🚀 Future Enhancements

-Integrate other forecasting models (Prophet, ARIMA, Hybrid) for comparison.
-Add hyperparameter tuning for better accuracy.
-Include technical indicators and financial news sentiment analysis.
-Deploy on Streamlit Cloud or Heroku for public access.
