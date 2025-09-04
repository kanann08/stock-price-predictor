

📈 Stock Price Predictor using Keras LSTM
🔍 Overview

This project is a Stock Price Prediction Web App that forecasts future stock prices of publicly traded companies using a Keras LSTM (Long Short-Term Memory) model. Users can enter stock ticker symbols (e.g., AAPL, MSFT) to view historical stock data, visualize trends, and see predicted prices for upcoming days.

The app is built using Python and Streamlit for a user-friendly web interface.

📁 Files & Structure
stock-price-predictor/
│
├── app.py              # Main Streamlit application
├── Latest_stock_price_model.keras  # Trained Keras LSTM model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

📚 Libraries & Frameworks Used

Keras - Deep learning (LSTM model)

Streamlit - Web app interface

Pandas & NumPy - Data handling and numerical computations

Matplotlib - Data visualization

yfinance - Fetch historical stock data from Yahoo Finance

scikit-learn - Data preprocessing (MinMaxScaler)

⚙️ How It Works
1. Data Collection

Users input a stock ticker (e.g., GOOG).

Historical stock data is fetched using the yfinance API.

2. Data Preprocessing

Missing values are handled.

Sliding windows are created for the LSTM model (using the last 100 days of stock prices).

Data is scaled using MinMaxScaler for the model.

3. LSTM Model

Input shape: (100, 1)

Architecture: LSTM layer with 50 units → Dense layer with 1 unit

Loss function: Mean Squared Error (mse)

Optimizer: Adam

4. Predictions & Visualization

Predicts future stock prices based on the LSTM model.

Plots original stock prices vs predicted prices.

Moving averages (100, 200, 250 days) can be visualized.

🖥️ Running the App

Clone the repository:

git clone https://github.com/<your-username>/stock-price-predictor.git
cd stock-price-predictor


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


Open the URL provided by Streamlit in your browser (usually http://localhost:8501).

📈 Example Use Cases

Forecast stock prices for investment decisions.

Compare predicted vs historical stock trends.

Educational tool for learning time series forecasting with LSTM.

🚀 Future Work

Include other models (Prophet, ARIMA, Hybrid) for comparison.

Add hyperparameter tuning for improved accuracy.

Integrate technical indicators and financial news sentiment analysis.

Deploy on Streamlit Cloud or Heroku for public access.