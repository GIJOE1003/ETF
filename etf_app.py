import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load predictions and data
df = pd.read_csv("predicted_etf_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Load trained models
rf_model = joblib.load("rf_classifier.joblib")
gbr_model = joblib.load("gbr_regressor.joblib")

st.title("📈 ETF Signal & Volatility Forecast Dashboard")

# Sidebar filters
selected_ticker = st.sidebar.selectbox("Select an ETF", df['Ticker'].unique())
date_range = st.sidebar.slider("Select Date Range", min_value=df['Date'].min().date(), max_value=df['Date'].max().date(), value=(df['Date'].min().date(), df['Date'].max().date()))

# Filter data
filtered_df = df[(df['Ticker'] == selected_ticker) & (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]

# Price + Indicators Plot
st.subheader(f"{selected_ticker} - Price with Technical Indicators")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(filtered_df['Date'], filtered_df['Close'], label='Close')
ax.plot(filtered_df['Date'], filtered_df['EMA20'], label='EMA 20')
ax.plot(filtered_df['Date'], filtered_df['EMA50'], label='EMA 50')
ax.plot(filtered_df['Date'], filtered_df['SMA200'], label='SMA 200')
ax.legend()
ax.set_title("Price with EMA/SMA")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(True)
st.pyplot(fig)

# Buy/Sell Signal Scatter
st.subheader("📍 Buy / Sell Signals")
buy_signals = filtered_df[filtered_df['Signal'] == 1]
sell_signals = filtered_df[filtered_df['Signal'] == -1]

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(filtered_df['Date'], filtered_df['Close'], label='Close', alpha=0.5)
ax2.scatter(buy_signals['Date'], buy_signals['Close'], color='green', marker='^', label='Buy', s=80)
ax2.scatter(sell_signals['Date'], sell_signals['Close'], color='red', marker='v', label='Sell', s=80)
ax2.legend()
ax2.set_title("Buy/Sell Signals")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.grid(True)
st.pyplot(fig2)

# Run Prediction Buttons
if st.sidebar.button("Run Signal Prediction"):
    st.subheader("📈 Predicted Signals")
    X_features = filtered_df[['Close', 'High', 'Low', 'Return', 'Volatility7', 'MA5', 'Close_MA5_diff',
                              'SMA20', 'EMA20', 'SMA50', 'EMA50', 'SMA200', 'EMA200', 'TR', 'ATR7']]
    filtered_df['Predicted_Signal'] = rf_model.predict(X_features)
    st.write(filtered_df[['Date', 'Close', 'Signal', 'Predicted_Signal']].tail(10))

if st.sidebar.button("Forecast Volatility"):
    st.subheader("📉 Forecasted Volatility (ATR7%)")
    X_features = filtered_df[['Close', 'High', 'Low', 'Return', 'Volatility7', 'MA5', 'Close_MA5_diff',
                              'SMA20', 'EMA20', 'SMA50', 'EMA50', 'SMA200', 'EMA200', 'TR', 'ATR7']]
    filtered_df['Predicted_ATR7_pct'] = gbr_model.predict(X_features)
    st.line_chart(filtered_df.set_index('Date')[['ATR7_pct', 'Predicted_ATR7_pct']])

# Optional: Show Raw Data
if st.checkbox("Show raw data"):
    st.write(filtered_df[['Date', 'Ticker', 'Close', 'Signal', 'ATR7_pct']])
