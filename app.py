# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime, timedelta
# import pickle
# from keras.models import load_model
# from nselib import capital_market
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# # --- Streamlit Page Config ---
# st.set_page_config(
#     page_title="Stock Predictor App",
#     layout="wide",
#     page_icon="📈",
# )

# # --- Background ---
# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://images.unsplash.com/photo-1526378722484-bd91ca387e72?fit=crop&w=1950&q=80");
#              background-size: cover;
#              background-attachment: fixed;
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# add_bg_from_url()

# # --- Title ---
# st.markdown("<h1 style='text-align: center; color: white;'>📈 Stock Price Predictor - RNN Model</h1>", unsafe_allow_html=True)

# # --- Sidebar Inputs ---
# st.sidebar.header("⚙️ Input Configuration")
# symbol = st.sidebar.text_input("Enter NSE Ticker Symbol", value="SBIN")
# from_date = st.sidebar.date_input("From Date", datetime(2021, 1, 1))
# to_date = st.sidebar.date_input("To Date", datetime.today())

# # --- Paths ---
# DATA_PATH = "Data/raw.csv"
# SCALER_PATH = "scaler.pkl"
# MODEL_PATH = "simple_rnn_base.h5"

# # --- Function: Fetch & Clean Data ---
# @st.cache_data


# # Import your data fetching library, e.g. capital_market
# # from some_module import capital_market  

# def fetch_and_save_nse_data(symbol, start_date='01-01-2021', end_date='30-09-2025', save_path=None):
#     # Fetch data
#     data = capital_market.price_volume_data(symbol=symbol, from_date=start_date, to_date=end_date)
    
#     # Filter EQUITY SERIES only
#     data_eq = data[data['Series'] == 'EQ']
    
#     # Select relevant columns
#     cols = ['Symbol', 'Series', 'Date', 'PrevClose', 'OpenPrice', 'HighPrice',
#             'LowPrice', 'ClosePrice', 'TotalTradedQuantity']
#     data_eq = data_eq[cols]
    
#     # Convert columns to float
#     n_cols = ['PrevClose', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice']
#     for col in n_cols:
#         data_eq[col] = data_eq[col].astype(float)
    
#     # Convert volume (remove commas and convert to numeric)
#     data_eq['Volume'] = pd.to_numeric(data_eq['TotalTradedQuantity'].str.replace(',', ''), errors='coerce').astype('Int64')
    
#     # Convert Date column to datetime
#     data_eq['Date'] = pd.to_datetime(data_eq['Date'], format='%d-%b-%Y', errors='coerce')
    
#     # Drop the original TotalTradedQuantity column
#     data_eq = data_eq.drop(columns='TotalTradedQuantity', axis=1)
    
#     print(f'Data preprocessing (`Data Types`) conversion is done...')

#     # Select only Date, ClosePrice, Volume
#     df = data_eq[['Date', 'ClosePrice']]
    
#     # Set Date as index
#     df.set_index('Date', inplace=True)
    
#     # Set default save path if not provided
#     if save_path is None:
#         # Create folder if not exists
#         os.makedirs('Data', exist_ok=True)
#         save_path = f'Data/{symbol}_raw.csv'
    
#     data_eq.to_csv(save_path, index=False, header=True)
#     print(f'Data for {symbol} saved to {save_path}!')
    
#     return df  # return processed df for plotting/use in Streamlit

# # Streamlit app starts here
# st.title("📊 NSE Stock Data Fetch & Prediction Dashboard")

# # Sidebar for inputs
# with st.sidebar:
#     st.header("Enter Stock Details")
#     ticker = st.text_input("Ticker Symbol (e.g., SBIN)")
#     from_date = st.date_input("From Date", value=pd.to_datetime("2021-01-01"))
#     to_date = st.date_input("To Date", value=pd.to_datetime("2025-09-30"))
    
#     if st.button("Fetch & Save Data"):
#         if ticker:
#             # Format dates as string in 'DD-MM-YYYY' or 'DD-MMM-YYYY' if needed by function
#             start_str = from_date.strftime('%d-%m-%Y')
#             end_str = to_date.strftime('%d-%m-%Y')
            
#             st.write(f"Fetching data for {ticker} from {start_str} to {end_str} ...")
            
#             try:
#                 df = fetch_and_save_nse_data(ticker, start_date=start_str, end_date=end_str)
#                 st.success(f"Data fetched and saved for {ticker}!")
                
#                 # Plot the ClosePrice line plot
#                 st.line_chart(df['ClosePrice'])
                
#                 # Save path info
#                 st.write(f"Saved CSV file: Data/{ticker}_raw.csv")
                
#             except Exception as e:
#                 st.error(f"Error fetching data: {e}")
#         else:
#             st.warning("Please enter a ticker symbol.")

# # Continue with other tabs for prediction, etc. as per your full app design


# # --- Prediction ---
# st.markdown("## 🔮 Predict Next Day Close Price")

# def predict_next_day(df, lookback=20, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
#     # Load model and scaler
#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)
#     model = load_model(model_path, compile=False)

#     # Prepare last N days
#     last_data = df[-lookback:]
#     data_array = last_data.values
#     data_scaled = scaler.transform(data_array)
#     input_seq = data_scaled.reshape(1, lookback, data_scaled.shape[1])

#     pred_scaled = model.predict(input_seq)
    
#     dummy = np.tile(data_scaled[-1], (pred_scaled.shape[1], 1))
#     dummy[:, 0] = pred_scaled.flatten()
#     pred_inverse = scaler.inverse_transform(dummy)[:, 0]
    
#     return pred_inverse[0]

# # Predict Button
# if st.button("🚀 Predict Next Day Price"):
#     if os.path.exists(DATA_PATH):
#         df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
#         try:
#             pred = predict_next_day(df)
#             st.success(f"✅ Predicted Close Price for next day: ₹{pred:.2f}")
#         except Exception as e:
#             st.error(f"❌ Prediction failed: {e}")
#     else:
#         st.warning("Please fetch data first!")

# # --- Disclaimer ---
# st.markdown("---")
# st.markdown(
#     "<p style='text-align: center; color: white; font-size: 14px;'>⚠️ This dashboard is for <strong>educational purposes only</strong>. Do not use for real trading or investment decisions.</p>",
#     unsafe_allow_html=True
# )

import streamlit as st
import pandas as pd
import numpy as np
import os,sys
from datetime import datetime
import pickle
from keras.models import load_model
from nselib import capital_market
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import warnings
import sys

print("warnings in sys.modules:", 'warnings' in sys.modules)  # Should print True
print("warnings module:", warnings)


# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Stock Predictor App",
    layout="wide",
    page_icon="📈",
)

# --- Title ---
st.markdown("<h1 style='text-align: center; color: red;'>📈 Stock Price Predictor - RNN Model</h1>", unsafe_allow_html=True)

# --- Paths ---
DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "simple_rnn_base.keras"

# --- Function: Fetch & Clean Data ---
@st.cache_data
def fetch_and_save_nse_data(symbol, start_date, end_date, save_path=None):
    data = capital_market.price_volume_data(symbol=symbol, from_date=start_date, to_date=end_date)
    data_eq = data[data['Series'] == 'EQ']
    cols = ['Symbol', 'Series', 'Date', 'PrevClose', 'OpenPrice', 'HighPrice',
            'LowPrice', 'ClosePrice', 'TotalTradedQuantity']
    data_eq = data_eq[cols]
    
    for col in ['PrevClose', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice']:
        data_eq[col] = data_eq[col].astype(str).str.replace(',', '').astype(float)
    data_eq['Volume'] = pd.to_numeric(data_eq['TotalTradedQuantity'].astype(str).str.replace(',', ''), errors='coerce').astype('Int64')
    data_eq['Date'] = pd.to_datetime(data_eq['Date'], format='%d-%b-%Y', errors='coerce')
    data_eq.drop(columns='TotalTradedQuantity', inplace=True)

    if save_path is None:
        save_path = os.path.join(DATA_DIR, f"{symbol}_raw.csv")
    data_eq.to_csv(save_path, index=False)

    df = data_eq[['Date', 'ClosePrice']].set_index('Date')
    return df, save_path

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Input Configuration")
symbol = st.sidebar.text_input("Enter NSE Ticker Symbol", value="SBIN")
from_date = st.sidebar.date_input("From Date", datetime(2025, 9, 1))
to_date = st.sidebar.date_input("To Date", datetime.today())

csv_path = os.path.join(DATA_DIR, f"{symbol}_raw.csv")
df = pd.DataFrame()
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'])

if st.sidebar.button("Fetch & Save Data"):
    if symbol:
        start_str = from_date.strftime('%d-%m-%Y')
        end_str = to_date.strftime('%d-%m-%Y')
        st.write(f"Fetching data for {symbol} from {start_str} to {end_str} ...")
        try:
            df, _ = fetch_and_save_nse_data(symbol, start_date=start_str, end_date=end_str, save_path=csv_path)
            st.success(f"Data fetched and saved for {symbol}!")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

if not df.empty:
    st.subheader("Stock Data")
    st.write(df.head(20))

    # Candlestick Chart
    full_df = pd.read_csv(csv_path, parse_dates=['Date'])
    fig_candle = go.Figure(data=[go.Candlestick(
        x=full_df['Date'],
        open=full_df['OpenPrice'],
        high=full_df['HighPrice'],
        low=full_df['LowPrice'],
        close=full_df['ClosePrice']
    )])
    fig_candle.update_layout(
        title=f'{symbol} Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_candle)

    # Line Chart of Closing Prices
    st.subheader("Stock Closing Prices")
    fig_line, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['ClosePrice'], color='blue', linewidth=2)
    ax.set_title(f"{symbol} Stock Closing Prices", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Closing Price (INR)", fontsize=12)
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_line)

# --- Prediction ---
st.markdown("## 🔮 Predict Next Day Close Price")

def predict_next_day(df, lookback=20, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    model = load_model(model_path, compile=False)

    last_data = df[-lookback:]
    data_array = last_data.values
    data_scaled = scaler.transform(data_array)
    input_seq = data_scaled.reshape(1, lookback, data_scaled.shape[1])

    pred_scaled = model.predict(input_seq)

    dummy = np.tile(data_scaled[-1], (pred_scaled.shape[1], 1))
    dummy[:, 0] = pred_scaled.flatten()
    pred_inverse = scaler.inverse_transform(dummy)[:, 0]

    return pred_inverse[0]

if st.button("🚀 Predict Next Day Price"):
    if os.path.exists(csv_path) and not df.empty:
        df_pred = pd.read_csv(csv_path, parse_dates=['Date'])
        df_pred = df_pred[['Date', 'ClosePrice']]  # include Date here

        try:
            pred = predict_next_day(df_pred[['ClosePrice']])
            
            last_date = df_pred['Date'].iloc[-1]
            last_close = df_pred['ClosePrice'].iloc[-1]
            
            # Calculate the next date (assuming daily frequency, skip weekends if needed)
            next_date = last_date + pd.Timedelta(days=1)
            
            st.success(
                f"✅ Previous Date: {last_date.date()} | Close Price: ₹{last_close:.2f}\n\n"
                f"🔮 Predicted Next Date: {next_date.date()} | Predicted Close Price: ₹{pred:.2f}"
            )
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
    else:
        st.warning("Please fetch data first!")


# --- Disclaimer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: black; font-size: 14px;'>⚠️ This dashboard is for <strong>educational purposes only</strong>. Do not use for real trading or investment decisions.</p>",
    unsafe_allow_html=True
)

