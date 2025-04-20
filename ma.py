import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Optional Prophet dependency
try:
    from prophet import Prophet
    prophet_installed = True
except ModuleNotFoundError:
    prophet_installed = False

# Optional Streamlit dependency
try:
    import streamlit as st
    from dotenv import load_dotenv
    streamlit_installed = True
    load_dotenv()
except ModuleNotFoundError:
    streamlit_installed = False
    st = None  # Dummy fallback

if streamlit_installed:
    st.set_page_config(page_title="\U0001F4CA Finance Predictor", layout="wide")
    st.title("\U0001F680 AI-Driven Finance Predictor")
    st.markdown("Use deep learning and technical analysis to forecast stock movements.")


def fetch_stock_data(ticker, start=None, end=None):
    try:
        df = yf.download(ticker, start=start, end=end)
        df.index = pd.to_datetime(df.index)
        if df.empty:
            raise ValueError("No data found for the given stock and date range.")
        if streamlit_installed:
            st.write("\U0001F4CA Raw Fetched Data Preview:", df.head())
        return df
    except Exception as e:
        if streamlit_installed:
            st.error(f"Error fetching stock data: {e}")
        return None


def add_technical_indicators(df):
    try:
        df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = compute_rsi(df['Close'], window=14)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        if streamlit_installed:
            st.error(f"Error adding technical indicators: {e}")
        return df


def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_data(df, lookback=60):
    try:
        if df.shape[0] <= lookback:
            raise ValueError(f"Insufficient data: only {df.shape[0]} rows available, but {lookback + 1} needed. Please select a longer date range.")
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[["Close"]])
        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i - lookback:i])
            y.append(scaled[i][0])
        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)
        return X, y, scaler
    except Exception as e:
        if streamlit_installed:
            st.error(f"Error preparing data: {e}")
        return None, None, None


def train_model(X, y, epochs=5, batch_size=32):
    try:
        from keras.models import Sequential
        from keras.layers import Dense, LSTM

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)

        if streamlit_installed:
            st.subheader("\U0001F4CA Training Loss Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.history['loss'], name='Loss'))
            fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'))
            fig.update_layout(xaxis_title='Epochs', yaxis_title='Loss')
            st.plotly_chart(fig, use_container_width=True)

        return model
    except Exception as e:
        if streamlit_installed:
            st.error(f"Error training the model: {e}")
        return None

if streamlit_installed:
    st.sidebar.title("⚙️ Settings")
    epochs = st.sidebar.slider("Epochs", 1, 100, 5)
    batch_size = st.sidebar.slider("Batch Size", 8, 128, 32, step=8)
    lookback = st.sidebar.slider("Lookback Days", 30, 120, 60)

    tickers_input = st.text_input("🔍 Enter Stock Symbols (comma-separated)", value="AAPL,TSLA")
    symbols = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    start_date = st.date_input("📅 Start Date", datetime.date.today() - datetime.timedelta(days=365))
    end_date = st.date_input("📅 End Date", datetime.date.today())
    model_choice = st.radio("📌 Choose Prediction Model", ["LSTM", "Prophet"] if prophet_installed else ["LSTM"])
    threshold = st.slider("📣 Alert Threshold (USD)", 1, 20, 5)

    if st.button("📈 Predict"):
        for symbol in symbols:
            st.header(f"📊 {symbol} Forecast")
            df = fetch_stock_data(symbol, start=start_date, end=end_date)

            if df is not None and not df.empty:
                df = add_technical_indicators(df)

                if df is not None and df.shape[0] > 0:
                    if model_choice == "LSTM":
                        X, y, scaler = prepare_data(df, lookback=lookback)
                        if X is not None and X.shape[0] > 0:
                            model = train_model(X, y, epochs=epochs, batch_size=batch_size)
                            if model is not None:
                                last_window = df[["Close"]].tail(lookback).values
                                X_test = scaler.transform(last_window).reshape(1, lookback, 1)
                                forecast = []
                                input_seq = X_test
                                for _ in range(7):
                                    pred = model.predict(input_seq, verbose=0)
                                    forecast.append(pred[0][0])
                                    input_seq = np.append(input_seq[:, 1:, :], [[pred[0]]], axis=1)
                                forecast_inv = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

                                predicted_price = float(forecast_inv[0])
                                last_price = float(df['Close'].iloc[-1])
                                st.metric("📌 Next Day Prediction", f"${predicted_price:.2f}", delta=f"{predicted_price - last_price:+.2f}")
                                st.line_chart(pd.Series(forecast_inv, name="Forecasted Price"))

                    elif model_choice == "Prophet" and prophet_installed:
                        st.subheader("📈 Prophet Forecast (7 Days)")
                        df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                        df_p['ds'] = pd.to_datetime(df_p['ds'])
                        df_p['y'] = pd.to_numeric(df_p['y'], errors='coerce')
                        df_p.dropna(subset=['ds', 'y'], inplace=True)

                        m = Prophet()
                        m.fit(df_p)

                        future = m.make_future_dataframe(periods=int(7))
                        forecast = m.predict(future)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
                        fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name='Actual'))
                        st.plotly_chart(fig, use_container_width=True)

                st.subheader("📊 Data Preview")
                st.dataframe(df.tail(10))
            else:
                st.warning(f"⚠️ No data for {symbol}. Try a different symbol or a longer date range.")

if __name__ == "__main__":
    os.system("streamlit run m.py")
