
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout

# ========== 1. Download Data ==========
stock = "RELIANCE.NS"   # <--- change if needed
data = yf.download(stock, start="2018-01-01", end="2024-12-31")
close_prices = data[['Close']]

# ========== 2. Scaling ==========
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(close_prices)

# ========== 3. Create sequences ==========
X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)

# reshape for RNN
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ========== 4. LSTM MODEL ==========
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X, y, epochs=20, batch_size=32, verbose=0)

# ========== 5. GRU MODEL ==========
gru_model = Sequential([
    GRU(50, return_sequences=True, input_shape=(X.shape[1],1)),
    Dropout(0.2),
    GRU(50, return_sequences=True),
    Dropout(0.2),
    GRU(50),
    Dropout(0.2),
    Dense(1)
])
gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.fit(X, y, epochs=20, batch_size=32, verbose=0)

# ========== 6. Predict next 7 days ==========
def forecast_next_7(model):
    last_60 = scaled[-60:]
    prediction_list = []
    seq = last_60.reshape(1,60,1)
    for _ in range(7):
        pred = model.predict(seq, verbose=0)[0][0]
        prediction_list.append(pred)
        seq = np.append(seq[:,1:,:], [[[pred]]], axis=1)
    return scaler.inverse_transform(np.array(prediction_list).reshape(-1,1))

lstm_pred = forecast_next_7(lstm_model)
gru_pred  = forecast_next_7(gru_model)

# ========== 7. Print ==========
print(" LSTM next 7 days prediction:")
print(lstm_pred)
print("\nGRU next 7 days prediction:")
print(gru_pred)

# ========== 8. Plot ==========
plt.figure(figsize=(10,5))
plt.plot(close_prices.index, close_prices['Close'], label="Historical Price")
plt.scatter(range(len(close_prices), len(close_prices)+7), lstm_pred, label="LSTM Forecast", marker="o")
plt.scatter(range(len(close_prices), len(close_prices)+7), gru_pred, label="GRU Forecast", marker="x")
plt.title(f"{stock} LSTM vs GRU Forecast (7 days)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()











