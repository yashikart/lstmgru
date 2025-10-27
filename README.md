import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

# --- Generate simple sine wave data ---
x = np.linspace(0, 100, 1000)
y = np.sin(x)

# Prepare sequence data (past 20 steps → next value)
def create_dataset(y, step=20):
    X, Y = [], []
    for i in range(len(y)-step):
        X.append(y[i:i+step])
        Y.append(y[i+step])
    return np.array(X), np.array(Y)

X, Y = create_dataset(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

# --- LSTM Model ---
lstm_model = Sequential([
    LSTM(50, activation='tanh', input_shape=(20,1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X, Y, epochs=10, verbose=0)

# --- GRU Model ---
gru_model = Sequential([
    GRU(50, activation='tanh', input_shape=(20,1)),
    Dense(1)
])
gru_model.compile(optimizer='adam', loss='mse')
gru_model.fit(X, Y, epochs=10, verbose=0)

# --- Compare Predictions ---
pred_lstm = lstm_model.predict(X, verbose=0)
pred_gru = gru_model.predict(X, verbose=0)

# --- Plot results ---
plt.figure(figsize=(10,4))
plt.plot(y, label='True')
plt.plot(np.arange(20, len(pred_lstm)+20), pred_lstm, label='LSTM')
plt.plot(np.arange(20, len(pred_gru)+20), pred_gru, label='GRU')
plt.legend()
plt.title("LSTM vs GRU for Time-Series Forecasting")
plt.show()
