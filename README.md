
sirs practical 


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from sklearn.manifold import TSNE

# 1. Load and Preprocess the MNIST Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# Flatten the 28x28 images into vectors of 784 elements
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# 2. Build the Autoencoder Model
# This is the size of our encoded representations
encoding_dim = 32 # 32 floats -> compression of factor 24.5, assuming the 
# This is our input placeholder
input_img = Input(shape = (784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation = 'relu')(input_img)
# "decoded" is the lossy reconstruction of the input
# CORRECTED THIS LINE: The input to this layer should be 'encoded'
decoded = Dense(784, activation = 'sigmoid')(encoded)
# This model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# This model maps an input to its encoded representation 
encoder = Model(input_img, encoded)
# Create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape = (encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
# Compile the autoencoder
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
# 3. Train the Autoencoder
history = autoencoder.fit(x_train, x_train,
 epochs = 50,
 batch_size = 256, 
 shuffle = True,
 validation_data = (x_test, x_test))
# 4. Predict on the test data to get the reconstructed images
decoded_imgs = autoencoder.predict(x_test)
# 5. Visualize the Original and Reconstructed Images
n = 10 # How many digits we will display
plt.figure(figsize = (20, 4))
for i in range(n):
 # Display original
 ax = plt.subplot(2, n, i + 1)
 plt.imshow(x_test[i].reshape(28, 28))
 plt.gray()
 ax.get_xaxis().set_visible(False)
 ax.get_xaxis().set_visible(False)
  if i == 0:
 ax.set_title('Original Images', loc = 'left')
 # Display reconstruction
 ax = plt.subplot(2, n, i + 1 + n)
 plt.imshow(decoded_imgs[i].reshape(28, 28))
 plt.gray()
 ax.get_xaxis().set_visible(False)
 ax.get_xaxis().set_visible(False)
 if i == 0:
 ax.set_title('Original Images', loc = 'left')
plt.show()
# 7. Visualize the Latent Space using t-SNE
# Use the encoder to get the latent representation of the test data
encoded_imgs = encoder.predict(x_test)
# Use t-SNE to reduce the dimensionality of the latent space to 2D
tsne = TSNE(n_components = 2, random_state = 42)
encoded_imgs_2d = tsne.fit_transform(encoded_imgs)
# Plot the 2D latent space
plt.figure(figsize = (12, 10))
plt.scatter(encoded_imgs_2d[:, 0], encoded_imgs_2d[:, 1], c = y_test, cmap
plt.colorbar()
plt.title('t-SNE visualization of the MNIST latent space')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.show()

















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











