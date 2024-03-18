import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras import layers, models

# Generate synthetic data
def generate_data(num_samples):
    x = np.linspace(0, 10, num_samples)
    y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)
    return x, y

# Wavelet transform function
def wavelet_transform(data, wavelet='db4', level=5):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return np.concatenate(coeffs)

# Create WNN model
def create_wnn(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(1))  # Output layer for regression
    return model

# Generate data
x_train, y_train = generate_data(1000)
x_test, y_test = generate_data(200)

# Perform wavelet transform on input data
x_train_wavelet = np.array([wavelet_transform(x) for x in x_train])
x_test_wavelet = np.array([wavelet_transform(x) for x in x_test])

# Create and compile WNN model
input_shape = x_train_wavelet.shape[1:]
wnn_model = create_wnn(input_shape)
wnn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
wnn_model.fit(x_train_wavelet, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

# Evaluate the model
loss = wnn_model.evaluate(x_test_wavelet, y_test)
print('Test Loss:', loss)
