import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define Logicon Projection Network
def logicon_projection_network(input_dim, latent_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(latent_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    return autoencoder, encoder

# Set hyperparameters
input_dim = X_train.shape[1]
latent_dim = 2  # 2-dimensional latent space for visualization purposes

# Create and compile the model
autoencoder, encoder = logicon_projection_network(input_dim, latent_dim)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Use the trained encoder for dimensionality reduction
encoded_X_train = encoder.predict(X_train)
encoded_X_test = encoder.predict(X_test)

# Visualize the results
plt.scatter(encoded_X_train[:, 0], encoded_X_train[:, 1], c=y_train, cmap='viridis')
plt.colorbar()
plt.xlabel('Latent Feature 1')
plt.ylabel('Latent Feature 2')
plt.title('Logicon Projection Network')
plt.show()
