import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define Cascade Correlation Neural Network
class CascadeCorrelationNN(nn.Module):
    def __init__(self, input_size):
        super(CascadeCorrelationNN, self).__init__()
        self.input_size = input_size
        self.output_size = 1  # Binary classification
        self.model = nn.Sequential(
            nn.Linear(input_size, 10),  # Initial layer with 10 neurons
            nn.Sigmoid(),
            nn.Linear(10, self.output_size)  # Output layer
        )
        self.cascade_threshold = 0.1  # Threshold for adding new neurons
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss_fn = nn.BCELoss()  # Binary cross-entropy loss

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.loss_fn(output.view(-1), y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if loss.item() < self.cascade_threshold:  # Add new neuron if loss is below threshold
                new_neuron = nn.Linear(self.input_size, 1)
                self.model.add_module(f'neuron_{epoch}', new_neuron)
                self.cascade_threshold *= 0.9  # Update threshold for next iteration

    def predict(self, X):
        with torch.no_grad():
            output = self.forward(X)
            predictions = (output >= 0.5).int().view(-1)
            return predictions.numpy()

# Instantiate and train the CCNN
input_size = X_train.shape[1]
ccnn = CascadeCorrelationNN(input_size)
ccnn.fit(X_train_tensor, y_train_tensor)

# Evaluate the model
y_pred = ccnn.predict(X_test_tensor)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
