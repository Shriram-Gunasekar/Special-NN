import numpy as np

class SpikingNeuron:
    def __init__(self, threshold=1.0, tau=10, alpha=0.1):
        self.threshold = threshold  # Spike threshold
        self.tau = tau  # Membrane time constant
        self.alpha = alpha  # Learning rate
        self.membrane_potential = 0
        self.input_weights = None
        self.output_spike = 0

    def integrate(self, input_spike):
        # Integrate input spike
        self.membrane_potential -= self.membrane_potential / self.tau
        self.membrane_potential += input_spike

        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.output_spike = 1
            self.membrane_potential = 0
        else:
            self.output_spike = 0

    def update_weights(self, input_spikes, target_spike):
        # Compute error
        error = target_spike - self.output_spike

        # Update weights using SpikeProp rule
        self.input_weights += self.alpha * error * input_spikes

class SpikingNetwork:
    def __init__(self, num_inputs, num_outputs, threshold=1.0, tau=10, alpha=0.1):
        self.neuron = SpikingNeuron(threshold, tau, alpha)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neuron.input_weights = np.random.randn(num_inputs)  # Initialize weights

    def forward_pass(self, input_spikes):
        self.neuron.integrate(np.dot(input_spikes, self.neuron.input_weights))
        return self.neuron.output_spike

    def train(self, input_spikes, target_spike):
        self.forward_pass(input_spikes)
        self.neuron.update_weights(input_spikes, target_spike)

# Example usage
input_spikes = np.array([0, 1, 0])  # Input spike train
target_spike = 1  # Desired output spike

# Create a spiking neural network
snn = SpikingNetwork(num_inputs=len(input_spikes), num_outputs=1)

# Train the network
for _ in range(100):
    snn.train(input_spikes, target_spike)

# Test the trained network
output_spike = snn.forward_pass(input_spikes)
print('Output spike:', output_spike)
