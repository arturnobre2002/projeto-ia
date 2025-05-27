import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_architecture, hidden_activation, output_activation):
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.hidden_weights = []
        self.hidden_biases = []
        self.output_weights = None
        self.output_bias = None

    def compute_num_weights(self):
        # Número total de pesos e biases
        total_weights = 0
        input_size = self.input_size
        for n in self.hidden_architecture:
            total_weights += (input_size + 1) * n  # pesos + bias
            input_size = n
        # +1 para bias da saída, +n para pesos da camada anterior
        total_weights += 1 + input_size
        return total_weights

    def load_weights(self, weights):
        w = np.array(weights)

        self.hidden_weights = []
        self.hidden_biases = []

        start_w = 0
        input_size = self.input_size
        for n in self.hidden_architecture:
            end_w = start_w + (input_size + 1) * n
            self.hidden_biases.append(w[start_w:start_w + n])
            self.hidden_weights.append(w[start_w + n:end_w].reshape(input_size, n))
            start_w = end_w
            input_size = n

        self.output_bias = w[start_w]
        self.output_weights = w[start_w + 1:]

    def forward(self, x):
        a = np.array(x, dtype=float)

        for w, b in zip(self.hidden_weights, self.hidden_biases):
            z = np.dot(a, w) + b
            a = self.hidden_activation(z)

        # camada de saída
        z = np.dot(a, self.output_weights) + self.output_bias
        return self.output_activation(z)


def create_network_architecture(input_size):
    hidden_fn = lambda x: 1 / (1 + np.exp(-x))  # sigmoid
    output_fn = lambda x: 1 if x > 0 else -1    # perceptron-like output
    return NeuralNetwork(input_size, (8, 8), hidden_fn, output_fn)
