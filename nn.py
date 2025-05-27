import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_architecture, hidden_activation, output_activation):
        self.input_size = input_size
        # hidden_architecture is a tuple with the number of neurons in each hidden layer
        # e.g. (5, 2) corresponds to a neural network with 2 hidden layers in which the first has 5 neurons and the second has 2
        self.hidden_architecture = hidden_architecture
        # The activations are functions 
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def compute_num_weights(self):
        # Implement this. Remember to account for the biases.
        total = 0
        input_size = self.input_size

        for n in self.hidden_architecture:
            total += (input_size + 1) * n
            input_size = n

        total += (input_size + 1) * 1  # saída
        return total

    def load_weights(self, weights):
        w = np.array(weights)

        self.hidden_weights = []
        self.hidden_biases = []

        start_w = 0
        input_size = self.input_size
        for n in self.hidden_architecture:
            end_w = start_w + (input_size + 1) * n
            self.hidden_biases.append(w[start_w:start_w+n])
            self.hidden_weights.append(w[start_w+n:end_w].reshape(input_size, n))
            start_w = end_w
            input_size = n

        self.output_bias = w[start_w]
        self.output_weights = w[start_w+1:]

    #A função nn.forward(state) devolve -1 ou 1 com base na entrada state, que representa o estado do jogo.
    #state = np.array([cesta_x,item1_x, item1_y, item1_tipo, item2_x, item2_y, item2_tipo, item3_x, item3_y, item3_tipo])
    #-1 esquerda 1 direita serve para ver para onde mover o cesto
    def forward(self, x):
        x = np.array(x)
        for i in range(len(self.hidden_architecture)):
            W = self.hidden_weights[i]
            b = self.hidden_biases[i]
            x = self.hidden_activation(np.dot(x, W) + b)

        y = np.dot(x, self.output_weights) + self.output_bias
        return self.output_activation(y)
        

def create_network_architecture(input_size):

    # Replace with your configuration

    hidden_fn = lambda x: 1 / (1 + np.exp(-x))
    output_fn = lambda x: 1 if x > 0 else -1
    return NeuralNetwork(input_size, (10,), hidden_fn, output_fn) #so c 1 neuronio, input size vai ser o state size do main que é 10 para um hidden layer meter (N,)