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

    def compute_num_weights(self): # calcula o nr de pesos e biases que a rede neuronal vai ter. nao conta com o input.
        # Implement this. Remember to account for the biases.
        total = 0
        input_size = self.input_size # tamanho do input

        for n in self.hidden_architecture: # para cada layer com n neuronios
            total += (input_size + 1) * n # adiciona ao total o tamanho do input * n (nr de pesos) + n (biases)
            input_size = n # atualiza o tamanho do input para processar o proximo layer

        total += (input_size + 1) * 1  # quando acaba de processar os hidden layers faz o mesmo para a saida
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

    
    # funcao que gera o output da rede neuronal  (x e o estado do jogo - input)
    def forward(self, x):
        x = np.array(x)
        for i in range(len(self.hidden_architecture)): # percorre os hidden layers 
            W = self.hidden_weights[i]
            b = self.hidden_biases[i]
            x = self.hidden_activation(np.dot(x, W) + b) # vai calculando os outputs para cada layer. cada layer recebe como input o output do layer anterior.

        y = np.dot(x, self.output_weights) + self.output_bias
        return self.output_activation(y) # output final com a funcao de ativacao de output
        

def create_network_architecture(input_size):

    # Replace with your configuration

    hidden_fn = lambda x: 1 / (1 + np.exp(-x))
    output_fn = lambda x: 1 if x > 0 else -1
    return NeuralNetwork(input_size, (10,), hidden_fn, output_fn) # 1 hidden layer com 10 neuronios