import csv
from nn import create_network_architecture
from genetic import genetic_algorithm

# 1. Lê os dados do CSV
def read_csv(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        data = list(reader)
    return data

# 2. One-hot encoding simples
def encode_data(data):
    X = []
    y = []

    features = ['name', 'color', 'format']
    categories = {f: list(sorted(set(d[f] for d in data))) for f in features}

    for row in data:
        xi = []
        for f in features:
            one_hot = [0] * len(categories[f])
            idx = categories[f].index(row[f])
            one_hot[idx] = 1
            xi.extend(one_hot)
        X.append(xi)
        y.append(int(row['is_fruit']))
    return X, y, sum(len(v) for v in categories.values())

# 3. Função de fitness
def fitness_function(weights, seed=None):
    net = create_network_architecture(input_size)
    net.load_weights(weights)

    correct = 0
    for xi, yi in zip(X, y):
        if net.forward(xi) == yi:
            correct += 1
    return correct / len(y)

# -------------------------
# EXECUÇÃO
# -------------------------

data = read_csv("train.csv")
X, y, input_size = encode_data(data)

net = create_network_architecture(input_size)
individual_size = net.compute_num_weights()

best_weights, best_fitness = genetic_algorithm(
    individual_size=individual_size,
    population_size=100,
    fitness_function=fitness_function,
    target_fitness=1.0,
    generations=200
)

print("Melhor fitness encontrado:", best_fitness)
