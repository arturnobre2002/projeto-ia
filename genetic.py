import random
#evolui um vetor de pesos (lista de floats) para serem usados na rede neuronal
def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]
#Tipo: Crossover de 1 ponto (one-point crossover) escolhe um ponto de corte aleatório, pega na primeira parte do parent1, junta com a segunda parte do parent2
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1) #refletir se usamos corte aleatório
    return parent1[:point] + parent2[point:]

#inclui aqui varios tipos de mutacao para depois escolhermos estas 2 foram ensinadas em aula

def swap_mutation(individual, mutation_rate): #Troca dois pesos de lugar → muda a topologia sem alterar os valores
    ind = individual[:]  # cópia
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind
def scramble_mutation(individual, mutation_rate): #Pega num pedaço dos pesos e embaralha a ordem (mantém os valores, mas muda a estrutura)
    ind = individual[:]
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(ind)), 2))
        sub = ind[i:j]
        random.shuffle(sub)
        ind[i:j] = sub
    return ind
#esta foi a q o chat fez: mutacao gaussiana simplificada Com probabilidade mutation_rate, soma um valor aleatório entre -0.1 e +0.1
def mutate(individual, mutation_rate):
    return [
        gene + random.uniform(-0.1, 0.1) if random.random() < mutation_rate else gene
        for gene in individual
    ]


def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.2, mutation_rate=0.05):
    population = generate_population(individual_size, population_size)
    best_individual = None
    #debater este código
    for gen in range(generations):
        scored = [(ind, fitness_function(ind)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        if best_individual is None or scored[0][1] > best_individual[1]:
            best_individual = scored[0]

        if best_individual[1] >= target_fitness:
            break

        elite_count = int(population_size * elite_rate)
        new_population = [ind for ind, _ in scored[:elite_count]]

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(scored[:elite_count], k=2)
            child = crossover(parent1[0], parent2[0])
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_individual # This is expected to be a pair (individual, fitness)

#vr do ficheiro best_individuals.txt dps guardar a melhor geração