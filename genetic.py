import random


def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

# one-point crossover
def crossover(parent1, parent2):
    point = len(parent1) // 2 # ponto que parte ao meio os parents
    return parent1[:point] + parent2[point:] # pega na primeira parte do parent1, junta com a segunda parte do parent2

# scramble mutation
def scramble_mutation(individual, mutation_rate): #Pega num pedaço dos pesos e baralha a ordem
    ind = individual[:]
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(ind)), 2)) # escolhe duas posicoes aleatorias e ordena-as para garantir i< j
        sub = ind[i:j] # vai buscar o segmento entre i e j
        random.shuffle(sub) # baralha
        ind[i:j] = sub # substitui o segmento antigo pelo baralhado
    return ind

# algoritmo genetico
def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.2, mutation_rate=0.05):
    population = generate_population(individual_size, population_size)
    best_individual = None 
    
    for gen in range(generations): # para cada geracao
        scored = [(ind, fitness_function(ind)) for ind in population] # scored e uma lista de pares (individuo, fitness), tem todos os individuos da populacao
        scored.sort(key=lambda x: x[1], reverse=True) # ordena scored pela ordem decrescente de fitness

        if best_individual is None or scored[0][1] > best_individual[1]: 
            best_individual = scored[0] # atualiza o best_individual se tiver maior fitness que o anterior

        if best_individual[1] >= target_fitness: # se atingiu o target_fitness devolve esse individuo
            break

        elite_count = int(population_size * elite_rate) # numero de individuos a manter
        new_population = [ind for ind, _ in scored[:elite_count]] # guarda os melhores individuos na nova populacao (ignora o fitness)

        while len(new_population) < population_size: # enquanto o tamanho da nova populacao nao chegar ao tamanho da populacao inicial
            parent1, parent2 = random.choices(scored[:elite_count], k=2) # escolhe dois individuos aleatoriamente da lista dos melhores individuos para serem parents 
            child = crossover(parent1[0], parent2[0]) # produz a child a partir dos parents
            child = scramble_mutation(child, mutation_rate) # faz a mutacao da child
            new_population.append(child) # adiciona a child mutada a nova populacao

        population = new_population # atualiza a populacao

    return best_individual # This is expected to be a pair (individual, fitness)

