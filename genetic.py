import random
import copy

def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

def evaluate_population(population, fitness_function, seed=None):
    scored = []
    for individual in population:
        if seed is not None:
            score = fitness_function(individual, seed)
        else:
            score = fitness_function(individual)
        scored.append((individual, score))
    return scored

def select_elite(scored_population, elite_rate):
    scored_population.sort(key=lambda x: x[1], reverse=True)
    elite_count = max(1, int(len(scored_population) * elite_rate))
    return scored_population[:elite_count]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual, mutation_rate):
    return [
        gene + random.uniform(-0.5, 0.5) if random.random() < mutation_rate else gene
        for gene in individual
    ]

def genetic_algorithm(
    individual_size,
    population_size,
    fitness_function,
    target_fitness,
    generations,
    elite_rate=0.2,
    mutation_rate=0.1,
    seed=None
):
    # Gerar população inicial
    population = generate_population(individual_size, population_size)

    for generation in range(generations):
        # Avaliar fitness
        scored_population = evaluate_population(population, fitness_function, seed)

        # Verificar se alguém atingiu o fitness alvo
        best_individual, best_fitness = max(scored_population, key=lambda x: x[1])
        print(f"Geração {generation}, Melhor Fitness: {best_fitness:.4f}")
        if best_fitness >= target_fitness:
            return best_individual, best_fitness

        # Selecionar elite
        elite = select_elite(scored_population, elite_rate)
        new_population = [copy.deepcopy(ind[0]) for ind in elite]

        # Gerar novos indivíduos por crossover + mutação
        while len(new_population) < population_size:
            parent1 = random.choice(elite)[0]
            parent2 = random.choice(elite)[0]
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        # Substituir população
        population = new_population

    # Após todas as gerações, devolver o melhor
    final_scored = evaluate_population(population, fitness_function, seed)
    best_individual, best_fitness = max(final_scored, key=lambda x: x[1])
    return best_individual, best_fitness
