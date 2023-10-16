import random

# Definición de la instancia del problema TSP
cities = {
    "A": (0, 0),
    "B": (2, 4),
    "C": (5, 2),
    "D": (7, 6),
    "E": (3, 8)
}

# Parámetros del algoritmo genético
population_size = 100
generations = 1000
mutation_rate = 0.1

# Función para calcular la distancia entre dos ciudades
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# Función para calcular la longitud del recorrido (fitness)
def total_distance(route):
    return sum(distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1))

# Genera una población inicial aleatoria
def generate_population(size):
    return [random.sample(cities.keys(), len(cities)) for _ in range(size)]

# Función de selección de padres (ruleta)
def select_parents(population):
    fitness_values = [1 / total_distance(individual) for individual in population]
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    return random.choices(population, probabilities, k=2)

# Función de cruce (orden)
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    for i in range(start, end + 1):
        child[i] = parent1[i]
    remaining_cities = [city for city in parent2 if city not in child]
    j = 0
    for i in range(len(parent1)):
        if child[i] is None:
            child[i] = remaining_cities[j]
            j += 1
    return child

# Función de mutación (intercambio)
def mutate(route):
    index1, index2 = random.sample(range(len(route)), 2)
    route[index1], route[index2] = route[index2], route[index1]

# Algoritmo genético
population = generate_population(population_size)
for generation in range(generations):
    population = sorted(population, key=total_distance)
    new_population = [population[0]]  # Elitismo: mantenemos al mejor individuo
    while len(new_population) < population_size:
        parent1, parent2 = select_parents(population)
        child = crossover(parent1, parent2)
        if random.random() < mutation_rate:
            mutate(child)
        new_population.append(child)
    population = new_population

best_route = population[0]
best_distance = total_distance(best_route)
print("Mejor recorrido:", best_route)
print("Distancia del mejor recorrido:", best_distance)
