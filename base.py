import numpy as np
import random
import time

# Cargar las coordenadas de las ciudades desde el archivo .npy
cities = np.load('ciudades.npy')

# Parámetros del algoritmo genético
population_size = 100
generations = 1000
mutation_rate = 0.1

# Función para calcular la distancia entre dos ciudades
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Función para calcular la longitud del recorrido (fitness)
def total_distance(route):
    return np.sum(distance(cities[route], np.roll(cities[route], shift=-1, axis=0), axis=1))

# Genera una población inicial aleatoria
def generate_population(size):
    return np.array([np.random.permutation(len(cities)) for _ in range(size)])

# Función de selección de padres (ruleta)
def select_parents(population):
    fitness_values = 1 / np.array([total_distance(individual) for individual in population])
    probabilities = fitness_values / fitness_values.sum()
    return np.array(random.choices(population, probabilities, k=2))

# Función de cruce (orden)
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = np.array([None] * len(parent1))
    child[start:end+1] = parent1[start:end+1]
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
start_time = time.time()

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

end_time = time.time()
elapsed_time = end_time - start_time

# Guardar la solución en un archivo de texto
with open('ruta_optima.txt', 'w') as file:
    file.write("Mejor recorrido: " + " -> ".join(map(str, best_route)))
    file.write("\nDistancia del mejor recorrido: " + str(best_distance))
    file.write(f"\nTiempo empleado: {elapsed_time:.4f} segundos")

print("Mejor recorrido:", best_route)
print("Distancia del mejor recorrido:", best_distance)
print(f"Tiempo empleado: {elapsed_time:.4f} segundos")
