import numpy as np
from deap import base, creator, tools, algorithms
import random
import time

# Carga las coordenadas de las ciudades desde el archivo .npy
cities = np.load("ciudades.npy")

# Define la función de distancia entre dos ciudades
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Define el problema de optimización
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Crea las herramientas del algoritmo genético
toolbox = base.Toolbox()

# Define una función heurística de inserción más cercano
def insertion_heuristic(cities):
    remaining_cities = list(range(len(cities))
    initial_city = np.random.choice(remaining_cities)
    remaining_cities.remove(initial_city)
    tour = [initial_city]

    while remaining_cities:
        current_city = tour[-1]
        nearest_city = min(remaining_cities, key=lambda city: distance(cities[current_city], cities[city]))
        remaining_cities.remove(nearest_city)
        tour.append(nearest_city)

    return tour

toolbox.register("individual", tools.initIterate, creator.Individual, insertion_heuristic, cities)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_tsp(ind):
    return (distance(cities[ind[i]], cities[ind[i - 1]]) for i in range(len(ind)),)

toolbox.register("evaluate", evaluate_tsp)

def custom_crossover(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size)
    if cxpoint2 < cxpoint1:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    subset = ind1[cxpoint1:cxpoint2]  # Extraer una sección de ind1
    missing = [city for city in ind2 if city not in subset]  # Ciudades faltantes de ind2

    child = []
    for city in ind2:
        if city in subset:
            child.append(city)
        else:
            child.append(missing.pop(0))

    return creator.Individual(child),

toolbox.register("mate", custom_crossover)

def custom_mutate(ind):
    if random.random() < 0.2:
        idx1, idx2 = random.sample(range(len(ind)), 2)
        ind[idx1], ind[idx2] = ind[idx2], ind[idx1]

toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Configura los parámetros del algoritmo
population_size = 100
generations = 1000

# Medir el tiempo
start_time = time.time()

# Crea la población inicial
population = toolbox.population(n=population_size)

# Realiza la evolución
algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=2 * population_size,
                         cxpb=0.7, mutpb=0.2, ngen=generations, stats=None, halloffame=None)

# Obtiene el mejor individuo
best_individual = tools.selBest(population, 1)[0]

# Calcula el tiempo empleado
elapsed_time = time.time() - start_time

# Imprime la mejor ruta y su distancia
print("Mejor recorrido:", best_individual)
print("Distancia del mejor recorrido:", sum(best_individual.fitness.values))
print(f"Tiempo empleado: {elapsed_time} segundos")
