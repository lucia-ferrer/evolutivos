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

# Ajuste fino de hiperparámetros
population_size = 150  # Tamaño de la población
generations = 1000  # Número de generaciones
crossover_prob = 0.8  # Probabilidad de cruce
mutation_prob = 0.2  # Probabilidad de mutación
tournament_size = 3  # Tamaño del torneo para la selección
elitism = True  # Habilitar elitismo

# Medir el tiempo
start_time = time.time()

# Crea la población inicial
population = toolbox.population(n=population_size)

# Realiza la evolución
algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=2 * population_size,
                         cxpb=crossover_prob, mutpb=mutation_prob, ngen=generations,
                         stats=None, halloffame=None, tournsize=tournament_size, halloffame_size=1 if elitism else 0)

# Obtiene el mejor individuo
best_individual = tools.selBest(population, 1)[0]

# Calcula el tiempo empleado
elapsed_time = time.time() - start_time

# Imprime la mejor ruta y su distancia
print("Mejor recorrido:", best_individual)
print("Distancia del mejor recorrido:", sum(best_individual.fitness.values))
print(f"Tiempo empleado: {elapsed_time} segundos")
