import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
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
toolbox.register("indices", np.random.permutation, len(cities))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: (distance(cities[ind[i]], cities[ind[i - 1]]) for i in range(len(ind)),))
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Configura los parámetros del algoritmo
population_size = 100
generations = 1000
crossover_prob = 0.7
mutation_prob = 0.2

# Medir el tiempo
start_time = time.time()

# Crea la población inicial
population = toolbox.population(n=population_size)

# Realiza la evolución
algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=2 * population_size,
                         cxpb=crossover_prob, mutpb=mutation_prob, ngen=generations, stats=None, halloffame=None)

# Obtiene el mejor individuo
best_individual = tools.selBest(population, 1)[0]

# Calcula el tiempo empleado
elapsed_time = time.time() - start_time

# Imprime la mejor ruta y su distancia
print("Mejor recorrido:", best_individual)
print("Distancia del mejor recorrido:", sum(best_individual.fitness.values))
print(f"Tiempo empleado: {elapsed_time} segundos")

# Visualizar la ruta
best_route = np.array(best_individual)
plt.scatter(cities[:, 0], cities[:, 1])
plt.plot(cities[best_route, 0], cities[best_route, 1], 'r-')
plt.show()
