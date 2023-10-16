import random
import numpy as np
from deap import base, creator, tools, algorithms
import time

# Cargar las coordenadas de las ciudades desde el archivo .npy
cities = np.load('ciudades.npy')

# Parámetros del algoritmo genético
population_size = 100
generations = 100
mutation_rate = 0.1

# Función para calcular la distancia entre dos ciudades
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Función para calcular la longitud del recorrido (fitness)
def total_distance(route):
    return sum(distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1))

# Crear tipos de individuos y la población
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(cities)), len(cities))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de selección de padres (ruleta)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutation_rate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Función de evaluación
def evaluate(individual):
    return (total_distance(individual),)

toolbox.register("evaluate", evaluate)

# Algoritmo genético
start_time = time.time()

population = toolbox.population(n=population_size)
algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size, 
                         cxpb=0.7, mutpb=0.3, ngen=generations, stats=None, halloffame=None, verbose=True)

best_individual = tools.selBest(population, k=1)[0]
best_distance = total_distance(best_individual)

end_time = time.time()
elapsed_time = end_time - start_time

# Guardar la solución en un archivo de texto
with open('ruta_optima.txt', 'w') as file:
    file.write("Mejor recorrido: " + " -> ".join(map(str, best_individual)))
    file.write("\nDistancia del mejor recorrido: " + str(best_distance))
    file.write(f"\nTiempo empleado: {elapsed_time:.4f} segundos")

print("Mejor recorrido:", best_individual)
print("Distancia del mejor recorrido:", best_distance)
print(f"Tiempo empleado: {elapsed_time:.4f} segundos")
