import numpy as np
import os
import json
import random
import argparse
import matplotlib.pyplot as plt
import timeout_decorator
import time
import pandas as pd

"""
    ###########################  UTILS ##################################################

"""
def get_config():
    # Creates a parser to receive the input argument.
    parser = argparse.ArgumentParser(description="Ejemplo de programa con argumentos")
    parser.add_argument("file", help="Nombre del archivo (argumento obligatorio)")
    parser.add_argument("--numPop", type=int, default=60, help="Numero de individuos de la poblacion")
    parser.add_argument("--mutRate", type=float, default=0.3, help="Ratio de mutacion")
    parser.add_argument("--xRate", type=float, default=0.5, help="Ratio de seleccion")
    parser.add_argument("--numElite", type=int, default=20, help="Numero de cromosmas elites")
    parser.add_argument("--numKeep", type=int, default=50, help="Numero de cromosomas para genersas sucesores")
    parser.add_argument("--itMax", type=int, default=2000, help="Numero maximo de generaciones / iteraciones.")
    return  parser.parse_args()


#deep copy
def copy_list(old_arr: [int]):
    return [e for e in old_arr]

""" #######################   AG   #########################################"""
#_________________________________________________________________________
# _________________________ GENOMA _______________________________________
class Individual:
    def __init__(self, genoma=None):
        if genoma is None: # Generar una secuencia de manera aleatoria si no se proporciona
            genoma = [i for i in range(numGenes)]
            random.shuffle(genoma)
        self.genoma = genoma
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
         # Calcular la aptitud de un individuo
        fitness = 0.0
        for i in range(numGenes - 1):
            fitness += matrix_dist[self.genoma[i], self.genoma[i + 1]]
        # Conectar la última ciudad con la primera ciudad
        fitness += matrix_dist[self.genoma[-1], self.genoma[0]]
        return fitness

#_________________________________________________________________________
# _________________________ ALGORITHM _______________________________________
class AG_PER_DEC:
    def __init__(self, input_):
        self.best = None # Mejor individuo de cada generación
        self.gen_individuals = []  # Esta lista almacena a los individuos de la generación actual
        self.final_individuals = []  # Esta lista almacena a los individuos finales (mejores) de cada generación
        self.final_fitnesses = []  # Esta lista almacena la aptitud (costo) del mejor individuo de cada generación
        self.it = 0
    
    
    def cross(self):
        new_gen = []
        random.shuffle(self.gen_individuals)
        for i in range(0, numPop - 1, 2):
             # Genoma del padre original del individuo y del siguiente (ORDENADO DE MANERA ALEATORIA)
            genoma1 = copy_list(self.gen_individuals[i].genoma)
            genoma2 = copy_list(self.gen_individuals[i + 1].genoma)

            # Rango mínimo y máximo para ser intercambiados/cruzados de manera aleatoria
            idx1 = random.randint(0,numGenes - 2) # mantener la última posición vacía para asegurar al menos 1
            idx2 = random.randint(idx1, numGenes - 1)
             # Posiciones de las ciudades dentro del genoma
            pos1_recorder = {city: idx for idx, city in enumerate(genoma1)}
            pos2_recorder = {city: idx for idx, city in enumerate(genoma2)}

             # Cruce (crossover)
            for j in range(idx1, idx2):
                 # Almacenar los valores de ambos individuos
                city1, city2 = genoma1[j], genoma2[j]
                pos1, pos2 = pos1_recorder[city2], pos2_recorder[city1]

                # Intercambiar con el índice aleatorio y los valores correspondientes
                genoma1[j], genoma1[pos1] = genoma1[pos1], genoma1[j]
                genoma2[j], genoma2[pos2] = genoma2[pos2], genoma2[j]

                pos1_recorder[city1], pos1_recorder[city2] = pos1, j
                pos2_recorder[city1], pos2_recorder[city2] = j, pos2
                
            # En la nueva generación, agregar a los sucesores
            new_gen.append(Individual(genoma1))
            new_gen.append(Individual(genoma2))
        return new_gen

    def mutate(self, new_gen):
        for individual in new_gen:
            if random.random() < mutRate:
                old_genoma = copy_list(individual.genoma)
                # Rango del genoma a mutar/invertir
                idx1 = random.randint(0, numGenes - 2)
                idx2 = random.randint(idx1, numGenes - 1)
                 # Invertir un fragmento
                genoma_mutate = old_genoma[idx1:idx2]
                genoma_mutate.reverse()
                 # Si los índices no eran 0 o -1, mantener las partes originales
                individual.genoma = old_genoma[:idx1] + genoma_mutate + old_genoma[idx2:]
        # Combinar dos generaciones
        self.gen_individuals += new_gen

    def select(self):
        # Selección por torneo ~ Cross validation
        group_num = 10  # Number of groups
        group_size = 10  # Number of individuals in each group
        group_winner = numPop // group_num  # Number of winners in each group
        winners = []  # Tournament best performing individuals. 
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # Formar aleatoriamente un grupo
                player = random.choice(self.gen_individuals)
                player = Individual(player.genoma)
                group.append(player)
            # Obtener a los ganadores
            group = AG_PER_DEC.rank(group)
            winners += group[:group_winner]
        # Actualizar los individuos solo con los ganadores <- ELISTISMO
        self.gen_individuals = winners

    @staticmethod
    def rank(group):
         # Ordenamiento de burbuja (bubble sort)
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def sucessor_gen(self):
        # (1) Crossover
        new_gen = self.cross()
        # (2) Mutation
        self.mutate(new_gen)
        # (3) Selection
        self.select()
        # Calcular y almacenar al mejor individuo de desempeño (ganador)
        for individual in self.gen_individuals:
            if individual.fitness < self.best.fitness:
                self.best = individual

    # def stop(self):
    #     if self.final_fitnesses[-500] - self.final_fitnesses[-1] < 7 : 
    #         return True 
    #     return False
    
    def has_converged(self, window_size=20, threshold=1e1):
        if len(self.final_fitnesses) < window_size * 2:
            ## No hay suficientes datos para la comparación
            return False

        current_window = self.final_fitnesses[-window_size:]
        previous_window = self.final_fitnesses[-2 * window_size:-window_size]

        current_average = sum(current_window) / len(current_window)
        previous_average = sum(previous_window) / len(previous_window)

        return abs(current_average - previous_average) < threshold

    @timeout_decorator.timeout(TIMEOUT)
    def train(self):
        # Población inicial aleatoria y costo
        self.gen_individuals = [Individual() for _ in range(numPop)]
        self.best = self.gen_individuals[0]
        # Iteration
        for i in range(gen_num):
            self.it +=1

            self.sucessor_gen()
            result = copy_list(self.best.genoma)
            result.append(result[0])
            self.final_individuals.append(result)
            self.final_fitnesses.append(self.best.fitness)
            self.best = self.gen_individuals[0]
            if i>1 and i%500==0 and self.has_converged(): 
                return self.final_individuals, self.final_fitnesses
        return self.final_individuals, self.final_fitnesses

"""
######################### MAIN #################################
"""

def read_file_matrix(f):
    file_name = root + '/files/' + f
    return np.load(file_name)

if __name__ == '__main__':

    current_file_path = os.path.abspath(__file__)
    folder = os.path.basename(os.path.dirname(current_file_path))
    root = os.get.pathdir(os.getcwd()) 
    args = get_config()
    random.seed(4)

    matrix_dist = read_file_matrix(args.file)
    numGenes = m_dist.shape[0] 
    numPop = args.numPop 
    gen_num = args.itMax
    mutRate = args.mutRate 

    TIMEOUT =  11000 # seconds ~ 2 h.

    start_time = time.time()

    ga = AG_PER_DEC(m_dist)


    try:
        result_list, fitness_list = ga.train()
        end_time = time.time()
    except timeout_decorator.TimeoutError:
        end_time = time.time()
    
    #TO CSV SOLUTION
    print(f'PROBLEM: {args.file}, SOLVER: {folder}, ITER: {ga.it}, TIME: {end_time-start_time}, BEST_COST: {ga.best.fitness}, BEST_PATH:{ga.best.genoma}' )
    ok = pd.DataFrame({'PROBLEM':args.file, 'ARGS': str(args), 'SOLVER':folder, 'iter':ga.it, 'TIME':end_time-start_time, 'BEST_COST':ga.best.fitness, 'BEST_PATH':str(ga.best.genoma)}, index=[0])
    df = pd.read_csv('results.csv')
    df = pd.concat([df,ok])
    df.to_csv('results.csv', index=False)
    
    
    #PLOT FITTNESS
    fig, ax = plt.subplots()
    plt.plot(fitness_list)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness: cost")
    plt.title(f"{folder}-{args}")
    plt.savefig(root + f'/figures/{args}.fit.{folder}.png')
    plt.show()

    #PLOT GRAPH IF SMALL
    if numGenes < 50 : # Create a graph from the adjacency matrix
        G = nx.Graph(matrix_dist)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=10)
        nodes = [i-1 for i in result_list[-1]]
        labels = {i: str(i) for i in nodes}
        nx.draw_networkx_labels(G, pos, labels)

        # Create the path by connecting nodes in the order of the list
        edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=2)

        plt.title("Path based on Adjacency Matrix")
        plt.axis('off')
        plt.show()