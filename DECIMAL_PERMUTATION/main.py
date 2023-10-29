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
    Parser : file, numPop (members), generatioon_num (iterations), mutation prob(0.25)
    -> get_config
    
    Others : 
    -> deep copy with : copy_list
    -> visualize path : draw_result
    -> visualize learning : draw_fit
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

"""
    ###########################  SOLUCION GENETICA TSP #######################################
    
    AG class con el desarrolllo de los operadores : cruce y mutacion, más seleccion por torneo.
    Individual class : contiente las caracteristicas de un individuo. 
"""

# Individual class for representing a solution
class Individual:
    def __init__(self, genoma=None):
        # Randomly generate a sequence if not provided
        if genoma is None:
            genoma = [i for i in range(numGenes)]
            random.shuffle(genoma)
        self.genoma = genoma
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        #  Calculate the fitness of an individual
        fitness = 0.0
        for i in range(numGenes - 1):
            fitness += matrix_dist[self.genoma[i], self.genoma[i + 1]]
         # Connect the last city to the first city
        fitness += matrix_dist[self.genoma[-1], self.genoma[0]]
        return fitness

# Genetic Algorithm class
class AG_PER_DEC:
    def __init__(self, input_):
        self.best = None # Best individual of each generation
        self.gen_individuals = []  # This list stores the individuals of the current generation. 
        self.final_individuals = []  # This list stores the final individuals (best-performing individuals) from each generation.
        self.final_fitnesses = []  # This list stores the fitness (cost) of the best individual from each generation.
        self.it = 0
    
    
    def cross(self):
        new_gen = []
        random.shuffle(self.gen_individuals)
        for i in range(0, numPop - 1, 2):
            # Parent genoma original individuo and next one. (RANDOMLY ORDERED)
            genoma1 = copy_list(self.gen_individuals[i].genoma)
            genoma2 = copy_list(self.gen_individuals[i + 1].genoma)

            #Random min and max range to be interchanged / cruzado. 
            idx1 = random.randint(0,numGenes - 2) #keep last position empty so as to have at least 1. 
            idx2 = random.randint(idx1, numGenes - 1)
            #Positions of cities within genoma.
            pos1_recorder = {city: idx for idx, city in enumerate(genoma1)}
            pos2_recorder = {city: idx for idx, city in enumerate(genoma2)}

             # Crossover
            for j in range(idx1, idx2):
                #store values of both individuals. 
                city1, city2 = genoma1[j], genoma2[j]
                pos1, pos2 = pos1_recorder[city2], pos2_recorder[city1]

                #interchange with the random indx, and corresponding values.
                genoma1[j], genoma1[pos1] = genoma1[pos1], genoma1[j]
                genoma2[j], genoma2[pos2] = genoma2[pos2], genoma2[j]

                pos1_recorder[city1], pos1_recorder[city2] = pos1, j
                pos2_recorder[city1], pos2_recorder[city2] = j, pos2
                
            #In the new generation add the sucessors. 
            new_gen.append(Individual(genoma1))
            new_gen.append(Individual(genoma2))
        return new_gen

    def mutate(self, new_gen): #with probability 0.25 by default. 
        for individual in new_gen:
            if random.random() < mutRate:
                # deep copy :) cus if not it is hard. 
                old_genoma = copy_list(individual.genoma)
                #Range of the genoma to mutate / reverse
                idx1 = random.randint(0, numGenes - 2) #again at least 1. 
                idx2 = random.randint(idx1, numGenes - 1)
                # Reverse a slice
                genoma_mutate = old_genoma[idx1:idx2]
                genoma_mutate.reverse()
                #if the idx where not 0 or -1 keep original parts. 
                individual.genoma = old_genoma[:idx1] + genoma_mutate + old_genoma[idx2:]
        # Combine two generations
        self.gen_individuals += new_gen

    def select(self):
        # Tournament selection ~ Cross validation
        group_num = 10  # Number of groups
        group_size = 10  # Number of individuals in each group
        group_winner = numPop // group_num  # Number of winners in each group
        winners = []  # Tournament best performing individuals. 
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # Randomly form a group
                player = random.choice(self.gen_individuals)
                player = Individual(player.genoma)
                group.append(player)
            group = AG_PER_DEC.rank(group)
             # Get the winners
            winners += group[:group_winner]
        #update the individuals with only the winners. 
        self.gen_individuals = winners

    @staticmethod
    def rank(group):
        # Bubble sort <- straight :)
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
        # Fit and store the best performing / winner. 
        for individual in self.gen_individuals:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def stop(self):
        if self.final_fitnesses[-500] - self.final_fitnesses[-1] < 7 : 
            return True 
        return False
    
    def has_converged(self, window_size=20, threshold=1e1):
        if len(self.final_fitnesses) < window_size * 2:
            # Not enough data for comparison
            return False

        current_window = self.final_fitnesses[-window_size:]
        previous_window = self.final_fitnesses[-2 * window_size:-window_size]

        current_average = sum(current_window) / len(current_window)
        previous_average = sum(previous_window) / len(previous_window)

        return abs(current_average - previous_average) < threshold

    @timeout_decorator.timeout(TIMEOUT)
    def train(self):
        self.it +=1
        # Initial random population & cost
        self.gen_individuals = [Individual() for _ in range(numPop)]
        self.best = self.gen_individuals[0]
        # Iteration
        for i in range(gen_num):
            self.it +=1

            self.sucessor_gen()
             # Connect the last city to the first city
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
    file_name = os.getcwd() + '/' + f
    return np.load(file_name)

if __name__ == '__main__':

    current_file_path = os.path.abspath(__file__)
    folder = os.path.basename(os.path.dirname(current_file_path))
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
    
    print(fitness_list)
    print()
    print(f'PROBLEM: {args.file}, SOLVER: {folder}, ITER: {ga.it}, TIME: {end_time-start_time}, BEST_COST: {ga.best.fitness}, BEST_PATH:{ga.best.genoma}' )
    ok = pd.DataFrame({'PROBLEM':args.file, 'ARGS': str(args), 'SOLVER':folder, 'iter':ga.it, 'TIME':end_time-start_time, 'BEST_COST':ga.best.fitness, 'BEST_PATH':str(ga.best.genoma)}, index=[0])
    df = pd.read_csv('results.csv')
    df = pd.concat([df,ok])
    df.to_csv('results.csv', index=False)
    
    
    
    fig, ax = plt.subplots()
    plt.plot(fitness_list)
    fig = plt.gcf()
    plt.xlabel("Iterations")
    plt.ylabel("Fitness: cost")
    plt.title("Line Plot of fitness evolution")
    plt.savefig(f'{args.file}.fit.{folder}.png')
    plt.show()

    if numGenes < 20 : # Create a graph from the adjacency matrix
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

    record = pd.read_csv(os.getcwd() + "/results.csv")
    new = pd.DataFrame(results)
    record = pd.concat([record, new], ignore_index=True)
    record.to_csv(os.getcwd() + "/results.csv")
    