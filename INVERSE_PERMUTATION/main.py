import numpy as np
import os
import json
import random
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import timeout_decorator
import time
import pandas as pd
import math

TIMEOUT =  11000 # seconds ~ 2 h.
random.seed(4)
""" ###########################  UTILS ##################################################"""
def get_config():
    # Creates a parser to receive the input argument.
    parser = argparse.ArgumentParser(description="Ejemplo de programa con argumentos")
    parser.add_argument("file", help="Nombre del archivo (argumento obligatorio)")
    parser.add_argument("--numPop", type=int, default=60, help="Numero de individuos de la poblacion")
    parser.add_argument("--mutRate", type=float, default=0.5, help="Ratio de mutacion")
    parser.add_argument("--xRate", type=float, default=0.5, help="Ratio de seleccion")
    parser.add_argument("--numElite", type=int, default=20, help="Numero de cromosmas elites")
    parser.add_argument("--numKeep", type=int, default=50, help="Numero de cromosomas para genersas sucesores")
    parser.add_argument("--itMax", type=int, default=1500, help="Numero maximo de generaciones / iteraciones.")
    return  parser.parse_args()

#deep copy
def copy_list(old_arr: [int]):
    return [e for e in old_arr]



"""  ###########################  SOLUCION GENETICA TSP ####################################### """
#_________________________________________________________________________
# _________________________ GENOTIPO _______________________________________

class Individual:
    def __init__(self, genoma=None, fenoma=None):
        if genoma is None:
            genoma = [i for i in range(1, pHi+1)]
            self.fenoma = copy_list(genoma)       # permutation solution.       
            random.shuffle(genoma) 
            self.genoma = Individual.encode(genoma) #inverse permutation

        else: 
            self.genoma = genoma 
            self.fenoma = Individual.decode(genoma)

        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        #  Calculate the fitness of an individual
        fitness = 0.0
        
        for i in range(pHi - 1):
            fitness += matrix_dist[self.fenoma[i]-1, self.fenoma[i + 1]-1]
         # Connect the last city to the first city
        fitness += matrix_dist[self.fenoma[-1]-1, self.fenoma[0]-1]
        return fitness

    @staticmethod
    def encode(perm):
        """
        input : permutation i0...iN
        output: inverse permutation a0 ... aN
        For a permutation i1, i2, . . . , iN of the set {1, 2, . . . , N } we let aj denote the number of integers in the permutation which precede j but are greater than j. 
        So, aj is a measure of how much out of order j is. 
        The sequence of numbers a1, a2, . . . , aN is called the inversion sequence of the permutation i1, i2, . . . , iN . 
        The inversion sequence a1, a2, . . . , aN satisfies the conditions 0 ≤ ai ≤ N − i for i = 1, 2, . . . , N
        As seen there is no restriction on the elements which says ai = aj is forbidden for i different of j.
        This is of course very convenient for the crossover and mutation operations in GA.
        """
        N = len(perm)
        inv = [0] * N  # Initialize the inversion sequence with zeros

        for i in range(N):
            inv_i = 0
            m = 1
            while perm[m - 1] != (i + 1):
                if perm[m - 1] > (i + 1):
                    inv_i += 1
                m += 1
            inv[i] = inv_i

        return inv


    @staticmethod
    def decode(inv):
        """
        does the inverse of the method encodr, such that 
        given the inversed permutation a0...aN, it returns the original permutation i0...iN.
        """
        N = len(inv)
        permutation = [0] * N

        for i in range(N):
            count = inv[i]
            j = 0
            while count > 0 or permutation[j] != 0:
                if permutation[j] == 0:
                    count -= 1
                j += 1
            permutation[j] = i + 1
        return permutation    
   
#_________________________________________________________________________
# _________________________ ALGORITMO _______________________________________

class AG_PER_INV:
    def __init__(self, input_):
        self.best = None # Best individual of each generation
        self.gen_individuals = []  # This list stores the individuals of the current generation. (POPULATION)
        self.final_individuals = []  # This list stores the final individuals (best-performing individuals) from each generation.
        self.final_fitnesses = []  # This list stores the fitness (cost) of the best individual from each generation.
        self.it = 0

    #_____________________________________________________________________________
    #_____________________________ GENETIC OPERATORS _____________________________
    def matePairwise(self, parents):
        def crossover(cp, ma, pa):        
            offspring1 = Individual(genoma=ma.genoma[:cp] + pa.genoma[cp:])
            offspring2 = Individual(genoma=pa.genoma[:cp] + ma.genoma[cp:])           
            return [offspring1, offspring2]
        
        def two_point_ordered_crossover(parent1, parent2):
            n = len(parent1.genoma)           
            # Choose two distinct crossover points
            point1, point2 = random.sample(range(n), 2)
            start, end = min(point1, point2), max(point1, point2)
            
            # Initialize offspring as copies of parents
            offspring1 = copy_list(parent1.genoma)
            offspring2 = copy_list(parent2.genoma)
            
            # Perform crossover
            for i in range(start, end + 1):
                offspring1[i] = parent2.genoma[i]
                offspring2[i] = parent1.genoma[i]
            
            return [Individual(genoma=offspring1), Individual(genoma=offspring2)]

        def mate_recursive(population):

            if not population:
                return []

            if len(population) == 1:
                return [population[0]]

            # randomly draw a mother and a father chromosome from a subpopulation 
            # consisting of the numKeep best chromosomes in a population
            selected_chromosomes = population[:numKeep]
            ma, pa, cs = selected_chromosomes[0],selected_chromosomes[1],selected_chromosomes[2:] 
            # print('Length after mate_Recursive quit ma and pa', len(cs))
            cp = random.randint(0, len(ma.genoma) - 1)

            # offspring = crossover(cp, ma, pa)
            offspring = two_point_ordered_crossover(ma,pa)

            rest_offspring = mate_recursive(cs)

            return offspring + rest_offspring

        # Call mate_recursive with the current population
        final_population = mate_recursive(parents)

        return final_population


    def mutate(self, new_gen):
        # mutatePop that given a list of indices, mutates all its chromosomes at those indices.

        def mutate_chrom(chrom): #The mutateChrom function mutates a randomly selected gene among its list of genes

            num_mutations = random.randint(1, gHi//4) 
            idxsGene = random.sample(range(gHi), num_mutations) #idxGene = random.randint(0, gHi-1) #posicion dentro del cromosoma


            for ind in idxsGene:
                mutGene = random.randint(0, gHi-ind-1) #gen a insertar (dentro de un rango aswell)  
                chrom.genoma[ind] = mutGene
            return chrom

        def mutate_chrom_in_pop(n, pop): #The function mutateChromInPop mutates a chromosome at index n in population pop
            mutChrom = mutate_chrom(pop[n])
            # new_pop = pop.copy()
            pop[n] = mutChrom
            return pop
        
        #print('Len of pop with mut_indices', len(new_gen)-numElite, ' and original gen size: ', len(new_gen))        

        mut_indices = random.sample(range(numElite, len(new_gen)), numMut) # we do not want to mutate any of the elite chromosomes
        new_population = copy_list(new_gen)

        for n in mut_indices:
            new_population = mutate_chrom_in_pop(n, new_population)
        return new_population


    def select_parents(self):
        sorted_pop = sorted(self.gen_individuals, key=lambda x: x.fitness)  # Sort the self.gen_individuals by cost

        # Calculate the number of elite chromosomes to keep
        elite_chromosomes = sorted_pop[:numElite]

        # Calculate the number of chromosomes to select
        numSelect = numKeep - numElite

        # print(f'Num select: {numSelect}, keep: {numKeep}, elite: {numElite}')

        # Select additional chromosomes using selection rate
        selected_chromosomes = random.sample(sorted_pop[numElite:], numSelect)

        # Combine elite and selected chromosomes to get parents
        parents = elite_chromosomes + selected_chromosomes

        return parents

    
    @staticmethod
    def rank(group):
        # Bubble sort <- straight from google and efficient :)
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def sucessor_gen(self): #evolvePopOnce
        #print('TO CREATE A NEW GEN')
        parents = self.select_parents()
        #print(f' (0) Parents to be mutated with : {len(parents)}, such as {parents[0].genoma}')
        offspring = self.matePairwise(copy_list(parents)) # The function matePairwise then create offspring using single point crossover on the chromosomes in parents.
        #print(f' (1) Crossovered offspring : {len(offspring)}, such as {offspring[0].genoma}')
        new_gen = parents + offspring
        #print(f' (2) Chroms to be mutated with : {len(new_gen)}, such as {new_gen[0].genoma}')
        self.gen_individuals = self.mutate(new_gen)
        #print(f' (3) Mutated to be in Tournment Selection with : {len(self.gen_individuals)}, such as {self.gen_individuals[0].genoma}')
        self.select()
        #print(f' Final gen with {len(self.gen_individuals)} and best {self.gen_individuals[0].genoma}')
        # Fit and store the best performing / winner. 
        for individual in self.gen_individuals:
            if individual.fitness < self.best.fitness:
                self.best = individual
    
    def select(self):
        # Tournament selection 
        group_num = numElite  # Number of groups
        group_size = 10  # Number of individuals in each group
        group_winner = numKeep #numPop // group_num  # Number of winners in each group
        winners = []  # Tournament best performing individuals. 
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # Randomly form a group
                player = random.choice(self.gen_individuals)
                player = Individual(genoma=player.genoma)
                group.append(player)
            group = AG_PER_INV.rank(group)
             # Get the winners
            winners += group[:group_winner]
        #update the individuals with only the winners. 
        self.gen_individuals = sorted(winners, key=lambda x: x.fitness)
    
    def has_converged(self, window_size=50, threshold=1):
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
        # Initial random population & cost
        self.gen_individuals = [Individual() for _ in range(numPop)]
        self.best = self.gen_individuals[0]
        # Iteration
        for i in range(int(itMax)):
            self.it += 1
            self.sucessor_gen()
            # Connect the last city to the first city
            result = copy_list(self.best.fenoma)
            result.append(result[0])
            self.final_individuals.append(result)
            self.final_fitnesses.append(self.best.fitness)
            self.best = self.gen_individuals[0]
            if i>1 and i%500==0 and self.has_converged(): 
                return self.final_individuals, self.final_fitnesses
        return self.final_individuals, self.final_fitnesses


""" ######################### MAIN ################################# """

def read_file_matrix(f):
    file_name = root + '/files/'+ f
    return np.load(file_name)

if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    folder = os.path.basename(os.path.dirname(current_file_path))
    root = os.get.pathdir(os.getcwd()) 
    args = get_config()
    print(args)
    matrix_dist = read_file_matrix(args.file)
    pLo, pHi = 1, matrix_dist.shape[0]
    gLo, gHi = pLo, pHi

    numPop = args.numPop
    mutRate = args.mutRate
    xRate = args.xRate
    numElite = args.numElite
    itMax = args.itMax
    numKeep = args.numKeep

    numKeep_prime = math.ceil(xRate * numPop) 
    if numKeep_prime > numPop: numKeep = numPop - (numPop % 2)
    elif numKeep_prime < 2: numKeep = 2
    else: numKeep = numKeep_prime
    
    numMut = math.ceil(mutRate * float(numPop))
    
    args.numKeep = numKeep
        
    start_time = time.time()
    
    ga = AG_PER_INV(matrix_dist)

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

