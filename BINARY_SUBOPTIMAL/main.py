import timeout_decorator
import os 
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd
import math

TIMEOUT =  11000 # seconds ~ 2 h.

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
    parser.add_argument("--numCerc", type=int, default=4, help="Numerocercanos entre los que buscar.")
    return  parser.parse_args()

#deep copy
def copy_list(old_arr: [int]):
    return [e for e in old_arr]
"""  ###########################  SOLUCION GENETICA TSP ####################################### """
#_________________________________________________________________________
# _________________________ GENOMA _______________________________________
class Individual:
    def __init__(self, genoma=None, fenoma=None):

        self.gene_len = int(math.log(m,2))
        # print(f"M vecinos: {m}, en codificacions: {self.gene_len}")
        
        if genoma is None:
             #se empieza siempre en 1. y la última no hace falta codificarla pq se resta. 
             # Cada gen del genoma, excepto las úlimas 2 posiciones, requieren log_2 (m) bits, y las dos últimas 1. 
            genes = [i for i in range(nHi-1, self.gene_len-1, -1)] # i.e. 9,8,7,6,5,4,3,2 <- ciudades de las que elegir <-> GENES. 
            self.genoma, self.fenoma, self.solucion = self.createIndividual(genes)        

        else: 
            self.genoma = genoma #"01010010011011" <-> Str : Representación binaria sub-óptima del individuo
            self.fenoma, self.solucion = Individual.decode(genoma)
        
        self.fitness = self.evaluate_fitness()
        # print('GENOMA FINISHED AND CREATED: ', self.genoma, 'with len: ', len(self.genoma))
        # print('SOLUTION FINISHED AND CREATED: ', self.solucion, 'with len: ', len(self.solucion))
        # print('FENOMA FINISHED AND CREATED: ', self.fenoma, 'with len: ', len(self.fenoma))
        # print('FITNES : ', self.fitness)


    def evaluate_fitness(self):
        #  Calculate the fitness of an individual
        fitness = 0.0

        for i, a in enumerate(self.fenoma[:-2]):
            b = self.fenoma[i + 1]
            fitness += matrix_dist[a-1, b-1]
        
        # Connect the last city to the first city
        fitness += matrix_dist[self.fenoma[-1]-1, self.fenoma[0]-1]        
        return fitness

    def createIndividual(self, genes):
        fenoma = [1]
        genoma = ""
        solucion = []

        for i in genes:
            # print(f'fenoma: {fenoma}, solucion: {solucion}, genoma:{genoma}')
            
            c_idx = fenoma[-1]                          # <- la ciudad - 1 = index en la tabla de cercanias. 
            not_visited_nn = [nn_c for nn_c in ordenadas_ids[c_idx-1] if nn_c not in fenoma]
            
            if i > self.gene_len+1:                       # <- index de 1 a m vecinos cercanos. 
                nn_idx = random.randint(0,m-1)

            elif i <= self.gene_len+1 : 
                nn_idx = random.randint(0,1)

            nn_c = not_visited_nn[nn_idx]               # <- ciudad cercana no Visitada. 
            fenoma.append(nn_c)
            solucion.append(nn_idx)
            gene = bin(nn_idx)[2:] 

            if len(gene) < self.gene_len and i>3:
                # If the binary string is shorter than the desired length, pad with leading zeros
                gene = '0' * (self.gene_len - len(gene)) + gene

            genoma += gene

        #add the last location in the fenoma but no binary code needed. 
        last_c = [c for c in ordenadas_ids[fenoma[-1]-1] if c not in fenoma]
        fenoma.append(last_c[0])
        return genoma, fenoma, solucion

    @staticmethod
    def decode(individual, m=4):
        """
            Transforma un individuo/genoma en una solución y luego en un camino/fenoma utilizando una matriz de distancias.

            Args:
                individual (str): La representación binaria sub-óptima del individuo, es decir self.genoma.

            Returns:
                tuple: Una tupla que contiene el camino (secuencia de ciudades visitadas) y la solución final.
        """
        genoma = individual
        gene_len = int(math.log(m, 2))
        fenoma = [1]  # Comenzamos desde la ciudad 1
        n = matrix_dist.shape[0]


        #(1) decodificamos los indices de binario para sacar la solucion, indices en decimal. 
        genes_bit = [genoma[i:i+gene_len] for i in range(0,len(genoma)-gene_len, gene_len)]
        genes_sol = [int(group, gene_len) for group in genes_bit]

        ulti_bit = [genoma[i] for i in range(-gene_len, 0, +1)]
        ulti_sol = [int(bit, gene_len) for bit in ulti_bit]
        
        # print(f'Genoma: {genoma}\nGenesBina: {genes_bit}\nGenesDeci: {genes_sol}\nUltBin: {ulti_bit}\nUltDec: {ulti_sol}')


        solucion = genes_sol + ulti_sol

        #(2) de la solucion sacamos el path.

        for i in range(len(solucion)):
            c_idx = fenoma[-1] - 1  # La ciudad actual en el índice de la tabla de cercanías                    
            not_visited_nn = [nn_c for nn_c in ordenadas_ids[c_idx] if nn_c not in fenoma] # no visitadas
            nn_idx = solucion[i]
            if nn_idx >= len(not_visited_nn):
                nn_c = not_visited_nn[0]  # Si el índice es mayor o igual al número de no visitados, elegimos el primero
            else:
                nn_c = not_visited_nn[nn_idx]

            fenoma.append(nn_c)
        

        # Añadir al final la ciudad no visitada
        not_visited_cities = [c for c in ordenadas_ids[fenoma[-1] - 1] if c not in fenoma]
        fenoma.append(not_visited_cities[0])

        return fenoma, solucion

#_________________________________________________________________________
# _________________________ ALGORITMO _______________________________________    
class AG_BIN:
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
        
        # def two_point_ordered_crossover(parent1, parent2):
        #     n = len(parent1.genoma)           
        #     # Choose two distinct crossover points
        #     point1, point2 = random.sample(range(n), 2)
        #     start, end = min(point1, point2), max(point1, point2)
            
        #     # Initialize offspring as copies of parents
        #     offspring1 = copy_list(parent1.genoma)
        #     offspring2 = copy_list(parent2.genoma)
            
        #     # Perform crossover
        #     for i in range(start, end + 1):
        #         offspring1[i] = parent2.genoma[i]
        #         offspring2[i] = parent1.genoma[i]
            
        #     return [Individual(genoma=offspring1), Individual(genoma=offspring2)]

        def mate_recursive(population):

            if not population:
                return []

            if len(population) == 1:
                return [population[0]]

            # randomly draw a mother and a father chromosome from a subpopulation 
            # consisting of the numKeep best chromosomes in a population
            selected_chromosomes = population[:numKeep]
            ma, pa, cs = selected_chromosomes[0],selected_chromosomes[1],selected_chromosomes[2:] 
            
            #Since the genes are not uniform and occupy more than 1 digit. 
            # cp must be selected from the points that separete the genes. 
            possible_cp = list(range(ma.gene_len, len(ma.genoma)-2, ma.gene_len)) + [-2] #excluimos ademas posicion 0 y úlitma. 
            cp = random.choice(possible_cp)

            offspring = crossover(cp, ma, pa)
            # offspring = two_point_ordered_crossover(ma,pa)

            rest_offspring = mate_recursive(cs)

            return offspring + rest_offspring

        # Call mate_recursive with the current population
        final_population = mate_recursive(parents)

        return final_population


    def mutate(self, new_gen):
        # mutatePop that given a list of indices, mutates all its chromosomes at those indices.

        def mutate_chrom(chrom): #The mutateChrom function mutates a randomly selected gene among its list of genes
            n = chrom.gene_len
            x = len(chrom.genoma)
            genes_positions = list(range(0, len(chrom.genoma)-2, n)) + [-2, -1]

            num_mutations = random.randint(1, int(mutRate*len(genes_positions))) #mutations within the chromosoma
            idxsGene = random.sample(genes_positions, num_mutations) #idxGene = random.randint(0, gHi-1) #posicion dentro del cromosoma

            for ind in idxsGene:
                if ind >= 0:
                    mutGene = ''.join(random.sample(['0','1'], n))
                    chrom.genoma = chrom.genoma[:ind] + mutGene + chrom.genoma[ind + n:]
                else:
                    mutGene = random.choice(['0','1'])
                    if ind == -1:               
                        chrom.genoma = chrom.genoma[:-1] + mutGene
                    else: #-2
                        last = chrom.genoma[-1]
                        chrom.genoma = chrom.genoma[:-2] + mutGene + last

                if x!=len(chrom.genoma):
                    print(f'WRONG MUTATION for: {mutGene}, to be inserted in {ind}, with len x: {x}, chrom : {len(chrom.genoma)}' )
                    print(genes_positions)
                    quit()
            return chrom

        def mutate_chrom_in_pop(n, pop): #The function mutateChromInPop mutates a chromosome at index n in population pop
            mutChrom = mutate_chrom(pop[n])
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
            group = AG_BIN.rank(group)
             # Get the winners
            winners += group[:group_winner]
        #update the individuals with only the winners. 
        self.gen_individuals = sorted(winners, key=lambda x: x.fitness)
    
    def has_converged(self, window_size=10, threshold=1e-6):
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
        for i in range(itMax):
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

def matrix_cercanas_ciudad(adj_matrix):
    n = len(adj_matrix)
    orden_ciudades = np.zeros((n, n), dtype=int)

    for i in range(n):
        # Calcula la distancia de la ciudad i a todas las demás ciudades
        distancias = adj_matrix[i]
        ciudades_posibles = range(1, len(distancias)+1)
        # Enumera las ciudades en función de sus distancias a la ciudad i
        ordenadas = sorted(ciudades_posibles, key=lambda j: distancias[j-1])

        # Almacena los índices de las ciudades ordenadas en la matriz de salida
        orden_ciudades[i] = ordenadas

    # Convierte los índices de las ciudades en nombres de ciudades
    ciudades_ordenadas = []
    
    for i in range(n):
        fila = [ciudades_posibles[j-1] for j in orden_ciudades[i]][1:]
        ciudades_ordenadas.append(fila)
        # print(f'ciudad; {i+1}, ciudades_ordenadas:{fila}')
    return ciudades_ordenadas

if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    folder = os.path.basename(os.path.dirname(current_file_path))
    root = os.get.pathdir(os.getcwd()) 
    args = get_config()
    print(args)
    matrix_dist = read_file_matrix(args.file)
    ordenadas_ids = matrix_cercanas_ciudad(matrix_dist)
    m= args.numCerc 
    if (m & (m - 1)) != 0: m = 8 
    args.numCerc = m

    pLo, nHi = 0, matrix_dist.shape[0]
    gLo, gHi = 0, 1

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
    
    start_time = time.time()
    
    ga = AG_BIN(matrix_dist)

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


