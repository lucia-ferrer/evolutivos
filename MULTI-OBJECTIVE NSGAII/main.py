import numpy as np
import os
import json
import random
import argparse
import matplotlib.pyplot as plt
import timeout_decorator
import time
import pandas as pd

TIMEOUT = 14000
""" #######################   UTILS   #########################################"""
def get_config():
    # Creates a parser to receive the input argument.
    parser = argparse.ArgumentParser(description="Ejemplo de programa con argumentos")
    parser.add_argument("file1", help="Nombre del archivo 1 (argumento obligatorio)")
    parser.add_argument("file2", help="Nombre del archivo 2 (argumento obligatorio)")
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

        if genoma is None:
            self.genoma = [i for i in range(1, numGenes+1)]
            random.shuffle(self.genoma) 
        else: 
            self.genoma = genoma 
        
        self.dominates_list = None  # List of individuals that this individual dominates
        self.inverse_domination_count = float('-inf')  # Number of individuals that dominate this individual
        self.rank = -1  # Member of the n'th pareto front; 0 being the best
        self.crowding_distance = -1            
        self.fitnesses = [0,0] # <- ahora hay dominate
    
    def evaluate_fitness(self):
        total_distance = 0
        total_cost = 0
        for i, city_a in enumerate(self.genoma[:-1]):
            city_b = self.genoma[i+1]
            #  The cost of travelling from a to b is equal to the cost of travelling from b to a
            total_distance += matrix_dist[city_a-1][city_b-1]
            total_cost += matrix_cost[city_a-1][city_b-1]

        self.fitnesses[0] = total_distance
        self.fitnesses[1] = total_cost
   
    def dominates(self, individual_b):
        """
        Individual_a dominates individual_f if both:
            a is no worse than b in regards to all fitnesses
            a is strictly better than b in regards to at least one fitness
        Assumes that lower fitness is better, as is the case with cost-distance-TSP.
        ouput-> True if individual_a dominates individual_b
                 False if individual_b dominates individual_a or neither dominate each other
        """
        a_no_worse_b = 0  # a <= b
        a_strictly_better_b = 0  # a < b
        n_objectives = len(self.fitnesses)
        for fitness_i in range(n_objectives):
            f_a = self.fitnesses[fitness_i]
            f_b = individual_b.fitnesses[fitness_i]
            if f_a < f_b:
                a_no_worse_b += 1
                a_strictly_better_b += 1
            elif f_a == f_b:
                a_no_worse_b += 1
            else:
                return False
        return a_no_worse_b == n_objectives and a_strictly_better_b >= 1
    
    def __lt__(self, other):
        return self.fitnesses[0] < other.fitnesses[0] or (self.fitnesses[0] == other.fitnesses[0] and self.fitnesses[1] < other.fitnesses[1])

    def __gt__(self, other):
        return self.fitnesses[0] > other.fitnesses[0] or \
            (self.fitnesses[0] == other.fitnesses[0] and self.fitnesses[1] > other.fitnesses[1])

    def __eq__(self, other):
        return self.fitnesses[0] == other.fitnesses[0] and  self.fitnesses[1] == other.fitnesses[1]

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __ne__(self, other):
        return self.fitnesses[0] != other.fitnesses[0] and \
            self.fitnesses[1] != other.fitnesses[1]


#_________________________________________________________________________
# _________________________ ALGORITHM _______________________________________
class AG_MULTI():
    def __init__(self):
        self.best = None # Best individual of each generation
        self.gen_individuals = []  # This list stores the individuals of the current generation. 
        self.final_individuals = []  # This list stores the final individuals (best-performing individuals) from each generation.
        self.final_fitnesses = []  # This list stores the fitness (cost) of the best individual from each generation.
        self.it = 0

        self.rank_indexes = []
        self.f_mins = [float('inf'), float('inf')]
        self.f_maxes = [float('-inf'), float('-inf')]


    def evaluate_fitnesses(self):
            """
            Evaluate the fitnesses of the phenotypes.
            """
            for child in self.gen_individuals:
                
                child.evaluate_fitness()
                
                for i in range(len(child.fitnesses)):
                    f = child.fitnesses[i]
                    if f < self.f_mins[i]: self.f_mins[i] = f
                    if f > self.f_maxes[i]: self.f_maxes[i] = f
    
    def cross(self):

        new_gen = []
        gen_selected = self.select_mate()
        random.shuffle(gen_selected)

        for i in range(0, len(gen_selected) - 1, 2):
            # Parent genoma original individuo and next one. (RANDOMLY ORDERED)
            genoma1 = copy_list(gen_selected[i].genoma)
            genoma2 = copy_list(gen_selected[i + 1].genoma)

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

    def mutate(self, new_gen): #INVERSE RANGE OF CROMOSOMA.
            for individual in new_gen:
                if random.random() < mutate_prob:
                    old_genoma = copy_list(individual.genoma)
                    #Range of the genoma to mutate / reverse
                    idx1 = random.randint(0, numGenes - 2) #again at least 1. 
                    idx2 = random.randint(idx1, numGenes - 1)
                    # Reverse a slice
                    genoma_mutate = old_genoma[idx1:idx2]
                    genoma_mutate.reverse()
                    individual.genoma = old_genoma[:idx1] + genoma_mutate + old_genoma[idx2:]
            # Combine two generations
            self.gen_individuals += new_gen     
    
    def select_mate(self):
        """# Tournament selection PARENTS """
        group_num = 10  # Number of groups
        group_size = numPop // group_num  # Number of individuals in each group
        # group_winner = numPop // group_num  # Number of winners in each group
        winners = []  # Tournament best performing individuals. 
    
        for i in range(group_num):
            group = np.random.choice(self.gen_individuals, size=group_size)
            best = group[0]
            for individual in group[1:]:
                if individual.rank < best.rank:
                    best = individual
                elif individual.rank == best.rank and individual.crowding_distance > best.crowding_distance:
                    best = individual
            winners.append(best)
        return winners
    
    @staticmethod
    def rank_assign_sort(pool):
        """
        Very fast Non-Dominant sort with binary insertion as per NSGA-II.
        Assign ranks to the members of the pool and sort it by rank,
        """
        # Sort pool in ascending order by fitness function 1, and if two solutions
        # has the same fitness for fitness function 1,
        # then sort in ascending order by fitness function 2
        pool = sorted(pool)

        fronts = [[pool[0]]]
        # The individual with the lowest value for fitness func. 1 must belong to the first front
        pool[0].rank = 0
        current_rank = 0
        for ind_a in pool[1:]:
            for ind_b in fronts[current_rank]:
                if ind_b.dominates(ind_a):
                    current_rank += 1
                    fronts.append([ind_a])
                    ind_a.rank = current_rank
                    break
            else:
                # Find the lowest front, index/rank b, where ind_a is not dominated.
                b = AG_MULTI._bisect_fronts(fronts, ind_a)
                fronts[b].append(ind_a)
                ind_a.rank = b

        return fronts

    @staticmethod
    def _bisect_fronts(fronts, ind):
        lo = 0
        hi = len(fronts)
        while lo < hi:
            mid = (lo+hi)//2
            if not fronts[mid][-1].dominates(ind):
                hi = mid
            else:
                lo = mid+1
        return lo
    
    def select(self):
        
        """
        Select adults from a composite pool of children and their parents
        (from the previous generation) based on first on rank and second on
        crowding distance. This ensures elitism.
        """

        self.evaluate_fitnesses() # <- actualiza f_max, f_min.
        #Parents [elitism] + offspring stored after mutation. 
        n_adults = len(self.gen_individuals)
        pool = AG_MULTI.rank_assign_sort(self.gen_individuals)
        survival_gen = []
        rank_indexes = []
        for non_dominated_front in pool:
            # Crowding distance needs to be calculated for all fronts since it will later be used for tournament selection.
            AG_MULTI.crowding_distance_assign(non_dominated_front, self.f_mins, self.f_maxes)

            # Add non-dominated fronts to empty adult pool until no more complete fronts can be added.
            if len(survival_gen) + len(non_dominated_front) <= n_adults:
                survival_gen.extend(non_dominated_front)
                rank_indexes.append(len(survival_gen))
            # Then, add individuals from the last front based on their crowding distance.
            else:
                non_dominated_front = sorted(non_dominated_front,
                                            key=lambda ind: ind.crowding_distance,
                                            reverse=True)
                survival_gen.extend(non_dominated_front[:n_adults - len(survival_gen)])
                rank_indexes.append(len(survival_gen))
                break
        
        self.gen_individuals = survival_gen
        self.rank_indexes = rank_indexes
        
    def sucessor_gen(self):
        # (1) Crossover
        new_gen = self.cross()
        
        # (2) Mutation
        self.mutate(new_gen)

        # (3) Selection
        self.select()

        # Save the best performing / winner. 
        for individual in self.gen_individuals:
            if individual < self.best:
                self.best = individual

    def stop(self):
        if self.final_fitnesses[-500] - self.final_fitnesses[-1] < 7 : 
            return True 
        return False
    
    def has_converged(self, window_size=20, threshold=1e1):
        if len(self.final_fitnesses[0]) < window_size * 2:
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
        for i in range(itMax):

            self.sucessor_gen()

            #STORING GENERATION 
            result = copy_list(self.best.genoma)
            result.append(result[0])
            self.final_individuals.append(result)
            self.final_fitnesses.append(self.best.fitnesses)

            if i>1 and i%500==0 and self.has_converged(): 
                return self.final_individuals, self.final_fitnesses

        return self.final_individuals, self.final_fitnesses


    @staticmethod
    def crowding_distance_assign(non_dominated_front, f_mins, f_maxes):
        """
        Assign crowding distance to the individuals in a non-dominated front
        and return a sorted front
        :param non_dominated_front:
        :return: The non-dominated front sorted by crowding distance in
        descending order (higher is better)
        """
        for individual in non_dominated_front:
            individual.crowding_distance = 0

        for objective_i in range(len(non_dominated_front[0].fitnesses)):
            # Sort front by current fitness function
            objective_sorted = sorted(non_dominated_front,
                                    key=lambda ind: ind.fitnesses[objective_i])
            objective_sorted[0].crowding_distance = float('inf')
            objective_sorted[-1].crowding_distance = float('inf')
            #fitness_factor = objective_sorted[-1].fitnesses[objective_i] - objective_sorted[0].fitnesses[objective_i]
            fitness_factor = f_maxes[objective_i] - f_mins[objective_i]
            for individual_i in range(1, len(non_dominated_front)-1):
                prev_ind = objective_sorted[individual_i-1].fitnesses[objective_i]
                next_ind = objective_sorted[individual_i+1].fitnesses[objective_i]
                scaled_dist = (next_ind - prev_ind) / fitness_factor
                objective_sorted[individual_i].crowding_distance += scaled_dist


""" #######################   MAIN   #########################################"""
def read_file_matrix(f):
    file_name = root  + '/files/'+ f
    return np.load(file_name)

if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    folder = os.path.basename(os.path.dirname(current_file_path))
    root = os.get.pathdir(os.getcwd())
    args = get_config()
    random.seed(4)

    matrix_dist = read_file_matrix(args.file1)
    matrix_cost = read_file_matrix(args.file2)

    numGenes = matrix_dist.shape[0]
    numPop = args.numPop #60
    itMax = args.itMax #10000
    mutate_prob = args.mutRate 

    TIMEOUT =  11000 # seconds ~ 2 h.

    start_time = time.time()

    ga = AG_MULTI()

    try:
        result_list, fitness_list = ga.train()
        end_time = time.time()
    except timeout_decorator.TimeoutError:
        end_time = time.time()
    
    print(f'PROBLEM: {args.file}, SOLVER: {folder}, ITER: {ga.it}, TIME: {end_time-start_time}, BEST_COST: {ga.best.fitness}, BEST_PATH:{ga.best.genoma}' )
    #TO CSV
    ok = pd.DataFrame({'PROBLEM':'', 'ARGS': str(args), 'SOLVER':folder, 'iter':ga.it, 'TIME':end_time-start_time, 'BEST_COST':str(ga.best.fitness), 'BEST_PATH':str(ga.best.genoma)}, index=[0])
    df = pd.read_csv('results.csv')
    df = pd.concat([df,ok])
    df.to_csv('results.csv', index=False)
    
    #PLOT 
    fig, ax = plt.subplots()
    plt.plot(fitness_list[0], label='distance')
    plt.plot(fitness_list[1], label='cost')
    plt.xlabel("Iterations")
    plt.ylabel("Fitness: cost")
    plt.title( f"{folder}-{args}")
    plt.savefig(root +f'/figures/{args}.fit.{folder}.png')
    plt.show()

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

    
