import numpy as np
import networkx as nx
import random
from tqdm import tqdm

random.seed(3)


class TSPGenetic:
    def __init__(self, graph, m=0):
        self.graph = graph
        self.n = graph.number_of_nodes()
        if m == 0: m = self.n
        if m % 2 != 0: m += 1
        self.m = m
        self.reduccion = 0.75
        self.initialize_population()

    def initialize_population(self):
        rng = np.random.default_rng()
        pop = np.array([rng.permutation(np.arange(1, self.n)) for _ in range(self.m)])

        self.population = np.insert(pop, 0, 0, axis=1)
        self.fitness = np.array([self.fit(individuo) for individuo in self.population])
        self.best_cost = np.min(self.fitness)
        self.best_ind = self.population[np.argmin(self.fitness)]
    
    def fit(self, cand):
        try: 
            coste =  nx.path_weight(self.graph, cand, 'weight') 
        except nx.exception.NetworkXNoPath:
            print()
            print('Error because PATH does not exist')
            print(cand)
            quit()
        coste += self.graph[cand[-1]][cand[0]]['weight']
        return coste

    def selection_with_probabilities(self, max_cost):
        tmp = max_cost - self.fitness  #podrían existir probabilidades negativas si el nuevo fit es peor.

        # Mover la distribucion con minimo 0
        positive_tmp = tmp - np.min(tmp)

        # Normalizar para [0, 1]
        p = positive_tmp / np.sum(positive_tmp)

        red = int(self.m * self.reduccion)
        rng = np.random.default_rng()

        return rng.choice(self.population, p=p, size=red)

    def find_worst_edge(self, individuo):
        # Implement this function to find worst connections in an individual
        # el individuo es la permutacion de ciudades ~ nodos. 

        weights = np.array([self.graph.edges[edge]['weight'] for edge in nx.utils.pairwise(individuo, cyclic=True)])
        u = np.argmax(weights)
        v = (u + 1) % self.n

        weights[u] = -1
        weights[(u-1)%self.n] = -1
        weights[(u+1)%self.n] = -1
            
        u2 = np.argmax(weights) 
        w2 = weights[u2]
        v2 = (u2 + 1) % self.n

        return u, u2, v, v2

    def single_swap_improve(self):
        # Implement this function to improve individuals by swapping
        k = int(self.m * self.reduccion)
        new_pop =  np.zeros((k, self.n))
        i = 0
        for ind in self.population: #por cada individuo en la poblacion
            u, u2, v, v2 = self.find_worst_edge(ind)
            ori1 = ind[u]
            dest1 = ind[v]
            ori2 = ind[u2]
            dest2 = ind[v2]
            weights = np.array([
                self.graph[ori1][dest1]['weight'] +  self.graph[ori2][dest2]['weight'], #norm
                self.graph[ori1][ori2]['weight']  +  self.graph[dest1][dest2]['weight'], #quad
                self.graph[ori1][dest2]['weight'] +  self.graph[ori2][dest1]['weight'] #cross
            ])

            min_weight_idx = np.argmin(weights)

            new_ind = np.copy(ind)

            if min_weight_idx == 1:  
                new_ind[v], new_ind[u2] = new_ind[u2], new_ind[v] #Quad
            elif min_weight_idx == 2:  
                new_ind[v], new_ind[v2] = new_ind[v2], new_ind[v] #Cross

            new_pop[i] = np.copy(new_ind)
            i += 1

        return new_pop
    
    def random_mutation(self):
        # Implement this function for random mutation
        k = int(self.m * self.reduccion)
        new_pop =  np.zeros((k, self.n))
        i=0
        mutation_positions = np.random.choice(self.n, size=(k, 2), replace=True)
        for el, (first_pos, second_pos) in zip(self.population[:k], mutation_positions):
            new_el = el.copy()
            new_el[first_pos], new_el[second_pos] = new_el[second_pos], new_el[first_pos]
            new_pop[i] = new_el
            i+=1
        return new_pop

    def control_improvement(self, ref_coste=None):
        temp = 1e5       
        while True:
            if not ref_coste: 
                if np.any(self.fitness < self.best_cost ):
                    yield False
                else:
                    if temp > 100 : 
                        temp = (90 * temp) / 100
                        yield False
                    else : 
                        return True
            else: 
                if np.any(self.fitness < ref_coste + 1e3 ) or temp <= 100:
                    return True
                else: 
                    temp = (90 * temp) / 100
                    yield False


    def evolutionAG(self, max_iter=int(10e3), ref_coste = None):
        # padres
        cond = True
        self.control_improvement(ref_coste)
        for i in tqdm(range(max_iter)): 

            # sucesores
            self.population = self.selection_with_probabilities(self.best_cost)
            sucesores = self.single_swap_improve() #heuristica
            random_sucesores = self.random_mutation()

            self.population = np.concatenate((sucesores, random_sucesores), axis=0)
            self.fitness = np.array([self.fit(individuo) for individuo in self.population])
            
            condition_to_stop = next(self.control_improvement())

            if self.best_cost >= np.min(self.fitness):
                self.best_cost = np.min(self.fitness)
                self.best_ind = self.population[np.argmin(self.fitness)]

            if condition_to_stop:
                print('\nReached a good solution in', i, 'iterations')
                return self.best_cost, self.best_ind

        return self.best_cost, self.best_ind
