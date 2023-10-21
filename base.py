import utils
from clase_AG_real import TSPGenetic
import networkx as nx
import os
import time
import json

if __name__ == '__main__':
    # file_name = os.getcwd() + "/files/ulises.npy"
    file_name = os.getcwd() + "/files/tsp.sm.npy"
    g = utils.complete_graph_generator(file_name)

    start_time = time.time()
    # best_tour, nearest_cost = utils.nearest_neighbor_tour(g)
    best_tour = nx.approximation.christofides(g)
    nearest_cost = nx.path_weight(g, best_tour, 'weight')
    end_time = time.time()
    print('Solucion por nearest_neighbour:', best_tour)
    print('Coste:', nearest_cost)
    print('Time taken by nearest_neighbor_tour:', end_time - start_time, 'seconds')

    # Time the genetic function
    start_time = time.time()
    GA = TSPGenetic(g)
    try:
        cost, individuo = GA.evolutionAG(ref_coste=nearest_cost)
        print(json.dumps({  'ind':list(GA.best_ind),
                            'cost':GA.best_cost, 
                            'time': time.time()- start_time}))
    except KeyboardInterrupt:
        print(json.dumps({  'ind':list(GA.best_ind), 
                            'cost':GA.best_cost, 
                            'time': time.time()- start_time}))
