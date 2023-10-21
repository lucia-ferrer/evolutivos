import os
import random
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

random.seed(14)


def complete_graph_generator(file_name):
    G = nx.Graph()
    layout = np.load(file_name)
    print(layout)
    n = layout.shape[0]
    G.add_nodes_from(range(n))
        
    for i in G.nodes:
        max_per_node = 0
        for j in range(i+1,n):  # Usamos i+1 para evitar agregar duplicados
            coste = layout[i, j]
            if coste > 0: G.add_edge(i, j, weight=coste)
    print(G)
    return G

def check(number, list_of_numbers):
    for el in list_of_numbers:
        if number == el:
            return True
    return False

def nearest_neighbor_tour(graph):
    n = graph.number_of_nodes()
    
    if n == 0:
        print('Empty graph.')
        return
    if n == 1:
        print('Graph is a single node.')
        return
    
    best_tour = []
    max_dist = max(list(nx.get_edge_attributes(graph, 'weight').values()))
    best_cost = n*max_dist
    for i in graph.nodes:
        already_vis = [i]
        total_cost = 0
        while len(already_vis) < n:
            node = already_vis[len(already_vis)-1]
            best_weight = max_dist
            best_node = node
            for neighbor in graph.neighbors(node):
                tmp_node = neighbor
                if not check(tmp_node, already_vis):
                    tmp_weight = graph[node][tmp_node]['weight']
                    if tmp_weight <= best_weight:
                        best_weight = tmp_weight
                        best_node = tmp_node
            already_vis.append(best_node)
            total_cost += best_weight
        total_cost += graph[already_vis[len(already_vis)-1]][i]['weight'] # cost to return at the first node
        if total_cost <= best_cost:
            best_cost = total_cost
            best_tour = already_vis
    return best_tour, best_cost

def factorial(n):
    ret = 1
    for i in range(n):
        ret *= n-i
    return ret
