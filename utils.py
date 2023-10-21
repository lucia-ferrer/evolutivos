import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
random.seed(3)

def complete_graph_generator(file_name):
    G = nx.Graph()
    layout = np.load(file_name)
    n = layout.shape[0]
    max_total = 0 #digamo sque si los costes son uniformes sería n * max_cost.
    G.add_nodes_from(list(range(n)))
    # print(G.nodes)
    
    for i in G.nodes:
        max_per_node = 0
        for j in range(i+1,n):  # Usamos i+1 para evitar agregar duplicados
            coste = layout[i, j]
            G.add_edge(i, j, weight=coste)
            if coste > max_per_node : max_per_node = coste
        max_total += max_per_node
    
    
    # print('Graph with', G.number_of_nodes(), 'nodes and', G.number_of_edges(), 'edges successfully created.')
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.spring_layout(G)
    
    nx.draw(G, pos=pos, with_labels=True, node_size=50, node_color='skyblue', font_size=10, font_color='black')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=6)
    # plt.show()
    # plt.close('all')
    return G, layout, max_total

def check(number, list_of_numbers):
    for el in list_of_numbers:
        if number == el:
            return True
    return False

def nearest_neighbor_tour(graph, layout):
    n = graph.number_of_nodes()
    
    if n == 0:
        print('Empty graph.')
        return
    if n == 1:
        print('Graph is a single node.')
        return
    
    best_tour = []
    distances = np.array(list(nx.get_edge_attributes(graph, 'weight').values()))
    
    best_cost = min(n*max(distances), np.sum(distances))
    # print(graph.nodes)
    for i in graph.nodes:
        already_vis = [i]
        total_cost = 0
        while len(already_vis) < n:
            node = already_vis[len(already_vis)-1]
            best_weight = max(layout[i])#max(distances)+1
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
    
# ----------------------------------------------------------------------------
def print_pop(pop):
    for ind in pop:
        print(ind)

def fit(cand, graph):
    weights = [graph[cand[i]][cand[i + 1]]['weight'] for i in range(len(cand) - 1)]
    weights.append(graph[cand[-1]][cand[0]]['weight'])
    return sum(weights)#np.sum(np.array(weights))
    

def probabilities(pop, graph, max_cost):
    n = graph.number_of_nodes()
    if n != len(pop[0]):
        print('Errore')
    
    tmp = np.array([max_cost - fit(cand, graph) for cand in pop])
    s = np.sum(tmp)
    
    if s == 0:
        return [1 / len(pop)] * len(pop)
    
    return list(tmp / s)

def initialize_population(m, n):
    rng = np.random.default_rng()
    pop = np.array([rng.permutation(np.arange(1, n)) for _ in range(m)])
    pop = np.insert(pop, 0, 0, axis=1)
    return pop

   

def selection(pop, pro):
    members = len(pop)
    return random.choices(population=pop, weights=pro, k=members)

def control_and_return(pop, graph, ref_cost):
    # epsilon = 1
    for el in pop:
        fit_value = fit(el, graph)
        if fit_value <= ref_cost+1:  #- epsilon:
            return True, fit_value
    return False, fit_value

def find_two_maximal_pos_with_np(lista, graph):
    if not lista or len(lista) <2:
        return 0, 0, 0, 0

    n = len(lista)
    weights = np.array([graph[lista[i]][lista[(i + 1) % n]]['weight'] for i in range(n)])
    pos_1 = np.argmax(weights)
    arr_1 = (pos_1 + 1) % n

    weights[pos_1] = -1
    weights[(pos_1-1)%n] = -1
    weights[(pos_1+1)%n] = -1
        
    pos_2 = np.argmax(weights) 
    w2 = weights[pos_2]
    arr_2 = (pos_2 + 1) % n

    return pos_1, pos_2, arr_1, arr_2

def single_swap_improve(pop, graph):
    new_pop = []

    for mem in pop:
        pos_1, pos_2, arr_1, arr_2 = find_two_maximal_pos_with_np(mem, graph)
        # pos_1norm, pos_2norm, arr_1norm, arr_2norm = find_two_maximal_pos(mem, graph)
        node_1 = mem[pos_1]
        side_1 = mem[arr_1]
        node_2 = mem[pos_2]
        side_2 = mem[arr_2]
        weights = np.array([
            graph[node_1][side_1]['weight'],
            graph[node_2][side_2]['weight'],
            graph[node_1][node_2]['weight'],
            graph[side_1][side_2]['weight'],
            graph[node_1][side_2]['weight'],
            graph[node_2][side_1]['weight']
        ])

        min_weight_idx = np.argmin(weights)

        new_mem = mem.copy()

        if min_weight_idx == 2:  # Quad configuration
            new_mem[arr_1], new_mem[pos_2] = new_mem[pos_2], new_mem[arr_1]
        elif min_weight_idx == 3:  # Cross configuration
            new_mem[arr_1], new_mem[arr_2] = new_mem[arr_2], new_mem[arr_1]

        new_pop.append(new_mem)

    return new_pop

def random_mutation(pop):
    new_pop = []

    number_of_nodes = len(pop[0])
    mutation_positions = np.random.choice(number_of_nodes, size=(len(pop), 2), replace=True)

    for el, (first_pos, second_pos) in zip(pop, mutation_positions):
        new_el = el.copy()
        new_el[first_pos], new_el[second_pos] = new_el[second_pos], new_el[first_pos]
        new_pop.append(new_el)

    return new_pop

        
def genetic(graph, members, ref_cost, max_cost):
    n = graph.number_of_nodes()
    # cross takes an even number of members
    if members % 2 != 0:
        members += 1
    population = initialize_population(members, n) # members, size_of_a_member
    cont = 0
    cond = True
    while cond:
        cont += 1
        prob = probabilities(population, graph, max_cost)
        population = selection(population, prob)
        population = single_swap_improve(population, graph) # recombine edges to obtain a lower cost
        population = random_mutation(population)

        if cont % 50000 == 0:
            print()
            print_pop(population)
        condition_to_stop, a_fit_value = control_and_return(population, graph, ref_cost)
        if condition_to_stop:
            print('\nReached a good solution in', cont, 'iterations')
            print_pop(population)
            print('cost of the solution:', a_fit_value)
            cond = False
        elif cont >= 100000000:
            print('\nToo much iterations. Return the current solution.')
            print_pop(population)
            print('cost of the solution:', a_fit_value)
            cond = False    