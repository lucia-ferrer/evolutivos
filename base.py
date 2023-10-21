import utils
import os
import time

def main():
    file_name = os.getcwd() + "/files/ulises.npy"
    g, layout, max = utils.complete_graph_generator(file_name)

    start_time = time.time()
    best_tour, nearest_cost = utils.nearest_neighbor_tour(g, layout)
    end_time = time.time()
    print('Target sequence:', best_tour)
    print('Cost of the solution:', nearest_cost)
    print('Time taken by nearest_neighbor_tour:', end_time - start_time, 'seconds')

    # Time the genetic function
    start_time = time.time()
    utils.genetic(g, layout.shape[0], nearest_cost, max)
    end_time = time.time()
    print('Cost of the solution:', nearest_cost)
    print('Time taken by genetic:', end_time - start_time, 'seconds')

if __name__ == '__main__':
    main()
