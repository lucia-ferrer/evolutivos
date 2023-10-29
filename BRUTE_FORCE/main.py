import argparse
import itertools as it
import json
import numpy as np
import sys

bestRoute = None

def length (M, route):
    """
    Calculates the length of a route.

    @param M: the matrix with the distances.
    @param route: the route to calculate.
    @return the length of the route.
    """
    return sum([M[route[i], route[(i+1) % len(route)]] for i in range(len(M))])


def search (M):
    """
    Searches for the optimal path.
    So far, this is doing brute force.

    @param input: the input
      dataframe.
    @return a dataframe with two columns: ID and price.
    """
    global bestRoute
    bestLength = sys.maxsize
    np.random.seed(1)
    cities = list(range(1, len(M)))
    np.random.shuffle(cities)
    for route in it.permutations(cities):
        route = [0] + list(route)
        l = length(M, route)
        if l < bestLength:
            bestLength = l
            bestRoute = route


if __name__ == '__main__':

    # Creates a parser to receive the input argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Path to the data file.')
    args = parser.parse_args()

    # Read the argument and load the data.
    try:
        M = np.load('ulises.npy')
    except:
        print("Error: the input file does not have a valid format.", file=sys.stderr)
        exit(1)

    # Runs the search algorithm
    # NOTE: this is now brute force, must be GA instead.
    try:
        search(M)
    except KeyboardInterrupt:
        print(json.dumps(bestRoute))
        