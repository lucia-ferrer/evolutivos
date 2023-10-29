from itertools import chain
from nsga_utils import rank_assign_sort, crowding_distance_assign
import numpy as np
#______________________________________________________________________
#_________________________ EA ADULT SELECTION _________________________
def adult_select_pareto_crowding_distance(children, parents, f_mins, f_maxes):
    """
    Select adults from a composite pool of children and their parents
    (from the previous generation) based on first on rank and second on
    crowding distance. This ensures elitism.
    """
    # The number of parents is 0 the first iteration. So, assume that the child
    # pool has the same size as the parent pool.
    n_adults = len(children)
    # Combine children and adults into same pool. Sort the pool into non-dominated fronts.
    pool = rank_assign_sort(list(chain(parents, children)))
    adults = []
    rank_indexes = []
    for non_dominated_front in pool:
        # Crowding distance needs to be calculated for all fronts since it will later be used for tournament selection.
        crowding_distance_assign(non_dominated_front, f_mins, f_maxes)

        # Add non-dominated fronts to empty adult pool until no more complete fronts can be added.
        if len(adults) + len(non_dominated_front) <= n_adults:
            adults.extend(non_dominated_front)
            rank_indexes.append(len(adults))
        # Then, add individuals from the last front based on their crowding distance.
        else:
            non_dominated_front = sorted(non_dominated_front,
                                         key=lambda ind: ind.crowding_distance,
                                         reverse=True)
            adults.extend(non_dominated_front[:n_adults - len(adults)])
            rank_indexes.append(len(adults))
            break
    return adults, rank_indexes
#_____________________________________________________________________________
#_____________________________ GENETIC OPERATORS _____________________________

def displacement_mutation(genome, mutation_rate):
    """
    A sub-tour is selected at random, taken out and
    reinserted at random position.
    :param genome:
    :param mutation_rate:
    :return:
    """
    # todo this can probably be implemented faster
    if np.random.rand() < mutation_rate:
        i_left, i_right = _random_indices(len(genome))
        part = genome[i_left:i_right]
        mutated_genome = np.concatenate([genome[:i_left], genome[i_right:]])
        insert_pos = np.random.randint(0, len(mutated_genome)+1)
        mutated_genome = np.insert(mutated_genome, insert_pos, part)
        return mutated_genome
    return genome


def ordered_crossover(genome_a, genome_b, crossover_rate):
    """
    Ordered crossover ("OX-1")
    First, a sub-tour is selected at random from each route. The two sub-tours
    are at the same position in both routes,
    e.g. from visited city #3 to visited city #8 (0-indexed, high-exclusive).
    parent_a = [8, 4, 7, 3, 6, 2, 5, 1, 9, 0] -> child_a = [3, 6, 2, 5, 1]
    parent_b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> child_b = [3, 4, 5, 6, 7]
    Then, starting from the end position of the sub-tours, looping around to
    the end position of the sub-tours (exclusive), cities are added from the
    opposite parent to the child genome if the city is not yet present
    in the child genomes route. This ensures that a child has two sub-tours,
    one from each parent, and that each sub-tour has the same order
    as that in the parent.
    """
    if np.random.rand() >= crossover_rate:
        return [genome_a, genome_b]
    size = len(genome_a)
    i_left, i_right = _random_indices(size)
    child_a = np.ones(size, dtype='uint16')*-1
    child_b = np.ones(size, dtype='uint16')*-1
    child_a[i_left:i_right] = genome_a[i_left:i_right]
    child_b[i_left:i_right] = genome_b[i_left:i_right]

    # indexes in child to be filled from other parent
    child_chain = np.concatenate((np.arange(i_right, size),
                                  np.arange(0, i_left)))
    # indexes in parent in the order they should be added to the other child
    parent_chain = np.concatenate((np.arange(i_right, size),
                                   np.arange(0, i_right)))

    for chain_i, i in enumerate(child_chain):
        # Add genes from the opposite parent in the same order they appear in the opposite parent
        for j in parent_chain[chain_i:]:
            #  if genome_b[j] not in child_a:
            if not np.any(genome_b[j] == child_a):
                child_a[i] = genome_b[j]
                break
        for j in parent_chain[chain_i:]:
            #  if genome_a[j] not in child_b:
            if not np.any(genome_a[j] == child_b):
                child_b[i] = genome_a[j]
                break

    return [child_a, child_b]


def _random_indices(length):
    n1 = np.random.randint(0, length)
    n2 = np.random.randint(n1+1, length+1)
    return n1, n2


#_________________________________________________________________________
# _________________________ GENOMA _______________________________________
class TSPGenome:
    __slots__ = ['n_cities', 'fitnesses', 'dominates_list',
                 'inverse_domination_count', 'rank', 'crowding_distance',
                 'genotype']

    
    def __init__(self, n_cities, genotype=None):
        self.n_cities = n_cities     #pHi
        self.fitnesses = np.zeros(2, dtype='uint32')
        self.dominates_list = None  # List of individuals that this individual dominates
        self.inverse_domination_count = float('-inf')  # Number of individuals that dominate this individual
        self.rank = -1  # Member of the n'th pareto front; 0 being the best
        self.crowding_distance = -1

        if genotype is None:
            self.genotype = np.random.permutation(
                np.arange(self.n_cities, dtype='uint8'))
        else:
            self.genotype = genotype

    def dominates(self, individual_b):
        """
        Individual_a dominates individual_f if both:
            a is no worse than b in regards to all fitnesses
            a is strictly better than b in regards to at least one fitness
        Assumes that lower fitness is better, as is the case with cost-distance-TSP.
        :param self:
        :param individual_b:
        :return: True if individual_a dominates individual_b
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
        """ Even though A < B, that does not indicate that A.dominates(B),
        as A may have a lower value for fit. func. 1 but greater value for
        fit. func 2 and therefore neither dominate each other. """
        return self.fitnesses[0] < other.fitnesses[0] or \
            (self.fitnesses[0] == other.fitnesses[0] and
             self.fitnesses[1] < other.fitnesses[1])

    def __gt__(self, other):
        return self.fitnesses[0] > other.fitnesses[0] or \
            (self.fitnesses[0] == other.fitnesses[0] and
             self.fitnesses[1] > other.fitnesses[1])

    def __eq__(self, other):
        return self.fitnesses[0] == other.fitnesses[0] and \
            self.fitnesses[1] == other.fitnesses[1]

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __ne__(self, other):
        return self.fitnesses[0] != other.fitnesses[0] and \
            self.fitnesses[1] != other.fitnesses[1]


# __________________________________________________________________________
#_____________________________PARENT SELECTION  CON TOURNMETN_____________________________


def parent_select_crowding_tournament(adults, tournament_size=2):
    """
    Parent selection based on the crowded-comparison operator.
    Creates as many parents as there are adults.
    :param adults:
    :param tournament_size:
    :return:
    """
    parents = []
    n = len(adults)
    for i in range(n):
        group = np.random.choice(adults, size=tournament_size)
        best = group[0]
        for individual in group[1:]:
            if individual.rank < best.rank:
                best = individual
            elif individual.rank == best.rank and \
                    individual.crowding_distance > best.crowding_distance:
                best = individual
        parents.append(best)

    return parents

#_________________________________________________________________________________
#____________________________________ EVALUATE FITTNESS___________________________

class TSPPopulation:
    """
    A population of potential solutions (i.e. individuals)
    """

    def __init__(self, problem):
        self.problem = problem
        self.distances = problem.distances
        self.costs = problem.costs
        self.n_cities = len(self.distances)
        self.population_size = problem.population_size
        self.genome = problem.genome
        self.crossover = problem.crossover_method
        self.mutate = problem.mutation_method
        self.crossover_rate = problem.crossover_rate
        self.mutation_rate = problem.mutation_rate
        self.generation = 0
        self.children = [self.problem.genome(self.n_cities)
                         for _ in range(self.population_size)]
        self.adults = []
        self.parents = []
        # A list of indexes of the end position for each rank, such that
        # adults[:rank_indexes[0]] will yield the first front,
        # adults[rank_indexes[0]:rank_indexes[1]] the second and so forth
        self.rank_indexes = []
        self.f_mins = [float('inf'), float('inf')]
        self.f_maxes = [float('-inf'), float('-inf')]

    def evaluate_fitnesses(self):
        """
        Evaluate the fitnesses of the phenotypes.
        If the evaluation is a stochastic process, then adults should also be
        evaluated each run, in order to weed out phenotypes with
        a lucky evaluation.
        """
        for child in self.children:
            self.evaluate_fitness(child)
            for fitness_i in range(len(child.fitnesses)):
                f = child.fitnesses[fitness_i]
                if f < self.f_mins[fitness_i]:
                    self.f_mins[fitness_i] = f
                if f > self.f_maxes[fitness_i]:
                    self.f_maxes[fitness_i] = f

    def select_adults(self):
        self.adults, self.rank_indexes = self.problem.adult_select_method(self.children,
                                                                          self.adults,
                                                                          self.f_mins,
                                                                          self.f_maxes)

    def select_parents(self):
        """
        Select adults to become parents, e.g. to mate.
        """
        self.parents = self.problem.parent_select_method(self.adults,
                                                         **self.problem.parent_select_params)

    def reproduce(self):
        """
        Generate children from the selected parents by first
        crossing genes then mutating
        """
        # An individual can reproduce with itself. Probably not optimal.
        '''
        self.children = \
            [self.genome(self.n_cities, genotype=self.mutate(child_genome, self.problem.mutation_rate))
                for parent_a, parent_b in zip(islice(self.parents, 0, None, 2), islice(self.parents, 1, None, 2))
                for child_genome in self.crossover(parent_a.genotype, parent_b.genotype, self.crossover_rate)]
        '''
        # untested refactoring
        self.children = []
        couples = zip(islice(self.parents, 0, None, 2),
                      islice(self.parents, 1, None, 2))
        for parent_a, parent_b in couples:
            crossed_genomes = self.crossover(parent_a.genotype,
                                             parent_b.genotype,
                                             self.crossover_rate)
            for crossed_genome in crossed_genomes:
                mutated_genome = self.mutate(crossed_genome,
                                             self.problem.mutation_rate)
                self.children.append(self.genome(self.n_cities,
                                                 mutated_genome))

    def evaluate_fitness(self, child):
        total_distance = 0
        total_cost = 0
        for i in range(self.n_cities-1):
            city_a = child.genotype[i]
            city_b = child.genotype[i+1]
            #  The cost of travelling from a to b is equal to the cost of travelling from b to a
            total_distance += self.distances[city_a][city_b]
            total_cost += self.costs[city_a][city_b]

        child.fitnesses[0] = total_distance
        child.fitnesses[1] = total_cost

    @property
    def n_fronts(self):
        return len(self.rank_indexes)

    def get_front(self, rank):
        if rank >= 1:
            start = self.rank_indexes[rank - 1]
        else:
            start = 0
        return self.adults[start:self.rank_indexes[rank]]

    @staticmethod
    def area_metric(front):
        """
        Return a metric for the 'total fitness' of a pareto front. This metric
        can only be compared to the area metric
        of other fronts if the f_mins and f_maxes are the same.

        For two dimensional pareto fronts, the normalized area under the two
        pareto frontiers is a very nice metric.
        This means that whenever one Pareto front approximation dominates
        another, the are of the former is less (if both fitness functions are
        to be minimized) than that of the latter.
        """
        pass

    @staticmethod
    def min_fitness(pool, fitness_func_i):
        return min(pool, key=lambda i: i.fitnesses[fitness_func_i])

    @staticmethod
    def max_fitness(pool, fitness_func_i):
        return max(pool, key=lambda i: i.fitnesses[fitness_func_i])

    @staticmethod
    def n_unique(front):
        """
        Number of solutions with different fitness in the given front.

        :param front:
        :return:
        """
        a = np.asarray([ind.fitnesses for ind in front])
        unique = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))) \
            .view(a.dtype) \
            .reshape(-1, a.shape[1])
        return len(unique)


#__________________________________________________________________________________________
#______________________________________PROBLEMS____________________________________________


class InvalidDatasetError(Exception):
    pass


class TSPProblem:
    def __init__(self, distances, costs):
        if len(distances) != len(costs):
            raise InvalidDatasetError('Length of data sets not equal')

        self.costs = costs
        self.distances = distances
        self.n_cities = len(costs)

        self.population_size = 60
        self.generation_limit = 20

        self.genome = TSPGenome
        self.genome_params = {
        }

        self.adult_select_method = adult_select_pareto_crowding_distance
        self.parent_select_method = parent_select_crowding_tournament
        self.parent_select_params = {
            'tournament_size': 2
        }

        self.mutation_method = displacement_mutation
        self.crossover_method = ordered_crossover
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8


        # mut: 0.001, 0.005, 0.01, 0.05, 0.1
        # cross: 0.5, 0.6, 0.7, 0.8, 0.9

        # best: mut 0.01, cross: 0.8

        # pop/gen: 400/500, 200/1000, 100/2000, 50/4000
        # best: 100/2000


#_______________________________________________________________
# ___________________ RUNNER____________________________________
import logging
import pickle
import itertools
import timeit
from matplotlib import pyplot as plt
from utils import Loader

class EARunner:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.loader = Loader(False)
        self.run_problem()

    def run_problem(self):
        distances, costs = self.loader.load_dataset_a()
        problem = TSPProblem(distances, costs)
        save_path = str(problem.population_size) + ' ' \
                    + str(problem.generation_limit) + ' ' \
                    + str(problem.crossover_rate) + ' ' \
                    + str(problem.mutation_rate) + ' report'
        self.run(problem, plot=True)
        # self.run(problem, plot=True, save_path="../results/" + save_path)

    def run_true_front(self):
        distances, costs, front = self.loader.load_dataset_b()
        problem = TSPProblem(distances, costs)
        self.run(problem, plot=True, true_front=front)

    def load_results(self):
        paths = ["../results/50 4000 0.7 0.05 report-0.pickle",
                 "../results/100 2000 0.8 0.01 report-0.pickle",
                 "../results/200 1000 0.8 0.05 report-1.pickle"]
        self.load_results_stats(paths)
        self.load_results_plot(paths)

    @staticmethod
    def run(problem, true_front=None, plot=True, save_path=None):
        """
        :param problem:
        :param plot:
        :param true_front: actual optimal front (for comparison with discovered/calculated front)
        :param save_path: Save the first front of the final population to file with the given path
        :return:
        """
        # Generate the initial population
        population = TSPPopulation(problem)
        logging.info("Generations: %s, Pop. size: %s, Cross. rate: %s, Mut. rate: %s",
                     problem.generation_limit,
                     problem.population_size,
                     problem.crossover_rate, problem.mutation_rate)
        fronts = []

        def main_loop():
            while population.generation < problem.generation_limit:
                population.generation += 1
                population.evaluate_fitnesses()  # Calculate total cost and total distance for each route/individual
                population.select_adults()
                population.select_parents()
                population.reproduce()
                if population.generation % (problem.generation_limit / 5) == 0:
                    logging.info("\t\t Generation %s/%s", population.generation, problem.generation_limit)
                    fronts.append(population.get_front(0))

        logging.info("\tExecution time: %s", timeit.timeit(main_loop, number=1))
        logging.info("\t(Min/Max) Distance: %s/%s; Cost: %s/%s",
                     TSPPopulation.min_fitness(population.adults, 0).fitnesses[0],
                     TSPPopulation.max_fitness(population.adults, 0).fitnesses[0],
                     TSPPopulation.min_fitness(population.adults, 1).fitnesses[1],
                     TSPPopulation.max_fitness(population.adults, 1).fitnesses[1])

        if save_path:
            with open(save_path + "-" + str(np.random.randint(10)) + '.pickle', 'wb') as f:
                pickle.dump(population.get_front(0), f)
        if plot:
            EARunner.plot([population.adults], save_path=save_path)
            EARunner.plot([population.get_front(0)],
                          name='Fitnesses, final Pareto-front',
                          save_path=save_path)
            # EARunner.plot(fronts, true_front=true_front, dash=True,
            #              name='Fitnesses, final Pareto-front per 20% progress', save_path=save_path)
            plt.show()

    @staticmethod
    def plot(pools, true_front=None, dash=False, name='Fitnesses', save_path=None):
        """
        :param true_front:
        :param pools: NOT instance of TSPPopulations, but a list of lists of individuals (lists of population.adults)
        :param dash: dash lines between each individual in each pool
        :param name: Plot legend
        :param save_path:
        :return:
        """
        marker = itertools.cycle(('o', ',', '+', '.', '*'))
        color = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))

        if dash:
            linestyle = "--"
            for pool_i in range(len(pools)):
                pools[pool_i] = sorted(pools[pool_i],
                                       key=lambda ind: ind.fitnesses[0])
        else:
            linestyle = ""

        plt.figure()
        plt.title(name)

        for i, pool in enumerate(pools):
            c = next(color)
            plt.plot([individual.fitnesses[0] for individual in pool],
                     [individual.fitnesses[1] for individual in pool],
                     marker=next(marker), linestyle=linestyle, color=c,
                     label=str((i + 1) * 20) + "%-" + str(len(pool))
                           + "sols-" + str(TSPPopulation.n_unique(pool)) + "uniq")
            min_dist = TSPPopulation.min_fitness(pool, 0).fitnesses
            max_dist = TSPPopulation.max_fitness(pool, 0).fitnesses
            min_cost = TSPPopulation.min_fitness(pool, 1).fitnesses
            max_cost = TSPPopulation.max_fitness(pool, 1).fitnesses
            if not dash:
                c = 'r'
            plt.plot([min_dist[0]], [min_dist[1]], marker='D', color=c)
            plt.plot([max_dist[0]], [max_dist[1]], marker='D', color=c)
            plt.plot([min_cost[0]], [min_cost[1]], marker='D', color=c)
            plt.plot([max_cost[0]], [max_cost[1]], marker='D', color=c)
        if true_front is not None:
            plt.plot([i[0] for i in true_front], [i[1] for i in true_front],
                     linestyle="--", label="True front")
            # if dash:
            # plt.legend(loc="best")
        plt.xlabel("Distance")
        plt.xticks(np.arange(30000, 120001, 10000))
        plt.ylabel("Cost")
        plt.yticks(np.arange(300, 1401, 100))
        if save_path:
            plt.savefig(save_path + "-" + str(np.random.randint(10)) + ".png")

    @staticmethod
    def load_results_plot(paths):
        populations = []
        for path in paths:
            with open(path, 'rb') as f:
                population = pickle.load(f)
                populations.append(population)
        EARunner.plot(populations, dash=True,
                      name="Final pareto fronts, 3 configurations")
        plt.show()

    @staticmethod
    def load_results_stats(paths):
        for path in paths:
            with open(path, 'rb') as f:
                population = pickle.load(f)
                logging.info("\t(Min/Max) Distance: %s/%s; Cost: %s/%s",
                             TSPPopulation.min_fitness(population, 0).fitnesses[0],
                             TSPPopulation.max_fitness(population, 0).fitnesses[0],
                             TSPPopulation.min_fitness(population, 1).fitnesses[1],
                             TSPPopulation.max_fitness(population, 1).fitnesses[1])


if __name__ == "__main__":
    runner = EARunner()