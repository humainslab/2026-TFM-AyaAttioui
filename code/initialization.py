# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

from population import Population
from solution import Solution
import numpy as np

class PopulationCreator:
    """
    A class to create a population with solutions initialized at random
    """
    def __init__(self, pop_size, num_samples, max_clusters, distance_matrix, rnd, min_clusters):
        """
        Constructor to set class parameters
        :param pop_size: Size of the population
        :param num_samples: Number of data points comprising the data frame to be represented
        :param max_clusters: Maximum number of clusters to be assigned
        :param rnd: Random object
        """
        self.pop_size = pop_size
        self.num_samples = num_samples
        self.max_clusters = max_clusters
        self.distance_matrix = distance_matrix
        self.rnd = rnd
        self.min_clusters = min_clusters

    def create_population(self):
        """
        Create a random population of solutions. The population size is the one defined during initialization
        :return: A population (list of solutions) initialized at random
        """
        solutions = list()
        for i in range(0, self.pop_size):
            s = self.create_solution()
            solutions.append(s)
        population = Population(solutions)

        return population

    def create_solution(self):
        """
        Create a solution choosing a random number of clusters and setting one prototype point to each one at random
        :return: A solution with random genotype satisfying the constraints
        """
        genotype_length = self.num_samples
        genotype = np.full(genotype_length, 0, dtype=int)
        num_clusters = self.rnd.randint(self.min_clusters, self.max_clusters)
        genotype = self.fill_random_clusters(genotype, num_clusters)
        solution = Solution(genotype, self.max_clusters, self.distance_matrix)
        # Sanity check
        return solution

    def fill_random_clusters(self, genotype, num_clusters):
        """
        Fill the gen positions of cluster ids with random values between 0 and num_clusters. The method ensures
        that each cluster id appears at least once.
        :param genotype: The array representing the genotype
        :param num_clusters: The number of clusters to be represented
        :return: The genotype with the even positions filled with a cluster id
        """
        genotype_length = len(genotype)
        # Assign each cluster id to a random gen to ensure it contains one data point at least
    
        clusters_initialized = 0
        while clusters_initialized != num_clusters:
            rnd_index = self.rnd.randrange(0, genotype_length)
            if genotype[rnd_index] == 0:
                genotype[rnd_index] = 1
                clusters_initialized += 1
        return genotype
