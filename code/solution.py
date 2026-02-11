# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

import numpy as np

class Solution:
    """
    A class representing the clustering of error samples and the prototype associated to each cluster
    """

    def __init__(self, genotype, max_clusters, distance_matrix):
        """
        Create a solution from a given genotype
        :param genotype: Array with the genotype
        """
        self.distance_matrix = distance_matrix
        self.genotype = genotype
        self.genotype_length = len(genotype)
        self.max_clusters = max_clusters
        self.fitness = None
        self.silhouette_score = None
        self.xai_space_score = None
        self.labels_ = []

    def get_genotype(self):
        """
        Return the genotype of the solution
        :return: Array with the genotype
        """
        return self.genotype

    def set_genotype(self, genotype):
        """
        Set the genotype of the solution
        :param genotype: New genotype
        """
        self.genotype = genotype
        self.genotype_length = len(genotype)

    def get_genotype_length(self):
        """
        Return the genotype length
        :return: Length of the genotype
        """
        return self.genotype_length

    def set_cluster_gen(self, pos, value):
        """
        Set the id of the cluster to one gen
        :param pos: The position in the genotype, it should be an even index
        :param value: The value of the gen, it should be between 0 and max_clusters
        :return: True if the value is set, false otherwise
        """
        if pos < self.genotype_length:
            if 0 >= value < self.max_clusters:
                self.genotype[pos] = value
                return True
            else:
                return False
        else:
            return False

    def get_cluster_genes(self):
        """
        Get a copy of the genes in the positions representing the cluster id
        :return: Array with the half of the genotype containing the cluster ids
        """
        return self.genotype[1::2].copy()

    def get_number_clusters(self):
        """
        Get the number of clusters in the solution
        :return: Number of clusters in the solution
        """
        i = 0
        n = self.genotype_length
        n_clusters = 0
        while i < n:
            if self.genotype[i] == 1:
                n_clusters = n_clusters + 1
            i = i + 1
        return n_clusters

    def set_fitness(self, value):
        """
        Set the fitness value
        :param value: New fitness value
        """
        self.fitness = value

    def get_fitness(self):
        """
        Get the fitness value
        :return: Current fitness value
        """
        return self.fitness

    def is_evaluated(self):
        """
        Check whether the solution has been evaluated before
        :return: True if the solution has a fitness value, False otherwise
        """
        if self.fitness is None:
            return False
        else:
            return True

    def get_max_clusters(self):
        """
        Get the maximum number of clusters
        :return: Maximum number of clusters
        """
        return self.max_clusters

    def is_feasible(self):
        """
        Check if the solution has a valid genotype. Three constraints are checked:
        1. The solution has one cluster at least
        2. Each cluster has one prototype only
        :return: True if the solution is valid, False otherwise. The number of the violated constraint is returned too.
        """
        feasible = True
        constraint = 0

        # First constraint: one cluster at least
        num_clusters = self.get_number_clusters()
        if num_clusters < 1:
            feasible = False
            constraint = 1

        # Second condition: each cluster has only one prototype
        else:
            for i in range(0, num_clusters):
                if feasible:
                    num_cluster_prototype = 0
                    for j in range(0, self.genotype_length):
                        if self.genotype[j] == i and self.genotype[j + 1] == 1:
                            num_cluster_prototype = num_cluster_prototype + 1
                    if not num_cluster_prototype == 1:
                        feasible = False
                        constraint = 2
        return feasible, constraint

    def get_phenotype(self):
        """
        Return a string representation of the genotype
        :return: A string with the genotype information
        """
        phenotype = '{'
        i = 0
        n = self.genotype_length
        while i < n:
            phenotype = phenotype + '{' + str(self.genotype[i]) + ',' + str(self.genotype[i + 1]) + '},'
            i = i + 2

        phenotype = phenotype[:len(phenotype) - 1] + '}'
        return phenotype

    def copy(self):
        """
        Return a copy of the object
        :return: A new solution with the same genotype and fitness
        """
        new_genotype = self.genotype.copy()
        new_solution = Solution(new_genotype, self.max_clusters, self.distance_matrix)
        new_solution.set_fitness(self.get_fitness())
        return new_solution

    def get_clusters(self):
        # 1 - Obtener los puntos medoides y asignarle a cada uno un cluster
        number_clusters = self.get_number_clusters()
        clusters = []

        cluster = 0
        for i in range(0, self.genotype_length):
            if self.genotype[i] == 1:
                clusters.append([i])
        # 2 - Calcular la distancia de cada punto hacia los medoides y decidir a que cluster pertenece

        for i in range(0, self.genotype_length):
            if self.genotype[i] == 0:
                # Calculate the distance from the point to the medoids
                min_distance = float('inf')
                for j in range(0, number_clusters):
                    distance = self.get_precomputed_distance(i, clusters[j][0])
                    if distance < min_distance:
                        min_distance = distance
                        cluster = j
                clusters[cluster].append(i)
        return clusters

    def solution_to_eval(self, dataset):
        phen_clusters = self.get_clusters().copy()
        gen_clusters = self.get_clusters().copy()
        for i in range(0, len(phen_clusters)):
            for j in range(0, len(phen_clusters[i])):
                phen_clusters[i][j] = dataset[gen_clusters[i][j]]
        return phen_clusters

    def get_precomputed_distance(self, i, j):
        """
        Get the distance between two points in the dataset
        :param i: Index of the first point
        :param j: Index of the second point
        :return: Distance between the two points
        """
        return self.distance_matrix[i][j]

    def set_distance_matrix(self, distance_matrix):
        """
        Update the distance matrix
        :param distance_matrix: New distance matrix
        """
        self.distance_matrix = distance_matrix

    def get_distance_matrix(self):
        """
        Get the distance matrix
        :return: Distance matrix
        """
        return self.distance_matrix

    def get_point_indexes(self, dataset):
        clusters = self.get_clusters()
        points = [0]*len(dataset)
        index = 0
        for i in range(0, len(clusters)):
            for k in clusters[i]:
                points[k] = index
            index = index + 1
        return np.array(points)

    def get_medoid_indexes(self):
        #Select a random slice of the numner max_clusters indexes of the medoids (the ones with 1 in the genotype)
        medoid_indexes = []
        for i in range(0, self.genotype_length):
            if self.genotype[i] == 1:
                medoid_indexes.append(i)
        assert medoid_indexes != [] and isinstance(medoid_indexes, list)
        return medoid_indexes

