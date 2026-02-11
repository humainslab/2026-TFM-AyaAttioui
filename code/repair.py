# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

import numpy
import numpy as np

from solution import Solution

class RepairOperator:
    """
    A class to implement repair operators

    params:
        rnd: Random object
        min_clusters: Minimum number of clusters
        
    methods:
        repair_population: Repair a population
        check_genotype: Check if a solution satisfies the constraints and repair it if needed
        repair_overflow: Repair a solution with more clusters than allowed
        repair_underflow: Repair a solution with less clusters than allowed
    """
    def __init__(self, rnd, min_clusters):
        self.rnd = rnd
        self.min_clusters = min_clusters

    def repair_population(self, population):
        """
        Repair a population
        :param population: List of solutions
        :return: List of repaired solutions
        """
        new_population = []
        for s in population:
            repaired_solution = self.check_genotype(s)
            new_population.append(repaired_solution)
        return new_population

    def check_genotype(self, solution):
        """
        Check if a solution satisfies the constraints and repair it if needed
        
        params:
            solution: Solution object to check and repair

        return:
            solution: Solution object repaired
        """
        assert isinstance(solution, Solution), 'solution is not an object of class "Solution"'
        max_clusters = solution.get_max_clusters()
        min_clusters = self.min_clusters
        repaired_genotype = solution.get_genotype().copy()

        n_clusters = solution.get_number_clusters()
        if min_clusters > n_clusters:
            repaired_genotype = self.repair_underflow(solution.get_genotype(), min_clusters)

        elif n_clusters > max_clusters:
            repaired_genotype = self.repair_overflow(solution, max_clusters)

        solution.set_genotype(repaired_genotype)

        assert isinstance(solution, Solution), 'solution is not an object of class "Solution"'
        return solution

    def repair_overflow(self, solution, max_clusters):
        """
        Repair a solution with more clusters than allowed

        params:
            solution: Solution object to repair
            
        return:
            repaired_genotype: Genotype of the repaired solution
        """
        assert isinstance(solution, Solution), 'solution is not an object of class "Solution"'
        assert max_clusters is not None
        repaired_genotype = [0]*len(solution.get_genotype())

        medoid_indexes = solution.get_medoid_indexes()
        
        self.rnd.shuffle(medoid_indexes)
        medoid_indexes = medoid_indexes[:max_clusters]

        for i in medoid_indexes:
            repaired_genotype[i] = 1

        repaired_genotype = np.array(repaired_genotype)

        assert isinstance(repaired_genotype, numpy.ndarray), 'repaired_genotype is not an numpy.ndarray'
        return repaired_genotype

    def repair_underflow(self, genotype, min_clusters):
        """
        Repair a solution with less clusters than allowed

        params:
            genotype: Genotype of the solution to repair
            
        return:
            repaired_genotype: Genotype of the repaired solution
        """
        assert isinstance(genotype, numpy.ndarray), 'repaired_genotype is not an numpy.ndarray'

        repaired_genotype = [0]*len(genotype)
        posible_indexes = list(range(0, len(genotype)-1))

        for i in range(min_clusters):
            random_index = self.rnd.choice(posible_indexes)
            repaired_genotype[random_index] = 1
            posible_indexes.remove(random_index)

        repaired_genotype = np.array(repaired_genotype)

        assert isinstance(repaired_genotype, numpy.ndarray), 'repaired_genotype is not an numpy.ndarray'
        return repaired_genotype
