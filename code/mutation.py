# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

from solution import Solution

class Mutation:
    """
    A class to create mutation operators
    """
    def __init__(self, rnd, genotype_length, mut_probability=0.1, gen_probability=None):
        """
        Create a mutation operator
        :param rnd: The random number generator
        :param genotype_length: Length of the genotype to compute the gen_probability if it is None
        :param mut_probability: Mutation probability (change the solution), 0.1 by default
        :param gen_probability: Probability to change each gen, None by default
        """
        self.rnd = rnd
        self.mut_probability = mut_probability
        if gen_probability is None:
            self.gen_probability = 1.0 / genotype_length
        else:
            self.gen_probability = gen_probability

    def mutate_solutions(self, offspring):
        """
        Mutate each offspring with a given probability
        :param offspring: The list of offspring to be mutated
        :return: List of mutated offspring (same size)
        """
        mutants = list()

        for i in range(0, len(offspring)):
            # Mutate this solution with a given probability
            if self.rnd.random() < self.mut_probability:
                mutated_genotype = self.mutation(offspring[i].get_genotype(), offspring[i].get_number_clusters(), offspring[i].get_max_clusters())
                mutant = Solution(mutated_genotype, offspring[i].get_max_clusters(), offspring[i].get_distance_matrix())
            else:
                mutant = offspring[i].copy()
            mutants.append(mutant)
        return mutants

    def mutation(self, genotype, n_clusters, max_clusters):
        """
        Perform bit mutation at a configured probability
        :param genotype: The genotype to be mutated
        :param max_clusters: Maximum number of clusters
        :return A mutated genotype
        """
        genotype_length = len(genotype)
        mutated_genotype = genotype.copy()

        for i in range(0, genotype_length):
            """if self.rnd.random() > self.gen_probability:
                # Old Bit flip
                if mutated_genotype[i] == 0 and n_clusters < max_clusters:
                    mutated_genotype[i] = 1
                    n_clusters += 1
                elif mutated_genotype[i] == 1 and n_clusters > 2:
                    mutated_genotype[i] = 0
                    n_clusters -= 1"""

            if mutated_genotype[i] == 0 and self.rnd.random() < self.gen_probability/4:
                mutated_genotype[i] = 1
            elif mutated_genotype[i] == 1 and self.rnd.random() < self.gen_probability:
                mutated_genotype[i] = 0

        return mutated_genotype
