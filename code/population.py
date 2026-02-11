# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

class Population:
    """
    A class representing a population of solutions to be evolved, stored as a list
    """

    def __init__(self, solutions):
        """
        Create the population object
        :param solutions: Array of solutions to be added to the population
        """
        self.pop_size = len(solutions)
        self.population = list()
        self.fill_population(solutions)

    def fill_population(self, solutions):
        """
        Append solutions to the population
        :param solutions: Array of solutions to be added to the population list
        """
        for s in solutions:
            self.population.append(s)
        self.set_pop_size(len(self.population))

    def empty_population(self):
        """
        Remove all solutions from the population
        """
        self.population = list()
        self.set_pop_size(0)

    def set_pop_size(self, size):
        """
        Set the population size
        :param size: The new population size
        """
        self.pop_size = size

    def get_solutions(self):
        """
        Get the population members
        :return: The list of current solutions
        """
        return self.population

    def get_solution(self, index):
        """
        Get a solution at the given position
        :param index: Position of the solution in the population list
        :return: Solution at the given position if the index is a valid position, None otherwise
        """
        if 0 < index < self.pop_size:
            return self.population[index]
        else:
            return None

    def get_best_solution(self):
        """
        Get the solution with the best fitness
        :return: Solution with highest fitness in the population
        """
        max_fitness = -1
        max_index = -1
        for i in range(0, self.pop_size):
            fitness = self.population[i].get_fitness()
            if fitness > max_fitness:
                max_fitness = fitness
                max_index = i
        return self.population[max_index]

    def print_population(self):
        """
        Print the solutions comprising the population
        """
        for s in self.population:
            print(s.get_phenotype())
        print('--------------')

    def set_population(self, population):
        self.population = population
