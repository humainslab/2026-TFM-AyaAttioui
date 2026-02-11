# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

class ParentSelector:
    """
    A class to perform parent solution. Tournament size currently available.
    """
    def __init__(self, rnd, tournament_size=2):
        """
        Create the parent selector with some parameters
        :param rnd: Random number generator
        :param tournament_size: Tournament size, default value is 2
        """
        self.tournament_size = tournament_size
        self.rnd = rnd

    def tournament_selection(self, population):
        """
        Choose parents via tournament competition. The number of parents is equal to the population size
        :param population: Current population
        :return: List of parents
        """
        parents = list()
        pop_size = len(population)
        num_parents = pop_size
        for i in range(0, num_parents):
            candidates = list()
            for j in range(0, self.tournament_size):
                rnd_index = self.rnd.randint(0, pop_size-1)
                candidates.append(population[rnd_index])
            winner = self.compare_candidates(candidates)
            parents.append(winner)
        return parents

    def compare_candidates(self, candidates):
        """
        Compare candidate solutions to choose the tournament winner
        :param candidates: candidate solution compiting in the tournament
        :return: Best solutions according to the fitness function (maximize), None if solutions are not evaluated
        """
        best_index = -1
        best_fitness = -1.0
        for i in range(0, len(candidates)):
            candidate_fitness = candidates[i].get_fitness()
            if candidate_fitness > best_fitness:
                best_fitness = candidate_fitness
                best_index = i
        if best_index >= 0:
            best = candidates[best_index]
        else:
            print("[Selector] Candidate solutions do not have fitness value")
            best = None
        return best
