# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

from random import Random
from initialization import PopulationCreator
from repair import RepairOperator
from selection import ParentSelector
from crossover import Crossover
from mutation import Mutation
from evaluation import Evaluator
from utils import compute_xai_matrix, compute_distance_matrix
import time
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('always', UserWarning)

"""
    Evolutionary algorithm for optimal selection of cluster prototypes

    Parameters
    ----------
    max_clusters : int, default=5
        Maximum number of clusters to generate
    
    pop_size : int, default=30
        Size of the population
    
    max_iter : int, default=15
        Maximum number of iterations
    
    xai_object : list, default=None
        List with the xai matrix and the lime explanations
    
    show_times : bool, default=False
        If True, show the time spent in each step of the algorithm
        
    verbose : bool, default=False
        If True, show the best fitness in each iteration
    
    random_state : int, default=0
        Seed for the random number generator
    
    min_clusters : int, default=2
        Minimum number of clusters to generate
    
    tournament_size: int, default=2,     
        Tournament size in selector

    cross_prob: float, default = 0.9
        Crossover probability

    mut_prob: float, default = 0.3
        Mutation probability

    Attributes
    ----------
    __dataset : list
        List with the dataset
    
    __num_samples : int
        Number of samples in the dataset
    
    __num_features : int
        Number of features in the dataset
    
    __distance_matrix : list
        List with the distance matrix
        
    __isTrained : bool
        If True, the model is trained
    
    __sbest_phenotype : list
        List with the phenotype of the best solution
    
    __total_time : float
        Time spent in the algorithm
    
    __history : dict
        Dictionary with the history of the algorithm
    
    __best_solution : Solution
        Best solution found
        
    __lime_explanations : list
        List with the lime explanations
        
    __best_fitness_each_epoch : list
        List with the best fitness in each iteration
    
    __mean_fitness_each_epoch : list
        List with the mean fitness in each iteration
    
    __epochs : list
        List with the epochs
    
    Methods
    -------
    run(dataset, feature_names, model)
        Run the evolutionary algorithm to cluster the dataset
    
    summary()
        Show a summary of the algorithm
    
    getTotalTime()
        Get the total time spent in the algorithm
    
    getHistory(filename)
        Get the history of the algorithm
    
    getExplanations(custom_label)
        Get the explanations of the best solution
    
    getBestSolution()
        Get the best solution
    
    getBestFitness()
        Get the best fitness
        
    getBestSolutionClusters()
        Get the number of clusters of the best solution
"""
class ECELPE:
    
    def __init__(self, max_clusters: int = 5, pop_size: int = 30, max_iter: int = 15, xai_object: list = None,
                 show_times: bool = False, verbose: bool = False, random_state: int = 0, min_clusters = 2, 
                 tournament_size = 2, cross_prob: float = 0.9, mut_prob: float = 0.3):
        
        # Public attributes
        self.max_clusters = max_clusters
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.xai_object = xai_object
        self.show_times = show_times
        self.verbose = verbose
        self.random_state = random_state
        self.min_clusters = min_clusters
        self.tournament_size = tournament_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob

        # Private attributes
        self.__isTrained = False
        pass

    def run(self, dataset: list, feature_names: list, model):
        """
        Run the evolutionary algorithm to cluster the instances with wrong predictions
        """
        self.__dataset = dataset
        self.__num_samples = len(dataset)
        self.__num_features = len(feature_names)

        rnd = Random()
        rnd.seed(self.random_state)
        evaluator = Evaluator()
        selector = ParentSelector(rnd, self.tournament_size)
        crossover_op = Crossover(rnd, self.cross_prob)
        mutation_op = Mutation(rnd, genotype_length=self.__num_samples * 2, mut_probability=self.mut_prob)  
        repair_op = RepairOperator(rnd, self.min_clusters)

        total_time_start = time.time()
        
        if self.max_clusters >= self.__num_samples:
            # Enable warnings
            print(f"WARNING!! The number of cluster established ({self.max_clusters}) is greater than the number of instances" +
                f" provided ({self.__num_samples}). Shilhouette constraints the number of labels from 2 to labels - 1 (inclusive)," +
                f" max_clusters set to {self.__num_samples - 1}")
            self.max_clusters = self.__num_samples - 1
        
        self.__distance_matrix = compute_distance_matrix(dataset, self.__num_samples)
        if self.xai_object is None:
            xai_matrix, lime_explanations = compute_xai_matrix(dataset, feature_names, model) 
        else:
            xai_matrix, lime_explanations = self.xai_object
        creator = PopulationCreator(self.pop_size, self.__num_samples, self.max_clusters, self.__distance_matrix,
                                    rnd, self.min_clusters) 
        
        population = creator.create_population() 
        best_fitness = -1
        best_solution = population.get_solutions()[0]
        
        history = {'individuals': [], 'fitness': []}
        ev_ind = 1
        
        for iteration in range(self.max_iter):
            
            time_start = time.time()
            fitness_arr = list()
            
            for s in population.get_solutions():
                s.labels_ = s.get_point_indexes(dataset)
                n_clusters = len(set(s.labels_)) - (1 if -1 in s.labels_ else 0)
                if n_clusters != 1:
                    fitness = evaluator.evaluate_solution(s, self.__distance_matrix, xai_matrix)
                else:
                    fitness = 0  # if there is only one cluster, the xai score is -1 (the worst value)
                fitness_arr.append(fitness)
                s.set_fitness(fitness)
                if best_fitness < fitness:  # if the fitness of the solution is better than the best fitness, update the best
                    best_fitness = fitness
                    best_solution = s
                history['individuals'].append(ev_ind)
                history['fitness'].append(fitness)
                ev_ind += 1
                
            mean_fitness = np.mean(fitness) 

            time_stop = time.time()
            time_eval = time_stop - time_start

            time_start = time.time()
            parents = selector.tournament_selection(population.get_solutions())

            time_stop = time.time()
            time_select = time_stop - time_start

            time_start = time.time()
            offspring = crossover_op.recombine_parents(parents) 

            time_stop = time.time()
            time_crossover = time_stop - time_start

            time_start = time.time()
            mutants = mutation_op.mutate_solutions(offspring)
            time_stop = time.time()
            time_mutation = time_stop - time_start

            repaired_population = repair_op.repair_population(mutants)
            population.set_population(repaired_population)
            
            if iteration % 5 == 0 and self.verbose:
                print('\tGeneration: %i - Best fitness: %.4f - Mean fitness: %.4f' % (iteration, best_fitness, mean_fitness))
            

            ##### time summary #####
            if self.show_times:
                print('\t[TIME] Evaluation: ', time_eval)
                print('\t[TIME] Selection: ', time_select)
                print('\t[TIME] Crossover: ', time_crossover)
                print('\t[TIME] Mutation: ', time_mutation)
                print('\t[TIME] Total generation: ', time_eval + time_select + time_crossover + time_mutation)

        # Save important variables
        self.__sbest_phenotype = best_solution.solution_to_eval(dataset)  # get the phenotype of the best solution
        self.__total_time = time.time() - total_time_start
        self.__history = history
        self.__best_solution = best_solution
        self.__lime_explanations = lime_explanations
        self.__isTrained = True

    def summary(self):
        if self.__isTrained:
            print('\n')
            print('[SUMMARY]')
            print('\tNumber of samples: %i' % (self.__num_samples))
            print('\tNumber of features: %i' % (self.__num_features))
            print('\tTime spent: %i seconds' % (self.__total_time))
            print('\n[BEST SOLUTION INFO]')
            print('\tFitness: %.4f' % (self.__best_solution.get_fitness()))
            print('\tNumber of clusters: %i' % (self.__best_solution.get_number_clusters()))
            print('\tLabels: ', self.__best_solution.get_point_indexes(self.__dataset))
            print('___________________________________________________________')
        else:
            raise Exception('The algorithm has not been executed')

    def getTotalTime(self):
        if self.__isTrained:
            return self.__total_time
        else:
            raise Exception('The algorithm has not been executed')

    def getHistory(self, filename: str = None):
        if self.__isTrained:
            if filename is None:
                filename = datetime.now().strftime("%d_%m_%Y__%H-%M-%S")
            
            df = pd.DataFrame(self.__history)
            df.to_csv(filename + '.csv', index=False)
        else:
            raise Exception('The algorithm has not been executed')

    def getExplanations(self, custom_label: str = None):
        if self.__isTrained:
            n_explanation = 1  # number of explanations to show
            if custom_label is None:
                custom_label = datetime.now().strftime("%d_%m_%Y__%H-%M-%S")  # get the current date and time

            for i in range(len(self.__best_solution.get_genotype())):  # for each point in the best solution
                if self.__best_solution.get_genotype()[i] == 1:  # if the point is a medoid
                    with open(f'explanations/explanation{custom_label}#{n_explanation}.html', 'w', encoding='utf-8') as f:# create the file
                        f.write(self.__lime_explanations[i])  # write the explanation html in the file
                    n_explanation += 1  # update the number of explanations
        else:
            raise Exception('The algorithm has not been executed')
    
    def getBestSolution(self):
        if self.__isTrained:
            return self.__best_solution
        else:
            raise Exception('The algorithm has not been executed')
    
    def getBestFitness(self):
        if self.__isTrained:
            return self.__best_solution.get_fitness()
        else:
            raise Exception('The algorithm has not been executed')
    
    def getBestSolutionClusters(self):
        if self.__isTrained:
            return self.__best_solution.get_number_clusters()
        else:
            raise Exception('The algorithm has not been executed')
    
    def getBestSolutionPhenotype(self):
        if self.__sbest_phenotype:
            return self.__sbest_phenotype
        else:
            raise Exception('The algorithm has not been executed')