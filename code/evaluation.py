# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

import numpy as np
from sklearn.metrics import silhouette_score
from solution import Solution

class Evaluator:
    """
    A class to implement evaluation operations (fitness function)
    """
    def __init__(self):
    
    def calculate_silhouette_sklearn(labels: list, distance_matrix: list) -> float:
        """ Compute silhouette score for a solution
        :param distance_matrix:
        :param labels: array indicating which cluster each gen belongs to
        :return: sklearn silhouette score
        """
        silhouette = silhouette_score(X=distance_matrix, labels=labels)
        return silhouette


    def calculate_xai_score(solution: Solution, xai_matrix: list) -> float:
        """ Compute xai score for a solution
        :param xai_matrix: matrix with the xai scores for each point in the genotype
        :param solution: solution object
        :return: xai score
        """
        xai_score = list()
    
        for i in range(len(solution.labels_)):  # For each row in the distance matrix
            if solution.get_genotype()[i] == 1:  # If the sample is a medoid
                cluster_score = list()
                for j in range(len(solution.labels_)):  # For each column in the distance matrix
                    if solution.labels_[i] == solution.labels_[j]:  # If the samples are in the same cluster
                        cluster_score.append(xai_matrix[i][j])  # Add the xai score to the total xai score
                xai_score.append(np.mean(cluster_score))  # Get the mean of the xai scores

        xai_score = np.mean(xai_score)

        return xai_score


    def evaluate_solution(solution: Solution, distance_matrix, xai_matrix) -> float:
        """
        Evaluate a solution based on the silhouette score and the xai score
        :param solution: solution object
        :param distance_matrix: distance matrix between the points in the genotype
        :param xai_matrix: matrix with the xai scores for each point in the genotype
        :return: weiighted average of silhouette score and xai score
        """
        silhouette = calculate_silhouette_sklearn(labels=solution.labels_,
                                              distance_matrix=distance_matrix)

        solution.silhouette_score = silhouette
    
        silhouette = (silhouette + 1) / 2  

        xai_score = calculate_xai_score(solution=solution, xai_matrix=xai_matrix)
    
        solution.xai_space_score = xai_score

        return silhouette * xai_score
