# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

import numpy as np
from lime import lime_tabular
from numpy import ndarray

def compute_xai_matrix(samples: list, feature_names: list, model) -> ndarray:
    """
    Compute the xai matrix for a dataset
    :param samples: List of samples to explain
    :param feature_names: Name of the features
    :param model: Model trained with the dataset
    :return: XAI matrix
    """
    # obtain lime explanation array for each sample
    lime_explanation_array, lime_explanations = get_lime_explanation_array(samples=samples, feature_names=feature_names, model=model)  # get the lime explanation array

    # compute the xai matrix
    xai_matrix = np.zeros((len(samples), len(samples)))  # create the xai matrix
    for i in range(0, len(samples)):  # for each row in the xai matrix
        # print(f'XAI matrix: {xai_matrix}')
        for j in range(0, len(samples)):  # for each column in the xai matrix
            distance_sum = 0  # initialize the distance sum
            if isinstance(lime_explanation_array[i], list):  # if the lime explanation array is a list
                for k in range(0, len(samples[i])):  # for each element in the lime explanation array
                    distance_sum += get_distance_between_points(lime_explanation_array[i][k], lime_explanation_array[j][k])  # compute the distance between the two points
            else:  # if the lime explanation array is not a list
                distance_sum = get_distance(lime_explanation_array, i, j)  # compute the distance between the two points
            xai_matrix[i][j] = distance_sum  # set the distance sum in the xai matrix


    max_distance = np.max(xai_matrix)  # get the maximum distance
    xai_matrix = xai_matrix / max_distance  # normalize the xai matrix
    xai_matrix = 1 - xai_matrix  # invert the xai matrix

    return xai_matrix, lime_explanations  # return the xai matrix and the lime explanations as html


def get_lime_explanation_array(samples: list, feature_names, model) -> list:
    """
          Get the lime explanation array for each sample in the dataset
          :param model:
          :param samples: List of samples to explain
          :param dataset: Dataset to explain
          :param labels: Labels of the dataset
          :param feature_names: Name of the features
          :param class_names: Name of the classes
          :param categorical_features: List of categorical features
          :param categorical_names: List of categorical names
          :param kernel_width: Kernel width
          :param verbose: Verbose
          :param mode: Mode
          :param random_state: Random state
          :return: List of lime explanation arrays
          """
    
    # Disable warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    lime_explanation_array = []  # initialize the lime explanation array
    lime_explanations = []  # initialize the lime explanations
    for sample in samples:  # for each sample in the dataset
        """
        The steps to explain a sample are:
        1. Create an explainer object (This takes some time)
        2. Explain the sample (we need a predict function that returns the probability of each class)
        3. Get the explanation as a map (in order to get the features that are important for the explanation)
        and as html (in order to visualize the explanation)
        4. Append the explanation to the lime_explanation_array (sorted by its key, so that the features are in the same order)
        """
        explainer = lime_tabular.LimeTabularExplainer(training_data=np.asarray(samples), feature_names=feature_names, mode='classification')
        exp = explainer.explain_instance(data_row=np.asarray(sample), predict_fn=model.predict_proba, num_features=len(feature_names))
        lime_explanations.append(exp.as_html())
        explanation_map = exp.as_map()
        xai_list = []
        for key in sorted(explanation_map[1]):
            xai_list.append(key[1])
        lime_explanation_array.append(xai_list)
    assert len(lime_explanation_array) == len(samples)  # check that the length of the lime explanation array is the same as the length of the dataset
    return lime_explanation_array, lime_explanations


def compute_distance_matrix(dataset: list, genotype_length: int):
    """
    Compute the distance matrix between the points in the genotype with the dataset
    :param genotype_length: Length of the genotype
    :param dataset: Array with the points in the dataset
    :return: Distance matrix
    """
    distance_matrix = np.zeros((genotype_length, genotype_length))
    for i in range(0, genotype_length):
        for j in range(0, genotype_length):
            distance_sum = 0
            if isinstance(dataset[i], list):  # This is the most common case, but in case the dataset is not a list, we need to check it
                for k in range(0, len(dataset[i])):
                    distance_sum += get_distance_between_points(dataset[i][k], dataset[j][k])
            else:
                distance_sum = get_distance(dataset, i, j)
            distance_matrix[i][j] = distance_sum
    return distance_matrix


def get_distance(dataset: list, i: int, j: int):
    """
    Calculate the distance between two points in the dataset
    :param dataset: The dataset used to cluster
    :param i: Index of the first point
    :param j: Index of the second point
    :return: Distance between the two points
    """

    print('Dataset: ' + str(dataset))
    print('I: ' + str(i))
    print('J: ' + str(j))
    print('Dataset i: ' + str(dataset[i]))
    print('Dataset j: ' + str(dataset[j]))

    return np.linalg.norm(dataset[i] - dataset[j])


def get_distance_between_points(i: int, j: int):
    """Return the distance between two points
    Args:
        i (int): the first point
        j (int): the second point

    Returns:
        float: the distance between the two points i and j
    """
    return np.linalg.norm(i - j)


def calculate_xai_score(labels: list, xai_matrix: list, medoids: list) -> float:
    """ Compute xai score for a solution
    :param xai_matrix: matrix with the xai scores for each point in the genotype
    :param solution: solution object
    :return: xai score
    """
    xai_score = list()
    
    get_genotype = lambda x: [1 if i in x else 0 for i in range(len(labels))]  # This lambda function tries to emulate 
                                                                               # the get_genotype Solution class function                                                                             
        
    for i in range(len(labels)):  # For each row in the distance matrix
        if get_genotype(medoids)[i] == 1:  # If the sample is a medoid
            cluster_score = list()
            for j in range(len(labels)):  # For each column in the distance matrix
                if labels[i] == labels[j]:  # If the samples are in the same cluster
                    cluster_score.append(xai_matrix[i][j])  # Add the xai score to the total xai score
            xai_score.append(np.mean(cluster_score))  # Get the mean of the xai scores

    xai_score = np.mean(xai_score)  # Get the mean of the xai scores

    return xai_score


def xai_score(samples: list, feature_names: list, xai_matrix: list, silhouette_score: float, labels: list, medoids: list, model) -> float:
    """Calculate the xai score for a non Evolutive Clustering solution. The xai score is calculated as the silhouette score
    multiplied by the xai score.

    Args:
        samples (list): Array with the points in the dataset
        feature_names (list): Name of the features in the dataset
        silhouette_score (float): Silhouette score of the solution
        labels (list): Labels of the solution
        medoids (list): Medoids of the solution
        model (trained model): Model trained with the dataset

    Returns:
        float: XAI score of the solution (silhouette score * xai score)
    """
    xai_score = calculate_xai_score(labels=labels, xai_matrix=xai_matrix, medoids=medoids)

    silhouette_score = (silhouette_score + 1) / 2

    return xai_score * silhouette_score, xai_score
