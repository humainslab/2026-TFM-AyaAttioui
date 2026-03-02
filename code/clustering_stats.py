# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

import pandas as pd
import numpy as np
from clustering_result import ClusterResult

class ClusterStatistics:
    """
    A class to store statistics of several runs of clustering methods
    """
    def __init__(self):
        """
        Create the object
        """
        self.df = pd.DataFrame(columns=['dataset', 'classifier', 'fp-fn', 'silhouette', 'calinski', 'davies', 'num_clusters'])

    def add_row_values(self, dataset=None, classifier=None, fp_fn=None, silhouette=None, calinski=None, davies=None, num_clusters=None):
        """
        Add one value to each column
        :param dataset: Dataset name
        :param classifier: Classifier name
        :param fp_fn: False positive or false negative
        :param silhouette: Silhouette metric value
        :param calinski: Calinski metric value
        :param davies: Davies metric value
        :param num_clusters: Number of clusters
        """
        new_row = {
            'dataset': dataset,
            'classifier': classifier,
            'fp-fn': fp_fn,
            'silhouette': silhouette,
            'calinski': calinski,
            'davies': davies,
            'num_clusters': num_clusters
        }
        self.df.loc[len(self.df)] = new_row

    def save_statistics(self, filename):
        """
        Save the data frame with statistics in a CSV file
        :param filename: Name of the file
        """
        self.df.to_csv(filename, index=False)

    def save_clustering_results(self, filename, dataset_name, classifier_name, fp_or_fn, df_samples, data_columns, clusters, exemplars):
        """
        Save the clustering results in a text file and add the statistics to the data frame
        :param filename: Name of the file to save the results
        :param dataset_name: Name of the dataset
        :param classifier_name: Name of the classifier
        :param fp_or_fn: False positive or false negative
        :param df_samples: Data frame with the samples used for clustering
        :param data_columns: List of feature names
        :param clusters: List of clusters
        :param exemplars: List of exemplars (centroids) for each cluster
        """
        if clusters is None or exemplars is None:
            self.add_row_values(dataset_name, classifier_name, fp_or_fn, np.nan, np.nan, np.nan, np.nan)
        else:
            fp_cluster_results = ClusterResult(df_samples, data_columns, clusters, exemplars)
            fp_cluster_results.save_results(filename)
            num_clusters = fp_cluster_results.number_clusters()
            if num_clusters > 1:
                sil_coef = fp_cluster_results.compute_silhouette_coefficient()
                cal_sco = fp_cluster_results.compute_calinski_score()
                dav_ind = fp_cluster_results.compute_davies_index()
                self.add_row_values(dataset_name, classifier_name, fp_or_fn, sil_coef, cal_sco, dav_ind, num_clusters)
            else:
                self.add_row_values(dataset_name, classifier_name, fp_or_fn, np.nan, np.nan, np.nan, num_clusters)
