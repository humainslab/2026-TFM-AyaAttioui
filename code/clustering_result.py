# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

class ClusterResult:
    """
    A class to store clustering results and compute performance metrics
    """
    def __init__(self, samples, feature_names, clusters=None, centroids=None):
        """
        Create the object
        :param samples: Feature values of the samples used as input for clustering
        :param feature_names: Feature names of the data frame under analysis
        :param clusters: Indexes of the samples belonging to each cluster
        :param centroids: Indexed of the representative samples in each cluster
        """
        self.samples = samples
        self.feature_names = feature_names
        self.clusters = clusters
        self.centroids = centroids
        self.labels = None

    def number_clusters(self):
        """
        Get the number of clusters
        :return: Number of clusters
        """
        return len(self.clusters)

    def set_clusters_indexes(self, clusters):
        """
        Set the index of the samples belonging to each cluster
        :param clusters: list of indexes for each cluster
        """
        self.clusters = clusters

    def set_centroids_indexes(self, centroids):
        """
        Set the index of the samples selected as centroids in each cluster
        :param centroids: index for each cluster
        """
        self.centroids = centroids

    def get_cluster_indexes(self):
        """
        Get the list of sample indexes for all clusters
        :return: Current cluster division
        """
        return self.clusters

    def get_samples_indexes(self, cluster_index):
        """
        Get the index of the samples in a given cluster
        :param cluster_index: Index of the cluster
        :return: Indexes of the samples belonging to the cluster
        """
        return self.clusters[cluster_index]

    def get_centroids_indexes(self):
        """
        Get the indexes of the centroid samples
        :return: List of centroids
        """
        return self.centroids

    def get_sample(self, index):
        """
        Get a string representation of a given sample
        :param index: Index of the sample in the data frame
        :return: String representation with the feature names and values
        """
        sample = self.samples[index]
        sample_str = ''
        for i in range(0, len(self.feature_names)-1): #-1 to exclude the label column (assuming it is the last one)
            sample_str = sample_str + ' ' + self.feature_names[i] + '=' + str(sample[i])
        return sample_str

    def get_centroids_samples(self):
        """
        Get a string representation of all centroids
        :return: String representation with the feature names and values
        """
        list_centroids = list()
        for i in range(0, len(self.centroids)):
            sample = self.get_sample(self.centroids[i])
            list_centroids.append(sample)
        return list_centroids

    def get_cluster_samples(self, index):
        """
        Get a string representation of all samples in a given cluster
        :param index: Index of the cluster
        :return: String representation with the feature names and values
        """
        list_cluster = list()
        cluster = self.clusters[index]
        cluster_size = len(cluster)
        for i in range(0, cluster_size):
            sample = self.get_sample(cluster[i])
            list_cluster.append(sample)
        return list_cluster

    def get_clusters_samples(self):
        """
        Get a string representation of all samples in all clusters
        :return: String representation with the feature names and values
        """
        list_clusters = list()
        for i in range(0, len(self.clusters)):
            list_cl = self.get_cluster_samples(i)
            list_clusters.append(list_cl)
        return list_clusters

    def save_results(self, filename):
        """
        Save the cluster partition and centroids information in a file
        :param filename: Name of the file
        :return: True if the file is created, False otherwise
        """
        try:
            size = self.number_clusters()
            f = open(filename, "w")
            f.write('Number of clusters: {num}\n'.format(num=size))
            f.write('Clusters (instance indexes):\n')
            for i in range(0, size):
                f.write('\tcluster: {idx}, instances: {c}\n'.format(idx=i, c=self.clusters[i]))
            f.write('Prototype instances:\n')
            for i in range(0, size):
                f.write('\tcluster: {idx}, instance: {center}, values: {sample}\n'.format(
                    idx=i, center=self.centroids[i], sample=self.get_sample(self.centroids[i])))
            f.close()
            return True
        except IOError:
            return False

    def generate_label_samples_from_clusters(self):
        """
        Assign the cluster id as label of each sample in the dataframe. This allows computing performance metrics.
        """
        size = len(self.samples)
        num_clusters = self.number_clusters()
        self.labels = np.full(size, -1)
        for i in range(0, num_clusters):
            for j in range(0, len(self.clusters[i])):
                index = self.clusters[i][j]
                self.labels[index] = i

    def compute_silhouette_coefficient(self):
        """
        Compute the silhouette coefficient for each cluster
        :return: Silhouette coefficient values
        """
        if self.labels is None:
            self.generate_label_samples_from_clusters()
        try:
            result = silhouette_score(X=self.samples, labels=self.labels, metric='euclidean')
        except Exception: # Prevent cases when the metric cannot be computed(e.g. all clusters have one sample only)
            result = np.nan
        return result

    def compute_calinski_score(self):
        """
        Compute the calinski score for each cluster
        :return: Calinski score values
        """
        if self.labels is None:
            self.generate_label_samples_from_clusters()
        try:
            result = calinski_harabasz_score(X=self.samples, labels=self.labels)
        except Exception: # Prevent cases when the metric cannot be computed(e.g. all clusters have one sample only)
            result = np.nan
        return result

    def compute_davies_index(self):
        """
        Compute the Davies score for each cluster
        :return: Davies score values
        """
        if self.labels is None:
            self.generate_label_samples_from_clusters()
        try:
            result = davies_bouldin_score(X=self.samples, labels=self.labels)
        except Exception: # Prevent cases when the metric cannot be computed(e.g. all clusters have one sample only)
            result = np.nan
        return result
