# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

from sklearn.cluster import AffinityPropagation
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

class ClusterExplainer:
    """
    A class to generate clusters of error samples with different clustering methods
    """
    def __init__(self, error_samples):
        """
        Create the object with the samples to be analyzed
        :param error_samples: Misclassified samples (FP o FN)
        """
        self.error_samples = list(error_samples)

    def cluster_affinity(self, seed=0, damping=0.5, max_iter=200, conv_iter=15):
        """
        Group misclassified samples using the AffinityPropagation algorithm
        See https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation
        :param seed: Random seed
        :param damping: Damping parameter value
        :param max_iter: Maximum number of iterations
        :param conv_iter: Iterations until convergence
        :return: List of sample indexes in each cluster, list with the cluster "exemplars". If the algorithm
        does not converge, None is returned in both positions
        """
        alg = AffinityPropagation(random_state=seed, damping=damping, max_iter=max_iter, convergence_iter=conv_iter)
        alg.fit(self.error_samples)
        centroids = alg.cluster_centers_indices_
        num_clusters = len(centroids)
        if len(centroids) == 0:
            return None, None
        else:
            clusters = self.create_clusters_from_labels(alg.labels_, num_clusters)
            return clusters, centroids.tolist()

    def create_clusters_from_labels(self, labels, num_clusters):
        """
        Create a list of clusters, where each cluster is a list with the indexes of its samples
        :param labels: For each sample, the cluster to which is belongs
        :param num_clusters: Number of clusters (i.e., different label values)
        :return: A list with the samples distributed among clusters according to the labels
        """
        clusters = list()
        if num_clusters > 0 and len(labels) > 0:
            for i in range(0, num_clusters):
                clusters.append(list())
            for i in range(0, len(labels)):
                index = labels[i]
                clusters[index].append(i)
        return clusters
