# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

import os
import train_classifier
from pathlib import Path
from clustering_errors import ClusterExplainer
from clustering_stats import ClusterStatistics

if __name__ == "__main__":
    
    random_state=123456
    data_path = Path('data')
    results_path = Path('results/clustering/affinity_propagation')
    if data_path.exists():
        files = [f.name.split('.')[0] for f in data_path.iterdir() if f.is_file()]
    else:
        print(f"Directory '{data_path}' does not exist")
    
    if not results_path.exists():
        os.makedirs(results_path)

    classifiers = ['rf', 'svm', 'xgb']

    stats = ClusterStatistics()

    # Run all datasets and classifiers
    for dataset_name in files:
        for classifier_name in classifiers:
            
            # Load the results of the classifier for the given dataset
            classifier, fp_values, fn_values, data_columns = train_classifier.load_results(dataset_name, classifier_name)
           
            # Clustering false positives with affinity propagation
            if fp_values is None or len(fp_values) == 0:
                print(f"Dataset: {dataset_name}, Classifier: {classifier_name}, No false positives to cluster")
            else:
                explainer = ClusterExplainer(fp_values)
                clusters_fp, exemplars_fp = explainer.cluster_affinity(seed=random_state, damping=0.5, max_iter=200, conv_iter=15)
                filename = str(results_path) + '/' + dataset_name + '_' + classifier_name + '_fp.txt'
                stats.save_clustering_results(filename, dataset_name, classifier_name, 'fp', fp_values, data_columns, clusters_fp, exemplars_fp)
                print(f"Dataset: {dataset_name}, Classifier: {classifier_name}, FP Clusters: {len(clusters_fp)}, FP Exemplars: {exemplars_fp}")
    
            # Clustering false negatives with affinity propagation
            if fn_values is None or len(fn_values) == 0:
                print(f"Dataset: {dataset_name}, Classifier: {classifier_name}, No false negatives to cluster")
            else:
                explainer = ClusterExplainer(fn_values)
                clusters_fn, exemplars_fn = explainer.cluster_affinity(seed=random_state, damping=0.5, max_iter=200, conv_iter=15)
                print(f"Dataset: {dataset_name}, Classifier: {classifier_name}, FN Clusters: {len(clusters_fn)}, FN Exemplars: {exemplars_fn}")
                filename = str(results_path) + '/' + dataset_name + '_' + classifier_name + '_fn.txt'
                stats.save_clustering_results(filename, dataset_name, classifier_name, 'fn', fn_values, data_columns, clusters_fn, exemplars_fn)

    # This method save the accumulated statistics in a CSV file
    stats.save_statistics(str(results_path) + '/' + 'affinity_propagation_clustering_stats.csv')