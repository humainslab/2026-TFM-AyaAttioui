# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

import os
import pandas as pd
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

    grid_results = []

    # Run all datasets and classifiers
    files = ["usps_bin"]
    for dataset_name in files:
        for classifier_name in classifiers:
            
            # Load the results of the classifier for the given dataset
            classifier, fp_values, fn_values, data_columns = train_classifier.load_results(dataset_name, classifier_name)
           
            damping_values = [0.5, 0.7, 0.9]
            max_iter_values = [50, 200]
            conv_iter_values = [10, 15] 

            # Clustering false positives with affinity propagation
            if fp_values is None or len(fp_values) == 0:
                print(f"Dataset: {dataset_name}, Classifier: {classifier_name}, No false positives to cluster")
                continue 
            else:
                explainer = ClusterExplainer(fp_values)

            # We test different values of the parameters of the affinity propagation algorithm to see how they affect the clustering results. The parameters are: damping, max_iter and conv_iter.

            # Initialize best tracking
            best_num_clusters = -1
            best_params = None
            best_clusters_fp = None
            best_exemplars_fp = None

            for damping in damping_values:
                for max_iter in max_iter_values:
                    for conv_iter in conv_iter_values:

                        clusters_fp, exemplars_fp = explainer.cluster_affinity(
                            seed=random_state,
                            damping=damping,
                            max_iter=max_iter,
                            conv_iter=conv_iter
                        )

                        num_clusters = len(clusters_fp)

                        # Keep best configuration
                        if num_clusters > best_num_clusters:
                            best_num_clusters = num_clusters
                            best_params = (damping, max_iter, conv_iter)
                            best_clusters_fp = clusters_fp
                            best_exemplars_fp = exemplars_fp

                        print(f"damping={damping}, max_iter={max_iter}, conv_iter={conv_iter}")
                        
                        grid_results.append({
                            "dataset": dataset_name,
                            "classifier": classifier_name,
                            "type": "fp",
                            "damping": damping,
                            "max_iter": max_iter,
                            "conv_iter": conv_iter,
                            "num_clusters": num_clusters
                        })
            #best results for false positives           
            print(f"BEST FP PARAMS: {best_params} → clusters: {best_num_clusters}")

            #Best clustering for interpretation
            filename = str(results_path) + '/' + dataset_name + '_' + classifier_name + '_fp.txt'
            stats.save_clustering_results(filename, dataset_name, classifier_name, 'fp', fp_values, data_columns, best_clusters_fp, best_exemplars_fp)
            print(f"Dataset: {dataset_name}, Classifier: {classifier_name}, FP Clusters: {len(best_clusters_fp)}, FP Exemplars: {best_exemplars_fp}")




            # Clustering false negatives with affinity propagation
            if fn_values is None or len(fn_values) == 0:
                print(f"Dataset: {dataset_name}, Classifier: {classifier_name}, No false negatives to cluster")
                continue 
            else:
                explainer = ClusterExplainer(fn_values)
            # Initialize best tracking
            best_num_clusters = -1
            best_params = None
            best_clusters_fn = None
            best_exemplars_fn = None

            for damping in damping_values:
                for max_iter in max_iter_values:
                    for conv_iter in conv_iter_values:

                        clusters_fn, exemplars_fn = explainer.cluster_affinity(
                            seed=random_state,
                            damping=damping,
                            max_iter=max_iter,
                            conv_iter=conv_iter
                        )

                        num_clusters = len(clusters_fn)

                        
                        if num_clusters > best_num_clusters:
                            best_num_clusters = num_clusters
                            best_params = (damping, max_iter, conv_iter)
                            best_clusters_fn = clusters_fn
                            best_exemplars_fn = exemplars_fn

                        print(f"damping={damping}, max_iter={max_iter}, conv_iter={conv_iter}")


                        grid_results.append({
                            "dataset": dataset_name,
                            "classifier": classifier_name,
                            "type": "fn",
                            "damping": damping,
                            "max_iter": max_iter,
                            "conv_iter": conv_iter,
                            "num_clusters": len(clusters_fn)
                        })
            #best results for false negatives
            print(f"BEST FN PARAMS: {best_params} → clusters: {best_num_clusters}")

            #Best clustering for interpretation
            print(f"Dataset: {dataset_name}, Classifier: {classifier_name}, FN Clusters: {len(best_clusters_fn)}, FN Exemplars: {best_exemplars_fn}")
            filename = str(results_path) + '/' + dataset_name + '_' + classifier_name + '_fn.txt'
            stats.save_clustering_results(filename, dataset_name, classifier_name, 'fn', fn_values, data_columns, best_clusters_fn, best_exemplars_fn)

    # This method save the accumulated statistics in a CSV file
    stats.save_statistics(str(results_path) + '/' + 'affinity_propagation_clustering_stats.csv')

    df_grid = pd.DataFrame(grid_results)
    df_grid.to_csv("results/clustering/grid_search_results.csv", index=False)



    