from train_classifier import load_results
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import silhouette_score,calinski_harabasz_score, davies_bouldin_score 
import numpy as np
import pandas as pd
import os
import random
import mealpy
from mealpy import FloatVar
from mealpy import FloatVar
from EvoCluster import EvoCluster
from EvoCluster import optimizers
import matplotlib.pyplot as plt
from scipy.stats import mode



#EXPERIMENT SETTINGS
pop_sizes    = [20, 50]
epochs_list  = [30, 50]
SEEDS = [0, 7, 42, 99, 123, 256, 512, 777, 1000, 2024]


# Load Data
datasets = [
    {"name": "diabetes",          "path": "data/diabetes.csv"},
    {"name": "wdbc",              "path": "data/wdbc.csv"},
    {"name": "column",            "path": "data/column.csv"},
    {"name": "blood-transfusion", "path": "data/blood-transfusion.csv"},
    {"name": "isolet_bin",          "path": "data/isolet_bin.csv"},
    {"name": "usps_bin",          "path": "data/usps_bin.csv"},
    {"name": "phoneme",          "path": "data/phoneme.csv"},
    {"name": "kc1",          "path": "data/kc1.csv"},
    {"name": "ozone-level-8hr",          "path": "data/ozone-level-8hr.csv"},
    {"name": "hill_valley",          "path": "data/hill_valley.csv"},
]
classifiers = ["rf", "svm", "xgb"]
results = []


#Main loop

for data in datasets:
    
    if not os.path.exists(data["path"]):
        print(f"\n[SKIP] {data['name']} — file not found: {data['path']}")
        continue


    for classifier_name in classifiers:        
        dataset_name = data["name"]
        print(f"\nDATASET: {dataset_name} , CLASSIFIER: {classifier_name}")
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name}  |  CLASSIFIER: {classifier_name}")
        print(f"{'='*60}")


        # LOAD FP, FN
        _, fp_values, fn_values, _ = load_results(dataset_name, classifier_name)

        if len(fp_values) == 0 and len(fn_values) == 0:
            continue

        if len(fp_values) == 0:
            X = fn_values
        elif len(fn_values) == 0:
            X = fp_values
        else:
            X = np.vstack([fp_values, fn_values])


        # Normalisation
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        print("FP:", len(fp_values), "FN:", len(fn_values))
        print("Total used for clustering:", len(X))

        print("Number of samples:", len(X))
        if len(X) < 3:
            print("Not enough data for clustering")
            continue 


        # Affinity propagation clustering
        for seed in SEEDS:
            ap = AffinityPropagation(random_state=seed)
            labels_ap = ap.fit_predict(X)
            n_clusters_ap = len(set(labels_ap))


            if n_clusters_ap > 1: 
                results.append({
                    "dataset": dataset_name,
                    "classifier": classifier_name,
                    "method": "AffinityPropagation",
                    "seed":              seed,
                    "pop_size":          None,
                    "iterations":        None,
                    "num_clusters": n_clusters_ap,
                    "silhouette": silhouette_score(X, labels_ap),
                    "calinski_harabasz": calinski_harabasz_score(X, labels_ap),
                    "davies_bouldin": davies_bouldin_score(X, labels_ap)
                })




        # Evolutionary clustering

        # MetaCluster algorithms

        cluster_counts = []
        for s in SEEDS:
            ap_tmp = AffinityPropagation(random_state=s)
            labels_tmp = ap_tmp.fit_predict(X)
            cluster_counts.append(len(set(labels_tmp)))
 
        n_clusters_meta = int(mode(cluster_counts).mode)
        

        if n_clusters_meta < 2:
            print("  AP found only 1 cluster , skipping meta-heuristics")
            continue
           
        if n_clusters_meta > 20:
            n_clusters_meta = 20


        # To measures how good a clustering is
        def fitness_function(centers):
            centers = centers.reshape((n_clusters_meta, X.shape[1]))
            dist = ((X[:, None, :] - centers) ** 2).sum(axis=2)
            labels = np.argmin(dist, axis=1)
            # Avoid invalid clustering
            if len(np.unique(labels)) < 2:
                return 9999.0
            return davies_bouldin_score(X, labels)

        lb_flat = np.full(n_clusters_meta * X.shape[1], np.min(X))
        ub_flat = np.full(n_clusters_meta * X.shape[1], np.max(X))
        
        for pop in pop_sizes:
            for ep in epochs_list:
                for seed in SEEDS:
                    np.random.seed(seed)

                    print(f"    [{data['name']}] pop={pop} ep={ep} seed={seed}")

                    algorithms = [
                        ("GWO", mealpy.swarm_based.GWO.OriginalGWO(epoch=ep, pop_size=pop)),
                        ("PSO", mealpy.swarm_based.PSO.OriginalPSO(epoch=ep, pop_size=pop)),
                        ("WOA", mealpy.swarm_based.WOA.OriginalWOA(epoch=ep, pop_size=pop)),
                    ]

                    for name, optimizer in algorithms:
                        print(f"Running {name}...")

                        problem = {
                            "obj_func": fitness_function,
                            "bounds": FloatVar(
                                lb=lb_flat.tolist(),
                                ub=ub_flat.tolist()
                            ),
                            "minmax": "min",
                            "seed": seed,
                        }


                        agent = optimizer.solve(problem)
                        centers = agent.solution.reshape((n_clusters_meta, X.shape[1]))
                        dist = ((X[:, None, :] - centers) ** 2).sum(axis=2)
                        labels = np.argmin(dist, axis=1)


                        if len(set(labels)) > 1:
                            results.append({
                                "dataset": dataset_name,
                                "classifier": classifier_name,
                                "method": f"Meta_{name}",
                                "seed": seed,
                                "pop_size": pop,
                                "iterations": ep,
                                "num_clusters": n_clusters_meta,
                                "silhouette": silhouette_score(X, labels),
                                "calinski_harabasz": calinski_harabasz_score(X, labels),
                                "davies_bouldin": davies_bouldin_score(X, labels)
                            })


        # EvoCluster Algorithms
        # PSO
        try:
            from EvoCluster import EvoCluster 
            print("EvoCluster loaded successfully")
            evocluster_ok = True
        except Exception as e:
            print("EvoCluster import failed:", e)
            evocluster_ok = False


        if evocluster_ok :
            for pop in pop_sizes:
                for ep in epochs_list:
                    for seed in SEEDS:
                        np.random.seed(seed)
                        print(f"\nEvoCluster → pop={pop}, iter={ep}, seed={seed}")
                        
                        try:

                            def objf(solution, points=None, k=None):
                                centers = solution.reshape((n_clusters_meta, X.shape[1]))
                                dist = ((X[:, None, :] - centers) ** 2).sum(axis=2)
                                labels = np.argmin(dist, axis=1)

                                if len(set(labels)) < 2:
                                    return 9999, labels   
                                score = davies_bouldin_score(X, labels)
                                return score, labels  


                            
                            dim = n_clusters_meta * X.shape[1]

                            result = optimizers.PSO(
                                objf=objf,
                                lb=lb_flat,
                                ub=ub_flat,
                                dim=dim,
                                PopSize=pop,
                                iters=ep,
                                k=n_clusters_meta,
                                points=X,
                                metric="euclidean"
                            )

                            best_solution = result.bestIndividual   
                            best_fitness  = result.convergence[-1]  

                            centers = best_solution.reshape((n_clusters_meta, X.shape[1]))
                            dist = ((X[:, None, :] - centers) ** 2).sum(axis=2)
                            labels_evopso = np.argmin(dist, axis=1)

                

                            if len(set(labels_evopso)) > 1:
                                results.append({
                                    "dataset": dataset_name,
                                    "classifier": classifier_name,
                                    "method": "EvoCluster_PSO",
                                    "seed": seed,
                                    "pop_size": pop,
                                    "iterations": ep,
                                    "num_clusters": len(set(labels_evopso)),
                                    "silhouette": silhouette_score(X, labels_evopso),
                                    "calinski_harabasz": calinski_harabasz_score(X, labels_evopso),
                                    "davies_bouldin": best_fitness
                                })

                        except Exception as e:
                            print("EvoCluster PSO failed:", e)



#Results
df = pd.DataFrame(results)
os.makedirs("results/clustering", exist_ok=True)

df.to_csv("results/clustering/comparison_results.csv", index=False)

group_cols = ["dataset", "classifier", "method", "pop_size", "iterations"]
df_avg = df.groupby(group_cols, dropna=False).agg(
    num_clusters_mean = ("num_clusters",      "mean"),
    silhouette_mean   = ("silhouette",        "mean"),
    silhouette_std    = ("silhouette",        "std"),
    ch_mean           = ("calinski_harabasz", "mean"),
    ch_std            = ("calinski_harabasz", "std"),
    db_mean           = ("davies_bouldin",    "mean"),
    db_std            = ("davies_bouldin",    "std"),
    n_seeds           = ("seed",             "count"),
).reset_index()

df_avg.to_csv("results/clustering/results_avg.csv")
print("Results saved")

