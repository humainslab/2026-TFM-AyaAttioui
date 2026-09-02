from train_classifier import load_results
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import silhouette_score,calinski_harabasz_score, davies_bouldin_score 
import numpy as np
import pandas as pd
import os
import random
import mealpy
import optuna
from mealpy import FloatVar
from mealpy import FloatVar
from EvoCluster import EvoCluster
from EvoCluster import optimizers
import matplotlib.pyplot as plt
from scipy.stats import mode



#EXPERIMENT SETTINGS
N_TRIALS = 30
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
    {"name": "heart-disease",               "path": "data/heart-disease.csv"},
    {"name": "credit-approval",             "path": "data/credit-approval.csv"},
    {"name": "mammographic-mass",           "path": "data/mammographic-mass.csv"},
    #{"name": "bank-marketing",              "path": "data/bank-marketing.csv"},
    {"name": "contraceptive-method-choice", "path": "data/contraceptive-method-choice.csv"},
    {"name": "german-credit",               "path": "data/german-credit.csv"},
    #{"name": "adult-census-income",         "path": "data/adult-census-income.csv"},
    {"name": "horse-colic",                 "path": "data/horse-colic.csv"},
    {"name": "congressional-voting",        "path": "data/congressional-voting.csv"},
    {"name": "cylinder-bands",              "path": "data/cylinder-bands.csv"},
]
classifiers = ["rf", "svm", "xgb"]
results = []


def get_optimizer(algo_name, epoch, pop_size):
    if algo_name == "GWO":
        return mealpy.swarm_based.GWO.OriginalGWO(epoch=epoch, pop_size=pop_size)
    elif algo_name == "PSO":
        return mealpy.swarm_based.PSO.OriginalPSO(epoch=epoch, pop_size=pop_size)
    elif algo_name == "WOA":
        return mealpy.swarm_based.WOA.OriginalWOA(epoch=epoch, pop_size=pop_size)
    elif algo_name == "GA":
        return mealpy.evolutionary_based.GA.BaseGA(epoch=epoch, pop_size=pop_size)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


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
        cluster_counts = []
        for seed in SEEDS:
            ap = AffinityPropagation(random_state=seed)
            labels_ap = ap.fit_predict(X)
            n_clusters_ap = len(set(labels_ap))
            cluster_counts.append(n_clusters_ap)

        n_clusters_meta = int(mode(cluster_counts).mode)


        if n_clusters_meta < 2:
            print("  AP found only 1 cluster, skipping automatic clustering")
            continue
        if n_clusters_meta > 20:
            n_clusters_meta = 20

        def fitness_function(centers):
            centers = centers.reshape((n_clusters_meta, X.shape[1]))
            dist = ((X[:, None, :] - centers) ** 2).sum(axis=2)
            labels = np.argmin(dist, axis=1)
            if len(np.unique(labels)) < 2:
                return 9999.0
            return davies_bouldin_score(X, labels)

        lb_flat = np.full(n_clusters_meta * X.shape[1], np.min(X))
        ub_flat = np.full(n_clusters_meta * X.shape[1], np.max(X))



        def objective(trial):
            algo_name = trial.suggest_categorical("algo", ["GWO", "PSO", "WOA", "GA"])
            pop_size = trial.suggest_int("pop_size", 10, 100)
            epoch = trial.suggest_int("epoch", 10, 100)

            optimizer = get_optimizer(algo_name, epoch, pop_size)

            problem = {
                "obj_func": fitness_function,
                "bounds": FloatVar(lb=lb_flat.tolist(), ub=ub_flat.tolist()),
                "minmax": "min",
                "seed": trial.number,
            }

            try:
                agent = optimizer.solve(problem)
                centers = agent.solution.reshape((n_clusters_meta, X.shape[1]))
                dist = ((X[:, None, :] - centers) ** 2).sum(axis=2)
                labels = np.argmin(dist, axis=1)

                if len(set(labels)) < 2:
                    return -1.0

                sil = silhouette_score(X, labels)
                ch = calinski_harabasz_score(X, labels)
                db = davies_bouldin_score(X, labels)

                trial.set_user_attr("algo", algo_name)
                trial.set_user_attr("pop_size", pop_size)
                trial.set_user_attr("epoch", epoch)
                trial.set_user_attr("num_clusters", len(set(labels)))
                trial.set_user_attr("calinski_harabasz", ch)
                trial.set_user_attr("davies_bouldin", db)

                return sil
            except Exception as e:
                print(f"    [ERROR] trial failed: {e}")
                return -1.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS)

        # Save every trial as a result row
        for trial in study.trials:
            if trial.value is None or trial.value == -1.0:
                continue
            results.append({
                "dataset": dataset_name,
                "classifier": classifier_name,
                "method": "Optuna_Auto",
                "trial": trial.number,
                "algo": trial.user_attrs.get("algo"),
                "pop_size": trial.user_attrs.get("pop_size"),
                "iterations": trial.user_attrs.get("epoch"),
                "num_clusters": trial.user_attrs.get("num_clusters"),
                "silhouette": trial.value,
                "calinski_harabasz": trial.user_attrs.get("calinski_harabasz"),
                "davies_bouldin": trial.user_attrs.get("davies_bouldin"),
            })

        print(f"  Best trial: algo={study.best_trial.user_attrs.get('algo')}, "
              f"pop_size={study.best_trial.user_attrs.get('pop_size')}, "
              f"epoch={study.best_trial.user_attrs.get('epoch')}, "
              f"silhouette={study.best_value:.4f}")




#Results
df = pd.DataFrame(results)
os.makedirs("results/clustering", exist_ok=True)

df.to_csv("results/clustering/comparison_results_optuna.csv", index=False)

group_cols = ["dataset", "classifier", "algo"]
df_avg = df.groupby(group_cols, dropna=False).agg(
    num_clusters_mean = ("num_clusters",      "mean"),
    silhouette_mean   = ("silhouette",        "mean"),
    silhouette_std    = ("silhouette",        "std"),
    ch_mean           = ("calinski_harabasz", "mean"),
    ch_std            = ("calinski_harabasz", "std"),
    db_mean           = ("davies_bouldin",    "mean"),
    db_std            = ("davies_bouldin",    "std"),
    n_trials          = ("trial", "count"),
).reset_index()


df_avg.to_csv("results/clustering/results_avg_optuna.csv", index=False)
print("\nResults saved")

