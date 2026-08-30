from train_classifier import load_results_raw
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
import gower
import pandas as pd
import os

SEEDS = [0, 7, 42, 99, 123, 256, 512, 777, 1000, 2024]

datasets = [
    {"name": "heart-disease"},
    {"name": "credit-approval"},
    {"name": "mammographic-mass"},
    {"name": "contraceptive-method-choice"},
    {"name": "german-credit"},
    {"name": "horse-colic"},
    {"name": "congressional-voting"},
    {"name": "cylinder-bands"},
]
classifiers = ["rf", "svm", "xgb"]
results = []

for data in datasets:
    dataset_name = data["name"]

    for classifier_name in classifiers:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name}  |  CLASSIFIER: {classifier_name}")
        print(f"{'='*60}")

        # LOAD raw (non-encoded) FP, FN
        try:
            fp_values, fn_values, raw_columns = load_results_raw(dataset_name, classifier_name)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        if len(fp_values) == 0 and len(fn_values) == 0:
            print("  No FP/FN, skipping")
            continue

        X_raw = pd.DataFrame(fp_values + fn_values, columns=raw_columns)

        print("FP:", len(fp_values), "FN:", len(fn_values))
        print("Total used for clustering:", len(X_raw))

        if len(X_raw) < 3:
            print("Not enough data for clustering")
            continue


        for seed in SEEDS:
            ap = AffinityPropagation(affinity='precomputed', random_state=seed)
            labels_ap = ap.fit_predict(-distance_matrix)
            n_clusters_ap = len(set(labels_ap))

            if n_clusters_ap > 1:
                sil = silhouette_score(distance_matrix, labels_ap, metric='precomputed')
                results.append({
                    "dataset": dataset_name,
                    "classifier": classifier_name,
                    "method": "AffinityPropagation_Gower",
                    "seed": seed,
                    "num_clusters": n_clusters_ap,
                    "silhouette": sil,
                })
                print(f"  seed={seed}: {n_clusters_ap} clusters, silhouette={sil:.4f}")
            else:
                print(f"  seed={seed}: only 1 cluster, skipping")




df = pd.DataFrame(results)
os.makedirs("results/clustering", exist_ok=True)

df.to_csv("results/clustering/comparison_results_gower.csv", index=False)

df_avg = df.groupby(["dataset", "classifier", "method"]).agg(
    num_clusters_mean=("num_clusters", "mean"),
    silhouette_mean=("silhouette", "mean"),
    silhouette_std=("silhouette", "std"),
    n_seeds=("seed", "count"),
).reset_index()

df_avg.to_csv("results/clustering/results_avg_gower.csv", index=False)

print("\nResults saved to results/clustering/comparison_results_gower.csv")