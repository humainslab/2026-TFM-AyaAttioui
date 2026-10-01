import pickle, os, glob, pandas as pd, numpy as np
from train_classifier import load_results
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from collections import Counter
 
folder = "results/xai"
 
algorithms = ["AP", "GWO", "WOA", "EvoCluster_PSO", "PSO"]
classifiers = ["rf", "xgb", "svm"]
data = []
files = glob.glob(folder + "/*_metrics.pkl")
 
for path in files:
    filename = os.path.basename(path)
    name = filename.replace("_metrics.pkl", "")
 
    #find algorithm
    algorithm = None
    for a in algorithms:
        if name.endswith("_" + a):
            algorithm = a
            name = name[:-(len(a) + 1)]
            break
 
    if algorithm is None:
        print("algorithm not found for:", filename)
        continue
 
    #find classifier
    classifier = None
    for c in classifiers:
        if name.endswith("_" + c):
            classifier = c
            dataset = name[:-(len(c) + 1)]
            break
 
    if classifier is None:
        print("classifier not found for:", filename)
        continue
 
    with open(path, "rb") as f:
        metrics = pickle.load(f)
    if classifier not in metrics:
        continue
    clusters = metrics[classifier]
 
    for cluster_name in clusters:
        c_data = clusters[cluster_name]
 
        row = {
                "dataset": dataset,
                "classifier": classifier,
                "algorithm": algorithm,
                "cluster": cluster_name,
                "topk_BD_Shap": c_data.get("Mean_topk_BD_Shap"),
                "topk_BD_Lime": c_data.get("Mean_topk_BD_Lime"),
                "topk_Lime_Shap": c_data.get("Mean_topk_Lime_Shap"),
                "rank_BD_Shap": c_data.get("Mean_rank_BD_Shap"),
                "rank_BD_Lime": c_data.get("Mean_rank_BD_Lime"),
                "rank_Lime_Shap": c_data.get("Mean_rank_Lime_Shap"),
                "sign_BD_Shap": c_data.get("Mean_sign_BD_Shap"),
                "sign_BD_Lime": c_data.get("Mean_sign_BD_Lime"),
                "sign_Lime_Shap": c_data.get("Mean_sign_Lime_Shap"),
                "rank_sign_BD_Shap": c_data.get("Mean_rank_sign_BD_Shap"),
                "rank_sign_BD_Lime": c_data.get("Mean_rank_sign_BD_Lime"),
                "rank_sign_Lime_Shap": c_data.get("Mean_rank_sign_Lime_Shap"),
        }
        data.append(row)
 
df = pd.DataFrame(data)
print("total number of rows:", len(df))
 

 
datasets = [d.split('.')[0] for d in os.listdir('data') if d.endswith('.csv')]
size_lookup = {}
 
for dataset_name in datasets:
    for classifier_name in classifiers:
        _, fp_values, fn_values, _ = load_results(dataset_name, classifier_name)
 
        if len(fp_values) == 0 and len(fn_values) == 0:
            continue
        if len(fp_values) == 0:
            X = fn_values
        elif len(fn_values) == 0:
            X = fp_values
        else:
            X = np.vstack([fp_values, fn_values])
 
        if len(X) < 3:
            continue
 
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
 
        ap = AffinityPropagation(random_state=42)
        labels = ap.fit_predict(X)
 
        sizes = Counter(labels)
        for label, size in sizes.items():
            size_lookup[(dataset_name, classifier_name, int(label))] = size
 

print("\nsample cluster values in df:", df["cluster"].unique()[:10])
print("sample keys in size_lookup:", list(size_lookup.keys())[:10])
 
def get_size(row):
    candidates = [row["cluster"]]
    try:
        candidates.append(int(row["cluster"]))
    except (ValueError, TypeError):
        pass
    try:
        s = str(row["cluster"])
        if "_" in s:
            candidates.append(int(s.rsplit("_", 1)[-1]))
    except (ValueError, TypeError):
        pass
 
    for c in candidates:
        key = (row["dataset"], row["classifier"], c)
        if key in size_lookup:
            return size_lookup[key]
    return np.nan
 
df["cluster_size"] = df.apply(get_size, axis=1)
matched = df["cluster_size"].notna().sum()
print(f"\nmatched {matched} / {len(df)} rows ({100*matched/len(df):.1f}%)")
 
os.makedirs("results/analysis", exist_ok=True)
 
if matched / len(df) < 0.9:
    print("\nWARNING: matching rate looks low, don't trust the filtered tables below.")
    print("paste this whole output so we can fix the matching")
    df.to_csv("results/analysis/rq3_raw_per_cluster_with_sizes_DEBUG.csv", index=False)
else:
    df.to_csv("results/analysis/rq3_raw_per_cluster_with_sizes.csv", index=False)
 
    value_cols = [c for c in df.columns if c not in
                  ["dataset", "classifier", "algorithm", "cluster", "cluster_size"]]
 

    for min_size in [1, 2, 3, 5]:
        filtered = df[df["cluster_size"] >= min_size]
        print(f"\n--- min cluster size >= {min_size}: {len(filtered)} rows (dropped {len(df)-len(filtered)}) ---")
 
        if filtered.empty:
            print("nothing left at this threshold")
            continue
 
        df_avg = filtered.groupby(["dataset", "classifier", "algorithm"])[value_cols].mean().reset_index()
        comparison = df_avg.groupby("algorithm")[value_cols].agg(["mean", "std"])
        print(comparison)
 
        comparison.to_csv(f"results/analysis/rq3_algorithm_comparison_min{min_size}.csv")
 
print("done")