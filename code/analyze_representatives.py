import pickle, os, glob, pandas as pd

folder = "results/xai"

algorithms = ["AP", "GWO", "WOA","EvoCluster_PSO", "PSO",]
classifiers = ["rf", "xgb", "svm"]
data = []
files = glob.glob(folder + "/*_metrics.pkl")

for path in files:
    filename = os.path.basename(path)
    name = filename.replace("_metrics.pkl", "")

    #fidn algorithm
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


#Table
df = pd.DataFrame(data)
print("total number of rows:", len(df))

os.makedirs("results/analysis", exist_ok=True)
df.to_csv("results/analysis/rq3_raw_per_cluster.csv", index=False)
value_cols = [col for col in df.columns if col not in ["dataset", "classifier", "algorithm", "cluster"]]

df_avg = df.groupby(["dataset", "classifier", "algorithm"])[value_cols].mean().reset_index()
df_avg.to_csv("results/analysis/rq3_representative_methods_summary.csv", index=False)

#final comparison
comparison = df_avg.groupby("algorithm")[value_cols].agg(["mean", "std"])
print(comparison)

comparison.to_csv("results/analysis/rq3_algorithm_comparison.csv")
print("Files saved ")
