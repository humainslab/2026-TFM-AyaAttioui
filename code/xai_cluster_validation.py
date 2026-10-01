from train_classifier import load_results
import xai_functions as xf
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings('ignore')
 
SEED = 123456        # same as xai_experiment.py
CLUSTER_SEED = 42    # same as representatives.py
N_EXTRA = 3           # extra points per cluster besides the representative
rng = np.random.default_rng(42)
 
# same dataset list as xai_experiment.py
datasets = [
    {"name": "diabetes",          "path": "data/diabetes.csv"},
    {"name": "wdbc",              "path": "data/wdbc.csv"},
    {"name": "column",            "path": "data/column.csv"},
    {"name": "blood-transfusion", "path": "data/blood-transfusion.csv"},
    {"name": "isolet_bin",        "path": "data/isolet_bin.csv"},
    {"name": "usps_bin",          "path": "data/usps_bin.csv"},
    {"name": "phoneme",           "path": "data/phoneme.csv"},
    {"name": "kc1",               "path": "data/kc1.csv"},
    {"name": "ozone-level-8hr",   "path": "data/ozone-level-8hr.csv"},
    {"name": "hill_valley",       "path": "data/hill_valley.csv"},
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
methods = ["AP", "GWO", "PSO", "WOA", "EvoCluster_PSO"]
 
os.makedirs("results/xai_cluster_validation", exist_ok=True)
 
for data in datasets:
 
    if not os.path.exists(data["path"]):
        print(f"\n[SKIP] {data['name']} — file not found")
        continue
 
    dataset_name = data["name"]
 
    df = pd.read_csv(data["path"])
 
    if dataset_name == "usps_bin":
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
 
    valid_idx = y.notna()
    y = y[valid_idx]
    X = X[valid_idx]
 
    num_cols = X.select_dtypes(include=['number']).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    cat_cols = X.select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
 
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
 
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
 
    unique_labels = sorted(y.unique())
    if len(unique_labels) == 2 and unique_labels != [0, 1]:
        label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
        y = y.replace(label_mapping)
    elif len(unique_labels) > 2:
        majority_class = y.value_counts().idxmax()
        y = y.apply(lambda v: 0 if v == majority_class else 1)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED
    )
 
    num_features = X_train.shape[1]
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}  &  features: {num_features}")
 
    for classifier_name in classifiers:
        print(f"\n  CLASSIFIER: {classifier_name}")
 
        model_path = f"classifiers/{dataset_name}/{classifier_name}/{classifier_name}.joblib"
        if not os.path.exists(model_path):
            print(f"  [SKIP] model not found: {model_path}")
            continue
 
        model = joblib.load(model_path)
 
        # reload FP/FN, same as representatives.py
        _, fp_values, fn_values, _ = load_results(dataset_name, classifier_name)
 
        if len(fp_values) == 0 and len(fn_values) == 0:
            continue
        if len(fp_values) == 0:
            X_raw = np.array(fn_values)
        elif len(fn_values) == 0:
            X_raw = np.array(fp_values)
        else:
            X_raw = np.vstack([fp_values, fn_values])
 
        if len(X_raw) < 3:
            continue
 
        cluster_scaler = StandardScaler()
        X_scaled = cluster_scaler.fit_transform(X_raw)
 
        for method in methods:
 
            print(f"\n    METHOD: {method}")
 
            if method == "AP":
                points_path  = f"results/representatives/ap_exemplar_points_{dataset_name}_{classifier_name}.npy"
                indices_path = f"results/representatives/ap_exemplar_indices_{dataset_name}_{classifier_name}.npy"
            else:
                points_path  = f"results/representatives/nearest_points_{dataset_name}_{classifier_name}_{method}.npy"
                indices_path = f"results/representatives/nearest_indices_{dataset_name}_{classifier_name}_{method}.npy"
 
            if not os.path.exists(points_path):
                print(f"    [SKIP] representatives not found: {points_path}")
                continue
 
            rep_points  = np.load(points_path)
            rep_indices = np.load(indices_path).astype(int)
 
            print(f"    {len(rep_points)} representatives loaded")
 
            # cluster label for every FP/FN point
            if method == "AP":
                ap = AffinityPropagation(random_state=CLUSTER_SEED)
                labels = ap.fit_predict(X_scaled)
            else:
                # reuse saved centroids instead of re-running the optimization
                centroids_path = f"results/representatives/centroids_{dataset_name}_{classifier_name}_{method}.npy"
                if not os.path.exists(centroids_path):
                    print(f"    [SKIP] centroids not found: {centroids_path}")
                    continue
                centroids = np.load(centroids_path)
                dist = ((X_scaled[:, None, :] - centroids) ** 2).sum(axis=2)
                labels = np.argmin(dist, axis=1)
 
            index_dict = {}
 
            for cluster_id, rep_idx in enumerate(rep_indices):
 
                same_cluster = np.where(labels == cluster_id)[0]
                same_cluster = [i for i in same_cluster if i != rep_idx]
 
                if len(same_cluster) == 0:
                    continue  # singleton cluster, nothing to add
 
                n_pick = min(N_EXTRA, len(same_cluster))
                extra_raw_idx = rng.choice(same_cluster, size=n_pick, replace=False)
 
                raw_points_this_cluster = [rep_points[cluster_id]] + [X_raw[i] for i in extra_raw_idx]
 
                nearest_test_indices = []
                for raw_point in raw_points_this_cluster:
                    distances = np.sqrt(((X_test.values - raw_point) ** 2).sum(axis=1))
                    nearest_idx = X_test.index[np.argmin(distances)]
                    nearest_test_indices.append(nearest_idx)
 
                # remove duplicates (two raw points can map to same X_test row)
                nearest_test_indices = list(dict.fromkeys(nearest_test_indices))
 
                if len(nearest_test_indices) < 2:
                    continue  # still just 1 distinct point, nothing to compare
 
                index_dict[f"cluster_{cluster_id}"] = nearest_test_indices
 
            if not index_dict:
                print("    [SKIP] no cluster had extra points to add")
                continue
 
            models_dict = {classifier_name: model}
 
            try:
                metrics_dict, results_dict = xf.calculate_metrics_for_indices(
                    models_dict  = models_dict,
                    indexes_dict = index_dict,
                    X_test       = X_test,
                    X_train      = X_train,
                    y_train      = y_train,
                    num_features = num_features
                )
 
                save_path = f"results/xai_cluster_validation/{dataset_name}_{classifier_name}_{method}"
 
                with open(save_path + "_metrics.pkl", "wb") as f:
                    pickle.dump(metrics_dict, f)
 
                with open(save_path + "_results.pkl", "wb") as f:
                    pickle.dump(results_dict, f)
 
                print(f"    Results saved → {save_path}")
 
            except Exception as e:
                print(f"    [ERROR] {dataset_name} {classifier_name} {method}: {e}")
                continue
 
print("\n" + "="*60)
print(" results saved to results/xai_cluster_validation/")