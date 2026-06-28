from train_classifier import train
import xai_functions as xf
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')


SEED = 123456  

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
]

classifiers = ["rf", "svm", "xgb"]

# all 5 clustering methods
methods = ["AP", "GWO", "PSO", "WOA", "EvoCluster_PSO"]

os.makedirs("results/xai", exist_ok=True)

# MAIN LOOP
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

  
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # convert labels to 0/1 if needed
    unique_labels = sorted(y.unique())
    if len(unique_labels) == 2 and unique_labels != [0, 1]:
        label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
        y = y.replace(label_mapping)

    # same split as train_classifier.py
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

        # load representative points and their indices
        for method in methods:

            print(f"\n    METHOD: {method}")

            # load representative points and their indices
            if method == "AP":
                points_path  = f"results/representatives/ap_exemplar_points_{dataset_name}_{classifier_name}.npy"
                indices_path = f"results/representatives/ap_exemplar_indices_{dataset_name}_{classifier_name}.npy"
            else:
                points_path  = f"results/representatives/nearest_points_{dataset_name}_{classifier_name}_{method}.npy"
                indices_path = f"results/representatives/nearest_indices_{dataset_name}_{classifier_name}_{method}.npy"

            
            if not os.path.exists(points_path):
                print(f"    [SKIP] representatives not found: {points_path}")
                continue

            # load representative points and their indices
            rep_points  = np.load(points_path)
            rep_indices = np.load(indices_path).astype(int)

            print(f"    {len(rep_points)} representatives loaded")

           

            index_dict = {}

            for cluster_id, (rep_point, rep_idx) in enumerate(zip(rep_points, rep_indices)):

                # convert representative to DataFrame with correct column names
                rep_df = pd.DataFrame([rep_point], columns=X_train.columns)

                # find nearest row in X_test by euclidean distance
                distances  = np.sqrt(((X_test.values - rep_point) ** 2).sum(axis=1))
                nearest_idx = X_test.index[np.argmin(distances)]

                index_dict[f"cluster_{cluster_id}"] = [nearest_idx]


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

                
                save_path = f"results/xai/{dataset_name}_{classifier_name}_{method}"

                with open(save_path + "_metrics.pkl", "wb") as f:
                    pickle.dump(metrics_dict, f)

                with open(save_path + "_results.pkl", "wb") as f:
                    pickle.dump(results_dict, f)

                print(f"    Results saved → {save_path}")

            except Exception as e:
                print(f"    [ERROR] {dataset_name} {classifier_name} {method}: {e}")
                continue


print("\n" + "="*60)
print("All XAI results saved to results/xai/")
  