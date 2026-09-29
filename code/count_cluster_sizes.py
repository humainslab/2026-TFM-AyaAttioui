from train_classifier import load_results
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
import numpy as np
import os
from collections import Counter

datasets = [d.split('.')[0] for d in os.listdir('data') if d.endswith('.csv')]
classifiers = ["rf", "svm", "xgb"]

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

        # même seed que dans representatives.py
        ap = AffinityPropagation(random_state=42)
        labels = ap.fit_predict(X)

        cluster_sizes = Counter(labels)

        print(f"{dataset_name} | {classifier_name} | total points: {len(X)} | clusters: {len(cluster_sizes)} | sizes: {dict(cluster_sizes)}")