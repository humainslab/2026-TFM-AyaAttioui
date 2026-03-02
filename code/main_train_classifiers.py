# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

import pandas as pd
import os
import train_classifier
from pathlib import Path


def summarize_classification_metrics(base_dir: str = "classifiers") -> pd.DataFrame:
    """
    Read all `classification_metrics.csv` files under the given base directory
    and return a unified summary DataFrame.

    The expected directory layout is::

        base_dir/
            <dataset_name>/
                <classifier_name>/
                    classification_metrics.csv

    The returned DataFrame contains the columns ``dataset`` and ``classifier``
    columns plus the classification metrics that appear in the CSV file (accuracy,
    precision, recall, f1, tn, fp, fn, tp).
    """
    records = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"base directory '{base_dir}' does not exist")

    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for classifier in os.listdir(dataset_path):
            clf_path = os.path.join(dataset_path, classifier)
            if not os.path.isdir(clf_path):
                continue
            csv_path = os.path.join(clf_path, "classification_metrics.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    rec = {"dataset": dataset, "classifier": classifier}
                    rec.update(row.to_dict())
                    records.append(rec)
    if records:
        return pd.DataFrame(records)
    else:
        return pd.DataFrame(columns=["dataset", "classifier"])


if __name__ == "__main__":

    # Run all datasets and classifiers
    random_state=123456
    data_path = Path('data')
    if data_path.exists():
        files = [f.name.split('.')[0] for f in data_path.iterdir() if f.is_file()]
    else:
        print(f"Directory '{data_path}' does not exist")
    
    classifiers = ['rf', 'svm', 'xgb']

    for dataset_name in files:
        for classifier_name in classifiers:
            print(f"Training {classifier_name} on {dataset_name} dataset...")
            classifier, fp_values, fn_values, data_columns = train_classifier.train(dataset_name, seed=random_state, classifier_name=classifier_name, normalized=True, save=True)
    
    # Create a summary data frame
    try:
        summary_df = summarize_classification_metrics()
        results_path = "./results/classification"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        summary_df.to_csv(os.path.join(results_path, "classification_metrics_summary.csv"), index=False)
    except Exception as e:
        print(f"Could not build metrics summary: {e}")