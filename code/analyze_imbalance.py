import pandas as pd
import json
import os

datasets = [d.split('.')[0] for d in os.listdir('data') if d.endswith('.csv')]
classifiers = ['rf', 'svm', 'xgb']

rows = []

for dataset_name in datasets:

    df = pd.read_csv(f'data/{dataset_name}.csv')
    if dataset_name == "usps_bin":
        y = df.iloc[:, 0]
    else:
        y = df.iloc[:, -1]
    y = y.dropna()
    counts = y.value_counts()
    if len(counts) < 2:
        continue
    imbalance_ratio = counts.max() / counts.min()

    for classifier_name in classifiers:

        base_path = f'classifiers/{dataset_name}/{classifier_name}'
        results_path = f'{base_path}/results.json'
        metrics_path = f'{base_path}/classification_metrics.csv'

        if not os.path.exists(results_path) or not os.path.exists(metrics_path):
            continue

        # Count FP/FN
        with open(results_path) as f:
            results = json.load(f)
        n_fp = len(results['false_positives_values'])
        n_fn = len(results['false_negatives_values'])

        
        metrics = pd.read_csv(metrics_path).iloc[0]


        if imbalance_ratio <= 1.5:
            imbalance_category = "balanced"
        elif imbalance_ratio <= 3.0:
            imbalance_category = "slight"
        elif imbalance_ratio <= 10.0:
            imbalance_category = "moderate"
        elif imbalance_ratio <= 100.0:
            imbalance_category = "strong"
        else:
            imbalance_category = "extreme"

        rows.append({
            'dataset': dataset_name,
            'classifier': classifier_name,
            'imbalance_ratio': round(imbalance_ratio, 2),
            'imbalance_category': imbalance_category,
            'n_instances': len(y),
            'n_fp': n_fp,
            'n_fn': n_fn,
            'n_errors': n_fp + n_fn,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
        })


summary = pd.DataFrame(rows)
summary = summary.sort_values('imbalance_ratio', ascending=False)

os.makedirs('results/analysis', exist_ok=True)
summary.to_csv('results/analysis/imbalance_fpfn_summary.csv', index=False)

print(summary.to_string(index=False))
print(f"\nResults saved ")