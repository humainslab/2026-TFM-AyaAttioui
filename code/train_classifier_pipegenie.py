import pandas as pd
import os
import json
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pipegenie.classification import PipegenieClassifier

def load_dataset(filename):
    df = pd.read_csv(filename + '.csv')
    return df

def train_pipegenie(dataset_name, seed: int = 123456, grammar: str = "code/grammar_rf_svm_xgb.xml",
                     generations: int = 10, pop_size: int = 50, elite_size: int = 5,
                     n_jobs: int = 4, save: bool = True):
    """
    Train an AutoML (PipeGenie) model for a given dataset, using the same
    preprocessing logic as train_classifier.py, so results are directly
    comparable to the manual classifiers (rf, svm, xgb).
    """
    base_path = 'data/'
    dataset_path = base_path + dataset_name

    if os.path.exists(dataset_path + '.csv'):
        data = load_dataset(dataset_path)
    else:
        raise Exception('Dataset not found in folder "data".')

    if dataset_name == "usps_bin":
        y = data.iloc[:, 0]
        X = data.iloc[:, 1:]
    else:
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]

    # Drop rows with missing target
    valid_idx = y.notna()
    y = y[valid_idx].reset_index(drop=True)
    X = X[valid_idx].reset_index(drop=True)

    # Handle missing values
    num_cols = X.select_dtypes(include=['number']).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    cat_cols = X.select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encode categorical columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # Normalize (same as manual pipeline)
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Binarize target
    unique_labels = sorted(y.unique())
    if len(unique_labels) == 2:
        if unique_labels != [0, 1]:
            label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
            y = y.replace(label_mapping)
    elif len(unique_labels) > 2:
        majority_class = y.value_counts().idxmax()
        y = y.apply(lambda v: 0 if v == majority_class else 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    outdir = f"classifiers/{dataset_name}/pipegenie/pipegenie_run_seed{seed}"

    model = PipegenieClassifier(
        grammar=grammar,
        generations=generations,
        pop_size=pop_size,
        elite_size=elite_size,
        n_jobs=n_jobs,
        seed=seed,
        outdir=outdir,
    )

    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)

    false_positives = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_pred[i] != y_test.values[i]:
            false_positives.append(i)

    false_negatives = []
    for i in range(len(y_pred)):
        if y_pred[i] == 0 and y_pred[i] != y_test.values[i]:
            false_negatives.append(i)

    false_positives_values = [list(X_test.values[i]) for i in false_positives]
    false_negatives_values = [list(X_test.values[i]) for i in false_negatives]

    if save:
        path = f"classifiers/{dataset_name}/pipegenie"
        os.makedirs(path, exist_ok=True)

        with open(path + f'/results_seed{seed}.json', 'w') as f:
            json.dump({
                'false_positives_values': false_positives_values,
                'false_negatives_values': false_negatives_values,
                'data_columns': list(X.columns)
            }, f)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics_row = {
            'seed': seed, 'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
        metrics_path = path + '/classification_metrics.csv'
        if os.path.exists(metrics_path):
            df_metrics = pd.read_csv(metrics_path)
            df_metrics = pd.concat([df_metrics, pd.DataFrame([metrics_row])], ignore_index=True)
        else:
            df_metrics = pd.DataFrame([metrics_row])
        df_metrics.to_csv(metrics_path, index=False)

    return model, false_positives_values, false_negatives_values, X.columns


if __name__ == "__main__":
    train_pipegenie("heart-disease", seed=42, generations=5, pop_size=20, elite_size=3, n_jobs=2)
    print("Test run complete.")