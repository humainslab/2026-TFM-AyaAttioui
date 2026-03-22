# A. Ramírez, C. García, J.R. Romero.
# HUMAINS lab, University of Córdoba
# 2026

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import json

def load_dataset(filename):
    """
    Load the dataset from the filename
    
    :param filename: Name of the file to load
    :return: Dataframe with the dataset
    """
    df = pd.read_csv(filename + '.csv')
    return df


def train(dataset_name, seed: int =123456, classifier_name: str = 'rf', normalized: bool=True, save: bool=True):
    """
    Train a classifier for a given dataset. The last column of the dataset is the target variable. 
    The function returns the trained classifier, the list of false positives and false negatives (feature values) and the columns of the dataset.
    
    :param dataset_name: Name of the dataset
    :param seed: Seed for the random state
    :param classifier_name: Classification algorithm to use (rf, svm, xgb)
    :param normalized: True if the data is normalized, False otherwise
    :param save: If true, the classifier and list of FP/FN are save in disk (folder: /classifiers/dataset_name/classifier_name)

    :return: classifier: Trained classifier
    :return: false_positives_values: List of false positives (feature values)
    :return: false_negatives_values: List of false negatives (feature values)
    :return: data.columns: Columns of the dataset

    """
    base_path = 'data/'
    dataset_path = base_path + dataset_name
    
    if os.path.exists(dataset_path + '.csv'):
        data = load_dataset(dataset_path)
    else:
        raise Exception('Dataset not found in folder "data".')
    # --- ADDED BY AYA ---
    # USPS dataset has label in first column (not last), so we adjust X/y split.
    if dataset_name == "usps_bin":
        y = data.iloc[:, 0]
        X = data.iloc[:, 1:]
    else:
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]


    if normalized:
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    

    # Automatically convert binary labels to 0/1 if needed
    unique_labels = sorted(y.unique())

    if len(unique_labels) == 2:
        if unique_labels != [0, 1]:
            label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
            y = y.replace(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    if classifier_name =='rf':
        classifier = RandomForestClassifier(n_estimators=100, random_state=seed, min_samples_split=2,
                                 min_samples_leaf=1, max_features='sqrt')
    elif classifier_name == 'svm':
        classifier = SVC(kernel='linear', random_state=seed, probability=True)

    elif classifier_name == 'xgb':
        classifier = XGBClassifier()
    else:
        raise Exception('The name of the classification algorithm is not valid')

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    false_positives = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_pred[i] != y_test.values[i]:
            false_positives.append(i)

    false_negatives = []
    for i in range(len(y_pred)):
        if y_pred[i] == 0 and y_pred[i] != y_test.values[i]:
            false_negatives.append(i)

    false_positives_values = []
    for i in false_positives:
        false_positives_values.append(list(X_test.values[i]))

    false_negatives_values = []
    for i in false_negatives:
        false_negatives_values.append(list(X_test.values[i]))

    false_positives = [int(i) for i in false_positives]
    false_negatives = [int(i) for i in false_negatives]

    if save:
        save_results(dataset_name, classifier_name, classifier, false_positives_values, false_negatives_values, data.columns)
        save_classification_metrics(dataset_name, classifier_name, y_test, y_pred)
        
    return classifier, false_positives_values, false_negatives_values, data.columns

def save_results(dataset_name, classifier_name, classifier, fp_values, fn_values, data_columns):
    """
    Save results of the classifier performance in disk (folder: /classifiers/dataset_name/classifier_name)
    
    :param dataset_name: Name of the dataset
    :param classifier_name: Classification algorithm trained (rf, svm, xgb)
    :param: fp_values: List of false positives (feature values)
    :param: fn_values: List of false negatives (feature values)
    :param data_columns: Columns of the dataset
    """
    path = "classifiers/" + dataset_name + "/" + classifier_name
    if not os.path.exists(path):
        os.makedirs(path)
    joblib.dump(classifier, path + "/" + classifier_name + ".joblib")

    with open(path + '/results.json', 'w') as f:
        json.dump({'false_positives_values': fp_values, 'false_negatives_values': fn_values, 'data_columns': list(data_columns)}, f)


def save_classification_metrics(dataset_name, classifier_name, y_test, y_pred):
    """
    Save classification metrics in CSV (folder: /classifiers/dataset_name/classifier_name)
    
    :param dataset_name: Name of the dataset
    :param classifier_name: Classification algorithm trained (rf, svm, xgb)
    :param: y_test: True labels
    :param: y_pred: Predicted labels
    """
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    csv_path = "classifiers/" + dataset_name + "/" + classifier_name + '/classification_metrics.csv'
    row = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
        
    }
    df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


def load_results(dataset_name, classifier_name):
    """
    Load results of the classifier performance in disk (folder: /classifiers/dataset_name/classifier_name)
    
    :param dataset_name: Name of the dataset
    :param classifier_name: Classification algorithm trained (rf, svm, xgb)
    :return: fp_values: List of false positives (feature values)
    :return: fn_values: List of false negatives (feature values)
    :return data_columns: Columns of the dataset
    """
    path = "classifiers/" + dataset_name + "/" + classifier_name
    if os.path.exists(path + "/" + classifier_name + ".joblib"):
        classifier = joblib.load(path + "/" + classifier_name + ".joblib")
        with open(path + '/results.json', 'r') as f:
            results = json.load(f)
        fp_values = results['false_positives_values']
        fn_values = results['false_negatives_values']
        data_columns = results['data_columns']
        return classifier, fp_values, fn_values, data_columns
    else:
        raise Exception('Classifier and results not found in folder "classifiers".')


def load_metrics_summary(base_dir: str = "classifiers") -> pd.DataFrame:
    """
    Read all `classification_metrics.csv` files under the given base directory
    and return a unified summary DataFrame.

    The expected directory layout is::

        base_dir/
            <dataset_name>/
                <classifier_name>/
                    classification_metrics.csv

    The returned DataFrame always contains ``dataset`` and ``classifier``
    columns plus whatever metrics appear in the CSV files (accuracy,
    precision, recall, f1, tn, fp, fn, tp, etc.).
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