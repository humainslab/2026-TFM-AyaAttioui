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

    X = data.drop(data.columns[-1], axis=1)
    if normalized:
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    y = data[data.columns[-1]]

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