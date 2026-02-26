# CluEx
Evolutionary Clustering for Explaining Local Prediction Errors

## Code organization

The scripts to run the different steps of the approach are:

1. main_train_classifiers.py: Train and test different classifiers (currently RF, SVM and XGB), for all datasets in the data folder, using the train_classifier.py functions. The list of FP/FN is extracted and classification metrics are computed. Results are saved in /classifiers (sorted by dataset) and a summary of classification metrics is saved in /results.

2. TO-DO (run the evolutionary algorithm)

3. TO-DO (run other clustering algorithms)

4. TO-DO (scripts for specific RQs/experiments)

## Datasets

This is the list of datasets used for experimentation. The link is the original source (UCI or OpenML repositories). The CSV files included in the repository might be preprocessed (e.g. remove id column or change format from ARFF to CSV). They have numerical columns only and two class values (0/1).

- blood transfusion: https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center
- column: https://archive.ics.uci.edu/dataset/212/vertebral+column
- diabetes: https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=diabetes&id=37
- egg-eye-state: https://archive.ics.uci.edu/dataset/264/eeg+eye+state
- hill_valley: https://www.openml.org/search?type=data&status=active&id=1479
- kc1: https://www.openml.org/search?type=data&status=active&id=1067
- kc2: https://www.openml.org/search?type=data&status=active&id=1063
- ozone-level-8h: https://archive.ics.uci.edu/dataset/172/ozone+level+detection
- pc1: https://www.openml.org/search?type=data&status=active&id=1068
- phoneme: https://www.openml.org/search?type=data&status=active&id=1489
- qsar-biodeg: https://www.openml.org/search?type=data&status=active&id=1494
- spambase: https://archive.ics.uci.edu/dataset/94/spambase
- wdbc: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The script drop_id_column.py can be used to open the dataset, drop the column by name, and save the resulting dataset.