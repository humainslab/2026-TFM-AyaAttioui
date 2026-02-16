import train_classifier
from pathlib import Path

if __name__ == "__main__":
    
    # Example code to run one dataset and one classsifier
    #dataset_name = "blood-transfusion"
    #classifier_name = "rf"
    #classifier, fp_values, fn_values, data_columns = train_classifier.train(dataset_name, 
    #                                                                        seed=123456, 
    #                                                                        classifier_name='xgb', 
    #                                                                        normalized=True, 
    #                                                                        save=True)
    
    # Example code to load results of one dataset and one classsifier
    #classifier, fp_values, fn_values, data_columns = train_classifier.load_results(dataset_name, classifier_name='rf')/"

    #print("False positives values: ", fp_values)
    #print("False negatives values: ", fn_values)
    #print("Data columns: ", data_columns)

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
            train_classifier.train(dataset_name, seed=random_state, classifier_name=classifier_name, normalized=True, save=True)
