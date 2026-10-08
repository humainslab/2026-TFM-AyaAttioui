import sys
from train_classifier_pipegenie import train_pipegenie

dataset_name = sys.argv[1]
seed = int(sys.argv[2])
generations = int(sys.argv[3])
pop_size = int(sys.argv[4])
elite_size = int(sys.argv[5])
n_jobs = int(sys.argv[6])
 
try:
    train_pipegenie(
        dataset_name,
        seed=seed,
        generations=generations,
        pop_size=pop_size,
        elite_size=elite_size,
        n_jobs=n_jobs,
    )
except Exception as e:
    print(f"  [ERROR] {dataset_name} seed={seed}: {e}")