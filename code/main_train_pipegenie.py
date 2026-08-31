import pandas as pd
import os
from train_classifier_pipegenie import train_pipegenie

EXCLUDED = {"bank-marketing", "adult-census-income"}

SEEDS_30 = [0, 7, 42, 99, 123, 256, 512, 777, 1000, 2024,
            1, 8, 43, 100, 124, 257, 513, 778, 1001, 2025,
            2, 9, 44, 101, 125, 258, 514, 779, 1002, 2026]
SEEDS_10 = SEEDS_30[:10]

GENERATIONS = 20
POP_SIZE = 30
ELITE_SIZE = 5
N_JOBS = 4


def get_imbalance_category(dataset_name, data_path="data"):
    """Compute imbalance category the same way as analyze_imbalance_fpfn.py"""
    df = pd.read_csv(f"{data_path}/{dataset_name}.csv")
    if dataset_name == "usps_bin":
        y = df.iloc[:, 0]
    else:
        y = df.iloc[:, -1]
    y = y.dropna()
    counts = y.value_counts()
    if len(counts) < 2:
        return "balanced"
    ratio = counts.max() / counts.min()
    if ratio <= 1.5:
        return "balanced"
    elif ratio <= 3.0:
        return "slight"
    else:
        return "moderate_or_worse"


if __name__ == "__main__":
    datasets = [d.split('.')[0] for d in os.listdir('data') if d.endswith('.csv')]
    datasets = [d for d in datasets if d not in EXCLUDED]

    for dataset_name in datasets:
        category = get_imbalance_category(dataset_name)
        seeds = SEEDS_30 if category in ("balanced", "slight") else SEEDS_10

        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name} | imbalance: {category} | {len(seeds)} seeds")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n  --- seed {seed} ---")
            try:
                train_pipegenie(
                    dataset_name,
                    seed=seed,
                    generations=GENERATIONS,
                    pop_size=POP_SIZE,
                    elite_size=ELITE_SIZE,
                    n_jobs=N_JOBS,
                )
            except Exception as e:
                print(f"  [ERROR] {dataset_name} seed={seed}: {e}")
                continue

    print("\nAll PipeGenie runs complete.")
