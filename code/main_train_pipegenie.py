import pandas as pd
import os, sys, signal, subprocess
#from train_classifier_pipegenie import train_pipegenie

EXCLUDED = {"bank-marketing", "adult-census-income"}

SEEDS_30 = [0, 7, 42, 99, 123, 256, 512, 777, 1000, 2024,
            1, 8, 43, 100, 124, 257, 513, 778, 1001, 2025,
            2, 9, 44, 101, 125, 258, 514, 779, 1002, 2026]
SEEDS_10 = SEEDS_30[:10]

GENERATIONS = 20
POP_SIZE = 30
ELITE_SIZE = 5
N_JOBS = 1
SEED_TIMEOUT = 2 * 3600

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
            result_path = f"classifiers/{dataset_name}/pipegenie/results_seed{seed}.json"
            timeout_path = f"classifiers/{dataset_name}/pipegenie/timeout_seed{seed}.txt"
            if os.path.exists(result_path):
                print(f"  --- seed {seed} déjà fait, on saute ---")
                continue
            if os.path.exists(timeout_path):                                               
                print(f"  --- seed {seed} timeout avant, on saute ---")
                continue
             
            print(f"\n  --- seed {seed} ---")
            cmd = [sys.executable, "-u", "code/run_one_seed.py", dataset_name, str(seed),
                   str(GENERATIONS), str(POP_SIZE), str(ELITE_SIZE), str(N_JOBS)]
            p = subprocess.Popen(cmd, start_new_session=True)
            try:
                p.wait(timeout=SEED_TIMEOUT)

            except subprocess.TimeoutExpired:
                # kill the process and all its children (loky workers)
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                print(f"  [TIMEOUT] {dataset_name} seed={seed}: more than {SEED_TIMEOUT} sec, seed skipped", flush=True)

                # remember it, so we don't lose the same time at the next run
                os.makedirs(f"classifiers/{dataset_name}/pipegenie", exist_ok=True)
                with open(timeout_path, "w") as f:
                    f.write(f"timeout after {SEED_TIMEOUT} sec\n")

    print("\nAll PipeGenie runs complete.")
