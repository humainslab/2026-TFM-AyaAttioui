from train_classifier import load_results
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import davies_bouldin_score
import numpy as np
import os
import mealpy
from mealpy import FloatVar
from scipy.stats import mode

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================
POP  = 50
ITER = 50
SEED = 42

# ============================================================
# Load Datasets and classifiers
# ============================================================
datasets = [
    {"name": "diabetes",          "path": "data/diabetes.csv"},
    {"name": "wdbc",              "path": "data/wdbc.csv"},
    {"name": "column",            "path": "data/column.csv"},
    {"name": "blood-transfusion", "path": "data/blood-transfusion.csv"},
    {"name": "isolet_bin",        "path": "data/isolet_bin.csv"},
    {"name": "usps_bin",          "path": "data/usps_bin.csv"},
    {"name": "phoneme",           "path": "data/phoneme.csv"},
    {"name": "kc1",               "path": "data/kc1.csv"},
    {"name": "ozone-level-8hr",   "path": "data/ozone-level-8hr.csv"},
    {"name": "hill_valley",       "path": "data/hill_valley.csv"},
]
classifiers = ["rf", "svm", "xgb"]

os.makedirs("results/representatives", exist_ok=True)


# ============================================================
# Main loop
# ============================================================
for data in datasets:

    if not os.path.exists(data["path"]):
        print(f"\n[SKIP] {data['name']} — file not found: {data['path']}")
        continue

    for classifier_name in classifiers:
        dataset_name = data["name"]
        print(f"\n{'='*50}")
        print(f"DATASET: {dataset_name}  |  CLASSIFIER: {classifier_name}")
        print(f"{'='*50}")

        # === LOAD FP / FN ===
        _, fp_values, fn_values, _ = load_results(dataset_name, classifier_name)

        if len(fp_values) == 0 and len(fn_values) == 0:
            continue

        # FIX 1 — keep X_raw before normalization
        if len(fp_values) == 0:
            X_raw = np.array(fn_values)
        elif len(fn_values) == 0:
            X_raw = np.array(fp_values)
        else:
            X_raw = np.vstack([fp_values, fn_values])

        # Normalisation
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        print("FP:", len(fp_values), "FN:", len(fn_values))
        print("Total used for clustering:", len(X))

        if len(X) < 3:
            print("Not enough data for clustering")
            continue


        # ════════════════════════════════════════════════════
        # AFFINITY PROPAGATION
        # ════════════════════════════════════════════════════

        # FIX 2 — no loop, just one seed
        ap = AffinityPropagation(random_state=SEED)
        labels_ap = ap.fit_predict(X)
        n_clusters_ap = len(set(labels_ap))

        if n_clusters_ap > 1:
            exemplar_indices = ap.cluster_centers_indices_
            exemplar_points  = X_raw[exemplar_indices]  # real original data points

            # Save exemplar indices and real data points
            np.save(f"results/representatives/ap_exemplar_indices_{dataset_name}_{classifier_name}.npy", exemplar_indices)
            np.save(f"results/representatives/ap_exemplar_points_{dataset_name}_{classifier_name}.npy",  exemplar_points)

            print(f"  AP exemplars saved — {len(exemplar_indices)} representatives")
        else:
            print("  AP found only 1 cluster, skipping")


        # ════════════════════════════════════════════════════
        # COMPUTE K for metaheuristics
        # ════════════════════════════════════════════════════

        # FIX 3 — initialize cluster_counts before the loop
        cluster_counts = []
        for s in [0, 7, 42, 99, 123, 256, 512, 777, 1000, 2024]:
            ap_tmp = AffinityPropagation(random_state=s)
            labels_tmp = ap_tmp.fit_predict(X)
            cluster_counts.append(len(set(labels_tmp)))

        n_clusters_meta = int(mode(cluster_counts).mode)

        # FIX 4 — add missing k checks
        if n_clusters_meta < 2:
            print("  k < 2, skipping metaheuristics")
            continue

        if n_clusters_meta > 20:
            n_clusters_meta = 20

        k = n_clusters_meta
        print(f"\n  k for metaheuristics = {k}")


        # ════════════════════════════════════════════════════
        # FITNESS FUNCTION
        # ════════════════════════════════════════════════════
        def fitness_function(centers):
            centers = centers.reshape((k, X.shape[1]))
            dist = ((X[:, None, :] - centers) ** 2).sum(axis=2)
            labels = np.argmin(dist, axis=1)
            if len(np.unique(labels)) < 2:
                return 9999.0
            return davies_bouldin_score(X, labels)

        lb_flat = np.full(k * X.shape[1], np.min(X))
        ub_flat = np.full(k * X.shape[1], np.max(X))

        problem = {
            "obj_func": fitness_function,
            "bounds": FloatVar(
                lb=lb_flat.tolist(),
                ub=ub_flat.tolist()
            ),
            "minmax": "min",
            "seed": SEED,
        }

        # ════════════════════════════════════════════════════
        # GWO, PSO, WOA
        # ════════════════════════════════════════════════════

        # FIX 5 — use ITER and POP instead of ep and pop
        algorithms = [
            ("GWO", mealpy.swarm_based.GWO.OriginalGWO(epoch=ITER, pop_size=POP)),
            ("PSO", mealpy.swarm_based.PSO.OriginalPSO(epoch=ITER, pop_size=POP)),
            ("WOA", mealpy.swarm_based.WOA.OriginalWOA(epoch=ITER, pop_size=POP)),
        ]

        np.random.seed(SEED)

        for algo_name, optimizer in algorithms:
            print(f"\n  Running {algo_name}...")

            agent   = optimizer.solve(problem)

            # FIX 6 — use k instead of n_clusters_meta
            centers = agent.solution.reshape((k, X.shape[1]))

            # Assign each point to its nearest centroid
            dist   = ((X[:, None, :] - centers) ** 2).sum(axis=2)
            labels = np.argmin(dist, axis=1)

            if len(set(labels)) < 2:
                print(f"  {algo_name} produced only 1 cluster, skipping")
                continue

            # Save synthetic centroids
            np.save(f"results/representatives/centroids_{dataset_name}_{classifier_name}_{algo_name}.npy", centers)

            # Find nearest REAL point to each centroid
            nearest_indices = []
            nearest_points  = []

            for centroid in centers:
                distances   = np.sqrt(((X - centroid) ** 2).sum(axis=1))
                nearest_idx = np.argmin(distances)   # index of closest real point
                nearest_indices.append(nearest_idx)
                nearest_points.append(X_raw[nearest_idx])  # original unscaled point

            nearest_indices = np.array(nearest_indices)
            nearest_points  = np.array(nearest_points)

            # FIX 7 — save to results/representatives not results/rq4
            np.save(f"results/representatives/nearest_indices_{dataset_name}_{classifier_name}_{algo_name}.npy", nearest_indices)
            np.save(f"results/representatives/nearest_points_{dataset_name}_{classifier_name}_{algo_name}.npy",  nearest_points)

            print(f"  {algo_name} done , {k} centroids saved and {k} nearest real points saved")


        # ════════════════════════════════════════════════════
        # EVOCLUSTER PSO
        # ════════════════════════════════════════════════════

        # FIX 8 — move import inside try so it actually checks
        try:
            from EvoCluster import optimizers
            evocluster_ok = True
            print("EvoCluster loaded successfully")
        except Exception as e:
            print("EvoCluster import failed:", e)
            evocluster_ok = False

        if evocluster_ok:
            np.random.seed(SEED)
            print(f"\n  Running EvoCluster_PSO...")

            try:
                def objf(solution, points=None, k=None):
                    centers = solution.reshape((k, X.shape[1]))
                    dist = ((X[:, None, :] - centers) ** 2).sum(axis=2)
                    labels = np.argmin(dist, axis=1)
                    if len(set(labels)) < 2:
                        return 9999, labels
                    score = davies_bouldin_score(X, labels)
                    return score, labels

                dim = k * X.shape[1]

                result = optimizers.PSO(
                    objf=objf,
                    lb=lb_flat,
                    ub=ub_flat,
                    dim=dim,
                    PopSize=POP,
                    iters=ITER,
                    k=k,
                    points=X,
                    metric="euclidean"
                )

                best_solution = result.bestIndividual

                # FIX 9 — rename to centers_evo consistently
                centers_evo = best_solution.reshape((k, X.shape[1]))
                dist        = ((X[:, None, :] - centers_evo) ** 2).sum(axis=2)
                labels_evopso = np.argmin(dist, axis=1)

                # FIX 10 — condition was reversed
                if len(set(labels_evopso)) < 2:
                    print("  EvoCluster_PSO produced only 1 cluster, skipping")
                else:
                    np.save(f"results/representatives/centroids_{dataset_name}_{classifier_name}_EvoCluster_PSO.npy", centers_evo)

                    # Find nearest REAL point to each centroid
                    nearest_indices_evo = []
                    nearest_points_evo  = []

                    for centroid in centers_evo:
                        distances   = np.sqrt(((X - centroid) ** 2).sum(axis=1))
                        nearest_idx = np.argmin(distances)
                        nearest_indices_evo.append(nearest_idx)
                        nearest_points_evo.append(X_raw[nearest_idx])

                    nearest_indices_evo = np.array(nearest_indices_evo)
                    nearest_points_evo  = np.array(nearest_points_evo)

                    # FIX 11 — save to results/representatives not results/rq4
                    np.save(f"results/representatives/nearest_indices_{dataset_name}_{classifier_name}_EvoCluster_PSO.npy", nearest_indices_evo)
                    np.save(f"results/representatives/nearest_points_{dataset_name}_{classifier_name}_EvoCluster_PSO.npy",  nearest_points_evo)

                    print(f"  EvoCluster_PSO done — {k} centroids saved, {k} nearest real points saved")

            except Exception as e:
                print("EvoCluster PSO failed:", e)


print("\n" + "="*60)
print("All results saved to results/representatives/")
print("="*60)