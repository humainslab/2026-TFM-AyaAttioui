import os
import glob
import json
import pandas as pd
from scipy.stats import wilcoxon

# RQ1: PipeGenie (AutoML) vs manual classifiers (rf, svm, xgb)
# manual: only one run (seed 123456)
# pipegenie: one run per seed, we take the mean over the seeds

classifiers_dir = "classifiers"
imbalance_file = "results/analysis/imbalance_fpfn_summary.csv"
out_dir = "results/analysis"
manual_list = ["rf", "svm", "xgb"]

# our 18 working datasets (the list of the project, bank-marketing and adult are removed)
# PipeGenie also ran on 6 other datasets (cnae-9_bin, egg-eye-state, kc2, pc1, qsar-biodeg, spambase)
# put keep_only_18 = False to use all the datasets that have PipeGenie results (24)
keep_only_18 = True
datasets_18 = ["diabetes", "wdbc", "column", "blood-transfusion", "isolet_bin", "usps_bin",
               "phoneme", "kc1", "ozone-level-8hr", "hill_valley", "heart-disease",
               "credit-approval", "mammographic-mass", "contraceptive-method-choice",
               "german-credit", "horse-colic", "congressional-voting", "cylinder-bands"]
suffix = "_18datasets" if keep_only_18 else ""

# phoneme and egg-eye-state have a wrong ratio in the imbalance file
# real ratios: phoneme 2.41 (slight), egg-eye-state 1.23 (balanced)
# put fix_categories = False to keep the old categories
fix_categories = True

os.makedirs(out_dir, exist_ok=True)

#find imbalance category of each dataset
imb = pd.read_csv(imbalance_file)
category = {}
for i in range(len(imb)):
    category[imb["dataset"][i]] = imb["imbalance_category"][i]
if fix_categories:
    category["phoneme"] = "slight"
    category["egg-eye-state"] = "balanced"

rows = []
for dataset in sorted(os.listdir(classifiers_dir)):
    folder = f"{classifiers_dir}/{dataset}/pipegenie"
    json_files = sorted(glob.glob(f"{folder}/results_seed*.json"))
    if len(json_files) == 0:
        continue
    if keep_only_18 and dataset not in datasets_18:
        continue

    #find fp and fn of pipegenie for each seed
    if os.path.exists(f"{folder}/classification_metrics.csv"):
        pg = pd.read_csv(f"{folder}/classification_metrics.csv")
        source = "csv"
    else:
        # no csv for this dataset: we count fp and fn in the json files
        seeds = []
        fps = []
        fns = []
        for path in json_files:
            with open(path) as f:
                data = json.load(f)
            seeds.append(os.path.basename(path))
            fps.append(len(data["false_positives_values"]))
            fns.append(len(data["false_negatives_values"]))
        pg = pd.DataFrame({"seed": seeds, "fp": fps, "fn": fns, "f1": float("nan")})
        source = "json"

    pg_errors = pg["fp"] + pg["fn"]
    pg_fn_share = pg["fn"] / pg_errors

    #find manual classifier
    for clf in manual_list:
        path = f"{classifiers_dir}/{dataset}/{clf}/classification_metrics.csv"
        if not os.path.exists(path):
            continue
        man = pd.read_csv(path).iloc[0]
        man_errors = man["fp"] + man["fn"]

        row = {}
        row["dataset"] = dataset
        row["imbalance"] = category.get(dataset, "unknown")
        row["manual_clf"] = clf
        row["n_seeds"] = len(pg)
        row["pg_source"] = source
        row["pg_fp"] = pg["fp"].mean()
        row["pg_fn"] = pg["fn"].mean()
        row["pg_errors"] = pg_errors.mean()
        row["pg_errors_std"] = pg_errors.std()
        row["pg_fn_share"] = pg_fn_share.mean()
        row["pg_f1"] = pg["f1"].mean()
        row["man_fp"] = man["fp"]
        row["man_fn"] = man["fn"]
        row["man_errors"] = man_errors
        row["man_f1"] = man["f1"]
        # fn_share = part of the errors that are false negatives (0 to 1)
        if man_errors > 0:
            row["man_fn_share"] = man["fn"] / man_errors
            row["errors_change"] = (row["pg_errors"] - man_errors) / man_errors
        else:
            row["man_fn_share"] = float("nan")
            row["errors_change"] = float("nan")
        row["fn_share_diff"] = row["pg_fn_share"] - row["man_fn_share"]
        row["f1_diff"] = row["pg_f1"] - row["man_f1"]
        rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(f"{out_dir}/rq1_pipegenie_vs_manual_per_dataset{suffix}.csv", index=False)
print(table["dataset"].nunique(), "datasets,", len(table), "rows")


def summary(group, name):
    out = {"group": name, "n_datasets": len(group)}

    # errors_change < 0 means pipegenie makes fewer errors
    d = group["errors_change"].dropna()
    out["pg_fewer_errors"] = int((d < 0).sum())
    out["median_errors_change"] = d.median()
    try:
        out["p_errors"] = wilcoxon(d).pvalue
    except ValueError:
        out["p_errors"] = float("nan")

    # fn_share_diff < 0 means pipegenie has less FN (more FP) than the manual one
    d = group["fn_share_diff"].dropna()
    out["pg_less_fn_share"] = int((d < 0).sum())
    out["mean_fn_share_diff"] = d.mean()
    try:
        out["p_fn_share"] = wilcoxon(d).pvalue
    except ValueError:
        out["p_fn_share"] = float("nan")

    # f1_diff > 0 means pipegenie is better (only datasets with f1)
    d = group["f1_diff"].dropna()
    out["n_f1"] = len(d)
    out["pg_better_f1"] = int((d > 0).sum())
    out["mean_f1_diff"] = d.mean()
    try:
        out["p_f1"] = wilcoxon(d).pvalue
    except ValueError:
        out["p_f1"] = float("nan")
    return out


#find summary by manual classifier
rows = []
for clf in manual_list:
    rows.append(summary(table[table["manual_clf"] == clf], clf))
summary_table = pd.DataFrame(rows)
summary_table.to_csv(f"{out_dir}/rq1_pipegenie_vs_manual_summary{suffix}.csv", index=False)

#find summary by imbalance category
rows = []
for clf in manual_list:
    for cat in ["balanced", "slight", "moderate", "strong", "extreme"]:
        group = table[(table["manual_clf"] == clf) & (table["imbalance"] == cat)]
        if len(group) == 0:
            continue
        r = summary(group, clf + " | " + cat)
        r["manual_clf"] = clf
        r["imbalance"] = cat
        rows.append(r)
pd.DataFrame(rows).to_csv(f"{out_dir}/rq1_pipegenie_vs_manual_by_imbalance{suffix}.csv", index=False)

print(summary_table.round(3).to_string())