from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

datasets = {
    "contraceptive-method-choice": 30,
    "credit-approval":             27,
    "mammographic-mass":           161,
    "bank-marketing":               222,
    "heart-disease":                45,
    "german-credit":                144,
    "adult-census-income":          2,
    "horse-colic":                  47,
    "congressional-voting":         105,
    "cylinder-bands":               32,
}

for name, uid in datasets.items():
    print(f"Downloading: {name} (id={uid})")
    try:
        ds = fetch_ucirepo(id=uid)
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)
        df.to_csv(f"data/{name}.csv", index=False)
        print(f"  Saved: {df.shape}")
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")

print("\n Datasets processed.")