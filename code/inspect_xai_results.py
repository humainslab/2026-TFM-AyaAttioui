import pickle, os 

print(os.listdir("results/xai")[:20])
with open("results/xai/blood-transfusion_rf_AP_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)
print(type(metrics))
print(metrics)