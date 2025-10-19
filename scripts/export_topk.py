import pandas as pd, joblib, yaml, numpy as np
df = pd.read_csv("data/raw/creditcard.csv")
if "Class" in df.columns: df = df.drop(columns=["Class"])
pipe = joblib.load("models/inference.joblib")
scores = pipe.predict_proba(df)[:,1]
thr = yaml.safe_load(open("configs/thresholds.yaml"))
cut = thr["by_topk"]
df.assign(score=scores, is_fraud=(scores>=cut)).sort_values("score", ascending=False).head(500).to_csv("models/review_queue_topK.csv", index=False)
print("Wrote models/review_queue_topK.csv")
