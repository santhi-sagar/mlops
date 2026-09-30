from pathlib import Path
import json
import numpy as np
import pandas as pd

RAW_PATH = Path(r"C:\Users\91701\OneDrive\Desktop\EXCEL R PROJECT\ratings.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_ACTIVE_USERS = 1000
MIN_USER_RATINGS = 20

print("Loading the raw ratings file...")
df = pd.read_csv(
    RAW_PATH,
    dtype={
        "user_id": "string",
        "product_id": "string",
        "rating": "float32",
        "timestamp": "int64",
    },
)
df.columns = ["user_id", "product_id", "rating", "timestamp"]
raw_rows = len(df)
exact_duplicates = int(df.duplicated().sum())
df = df.dropna(subset=["user_id", "product_id", "rating", "timestamp"]).drop_duplicates()
df = (
    df.sort_values("timestamp", kind="mergesort")
    .drop_duplicates(["user_id", "product_id"], keep="last")
    .reset_index(drop=True)
)

user_counts = df["user_id"].value_counts()
eligible_users = user_counts[user_counts >= MIN_USER_RATINGS].index
sampled_users = eligible_users.to_series().sample(
    n=min(N_ACTIVE_USERS, len(eligible_users)), random_state=RANDOM_STATE
).tolist()
work = (
    df[df["user_id"].isin(sampled_users)]
    .sort_values(["user_id", "timestamp"], kind="mergesort")
    .copy()
)
test_raw = work.groupby("user_id", sort=False).tail(1).copy()
train = work.drop(index=test_raw.index).copy()
train_products = set(train["product_id"])
test = test_raw[test_raw["product_id"].isin(train_products)]

# Save only the compact training interactions required by the app.
train[["user_id", "product_id", "rating", "timestamp"]].to_csv(
    OUTPUT_DIR / "train_interactions.csv", index=False
)

# Preserve the verified offline benchmark for the app's Evaluation tab.
results_path = Path(r"C:\Users\91701\OneDrive\Desktop\EXCEL R PROJECT\model_results.json")
results = json.loads(results_path.read_text(encoding="utf-8"))["results"] if results_path.exists() else []
metadata = {
    "raw_rows": int(raw_rows),
    "exact_duplicates": exact_duplicates,
    "cleaned_rows": int(len(df)),
    "all_users": int(df["user_id"].nunique()),
    "all_products": int(df["product_id"].nunique()),
    "sampled_active_users": int(len(sampled_users)),
    "minimum_user_ratings": MIN_USER_RATINGS,
    "train_users": int(train["user_id"].nunique()),
    "train_products": int(train["product_id"].nunique()),
    "train_interactions": int(len(train)),
    "candidate_test_interactions": int(len(test_raw)),
    "evaluable_test_interactions": int(len(test)),
    "cold_start_test_interactions": int(len(test_raw) - len(test)),
    "matrix_sparsity": float(
        1
        - len(train)
        / (train["user_id"].nunique() * train["product_id"].nunique())
    ),
    "model": "TruncatedSVD user representation + KMeans cluster recommender",
    "cluster_count": 8,
    "random_state": RANDOM_STATE,
    "results": results,
}
(OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata, indent=2))
print(f"Wrote {OUTPUT_DIR / 'train_interactions.csv'}")
