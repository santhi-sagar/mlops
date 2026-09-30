from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
SEED_DIR = APP_DIR / "database_seed"
TRAIN_PATH = DATA_DIR / "train_interactions.csv"
METADATA_PATH = DATA_DIR / "metadata.json"
RANDOM_STATE = 42


def main() -> None:
    SEED_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(
        TRAIN_PATH,
        dtype={
            "user_id": "string",
            "product_id": "string",
            "rating": "float32",
            "timestamp": "int64",
        },
    )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    users = sorted(train["user_id"].unique())
    products = sorted(train["product_id"].unique())
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    product_to_idx = {product: idx for idx, product in enumerate(products)}

    matrix = csr_matrix(
        (
            train["rating"].to_numpy(dtype=np.float32),
            (
                train["user_id"].map(user_to_idx).to_numpy(),
                train["product_id"].map(product_to_idx).to_numpy(),
            ),
        ),
        shape=(len(users), len(products)),
        dtype=np.float32,
    )
    n_components = min(20, max(2, min(matrix.shape) - 1))
    latent = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE).fit_transform(matrix)
    labels = KMeans(n_clusters=8, n_init=10, random_state=RANDOM_STATE).fit_predict(latent)

    work = train.copy()
    work["user_idx"] = work["user_id"].map(user_to_idx)
    work["cluster"] = work["user_idx"].map(dict(enumerate(labels)))
    cluster = (
        work.groupby(["cluster", "product_id"])
        .agg(cluster_avg_rating=("rating", "mean"), cluster_rating_count=("rating", "size"))
        .reset_index()
    )
    cluster["recommendation_score"] = cluster["cluster_avg_rating"] * np.log1p(
        cluster["cluster_rating_count"]
    )
    cluster["item_idx"] = cluster["product_id"].map(product_to_idx)
    cluster = cluster.sort_values(
        ["cluster", "recommendation_score", "cluster_rating_count"],
        ascending=[True, False, False],
    )

    popularity = (
        train.groupby("product_id")
        .agg(rating_count=("rating", "size"), average_rating=("rating", "mean"))
        .reset_index()
    )
    popularity["recommendation_score"] = popularity["rating_count"] * np.log1p(
        popularity["average_rating"]
    )
    popularity = popularity.sort_values(
        ["recommendation_score", "rating_count"], ascending=False
    ).head(100)

    profiles = []
    recommendations = []
    for user_id in users:
        user_idx = user_to_idx[user_id]
        cluster_id = int(labels[user_idx])
        profiles.append(
            {
                "user_id": str(user_id),
                "cluster_id": cluster_id,
                "rating_count": int(matrix[user_idx].nnz),
            }
        )
        seen = set(matrix[user_idx].indices)
        candidates = cluster[
            (cluster["cluster"] == cluster_id) & (~cluster["item_idx"].isin(seen))
        ].head(20)
        for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
            recommendations.append(
                {
                    "user_id": str(user_id),
                    "rank": rank,
                    "product_id": str(row["product_id"]),
                    "source_cluster": f"KMeans Cluster {cluster_id}",
                    "source_method": "Cluster-based recommendation",
                    "cluster_avg_rating": float(row["cluster_avg_rating"]),
                    "cluster_rating_count": int(row["cluster_rating_count"]),
                    "recommendation_score": float(row["recommendation_score"]),
                }
            )

    popular_recommendations = [
        {
            "rank": rank,
            "product_id": str(row["product_id"]),
            "source_cluster": "N/A",
            "source_method": "Global popularity fallback",
            "recommendation_score": float(row["recommendation_score"]),
        }
        for rank, (_, row) in enumerate(popularity.iterrows(), start=1)
    ]
    user_ratings = [
        {
            "user_id": str(row.user_id),
            "product_id": str(row.product_id),
            "rating": float(row.rating),
            "timestamp": int(row.timestamp),
        }
        for row in train.itertuples(index=False)
    ]
    summary = {
        "model": metadata["model"],
        "cluster_count": metadata["cluster_count"],
        "train_users": metadata["train_users"],
        "train_products": metadata["train_products"],
        "train_interactions": metadata["train_interactions"],
        "results": metadata["results"],
    }

    database_path = DATA_DIR / "recommendations.db"
    if database_path.exists():
        database_path.unlink()
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            create table user_profiles(user_id text primary key, cluster_id integer not null, rating_count integer not null);
            create table recommendations(user_id text not null, rank integer not null, product_id text not null, source_cluster text not null, source_method text not null, cluster_avg_rating real, cluster_rating_count integer, recommendation_score real not null, primary key(user_id, rank));
            create table popular_recommendations(rank integer primary key, product_id text not null, source_cluster text not null, source_method text not null, recommendation_score real not null);
            create table user_ratings(user_id text not null, product_id text not null, rating real not null, timestamp integer, primary key(user_id, product_id));
            create table model_metadata(key text primary key, value text not null);
            """
        )
        connection.executemany("insert into user_profiles values(?,?,?)", [(x["user_id"], x["cluster_id"], x["rating_count"]) for x in profiles])
        connection.executemany("insert into recommendations values(?,?,?,?,?,?,?,?)", [(x["user_id"], x["rank"], x["product_id"], x["source_cluster"], x["source_method"], x["cluster_avg_rating"], x["cluster_rating_count"], x["recommendation_score"]) for x in recommendations])
        connection.executemany("insert into popular_recommendations values(?,?,?,?,?)", [(x["rank"], x["product_id"], x["source_cluster"], x["source_method"], x["recommendation_score"]) for x in popular_recommendations])
        connection.executemany("insert into user_ratings values(?,?,?,?)", [(x["user_id"], x["product_id"], x["rating"], x["timestamp"]) for x in user_ratings])
        connection.execute("insert into model_metadata values(?, ?)", ("summary", json.dumps(summary)))

    for name, rows in {
        "user_profiles": profiles,
        "recommendations": recommendations,
        "popular_recommendations": popular_recommendations,
        "user_ratings": user_ratings,
        "model_metadata": {"summary": summary},
    }.items():
        (SEED_DIR / f"{name}.json").write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")

    print(f"Created {database_path} with {len(profiles)} profiles, {len(recommendations)} recommendations, and {len(user_ratings)} ratings.")


if __name__ == "__main__":
    main()
