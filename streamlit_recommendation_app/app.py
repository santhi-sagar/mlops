from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
SQLITE_PATH = DATA_DIR / "recommendations.db"

st.set_page_config(
    page_title="Cluster-Aware Product Recommendations",
    page_icon="🛍️",
    layout="wide",
)

OUTPUT_COLUMNS = [
    "rank",
    "product_id",
    "product_name",
    "category",
    "brand",
    "price",
    "source_cluster",
    "source_method",
    "cluster_avg_rating",
    "cluster_rating_count",
    "recommendation_score",
    "image_url",
    "product_url",
    "description",
    "metadata_source",
]

CATALOG_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "description",
    "image_url",
    "product_url",
    "brand",
    "price",
    "metadata_source",
]


def _read_setting(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


class RecommendationDatabase:
    """Database interface with a remote Supabase REST path and local SQLite fallback."""

    def __init__(self) -> None:
        self.supabase_url = _read_setting("SUPABASE_URL").rstrip("/")
        self.supabase_key = _read_setting("SUPABASE_ANON_KEY") or _read_setting(
            "SUPABASE_PUBLISHABLE_KEY"
        )
        self.remote_enabled = bool(self.supabase_url and self.supabase_key)
        self.remote_error = ""

    def _remote_get(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        response = requests.get(
            f"{self.supabase_url}/rest/v1/{table}",
            headers={
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
            },
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected response from database table {table}.")
        return payload

    def _sqlite(self, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        if not SQLITE_PATH.exists():
            raise FileNotFoundError(
                "The bundled SQLite database is missing. Run build_database.py first."
            )
        with sqlite3.connect(SQLITE_PATH) as connection:
            return pd.read_sql_query(query, connection, params=params)

    def _with_fallback(
        self, remote_call: Callable[[], Any], sqlite_call: Callable[[], Any]
    ) -> tuple[Any, str]:
        if self.remote_enabled:
            try:
                return remote_call(), "Supabase PostgreSQL"
            except Exception as exc:
                self.remote_error = str(exc)
        return sqlite_call(), "Bundled SQLite fallback"

    def get_metadata(self) -> tuple[dict[str, Any], str]:
        def remote():
            rows = self._remote_get(
                "model_metadata", {"select": "value", "key": "eq.summary", "limit": "1"}
            )
            return rows[0]["value"] if rows else {}

        def local():
            frame = self._sqlite("select value from model_metadata where key = ?", ("summary",))
            return json.loads(frame.iloc[0]["value"]) if not frame.empty else {}

        return self._with_fallback(remote, local)

    def get_user_profile(self, user_id: str) -> tuple[dict[str, Any] | None, str]:
        def remote():
            rows = self._remote_get(
                "user_profiles",
                {
                    "select": "user_id,cluster_id,rating_count",
                    "user_id": f"eq.{user_id}",
                    "limit": "1",
                },
            )
            return rows[0] if rows else None

        def local():
            frame = self._sqlite(
                "select user_id, cluster_id, rating_count from user_profiles where user_id = ?",
                (user_id,),
            )
            return frame.iloc[0].to_dict() if not frame.empty else None

        return self._with_fallback(remote, local)

    def get_recommendations(self, user_id: str, k: int) -> tuple[pd.DataFrame, str]:
        base_columns = [
            "rank",
            "product_id",
            "source_cluster",
            "source_method",
            "cluster_avg_rating",
            "cluster_rating_count",
            "recommendation_score",
        ]

        def remote():
            rows = self._remote_get(
                "recommendations",
                {
                    "select": ",".join(base_columns),
                    "user_id": f"eq.{user_id}",
                    "order": "rank.asc",
                    "limit": str(k),
                },
            )
            return pd.DataFrame(rows, columns=base_columns)

        def local():
            return self._sqlite(
                "select rank, product_id, source_cluster, source_method, "
                "cluster_avg_rating, cluster_rating_count, recommendation_score "
                "from recommendations where user_id = ? order by rank limit ?",
                (user_id, k),
            )

        return self._with_fallback(remote, local)

    def get_popular_recommendations(self, k: int) -> tuple[pd.DataFrame, str]:
        def remote():
            rows = self._remote_get(
                "popular_recommendations",
                {
                    "select": "rank,product_id,source_cluster,source_method,recommendation_score",
                    "order": "rank.asc",
                    "limit": str(k),
                },
            )
            frame = pd.DataFrame(rows)
            frame["cluster_avg_rating"] = pd.NA
            frame["cluster_rating_count"] = pd.NA
            return frame[
                [
                    "rank",
                    "product_id",
                    "source_cluster",
                    "source_method",
                    "cluster_avg_rating",
                    "cluster_rating_count",
                    "recommendation_score",
                ]
            ]

        def local():
            return self._sqlite(
                "select rank, product_id, source_cluster, source_method, "
                "NULL as cluster_avg_rating, NULL as cluster_rating_count, "
                "recommendation_score from popular_recommendations order by rank limit ?",
                (k,),
            )

        return self._with_fallback(remote, local)

    def get_catalog(self, product_ids: list[str]) -> tuple[pd.DataFrame, str]:
        product_ids = [str(value) for value in dict.fromkeys(product_ids) if str(value)]

        def remote():
            rows: list[dict[str, Any]] = []
            # Keep requests comfortably below common URL-length limits.
            for start in range(0, len(product_ids), 100):
                ids = product_ids[start : start + 100]
                rows.extend(
                    self._remote_get(
                        "product_catalog",
                        {
                            "select": ",".join(CATALOG_COLUMNS),
                            "product_id": f"in.({','.join(ids)})",
                        },
                    )
                )
            return pd.DataFrame(rows, columns=CATALOG_COLUMNS)

        def local():
            if not product_ids:
                return pd.DataFrame(columns=CATALOG_COLUMNS)
            placeholders = ",".join("?" for _ in product_ids)
            try:
                return self._sqlite(
                    f"select {','.join(CATALOG_COLUMNS)} from product_catalog "
                    f"where product_id in ({placeholders})",
                    tuple(product_ids),
                )
            except Exception:
                # Older local SQLite bundles may not have the optional catalog table.
                return pd.DataFrame(columns=CATALOG_COLUMNS)

        return self._with_fallback(remote, local)


@st.cache_resource(show_spinner="Connecting to the recommendation database...")
def get_database() -> RecommendationDatabase:
    return RecommendationDatabase()


def enrich_recommendations(
    recommendations: pd.DataFrame, database: RecommendationDatabase
) -> tuple[pd.DataFrame, str]:
    recommendations = recommendations.copy()
    catalog, catalog_source = database.get_catalog(
        recommendations["product_id"].astype(str).tolist()
    )
    enriched = recommendations.merge(catalog, on="product_id", how="left")
    fallback_name = "Product " + enriched["product_id"].astype(str)
    enriched["product_name"] = enriched["product_name"].fillna(fallback_name)
    enriched["category"] = enriched["category"].fillna("Metadata unavailable")
    enriched["description"] = enriched["description"].fillna(
        "Product description is not available in the catalog."
    )
    enriched["metadata_source"] = enriched["metadata_source"].fillna(
        "Fallback product ID record"
    )
    for column in [
        "brand",
        "price",
        "image_url",
        "product_url",
    ]:
        enriched[column] = enriched[column].fillna("")
    for column in ["cluster_avg_rating", "cluster_rating_count", "recommendation_score"]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    return enriched[OUTPUT_COLUMNS], catalog_source


def render_product_cards(recommendations: pd.DataFrame) -> None:
    st.subheader("Product details")
    for start in range(0, len(recommendations), 2):
        columns = st.columns(2)
        for column, (_, row) in zip(columns, recommendations.iloc[start : start + 2].iterrows()):
            with column:
                image_url = str(row.get("image_url", ""))
                if image_url:
                    try:
                        st.image(image_url, width=180)
                    except Exception:
                        st.caption("Product image could not be loaded.")
                st.markdown(f"**#{int(row['rank'])} — {row['product_name']}**")
                if row.get("brand"):
                    st.caption(f"Brand: {row['brand']}")
                st.write(f"Category: {row['category']}")
                if row.get("price"):
                    st.write(f"Price listed in source catalog: {row['price']}")
                st.write(str(row["description"])[:500])
                st.caption(
                    f"Source: {row['source_cluster']} · {row['source_method']} · "
                    f"Catalog: {row['metadata_source']}"
                )
                if row.get("product_url"):
                    st.markdown(f"[View product listing]({row['product_url']})")


def main() -> None:
    st.title("Cluster-Aware Product Recommendation System")
    st.caption(
        "Recommendations are queried from a database and enriched with catalog metadata. "
        "Each result identifies its source cluster or the global popularity fallback."
    )

    database = get_database()
    try:
        metadata, metadata_source = database.get_metadata()
    except Exception as exc:
        st.error(f"Could not load model metadata: {exc}")
        st.stop()

    default_user = "A108X1JFG00VOC"
    with st.sidebar:
        st.header("Recommendation settings")
        user_id = st.text_input("Enter a user ID", value=default_user).strip()
        k = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
        st.divider()
        st.subheader("Database status")
        if database.remote_enabled and not database.remote_error:
            st.success("Connected to Supabase PostgreSQL")
        elif database.remote_enabled and database.remote_error:
            st.warning("Supabase unavailable; using SQLite fallback")
        else:
            st.info("Using bundled SQLite database")
        st.caption(f"Active source: {metadata_source}")

    profile, profile_source = database.get_user_profile(user_id)
    if profile:
        recommendations, recommendation_source = database.get_recommendations(user_id, k)
        cluster_id = int(profile["cluster_id"])
        st.success(
            f"Known user. Recommendations are generated from KMeans Cluster {cluster_id}."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("User history", f"{int(profile['rating_count'])} ratings")
        c2.metric("Source cluster", f"Cluster {cluster_id}")
        c3.metric("Recommendation count", len(recommendations))
    else:
        recommendations, recommendation_source = database.get_popular_recommendations(k)
        st.warning(
            "This user is not in the database. The app is using the global popularity fallback."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("User status", "Unknown user")
        c2.metric("Source cluster", "N/A")
        c3.metric("Recommendation count", len(recommendations))

    if database.remote_error:
        st.caption(f"Database fallback reason: {database.remote_error}")

    recommendations, catalog_source = enrich_recommendations(recommendations, database)
    st.subheader("Recommendations")
    st.caption(
        f"Recommendation query source: {recommendation_source} · "
        f"Product catalog source: {catalog_source}"
    )
    table_columns = [
        "rank",
        "product_id",
        "product_name",
        "category",
        "source_cluster",
        "source_method",
        "cluster_avg_rating",
        "recommendation_score",
    ]
    st.dataframe(
        recommendations[table_columns].style.format(
            {"cluster_avg_rating": "{:.2f}", "recommendation_score": "{:.3f}"},
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download recommendations as CSV",
        data=recommendations.to_csv(index=False).encode("utf-8"),
        file_name=f"recommendations_{user_id or 'unknown'}.csv",
        mime="text/csv",
    )
    render_product_cards(recommendations)

    tab1, tab2, tab3 = st.tabs(["How it works", "Evaluation", "Limitations"])
    with tab1:
        st.markdown(
            """
            1. The app looks up the user profile and cluster assignment in the database.
            2. For a known user, it queries precomputed recommendations for that user's KMeans cluster.
            3. For an unknown user, it queries the global popularity table instead of inventing a cluster.
            4. The response joins each recommendation with product catalog metadata by product ID.
            5. Supabase PostgreSQL is used when configured; the packaged SQLite database is a local fallback.
            """
        )
        st.info(
            "The cluster score is average rating × log(1 + cluster rating count). "
            "The recommendations are precomputed during the model-building pipeline and served by database queries."
        )
    with tab2:
        st.subheader("Initial offline benchmark")
        results = pd.DataFrame(metadata.get("results", []))
        if not results.empty:
            results["model"] = results["model"].str.replace(" recommender", "", regex=False)
            st.dataframe(
                results.style.format(
                    {
                        "precision_at_10": "{:.2%}",
                        "recall_at_10": "{:.2%}",
                        "hit_rate_at_10": "{:.2%}",
                        "map_at_10": "{:.2%}",
                        "catalog_coverage_at_10": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            "These are chronological-holdout results on the active-user sample, not production guarantees."
        )
    with tab3:
        st.markdown(
            """
            - The database contains the validated active-user sample, precomputed recommendations, and catalog enrichment rows.
            - A scheduled retraining job should refresh profiles and recommendations when new ratings arrive.
            - Public catalog metadata is incomplete for some products, so those rows show an explicit fallback record rather than inventing details.
            - The public database policies are read-only; loading and retraining should use a protected service role outside the app.
            """
        )


if __name__ == "__main__":
    main()
