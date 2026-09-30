from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

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
    "source_cluster",
    "source_method",
    "cluster_avg_rating",
    "cluster_rating_count",
    "recommendation_score",
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
            timeout=15,
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

    def _with_fallback(self, remote_call, sqlite_call):
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
                {"select": "user_id,cluster_id,rating_count", "user_id": f"eq.{user_id}", "limit": "1"},
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
        def remote():
            rows = self._remote_get(
                "recommendations",
                {
                    "select": ",".join(OUTPUT_COLUMNS),
                    "user_id": f"eq.{user_id}",
                    "order": "rank.asc",
                    "limit": str(k),
                },
            )
            return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

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
            return frame[OUTPUT_COLUMNS]

        def local():
            return self._sqlite(
                "select rank, product_id, source_cluster, source_method, "
                "NULL as cluster_avg_rating, NULL as cluster_rating_count, "
                "recommendation_score from popular_recommendations order by rank limit ?",
                (k,),
            )

        return self._with_fallback(remote, local)


@st.cache_resource(show_spinner="Connecting to the recommendation database...")
def get_database() -> RecommendationDatabase:
    return RecommendationDatabase()


def format_recommendations(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ["cluster_avg_rating", "cluster_rating_count", "recommendation_score"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[OUTPUT_COLUMNS]


def main() -> None:
    st.title("Cluster-Aware Product Recommendation System")
    st.caption(
        "Recommendations are queried from a database. Each result identifies its source cluster "
        "or the global popularity fallback."
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

    recommendations = format_recommendations(recommendations)
    st.subheader("Recommendations")
    st.caption(f"Recommendation query source: {recommendation_source}")
    st.dataframe(
        recommendations.style.format(
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

    tab1, tab2, tab3 = st.tabs(["How it works", "Evaluation", "Limitations"])
    with tab1:
        st.markdown(
            """
            1. The app looks up the user profile and cluster assignment in the database.
            2. For a known user, it queries precomputed recommendations for that user's KMeans cluster.
            3. For an unknown user, it queries the global popularity table instead of inventing a cluster.
            4. The response includes the exact source cluster, method, rating support, and score.
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
            - The current database contains the validated active-user sample and precomputed recommendations.
            - A scheduled retraining job should refresh profiles and recommendations when new ratings arrive.
            - Product metadata is not available, so a future content-based model is needed for completely new products.
            - The public database policies are read-only; loading and retraining should use a protected service role outside the app.
            """
        )


if __name__ == "__main__":
    main()
