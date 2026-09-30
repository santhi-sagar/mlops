# Database-Backed Cluster-Aware Product Recommendations

This Streamlit app serves precomputed cluster-aware recommendations from a database.

## Recommendation behavior

- Known user: looks up the user's KMeans cluster and queries recommendations stored for that user.
- Unknown user: queries `popular_recommendations` and labels every result `Global popularity fallback`.
- Every result includes `source_cluster`, `source_method`, cluster rating support, and recommendation score.

## Database architecture

The production path is Supabase PostgreSQL using the REST API. The schema contains:

- `user_profiles`: user-to-cluster assignments and rating history counts.
- `recommendations`: ranked recommendations for known users.
- `popular_recommendations`: cold-start fallback recommendations.
- `user_ratings`: historical interactions used by the training pipeline.
- `model_metadata`: model version and evaluation metrics.

A read-only Row Level Security policy is enabled for the app tables. Data loading and retraining must use a protected service-role process, never a public app key.

The local working copy also contains `data/recommendations.db`, a SQLite fallback that makes the app runnable locally without secrets. The public Streamlit deployment is configured to use Supabase PostgreSQL; the app can automatically use SQLite when that file is available and Supabase is missing or temporarily unavailable.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To use Supabase locally, set the following environment variables or add them to `.streamlit/secrets.toml`:

```toml
SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_ANON_KEY = "YOUR_PUBLISHABLE_OR_ANON_KEY"
```

The current database project is `recommendation-system-db`. Do not commit secrets to GitHub.

## Streamlit Community Cloud deployment

1. Open Streamlit Community Cloud and select **New app**.
2. Choose repository `santhi-sagar/mlops`.
3. Select branch `main`.
4. Set the main file path to `streamlit_recommendation_app/app.py`.
5. In Advanced settings, add:

```toml
SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_ANON_KEY = "YOUR_PUBLISHABLE_OR_ANON_KEY"
```

6. Deploy.

The app can start with the SQLite fallback, but adding the Supabase secrets makes the deployed app query the hosted PostgreSQL database.

## Rebuilding the database

`build_database.py` rebuilds the compact SQLite database and JSON seed files from `data/train_interactions.csv`. The generated seed files can be loaded into Supabase by a protected administrative script. Do not run a service-role loader inside the public Streamlit app.

## Validation examples

- Known user: `A108X1JFG00VOC` should show a KMeans cluster source.
- Unknown user: `UNKNOWN_USER` should show `Global popularity fallback`.
