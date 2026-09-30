# Database-Backed Cluster-Aware Product Recommendations

This Streamlit app serves precomputed cluster-aware recommendations from Supabase PostgreSQL and enriches each result with product catalog information.

## Recommendation behavior

- Known user: looks up the user's KMeans cluster and queries recommendations stored for that user.
- Unknown user: queries `popular_recommendations` and labels every result `Global popularity fallback`.
- Every result includes `source_cluster`, `source_method`, cluster rating support, and recommendation score.
- Each result is joined by `product_id` to `product_catalog` and can display product name, category, brand, price, description, image, and product link.
- If public metadata is unavailable for a product, the app shows an explicit `Fallback product ID record` instead of inventing product facts.

## Database architecture

The production path is Supabase PostgreSQL using the REST API. The schema contains:

- `user_profiles`: user-to-cluster assignments and rating history counts.
- `recommendations`: ranked recommendations for known users.
- `popular_recommendations`: cold-start fallback recommendations.
- `product_catalog`: product names, categories, descriptions, images, links, brand, and price metadata.
- `user_ratings`: historical interactions used by the training pipeline.
- `model_metadata`: model version and evaluation metrics.

The catalog was enriched from the public Amazon Electronics metadata archive for product IDs present in the training sample. Matching metadata is stored with source attribution; unmatched IDs have explicit fallback records so the application remains stable.

On 2026-09-30, an additional 33 fallback ASINs were verified against their public Amazon product pages and loaded into Supabase with the source label `Amazon public product page metadata`. Eighteen of these records include image URLs that returned valid JPEG content during validation. The reproducible upload payload is stored in `database_seed/catalog_enrichment_public.json`. Seven ASINs returned unavailable or 404 pages and remain explicit fallback records rather than receiving invented metadata.

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

6. Deploy or wait for the existing app to rebuild after a new commit.

The deployed app uses the hosted database when these secrets are configured. The public app key is read-only through Supabase RLS; catalog loading and model retraining remain administrative processes outside Streamlit.

## Rebuilding the database

`build_database.py` rebuilds the compact SQLite database and JSON seed files from `data/train_interactions.csv`. The generated seed files can be loaded into Supabase by a protected administrative script. Do not run a service-role loader inside the public Streamlit app.

## Validation examples

- Known user: `A108X1JFG00VOC` should show a KMeans cluster source and product metadata cards.
- Unknown user: `UNKNOWN_USER` should show `Global popularity fallback` and product metadata cards.
- A recommendation table row should include both the original product ID and its resolved catalog name/category.
