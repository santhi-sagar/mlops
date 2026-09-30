create table if not exists public.user_profiles (
  user_id text primary key,
  cluster_id integer not null,
  rating_count integer not null,
  created_at timestamptz not null default now()
);

create table if not exists public.recommendations (
  user_id text not null references public.user_profiles(user_id) on delete cascade,
  rank integer not null,
  product_id text not null,
  source_cluster text not null,
  source_method text not null,
  cluster_avg_rating double precision,
  cluster_rating_count integer,
  recommendation_score double precision not null,
  created_at timestamptz not null default now(),
  primary key (user_id, rank)
);

create table if not exists public.popular_recommendations (
  rank integer primary key,
  product_id text not null,
  source_cluster text not null default 'N/A',
  source_method text not null,
  recommendation_score double precision not null,
  created_at timestamptz not null default now()
);

create table if not exists public.user_ratings (
  user_id text not null references public.user_profiles(user_id) on delete cascade,
  product_id text not null,
  rating double precision not null,
  timestamp bigint,
  primary key (user_id, product_id)
);

create table if not exists public.model_metadata (
  key text primary key,
  value jsonb not null,
  updated_at timestamptz not null default now()
);

alter table public.user_profiles enable row level security;
alter table public.recommendations enable row level security;
alter table public.popular_recommendations enable row level security;
alter table public.user_ratings enable row level security;
alter table public.model_metadata enable row level security;

-- Public app access is read-only. Use a protected service-role process for loading data.
create policy public_read_user_profiles on public.user_profiles for select to anon, authenticated using (true);
create policy public_read_recommendations on public.recommendations for select to anon, authenticated using (true);
create policy public_read_popular_recommendations on public.popular_recommendations for select to anon, authenticated using (true);
create policy public_read_user_ratings on public.user_ratings for select to anon, authenticated using (true);
create policy public_read_model_metadata on public.model_metadata for select to anon, authenticated using (true);
