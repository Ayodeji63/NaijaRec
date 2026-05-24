FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY datasets/yelp-kimi/simulation/movie_detail.csv ./model_artifacts/datasets/yelp-kimi/simulation/movie_detail.csv
COPY datasets/yelp-kimi/simulation/all_personas_like_modify.csv ./model_artifacts/datasets/yelp-kimi/simulation/all_personas_like_modify.csv
COPY datasets/yelp-kimi/simulation/knowledge_graph_triples.csv ./model_artifacts/datasets/yelp-kimi/simulation/knowledge_graph_triples.csv
COPY storage/yelp-kimi/LightGCN/Test/rankings/full_rankings_70.npy ./model_artifacts/storage/yelp-kimi/LightGCN/Test/rankings/full_rankings_70.npy
COPY storage/yelp-kimi/LightGCN/Test/rankings/offline_balanced_rerank_70_best_rankings.npy ./model_artifacts/storage/yelp-kimi/LightGCN/Test/rankings/offline_balanced_rerank_70_best_rankings.npy
COPY storage/yelp-kimi/LightGCN/HybridRerankDiagnosticAll/rankings/full_rankings_821.npy ./model_artifacts/storage/yelp-kimi/LightGCN/HybridRerankDiagnosticAll/rankings/full_rankings_821.npy
COPY storage/yelp-kimi/LightGCN/HybridRerankDiagnosticAll/rankings/offline_exposure_small_821_best_rankings.npy ./model_artifacts/storage/yelp-kimi/LightGCN/HybridRerankDiagnosticAll/rankings/offline_exposure_small_821_best_rankings.npy
COPY recommenders/weights/yelp-kimi/LightGCN/0515_yelp_lgn_t0p0_Ks=20_patience=10_n_layers=2_batch_size=1024_neg_sample=1_lr=0.0005/epoch=84.checkpoint.pth.tar ./model_artifacts/recommenders/weights/yelp-kimi/LightGCN/0515_yelp_lgn_t0p0_Ks=20_patience=10_n_layers=2_batch_size=1024_neg_sample=1_lr=0.0005/epoch=84.checkpoint.pth.tar

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
