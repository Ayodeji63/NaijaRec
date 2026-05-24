# NaijaRec LightGCN Review Generator Web App

This web app is the containerized submission interface for the Nigerian-diaspora review/rating generator. It accepts a user persona and one or more restaurant/product descriptions, then returns persona-grounded ratings and review-style explanations.

The deployed model is the NaijaRec LightGCN + balanced hybrid reranker implemented in `app/review_engine.py`. It does not call Gemini, OpenAI, Kimi, or any external LLM API. The Docker image includes the trained LightGCN checkpoint, the Yelp-Kimi restaurant catalog, personas, knowledge graph triples, the base LightGCN ranking matrix, and the tuned balanced hybrid ranking matrix.

## Features

- FastAPI backend.
- Static web UI served from the same container.
- `/api/generate-review` JSON endpoint.
- `/api/model-info` endpoint describing the deployed model mode.
- Deployed LightGCN checkpoint and hybrid reranker artifacts.
- No external API calls or provider keys required.
- Persona fields for cultural group, taste profile, price sensitivity, dietary context, dining context, and review style.
- Product fields for name, category, city, price, rating, review count, summary, and optional image URL.

## Run Locally

```bash
pip install -r requirements-api.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/api/health
```

Model information:

```bash
curl http://localhost:8000/api/model-info
```

## Run With Docker

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000
```

The Docker image does not require `.env` or API keys. The app is fully self-contained.

### Docker Compose `.env` Troubleshooting

Docker Compose may automatically read a project-root `.env` file before building. If one is present, it must use normal dotenv syntax:

```text
REVIEW_APP_ENV=production
```

Do not place a bare quoted key on its own line, and do not quote the variable name:

```text
"production"             # invalid
"REVIEW_APP_ENV"=...     # invalid
```

If Compose fails with `unexpected character "\"" in variable name`, fix or temporarily rename the root `.env`:

```bash
mv .env .env.local
docker compose up --build
```

Alternatively, run the app with a clean env file:

```bash
cp .env.example .env.docker
docker compose --env-file .env.docker up --build
```

For local Python installs, `requirements-api.txt` uses Pydantic v2. If an older environment already has Pydantic v1 installed, upgrade with:

```bash
pip install --upgrade -r requirements-api.txt
```

### Docker Socket Permission Troubleshooting

If Docker is installed but Compose fails with:

```text
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

the current Linux user is not allowed to access the Docker daemon. For a quick one-time run:

```bash
sudo docker compose up --build
```

For the normal non-sudo workflow, add the user to the Docker group, refresh the shell group membership, and retry:

```bash
sudo usermod -aG docker $USER
newgrp docker
docker compose up --build
```

If Docker was installed through Snap and the group change does not apply immediately, log out and log back in, then run:

```bash
docker ps
docker compose up --build
```

The local Python install may show a warning about `gradio` requiring an older `websockets` package. That warning does not block this FastAPI app. For clean research runs, keep the web app in a separate virtual environment from the original Agent4Rec/Gradio environment.

## API Example

```bash
curl -X POST http://localhost:8000/api/generate-review \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "local",
    "persona": {
      "cultural_group": "Yoruba Nigerian diaspora",
      "taste_profile": "Likes bold pepper-forward food, generous portions, grilled meat or fish, rice dishes, good value, and lively vibes.",
      "price_sensitivity": "medium",
      "dietary_context": "",
      "dining_context": "Dinner after work",
      "review_style": "yoruba"
    },
    "products": [
      {
        "name": "Han Dynasty",
        "category": "Chinese, Szechuan, Restaurants",
        "city": "Philadelphia",
        "price_range": "2",
        "rating": 4.0,
        "review_count": 783,
        "summary": "A Szechuan restaurant known for bold heat, casual dining, and hearty dishes."
      }
    ]
  }'
```

## Research Positioning

This web app satisfies the required deployment interface: persona and product details are accepted as input, and reviews plus ratings are produced as output. The deployed app uses the research model artifacts instead of an external LLM or a template-only engine. For each request, the service selects the closest deployed avatar persona, matches the supplied product to the Yelp-Kimi restaurant catalog, retrieves LightGCN and hybrid reranker ranks, and produces a rating/explanation grounded in those model scores and metadata features.

## What Is Deployed?

The container deploys:

```text
FastAPI backend
Static web UI
Trained LightGCN checkpoint
Base LightGCN ranking matrix
Balanced hybrid reranking matrix
Yelp-Kimi restaurant catalog
Nigerian-diaspora personas
Knowledge graph triples
```

The container does not rerun LightGCN training or the full avatar simulation at request time. Instead, it deploys the trained checkpoint and saved LightGCN/reranker artifacts for fast inference in the web/API demo.
