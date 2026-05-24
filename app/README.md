# NaijaRec Submission App

This directory contains the containerized application submitted for judging.
The app satisfies the deployment requirement by accepting a user persona and
product details as input, then returning generated ratings and review text as
output.

The containerized app is a FastAPI service with:

- a browser UI at `/`
- a health endpoint at `/api/health`
- a model metadata endpoint at `/api/model-info`
- a JSON generation endpoint at `/api/generate-review`

The deployed service uses the packaged NaijaRec LightGCN checkpoint and saved
reranking artifacts. It does not call external LLM APIs at runtime.

## Docker Quick Start

Build the image:

```bash
docker build -t ayodeji63/naijarec:latest .
```

Run the container:

```bash
docker run --rm -p 8000:8000 ayodeji63/naijarec:latest
```

Open the app:

```text
http://localhost:8000
```

## Docker Compose

If Docker Compose is preferred:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000
```

## Sanity Checks

Health:

```bash
curl http://localhost:8000/api/health
```

Model info:

```bash
curl http://localhost:8000/api/model-info
```

Expected health response:

```json
{"status":"ok"}
```

## API Example

```bash
curl -X POST http://localhost:8000/api/generate-review \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "lightgcn",
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

## Notes for Judges

- No `.env` file or API key is required to run the submission app.
- The image is the submission interface, not the full 821-avatar experiment runner.
- The service depends on the packaged model artifacts copied by the root
  `Dockerfile`.

## Troubleshooting

If Docker requires elevated privileges on Linux:

```bash
sudo docker build -t ayodeji63/naijarec:latest .
sudo docker run --rm -p 8000:8000 ayodeji63/naijarec:latest
```

If port `8000` is already in use, map a different local port:

```bash
docker run --rm -p 8080:8000 ayodeji63/naijarec:latest
```

and open:

```text
http://localhost:8080
```
