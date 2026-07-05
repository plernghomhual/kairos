# Kairos Docker Deployment

Kairos runs the FastAPI server by default in Docker:

```bash
docker build -t kairos .
docker run --rm -p 8000:8000 -e KAIROS_API_KEY=change-me kairos
```

## Build

```bash
docker build -t kairos .
```

The image uses a multi-stage `python:3.12.11-slim` build, installs only runtime dependencies, and runs as the non-root `appuser`. It does not bake in a local DuckDB file; runtime data lives under `/data`.

## Run The API

```bash
docker run --rm -p 8000:8000 -e KAIROS_API_KEY=change-me kairos
```

Equivalent explicit command:

```bash
docker run --rm -p 8000:8000 -e KAIROS_API_KEY=change-me kairos \
  python -m uvicorn "kairos.api.server:create_app" \
  --factory --host 0.0.0.0 --port 8000
```

Then check:

```bash
curl http://127.0.0.1:8000/health
```

## Run One-Shot CLI

```bash
docker run --rm kairos kairos --asset BTC --no-tui
```

## Mount A Custom Database

Kairos reads `KAIROS_DB_PATH`, which defaults to `/data/kairos.db` in the image.

```bash
docker run --rm \
  -v kairos-data:/data \
  -p 8000:8000 \
  -e KAIROS_API_KEY=change-me \
  kairos
```

Use a different path if needed:

```bash
docker run --rm \
  -v "$(pwd)/kairos-data:/runtime-data" \
  -e KAIROS_DB_PATH=/runtime-data/kairos.db \
  -e KAIROS_API_KEY=change-me \
  -p 8000:8000 \
  kairos
```

## Set Environment Variables

Pass API keys at runtime. Do not bake secrets into the image.

```bash
docker run --rm -e GITHUB_TOKEN=xxx kairos
docker run --rm -e FRED_API_KEY=xxx kairos
docker run --rm -e SOLANA_RPC_URL=https://example.invalid kairos
```

## Health Check

The image health check runs:

```bash
python /kairos/kairos/healthcheck.py
```

For a named running container:

```bash
docker inspect --format='{{json .State.Health}}' kairos
```

The check exits 0 when the configured DuckDB database is readable and the local API server responds 200 on `/health`.

## Docker Compose

```bash
docker compose up -d
docker compose logs -f kairos
docker compose down
```

Compose mounts the `kairos-data` named volume to `/data`, exposes the API on `8000`, and passes through the supported provider and alerting environment variables from the host environment.
