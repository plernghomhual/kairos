# Kairos Docker Deployment

Kairos can run headlessly in Docker with the default one-shot CLI command:

```bash
docker build -t kairos .
docker run --rm kairos
```

## Build

```bash
docker build -t kairos .
```

The image uses a multi-stage `python:3.12.11-slim` build, installs only runtime dependencies, and runs as the non-root `appuser`.

## Run Headless

```bash
docker run --rm kairos
```

Equivalent explicit command:

```bash
docker run --rm kairos kairos --asset BTC --no-tui
```

## Run The API

The current API module exposes a FastAPI app factory. Run it with `uvicorn`:

```bash
docker run --rm -p 8000:8000 kairos \
  python -m uvicorn "kairos.api.server:create_app" \
  --factory --host 0.0.0.0 --port 8000
```

Then check:

```bash
curl http://127.0.0.1:8000/health
```

## Mount A Custom Database

Kairos reads `KAIROS_DB_PATH`, which defaults to `/kairos/kairos.db` in the image.

```bash
docker run --rm \
  -v "$(pwd)/kairos.db:/kairos/kairos.db" \
  kairos
```

Use a different path if needed:

```bash
docker run --rm \
  -v "$(pwd)/data:/data" \
  -e KAIROS_DB_PATH=/data/kairos.db \
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

The check exits 0 when the configured DuckDB database is readable or when a local API server responds 200 on `/health`.

## Docker Compose

```bash
docker compose up -d
docker compose logs -f kairos
docker compose down
```

Compose mounts `./kairos.db` to `/kairos/kairos.db` and passes through `FRED_API_KEY`, `GITHUB_TOKEN`, and `SOLANA_RPC_URL` from the host environment.
