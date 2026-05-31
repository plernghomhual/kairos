ARG PYTHON_VERSION=3.12.11

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY pyproject.toml ./
COPY kairos/ ./kairos/

RUN python -m pip install --prefix=/install --no-cache-dir --no-compile . \
    && find /install -type d \( -name __pycache__ -o -name tests -o -name test \) -prune -exec rm -rf '{}' + \
    && find /install -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KAIROS_DB_PATH=/kairos/kairos.db \
    PATH=/usr/local/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /home/appuser --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /kairos \
    && chown appuser:appuser /kairos

COPY --from=builder /install/ /usr/local/

WORKDIR /kairos

COPY --chown=appuser:appuser kairos/ ./kairos/
COPY --chown=appuser:appuser kairos.db ./kairos.db

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s --start-period=10s \
    CMD python /kairos/kairos/healthcheck.py

CMD ["kairos", "--asset", "BTC", "--no-tui"]
