FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv sync --no-install-project --locked --no-dev

COPY . /app

RUN uv sync --locked --no-dev

EXPOSE 8080

CMD ["/app/.venv/bin/app", "--port", "8080"]
