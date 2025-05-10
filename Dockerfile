FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ADD . /app
WORKDIR /app

RUN uv sync --locked --no-dev

EXPOSE 8080

CMD ["/app/.venv/bin/app", "--port", "8080"]
