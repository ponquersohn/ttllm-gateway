FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY alembic.ini .
COPY alembic/ alembic/

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh && pip install --no-cache-dir -e .

RUN useradd -r -s /bin/false ttllm
USER ttllm

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
