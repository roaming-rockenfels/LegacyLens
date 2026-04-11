FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/Reference-LAPACK/lapack.git /tmp/lapack && \
    cd /tmp/lapack && git sparse-checkout set SRC && \
    mkdir -p /app/data/lapack && mv /tmp/lapack/SRC /app/data/lapack/SRC && \
    rm -rf /tmp/lapack

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "legacylens.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
