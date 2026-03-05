FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY data/lapack/SRC/ data/lapack/SRC/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "legacylens.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
