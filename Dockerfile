# One image for both services (api and dashboard). Data and results are NOT copied in: they are mounted read-only at run time.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    PIP_NO_CACHE_DIR=1

# libgomp1: the OpenMP runtime LightGBM needs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# non-root user
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

COPY configs ./configs
COPY src ./src
COPY api ./api
COPY app ./app

USER appuser
EXPOSE 8000 8501
# default: the API (docker-compose overrides the command for the dashboard)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
