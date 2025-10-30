# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install minimal runtime libraries for scientific stack (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . /app

# Create non-root user and grant permissions
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/results && \
    chown -R appuser:appuser /app
USER appuser

# Defaults; Coolify will provide PORT at runtime
ENV PORT=8000 \
    WEB_CONCURRENCY=2 \
    GUNICORN_TIMEOUT=120 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_MAX_THREADS=2

# Persist user data
VOLUME ["/app/uploads", "/app/results"]

EXPOSE 8000

# Simple healthcheck against /health
HEALTHCHECK --interval=30s --timeout=2s --retries=5 CMD python - << 'PY' \
import os, urllib.request; \
port=os.environ.get('PORT','8000'); \
url=f"http://127.0.0.1:{port}/health"; \
try: \
    with urllib.request.urlopen(url, timeout=1) as r: \
        exit(0 if r.status==200 else 1) \
except Exception: \
    exit(1) \
PY

# Start with gunicorn, binding to the provided PORT
CMD ["sh", "-c", "exec gunicorn -w ${WEB_CONCURRENCY:-2} -k gthread -t ${GUNICORN_TIMEOUT:-120} -b 0.0.0.0:${PORT} run:app"]
