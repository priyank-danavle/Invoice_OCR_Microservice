# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

LABEL maintainer="OCR Microservice Team"
LABEL description="Mistral-powered invoice OCR microservice for textile ERP"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (separate layer for cache efficiency) ─────────
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY app/ ./app/
COPY ui/  ./ui/

# ── File ownership ────────────────────────────────────────────────────────────
RUN mkdir -p /app/logs \
    && chown -R appuser:appgroup /app

# ── Switch to non-root user ───────────────────────────────────────────────────
USER appuser

# ── Expose service port ───────────────────────────────────────────────────────
EXPOSE 8001

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# ── Launch application ────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "2", \
     "--log-level", "info", \
     "--access-log", \
     "--root-path", "/ocr-api"]

