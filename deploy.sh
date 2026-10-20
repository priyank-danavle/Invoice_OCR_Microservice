#!/usr/bin/env bash
set -e

echo "=========================================="
echo "  Deploying OCR Microservice"
echo "=========================================="

echo "[1/4] Stopping & removing existing containers..."
docker compose down --remove-orphans
docker rm -f ocr-microservice 2>/dev/null || true

echo "[2/4] Building docker image (--no-cache)..."
docker compose build --no-cache

echo "[3/4] Starting container in detached mode..."
docker compose up -d

echo "[4/4] Following container logs..."
docker logs ocr-microservice -f
