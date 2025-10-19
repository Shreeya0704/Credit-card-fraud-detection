#!/bin/bash
# This script serves the FastAPI application.

echo "Starting FastAPI server..."
echo "API will be available at http://127.0.0.1:8000"

uvicorn src.service.app:app --host 0.0.0.0 --port 8000
