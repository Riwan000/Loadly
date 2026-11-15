# ============================================================================
# Stage 1: Build React Frontend
# ============================================================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ============================================================================
# Stage 2: Python Backend & Production Runtime
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Set environment variables for Python and model caching
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=8000

# Install system utilities (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies:
# Install CPU-only PyTorch first to prevent downloading multi-GB CUDA runtime wheels (~2.5GB saved)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache Sentence-Transformers embedding model to eliminate cold-start latency
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source code and artifacts
COPY api/ ./api/
COPY artifacts/ ./artifacts/
COPY prompts/ ./prompts/
COPY utils/ ./utils/
COPY ui/ ./ui/
COPY config.py data_loading.py features.py models.py rag.py server.py visualization.py weather.py app.py ./

# Copy compiled frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Ensure runtime directory for FAISS vector index exists
RUN mkdir -p /app/data/rag_index

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/api/health || exit 1

# Start the unified FastAPI server
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]

