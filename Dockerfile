# ManualMind Dockerfile
# Optimized for local CPU inference, Hugging Face Spaces (Docker SDK), and Render
# Python 3.11-slim, pre-downloads all models at build time

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

# Install system dependencies (ffmpeg for whisper audio decoding, build-essential for C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install CPU-only PyTorch first (keeps image size lean ~1.2GB instead of 5GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependencies list and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Pre-download all models, spaCy pipeline, NLTK data, and generate default manual at build time
RUN python scripts/download_models.py

# Create non-root user for Hugging Face Spaces & security
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose standard port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Health check probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Start FastAPI application with uvicorn (dynamic port support for Render / Cloud hosts)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

