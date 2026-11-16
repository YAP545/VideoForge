# VideoForge Production Dockerfile
# Optimized for Railway, Render, Fly.io, or any container environment
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install FFmpeg and required runtime system libraries for HarfBuzz / FreeType text rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libfreetype6 \
    libfontconfig1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for caching efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create persistent storage directory mount point
RUN mkdir -p /data/videos

EXPOSE 8000

# Start Uvicorn with a single worker process (guarantees safe SQLite write-locks and in-memory queue consistency)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
