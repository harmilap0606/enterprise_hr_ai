FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application modules, models, datasets, frontend, and tests
COPY app/ app/
COPY data/ data/
COPY frontend/ frontend/
COPY models/ models/
COPY config/ config/
COPY tests/ tests/
COPY entrypoint.sh .
COPY README.md .

# Ensure startup script is executable
RUN chmod +x entrypoint.sh

# Expose ports: 7860 for Hugging Face Spaces Streamlit UI, 8000 for internal FastAPI
EXPOSE 7860 8000

# Health check against Streamlit health endpoint on 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
