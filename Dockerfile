# One image for both services: the FastAPI backend (default command) and
# the Streamlit frontend (command overridden in docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# PySpark runs on the JVM, so the image needs a JRE (same requirement as
# the "Install" step in the README).
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so this layer is cached until requirements.txt changes.
# The CPU-only torch wheel is installed explicitly: the default wheel pulls
# in several GB of CUDA libraries this project never uses.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY . .

# Run as a non-root user. /app/data is writable because the ETL writes the
# DuckDB warehouse and the FAISS index there. The model cache directory is
# created here so a volume mounted over it is owned by the same user.
RUN useradd --create-home appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
