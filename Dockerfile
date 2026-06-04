FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models \
    CHROMA_DIR=/app/chroma

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY config ./config
COPY scripts ./scripts
COPY src ./src
COPY models/metrics.json ./models/metrics.json

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -e .

RUN python -m invoice_iq.classifier.train --epochs 20 --out models

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/chroma \
    && chown -R appuser:appuser /app /home/appuser

USER appuser
EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn invoice_iq.serving.app:create_app --factory --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"]
