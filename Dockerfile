# Reproducible CPU environment for training, tests and the demo.
#   docker build -t retina .
#   docker run --rm -v "$PWD:/app" retina python -m pytest
#   docker run --rm -v "$PWD:/app" retina python -m src.train --config configs/multilabel_efficientnet.yaml
#   docker run --rm -p 7860:7860 -v "$PWD:/app" retina python -m app.demo --run outputs/runs/<run>
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLBACKEND=Agg \
    KERAS_HOME=/app/.keras

RUN apt-get update \
 && apt-get install -y --no-install-recommends git libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-dev.txt requirements-app.txt ./
RUN pip install -r requirements-dev.txt -r requirements-app.txt

COPY . .
RUN git config --global --add safe.directory /app

EXPOSE 7860
CMD ["python", "-m", "app.demo", "--host", "0.0.0.0", "--port", "7860"]
