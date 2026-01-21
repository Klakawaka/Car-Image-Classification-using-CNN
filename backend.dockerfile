FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install -U pip && pip install uv

# Copy dependency files first (better caching)
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

COPY README.md /app/README.md


# Copy source code BEFORE uv sync (so project exists)
COPY src /app/src
COPY service.py /app/service.py

# If you also need scripts:
COPY scripts /app/scripts

# Make sure ONNX model exists and is copied
COPY models /app/models



# Install deps
RUN uv sync --frozen

EXPOSE 4040
CMD ["uv", "run", "bentoml", "serve", "service:CarClassifierService", "--port", "4040", "--host", "0.0.0.0"]
