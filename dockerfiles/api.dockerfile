# Base image with uv package manager
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# Install dependencies (without BuildKit cache mount)
RUN uv sync --no-dev --no-install-project

# Copy source code and required files
COPY src/ src/
COPY README.md README.md
COPY LICENSE LICENSE


# Install the project itself
RUN uv sync --no-dev

RUN mkdir -p /app/models

ENV PORT=8000

# Expose port for FastAPI
EXPOSE 8000

# Set working directory to src for running the app
WORKDIR /app/src

# Run FastAPI server using PORT environment variable
CMD uv run uvicorn main:app --host 0.0.0.0 --port ${PORT}
