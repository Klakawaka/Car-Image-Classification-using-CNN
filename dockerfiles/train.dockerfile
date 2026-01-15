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

# Install dependencies (cached layer if dependencies don't change)
# Use --no-dev to skip dev dependencies, remove --frozen to allow platform resolution
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --no-install-project

# Copy source code and required files
COPY src/ src/
COPY README.md README.md
COPY LICENSE LICENSE

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev

# Copy data directory (training and test data)
# Note: In production, you might want to mount this as a volume instead
COPY raw/ raw/

# Create models directory for output
RUN mkdir -p models

# Run training script with unbuffered output (-u flag)
ENTRYPOINT ["uv", "run", "python", "-u", "src/car_image_classification_using_cnn/train.py"]
