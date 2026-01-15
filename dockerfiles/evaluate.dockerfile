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

# Note: Model and test data will be mounted as volumes at runtime
# Example usage:
# docker run --rm \
#   -v /path/to/models:/app/models \
#   -v /path/to/test/data:/app/raw/test \
#   evaluate:latest models/best_model.pth

# Run evaluation script with unbuffered output (-u flag)
ENTRYPOINT ["uv", "run", "python", "-u", "src/car_image_classification_using_cnn/evaluate.py"]
