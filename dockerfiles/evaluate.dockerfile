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

# Install dependencies 
RUN uv sync --no-dev --no-install-project

# Copy source code and required files
COPY src/ src/
COPY README.md README.md
COPY LICENSE LICENSE

# Install the project itself
RUN uv sync --no-dev

# Run evaluation script with unbuffered output (-u flag)
ENTRYPOINT ["uv", "run", "python", "-u", "src/car_image_classification_using_cnn/evaluate.py"]
