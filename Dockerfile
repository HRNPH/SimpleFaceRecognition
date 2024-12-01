# Define the dependency target (prod or dev)
ARG INSTALL_DEPENDENCIES=prod

# Base image
FROM python:3.12-slim AS base

# Install necessary system packages and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential python3-setuptools libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Stage for production dependencies
FROM base AS base-prod

# Set working directory
WORKDIR /app

# Copy requirements for production
COPY requirements.prod.txt ./

# Install production dependencies with caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.prod.txt

# Stage for development dependencies
FROM base-prod AS base-dev

# Copy additional development requirements
COPY requirements.dev.txt ./

# Install development dependencies with caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.dev.txt

# Final stage
FROM base-${INSTALL_DEPENDENCIES} AS final

# Set working directory
WORKDIR /app

# Copy application code
COPY . ./

# Set Prisma binary target in schema
RUN sed -i 's/binaryTargets = \[\]/binaryTargets = ["native", "debian-openssl-3.0.x"]/g' prisma/schema.prisma

# Install Prisma CLI for generating files
RUN npm install -g prisma

# Generate Prisma files
RUN prisma generate

# Make sure Prisma binary and cache are accessible
RUN chmod -R 777 /root/.cache/prisma-python

# Set environment variable for the application port
ENV PORT=8000

# Default user remains root for simplicity
USER root

# Expose the application port
EXPOSE $PORT

# Run the application with Gunicorn and Uvicorn workers
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "[::]:8000", "app.main:app", "--timeout", "120"]

# Add a healthcheck
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 CMD curl --fail http://127.0.0.1:$PORT/ || exit 1
