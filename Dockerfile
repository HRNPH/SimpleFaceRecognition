ARG INSTALL_DEPENDENCIES=prod

FROM python:3.12-slim AS base

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl git build-essential python3-setuptools \
    libgl1 \
    libglib2.0-0 \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/apt/lists/* \
    && rm -rf /var/cache/apt/*

FROM base AS base-prod

WORKDIR /app

COPY requirements.prod.txt ./

# Install production dependencies using pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.prod.txt

FROM base-prod AS base-dev

# Install development dependencies if any
COPY requirements.dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.dev.txt

# hadolint ignore=DL3006
FROM base-${INSTALL_DEPENDENCIES} AS final

# Copy all the application code
COPY . ./

# Generate Prisma files
RUN prisma generate

# Default user remains root
USER root

# Set environment variable for port
ENV PORT=8000

# Use exec form for CMD
CMD gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind [::]:$PORT app.main:app --timeout 120

# Healthcheck
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 CMD curl --fail http://0.0.0.0:$PORT/ || exit 1
