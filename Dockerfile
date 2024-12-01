ARG INSTALL_DEPENDENCIES=prod

FROM python:3.12-bookworm AS base

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential python3-setuptools libgl1 libglib2.0-0 libssl-dev libffi-dev \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Stage for Prisma and Node.js
FROM base AS prisma-builder

WORKDIR /app

# Copy Prisma schema and install Prisma CLI
COPY prisma ./prisma

# Install Prisma CLI python
# Copy production requirements and install dependencies
COPY requirements.prod.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.prod.txt

# Generate Prisma client
RUN python -m prisma generate

COPY . ./

# Expose port
ENV PORT=8000
EXPOSE $PORT

# Run the application
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "[::]:8000", "app.main:app", "--timeout", "120"]

# Add a healthcheck
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 CMD curl --fail http://127.0.0.1:$PORT/ || exit 1
