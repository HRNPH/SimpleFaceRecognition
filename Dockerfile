ARG INSTALL_DEPENDENCIES=prod

FROM python:3.12-slim AS base

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential python3-setuptools libgl1 libglib2.0-0 \
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

# Final production stage
FROM base AS base-prod

WORKDIR /app

# Copy Over installed dependencies python dependencies
COPY --from=prisma-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=prisma-builder /usr/local/bin /usr/local/bin

# Add application code and Prisma client
COPY . ./
COPY --from=prisma-builder /app/prisma ./prisma



# Expose port
ENV PORT=8000
EXPOSE $PORT

# Run the application
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "[::]:8000", "app.main:app", "--timeout", "120"]

# Add a healthcheck
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 CMD curl --fail http://127.0.0.1:$PORT/ || exit 1
