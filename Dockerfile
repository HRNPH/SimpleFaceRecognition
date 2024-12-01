ARG INSTALL_DEPENDENCIES=prod

FROM python:3.12-slim AS base

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl git build-essential python3-setuptools \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/apt/lists/* \
    && rm -rf /var/cache/apt/*

FROM base AS base-prod

WORKDIR /app

COPY requirements.prod.txt ./

# install production dependencies using pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.prod.txt

FROM base-prod AS base-dev

# install development dependencies if any
COPY requirements.dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.dev.txt

# hadolint ignore=DL3006
FROM base-${INSTALL_DEPENDENCIES} AS final

# copy all the application code
COPY . ./

# create a non-root user and switch to it, for security.
RUN addgroup --system --gid 1001 "app-user"
RUN adduser --system --uid 1001 "app-user"
USER "app-user"

ENTRYPOINT ["/bin/sh", "-c"]
# default port is 8000 but can be overridden
ENV PORT=8000
# log current port
RUN echo "Running on port $PORT"
# request to /
CMD ["gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind [::]:$PORT app.main:app --timeout 120"]
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 CMD curl --fail http://0.0.0.0:$PORT/ || exit 1
