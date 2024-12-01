#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Run Prisma generate
python -m prisma generate

# Execute the passed command (Gunicorn in this case)
exec "$@"
