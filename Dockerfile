FROM python:3.9-slim-bullseye as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    libmagic1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Second stage: Runtime
FROM python:3.9-slim-bullseye as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    TZ=UTC \
    PORT=5000 \
    DEBUG=False

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r forensicai && useradd -r -g forensicai forensicai

# Create required directories
RUN mkdir -p /app/uploads /app/debug_output /app/models

# Set ownership of app files
RUN chown -R forensicai:forensicai /app

# Copy virtual env from builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application code
COPY --chown=forensicai:forensicai . .

# Create log file and set permissions
RUN touch /app/forensic_ai.log && chown forensicai:forensicai /app/forensic_ai.log

# Switch to non-root user
USER forensicai

# Expose port for the application
EXPOSE 5000

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Run the application with Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --workers 4 --threads 2 --timeout 120 app:app 