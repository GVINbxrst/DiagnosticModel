# =============================================================================
# Multi-stage Dockerfile for DiagMod Streamlit Dashboard
# =============================================================================

# Base stage with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e ".[dev,test]"

# Copy source code
COPY --chown=app:app . .

# Switch to app user
USER app

# Expose port
EXPOSE 8501

# Command for development
CMD ["streamlit", "run", "src/dashboard/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Install only production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Dashboard
RUN pip install --no-cache-dir \
    reportlab==4.0.7 \
    streamlit-authenticator==0.2.3 \
    streamlit-option-menu==0.3.6

# Copy source code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app configs/ ./configs/

# Install package
RUN pip install --no-cache-dir -e .

# Create necessary directories and streamlit config
RUN mkdir -p /app/logs /app/.streamlit && \
    chown -R app:app /app

# Create Streamlit config
RUN echo '[server]\n\
port = 8501\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = "#FF6B6B"\n\
backgroundColor = "#FFFFFF"\n\
secondaryBackgroundColor = "#F0F2F6"\n\
textColor = "#262730"\n\
font = "sans serif"' > /app/.streamlit/config.toml

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Expose port
EXPOSE 8501

# Command for production
CMD ["streamlit", "run", "src/dashboard/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
