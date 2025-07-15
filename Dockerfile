FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ml_service.py .
COPY simple_models.py .
COPY models/ models/

# Create log directory for ml_service.py
RUN mkdir -p /var/log/mccva && chmod 755 /var/log/mccva

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /var/log/mccva
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the ML service directly with Python
CMD ["python", "ml_service.py"] 