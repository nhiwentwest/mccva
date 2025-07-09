FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files - UPDATED to use ml_service.py
COPY ml_service.py .
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

# Run the enhanced ML service with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "ml_service:app"] 