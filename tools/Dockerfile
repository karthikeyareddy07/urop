# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY TG-ReDial-main/ ./TG-ReDial-main/
COPY TG-ReDial-Data/ ./TG-ReDial-Data/

# Copy serving entrypoint and trained probe checkpoint
COPY serve_chinese.py ./serve_chinese.py
COPY out_probe/ ./out_probe/

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

# Serve the frozen-ALBERT Chinese movie recommender: loads out_probe/best.pt
# and exposes the chat_app HTTP server (includes /api/health).
CMD ["python", "serve_chinese.py", "--host", "0.0.0.0", "--port", "8000"]
