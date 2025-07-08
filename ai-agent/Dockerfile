# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port (if needed for future web interface)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import main; print('Agent is healthy')" || exit 1

# Default command to run the agent
CMD ["python", "main.py"]

# Alternative entry points
# For testing: docker run <image> python test_agent.py
# For interactive shell: docker run -it <image> /bin/bash

# Labels for metadata
LABEL name="pinecone-ai-agent"
LABEL version="1.0.0"
LABEL description="AI Agent with Pinecone memory and OpenAI GPT-4 reasoning"
LABEL maintainer="your-email@example.com"

# Environment variables documentation
# Required environment variables:
# - OPENAI_API_KEY: OpenAI API key for GPT-4 access
# - PINECONE_API_KEY: Pinecone API key for vector database
# 
# Optional environment variables:
# - PINECONE_ENVIRONMENT: Pinecone environment (default: auto-detected)
# - LOG_LEVEL: Logging level (default: INFO)