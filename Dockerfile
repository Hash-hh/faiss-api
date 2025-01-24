# Use NVIDIA CUDA 12.2 base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Create directory for mounting FAISS index
RUN mkdir /app_data

# Set environment variables
ENV CUDA_LAUNCH_BLOCKING=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .

# Install PyTorch with CUDA 12.2 first
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Make sure templates directory exists
RUN mkdir -p /app/app/templates

# Copy gunicorn config
COPY gunicorn.conf.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app/main.py
ENV FLASK_ENV=production
ENV FAISS_INDEX_PATH=/app_data/faiss_index

# Command to run the application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app.main:app"]
