# gunicorn.conf.py
import os

# Server socket configuration
bind = "0.0.0.0:5000"
workers = 1  # Single worker for GPU operations
threads = 8  # Multiple threads for concurrent requests
worker_class = "gthread"

# Timeout configuration
timeout = 300
keepalive = 2

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'faiss-search-api'

# Environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
