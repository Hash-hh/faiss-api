# gunicorn.conf.py
bind = "0.0.0.0:5000"
workers = 1  # Single worker for GPU usage
threads = 4  # Multiple threads for concurrent requests
worker_class = "gthread"
timeout = 300
keepalive = 2

# Remove all hooks - no post_fork, no on_starting