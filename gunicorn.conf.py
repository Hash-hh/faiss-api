# import multiprocessing
#
# # Gunicorn config
# bind = "0.0.0.0:5000"
# workers = multiprocessing.cpu_count() * 2 + 1
# threads = 4
# worker_class = "gthread"  # Changed from sync
# timeout = 300
# keepalive = 2
#
# # Update this to point to the app factory
# wsgi_app = "app.main:app"


import multiprocessing

# Gunicorn config
bind = "0.0.0.0:5000"
# When using GPU, we want fewer workers
workers = 2  # Just use 2 workers when using GPU
threads = 2
worker_class = "gthread"  # Changed from "sync" to "gthread"
timeout = 300
keepalive = 2

# Preload the application
preload_app = True  # Add this line

def on_starting(server):
    """Initialize FAISS before any workers are created"""
    from app.main import init_faiss
    init_faiss()
