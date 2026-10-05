"""
gunicorn.conf.py
================
Production Gunicorn configuration for the Financial Intelligence Platform dashboard.
"""

import os
import multiprocessing

# Server socket
port = os.environ.get("PORT", "8050")
bind = f"0.0.0.0:{port}"

# Worker processes
# Formula: 2 * CPU cores + 1  (recommended for I/O-bound workloads)
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
threads = 2

# Timeouts
timeout = 120          # Worker silent timeout (seconds)
keepalive = 5          # Keep-alive connection timeout

# Logging
accesslog = "-"        # stdout
errorlog = "-"         # stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "fip_dashboard"

# Graceful restart
max_requests = 1000
max_requests_jitter = 100

# Security: limit request line and headers
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
