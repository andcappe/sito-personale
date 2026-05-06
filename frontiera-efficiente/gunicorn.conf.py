import os
workers = 1
threads = 4
timeout = 120
keepalive = 5
worker_tmp_dir = "/dev/shm"
bind = "0.0.0.0:" + os.environ.get("PORT", "8080")
