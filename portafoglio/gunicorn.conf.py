import os
workers = 1
threads = 4
timeout = 120
keepalive = 5
bind = "0.0.0.0:" + os.environ.get("PORT", "8080")
