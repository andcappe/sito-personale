import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load(module_name, folder):
    folder_path = os.path.join(ROOT, folder)
    sys.path.insert(0, folder_path)
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(folder_path, "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod.app.server


portafoglio_srv = _load("_app_portafoglio", "portafoglio")
macro_srv       = _load("_app_macro",       "macro")
frontiera_srv   = _load("_app_frontiera",   "frontiera-efficiente")

from werkzeug.exceptions import NotFound

_ROUTES = [
    ("/portafoglio", portafoglio_srv),
    ("/macro",       macro_srv),
    ("/frontiera",   frontiera_srv),
    ("/",            portafoglio_srv),  # fallback: serve profilo + rotte Flask root
]


def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    for prefix, app in _ROUTES:
        if prefix == "/" or path == prefix or path.startswith(prefix + "/"):
            return app(environ, start_response)
    return NotFound()(environ, start_response)
