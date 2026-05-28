import os
import sys
import importlib.util

_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)

_spec = importlib.util.spec_from_file_location(
    "_sito_wsgi", os.path.join(_parent, "wsgi.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
application = _mod.application
