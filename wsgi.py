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

# ─── Autenticazione ───────────────────────────────────────────────────────────
from auth import check_credentials

SECRET_KEY = os.environ.get('SECRET_KEY', 'cambia-questa-chiave-in-produzione')

# Stessa chiave su tutti i server → condividono il cookie di sessione
for _srv in (portafoglio_srv, macro_srv, frontiera_srv):
    _srv.secret_key = SECRET_KEY

# Percorsi che non richiedono autenticazione
_PUBLIC_PREFIXES = (
    '/login', '/logout',
    '/_dash', '/assets/', '/_reload',
    '/portafoglio/_dash', '/frontiera/_dash', '/macro/_dash',
    '/portafoglio/assets', '/frontiera/assets', '/macro/assets',
)

_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Accesso – Dashboard Finanziaria</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: linear-gradient(135deg, #1a3a5c 0%, #0d2137 100%);
      min-height: 100vh;
      display: flex; align-items: center; justify-content: center;
    }
    .card {
      background: #fff; border-radius: 12px;
      padding: 40px 44px; width: 360px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.28);
    }
    .logo { text-align: center; margin-bottom: 28px; }
    .logo h1 { font-size: 22px; color: #1a3a5c; font-weight: 700; }
    .logo p  { font-size: 12px; color: #888; margin-top: 4px; }
    label { display: block; font-size: 12px; color: #555;
            font-weight: 600; margin-bottom: 4px; }
    input[type=text], input[type=password] {
      width: 100%; padding: 10px 12px; margin-bottom: 16px;
      border: 1px solid #dde3ec; border-radius: 6px;
      font-size: 14px; color: #1a3a5c;
      transition: border-color .2s;
    }
    input:focus { outline: none; border-color: #1a3a5c; }
    button {
      width: 100%; padding: 11px; background: #1a3a5c; color: #fff;
      border: none; border-radius: 6px; font-size: 14px;
      font-weight: 600; cursor: pointer; letter-spacing: .4px;
      transition: background .2s;
    }
    button:hover { background: #254e7a; }
    .error {
      background: #fdecea; color: #c0392b; border: 1px solid #f5c6cb;
      border-radius: 6px; padding: 9px 12px;
      font-size: 13px; margin-bottom: 16px;
    }
    .footer { text-align: center; font-size: 11px; color: #aaa; margin-top: 20px; }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">
      <h1>A·C Dashboard</h1>
      <p>Analisi Rischi di Portafoglio</p>
    </div>
    __ERROR__
    <form method="post">
      <label for="username">Username</label>
      <input id="username" name="username" type="text"
             autocomplete="username" autofocus placeholder="es. mario">
      <label for="password">Password</label>
      <input id="password" name="password" type="password"
             autocomplete="current-password" placeholder="••••••••">
      <button type="submit">Accedi</button>
    </form>
    <p class="footer">Sessione protetta · Solo utenti autorizzati</p>
  </div>
</body>
</html>
"""


def _is_public(path: str) -> bool:
    return any(path == p or path.startswith(p) for p in _PUBLIC_PREFIXES)


def _register_auth(flask_server, add_login_routes: bool = False):
    """Aggiunge controllo autenticazione a un Flask server Dash."""
    from flask import request, session, redirect

    if add_login_routes:
        @flask_server.route('/login', methods=['GET', 'POST'])
        def _login():
            error = ''
            if request.method == 'POST':
                u = request.form.get('username', '').strip()
                p = request.form.get('password', '')
                if check_credentials(u, p):
                    session['username'] = u
                    next_url = request.args.get('next', '/')
                    return redirect(next_url)
                error = 'Username o password non corretti.'
            err_html = (f'<div class="error">{error}</div>' if error else '')
            return _LOGIN_HTML.replace('__ERROR__', err_html)

        @flask_server.route('/logout')
        def _logout():
            session.clear()
            return redirect('/login')

    @flask_server.before_request
    def _require_login():
        if _is_public(request.path):
            return None
        if not session.get('username'):
            return redirect(f'/login?next={request.path}')


_register_auth(portafoglio_srv, add_login_routes=True)
_register_auth(macro_srv)
_register_auth(frontiera_srv)

# ─── Routing WSGI ─────────────────────────────────────────────────────────────
from werkzeug.exceptions import NotFound

_ROUTES = [
    ("/login",       portafoglio_srv),
    ("/logout",      portafoglio_srv),
    ("/portafoglio", portafoglio_srv),
    ("/macro",       macro_srv),
    ("/frontiera",   frontiera_srv),
    ("/",            portafoglio_srv),
]


def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    for prefix, app in _ROUTES:
        if prefix == "/" or path == prefix or path.startswith(prefix + "/"):
            return app(environ, start_response)
    return NotFound()(environ, start_response)
