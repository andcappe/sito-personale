import os
import sys
import importlib.util
import secrets
import smtplib
import urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Carica .env se presente (in sviluppo locale)
_dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_dotenv_path):
    with open(_dotenv_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _, _v = _line.partition('=')
                os.environ.setdefault(_k.strip(), _v.strip())

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─── Storage persistente (S3/R2): scarica i dati dal bucket PRIMA di montare le
#     app, così sopravvivono ai deploy/restart del filesystem effimero di DO.
#     No-op se le env vars S3_* non sono presenti (es. sviluppo locale).
try:
    import cloud_storage
    if cloud_storage.enabled():
        cloud_storage.pull_all()
    else:
        print("• [cloud] storage persistente disattivato (nessuna env var S3_*) — uso disco locale", flush=True)
except Exception as _e:
    print(f"⚠ [cloud] init fallito (uso disco locale): {_e}", flush=True)


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


def _safe_load(module_name, folder):
    try:
        return _load(module_name, folder)
    except Exception as _e:
        import traceback as _tb
        print(f"[ERROR] {folder} non caricata: {_e}", flush=True)
        _tb.print_exc()
        return None

portafoglio_srv = _safe_load("_app_portafoglio", "portafoglio")
macro_srv       = _safe_load("_app_macro",       "macro")
frontiera_srv   = _safe_load("_app_frontiera",   "frontiera-efficiente")
rendimenti_srv  = _safe_load("_app_rendimenti",  "rendimenti")
opzioni_srv     = _safe_load("_app_opzioni",     "opzioni")

# ─── Autenticazione ───────────────────────────────────────────────────────────
from auth import (check_credentials, register_user, register_oauth_user,
                  get_user, list_users, update_user, delete_user,
                  create_reset_token, verify_reset_token, consume_reset_token,
                  create_verify_token, consume_verify_token)

SECRET_KEY = os.environ.get('SECRET_KEY', 'cambia-questa-chiave-in-produzione')

# ─── Configurazione email e OAuth ─────────────────────────────────────────────
MAIL_FROM            = os.environ.get('MAIL_FROM', '')
MAIL_PASSWORD        = os.environ.get('MAIL_PASSWORD', '')
MAIL_SMTP_HOST       = os.environ.get('MAIL_SMTP_HOST', 'smtp-relay.brevo.com')
MAIL_SMTP_PORT       = int(os.environ.get('MAIL_SMTP_PORT', '587'))
MAIL_SMTP_USER       = os.environ.get('MAIL_SMTP_USER', '')
APP_URL              = os.environ.get('APP_URL', 'http://localhost:8080')
GOOGLE_CLIENT_ID     = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
FACEBOOK_APP_ID      = os.environ.get('FACEBOOK_APP_ID', '')
FACEBOOK_APP_SECRET  = os.environ.get('FACEBOOK_APP_SECRET', '')
ADMIN_EMAIL          = os.environ.get('ADMIN_EMAIL', 'admin@dashboard.local')
ADMIN_PASSWORD       = os.environ.get('ADMIN_PASSWORD', 'Cambia.Subito.123')

for _srv in (portafoglio_srv, macro_srv, frontiera_srv, rendimenti_srv, opzioni_srv):
    if _srv:
        _srv.secret_key = SECRET_KEY

# Percorsi esatti che non richiedono autenticazione
_PUBLIC_EXACT = {
    '/', '/foto.png',
    '/login', '/logout', '/register', '/suspended', '/setup',
    '/forgot-password',
    '/auth/google', '/auth/google/callback',
    '/auth/facebook', '/auth/facebook/callback',
}
# Prefissi dinamici pubblici (token variabile)
_PUBLIC_RESET_PREFIX  = '/reset-password/'
_PUBLIC_VERIFY_PREFIX = '/verify-email/'
# Prefissi che non richiedono autenticazione
_PUBLIC_PREFIXES = (
    '/_dash', '/assets/', '/_reload',
    '/portafoglio/_dash', '/frontiera/_dash', '/macro/_dash', '/rendimenti/_dash', '/opzioni/_dash',
    '/portafoglio/assets', '/frontiera/assets', '/macro/assets', '/rendimenti/assets', '/opzioni/assets',
)

# ─── Template HTML comune ─────────────────────────────────────────────────────

_BASE_STYLE = """\
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: linear-gradient(135deg, #1a3a5c 0%, #0d2137 100%);
      min-height: 100vh;
      display: flex; align-items: center; justify-content: center;
      padding: 20px;
    }
    .card {
      background: #fff; border-radius: 12px;
      padding: 40px 44px; width: 100%; max-width: 400px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.28);
    }
    .logo { text-align: center; margin-bottom: 28px; }
    .logo h1 { font-size: 22px; color: #1a3a5c; font-weight: 700; }
    .logo p  { font-size: 12px; color: #888; margin-top: 4px; }
    label { display: block; font-size: 12px; color: #555;
            font-weight: 600; margin-bottom: 4px; }
    input[type=text], input[type=password], input[type=email] {
      width: 100%; padding: 10px 12px; margin-bottom: 16px;
      border: 1px solid #dde3ec; border-radius: 6px;
      font-size: 14px; color: #222;
      background: #fff;
      transition: border-color .2s;
    }
    input:focus { outline: none; border-color: #1a3a5c; }
    .btn {
      width: 100%; padding: 12px; background: #1a3a5c; color: #fff;
      border: none; border-radius: 6px; font-size: 15px;
      font-weight: 700; cursor: pointer; letter-spacing: .4px;
      transition: background .2s; text-decoration: none;
      display: block; text-align: center; margin-top: 4px;
    }
    .btn:hover { background: #254e7a; }
    .btn-secondary {
      background: #6c757d; margin-top: 10px;
    }
    .btn-secondary:hover { background: #545b62; }
    .error {
      background: #fdecea; color: #c0392b; border: 1px solid #f5c6cb;
      border-radius: 6px; padding: 9px 12px;
      font-size: 13px; margin-bottom: 16px;
    }
    .success {
      background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7;
      border-radius: 6px; padding: 9px 12px;
      font-size: 13px; margin-bottom: 16px;
    }
    .footer { text-align: center; font-size: 11px; color: #aaa; margin-top: 20px; }
    .link { font-size: 13px; color: #555; text-align: center;
            display: block; margin-top: 16px; }
    .link a { color: #1a3a5c; font-weight: 700; text-decoration: underline; }
    .forgot { text-align: right; margin-top: -10px; margin-bottom: 14px; }
    .forgot a { font-size: 12px; color: #888; text-decoration: none; }
    .forgot a:hover { color: #1a3a5c; text-decoration: underline; }
    .divider { display: flex; align-items: center; gap: 10px;
               margin: 18px 0; color: #aaa; font-size: 12px; }
    .divider::before, .divider::after {
      content: ''; flex: 1; height: 1px; background: #e0e6ef;
    }
    .btn-oauth {
      width: 100%; padding: 10px 12px; border-radius: 6px; font-size: 14px;
      font-weight: 600; cursor: pointer; border: 1.5px solid #dde3ec;
      display: flex; align-items: center; justify-content: center; gap: 10px;
      text-decoration: none; margin-bottom: 10px; transition: background .15s;
      background: #fff; color: #333;
    }
    .btn-oauth:hover { background: #f5f7fa; }
    .btn-google  { border-color: #dde3ec; }
    .btn-facebook { border-color: #dde3ec; }
    .oauth-icon { width: 20px; height: 20px; display: inline-block; }
"""

_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Accesso – Dashboard Finanziaria</title>
  <style>
__STYLE__
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">
      <h1>A·C Dashboard</h1>
      <p>Analisi Rischi di Portafoglio</p>
    </div>
    __MSG__
    <form method="post">
      <label for="email">Email</label>
      <input id="email" name="email" type="email"
             autocomplete="email" autofocus placeholder="tua@email.com">
      <label for="password">Password</label>
      <input id="password" name="password" type="password"
             autocomplete="current-password" placeholder="••••••••">
      <div class="forgot"><a href="/forgot-password" id="forgot-link" onclick="var e=document.getElementById('email').value;if(e)this.href='/forgot-password?email='+encodeURIComponent(e)">Password dimenticata?</a></div>
      <button class="btn" type="submit">Accedi</button>
    </form>
    __OAUTH__
    <p class="link">Non hai un account? <a href="/register">Registrati</a></p>
    <p class="footer">Sessione protetta · Solo utenti autorizzati</p>
  </div>
</body>
</html>
""".replace('__STYLE__', _BASE_STYLE)

_REGISTER_HTML = """\
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Registrazione – Dashboard Finanziaria</title>
  <style>
__STYLE__
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">
      <h1>A·C Dashboard</h1>
      <p>Crea il tuo account</p>
    </div>
    __MSG__
    <form method="post">
      <label for="email">Email</label>
      <input id="email" name="email" type="email"
             autocomplete="email" autofocus placeholder="tua@email.com"
             value="__EMAIL__">
      <label for="password">Password</label>
      <input id="password" name="password" type="password"
             autocomplete="new-password" placeholder="min. 8 caratteri">
      <label for="confirm">Conferma password</label>
      <input id="confirm" name="confirm" type="password"
             autocomplete="new-password" placeholder="ripeti la password">
      <button class="btn" type="submit">Registrati</button>
    </form>
    <p class="link">Hai già un account? <a href="/login">Accedi</a></p>
    <p class="footer">Accesso gratuito · Funzioni avanzate disponibili in futuro</p>
  </div>
</body>
</html>
""".replace('__STYLE__', _BASE_STYLE)

_FORGOT_HTML = """\
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Recupera password – Dashboard Finanziaria</title>
  <style>
__STYLE__
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">
      <h1>A·C Dashboard</h1>
      <p>Recupera la tua password</p>
    </div>
    __MSG__
    <form method="post">
      <label for="email">Email del tuo account</label>
      <input id="email" name="email" type="email"
             autocomplete="email" autofocus placeholder="tua@email.com"
             value="__EMAIL__">
      <button class="btn" type="submit">Invia link di recupero</button>
    </form>
    <p class="link"><a href="/login">← Torna al login</a></p>
  </div>
</body>
</html>
""".replace('__STYLE__', _BASE_STYLE)

_RESET_HTML = """\
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nuova password – Dashboard Finanziaria</title>
  <style>
__STYLE__
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">
      <h1>A·C Dashboard</h1>
      <p>Imposta una nuova password</p>
    </div>
    __MSG__
    <form method="post">
      <label for="password">Nuova password</label>
      <input id="password" name="password" type="password"
             autocomplete="new-password" placeholder="min. 8 caratteri">
      <label for="confirm">Conferma password</label>
      <input id="confirm" name="confirm" type="password"
             autocomplete="new-password" placeholder="ripeti la password">
      <button class="btn" type="submit">Salva nuova password</button>
    </form>
  </div>
</body>
</html>
""".replace('__STYLE__', _BASE_STYLE)

_SUSPENDED_HTML = """\
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Account sospeso – Dashboard Finanziaria</title>
  <style>
__STYLE__
    .icon { font-size: 48px; text-align: center; margin-bottom: 16px; }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">
      <h1>A·C Dashboard</h1>
    </div>
    <div class="icon">🔒</div>
    <div class="error" style="text-align:center">
      Il tuo account è stato sospeso.<br>
      Contatta l'amministratore per maggiori informazioni.
    </div>
    <a class="btn btn-secondary" href="/logout">Esci</a>
  </div>
</body>
</html>
""".replace('__STYLE__', _BASE_STYLE)

_ADMIN_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Admin – Dashboard Finanziaria</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif;
           background: #0d2137; color: #e0e8f0; min-height: 100vh; }
    .topbar {
      background: #1a3a5c; padding: 12px 24px;
      display: flex; align-items: center; justify-content: space-between;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .topbar h1 { font-size: 18px; font-weight: 700; }
    .topbar a { color: #90b8d8; font-size: 13px; text-decoration: none; }
    .topbar a:hover { color: #fff; }
    .container { padding: 32px 24px; max-width: 1100px; margin: 0 auto; }
    h2 { font-size: 15px; font-weight: 600; margin-bottom: 16px;
         color: #90b8d8; text-transform: uppercase; letter-spacing: .6px; }
    table { width: 100%; border-collapse: collapse; background: #1a3a5c;
            border-radius: 8px; overflow: hidden; font-size: 13px; }
    th { background: #0d2137; padding: 10px 14px; text-align: left;
         color: #90b8d8; font-weight: 600; }
    td { padding: 10px 14px; border-top: 1px solid #254e7a; vertical-align: middle; }
    tr:hover td { background: rgba(255,255,255,0.04); }
    select { background: #0d2137; color: #e0e8f0; border: 1px solid #254e7a;
             border-radius: 4px; padding: 4px 8px; font-size: 12px; cursor: pointer; }
    .badge {
      display: inline-block; padding: 2px 8px; border-radius: 10px;
      font-size: 11px; font-weight: 600;
    }
    .badge-active   { background: #1b5e20; color: #a5d6a7; }
    .badge-pending  { background: #4a3800; color: #ffe082; }
    .badge-suspended{ background: #5c0000; color: #ef9a9a; }
    .btn-sm {
      padding: 4px 10px; border: none; border-radius: 4px;
      font-size: 12px; font-weight: 600; cursor: pointer;
      color: #fff; margin-left: 4px;
    }
    .btn-save   { background: #1a6a3c; }
    .btn-save:hover { background: #22884e; }
    .btn-del    { background: #8b1c1c; }
    .btn-del:hover  { background: #b52828; }
    .msg { padding: 10px 16px; border-radius: 6px; margin-bottom: 20px;
           font-size: 13px; }
    .msg-ok  { background: #1b5e20; color: #a5d6a7; }
    .msg-err { background: #5c0000; color: #ef9a9a; }
    .stats { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
    .stat-card {
      background: #1a3a5c; border-radius: 8px; padding: 16px 24px;
      flex: 1; min-width: 120px;
    }
    .stat-card .num { font-size: 28px; font-weight: 700; color: #90d8f8; }
    .stat-card .lbl { font-size: 11px; color: #90b8d8; margin-top: 4px; }
  </style>
</head>
<body>
"""


def _is_public(path: str) -> bool:
    return (path in _PUBLIC_EXACT
            or path.startswith(_PUBLIC_RESET_PREFIX)
            or path.startswith(_PUBLIC_VERIFY_PREFIX)
            or any(path.startswith(p) for p in _PUBLIC_PREFIXES))


# ─── Invio email ───────────────────────────────────────────────────────────────

def _send_reset_email(to_email: str, token: str) -> bool:
    if not MAIL_FROM or not MAIL_PASSWORD:
        return False
    smtp_user = MAIL_SMTP_USER or MAIL_FROM
    reset_url = f"{APP_URL}/reset-password/{token}"
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Recupero password – A·C Dashboard'
    msg['From']    = MAIL_FROM
    msg['To']      = to_email
    body = f"""\
<html><body style="font-family:Arial,sans-serif;color:#222;max-width:480px;margin:0 auto">
  <h2 style="color:#1a3a5c">A·C Dashboard – Recupero password</h2>
  <p>Hai richiesto il reset della password. Clicca il pulsante qui sotto:</p>
  <p style="margin:24px 0">
    <a href="{reset_url}" style="background:#1a3a5c;color:#fff;padding:12px 24px;
       border-radius:6px;text-decoration:none;font-weight:700">
      Reimposta password
    </a>
  </p>
  <p style="font-size:13px;color:#888">
    Il link scade tra <strong>1 ora</strong>.<br>
    Se non hai richiesto il reset, ignora questa email.
  </p>
</body></html>"""
    msg.attach(MIMEText(body, 'html'))
    try:
        if MAIL_SMTP_PORT == 465:
            with smtplib.SMTP_SSL(MAIL_SMTP_HOST, 465) as srv:
                srv.login(smtp_user, MAIL_PASSWORD)
                srv.sendmail(MAIL_FROM, to_email, msg.as_string())
        else:
            with smtplib.SMTP(MAIL_SMTP_HOST, MAIL_SMTP_PORT) as srv:
                srv.ehlo()
                srv.starttls()
                srv.login(smtp_user, MAIL_PASSWORD)
                srv.sendmail(MAIL_FROM, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[MAIL ERROR] {e}", flush=True)
        return False


def _send_verify_email(to_email: str, token: str) -> bool:
    if not MAIL_FROM or not MAIL_PASSWORD:
        return False
    smtp_user = MAIL_SMTP_USER or MAIL_FROM
    verify_url = f"{APP_URL}/verify-email/{token}"
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Conferma la tua email – A·C Dashboard'
    msg['From']    = MAIL_FROM
    msg['To']      = to_email
    body = f"""\
<html><body style="font-family:Arial,sans-serif;color:#222;max-width:480px;margin:0 auto">
  <h2 style="color:#1a3a5c">A·C Dashboard – Conferma email</h2>
  <p>Grazie per esserti registrato. Clicca il pulsante per attivare il tuo account:</p>
  <p style="margin:24px 0">
    <a href="{verify_url}" style="background:#1a3a5c;color:#fff;padding:12px 24px;
       border-radius:6px;text-decoration:none;font-weight:700">
      Conferma email
    </a>
  </p>
  <p style="font-size:13px;color:#888">
    Il link scade tra <strong>24 ore</strong>.<br>
    Se non ti sei registrato, ignora questa email.
  </p>
</body></html>"""
    msg.attach(MIMEText(body, 'html'))
    try:
        if MAIL_SMTP_PORT == 465:
            with smtplib.SMTP_SSL(MAIL_SMTP_HOST, 465) as srv:
                srv.login(smtp_user, MAIL_PASSWORD)
                srv.sendmail(MAIL_FROM, to_email, msg.as_string())
        else:
            with smtplib.SMTP(MAIL_SMTP_HOST, MAIL_SMTP_PORT) as srv:
                srv.ehlo()
                srv.starttls()
                srv.login(smtp_user, MAIL_PASSWORD)
                srv.sendmail(MAIL_FROM, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[MAIL ERROR] {e}", flush=True)
        return False


def _register_auth(flask_server, add_login_routes: bool = False):
    """Aggiunge controllo autenticazione a un Flask server Dash."""
    from flask import request, session, redirect, send_from_directory

    if add_login_routes:

        _PROFILO_DIR = os.path.join(ROOT, 'profilo')

        @flask_server.route('/')
        def _home():
            with open(os.path.join(_PROFILO_DIR, 'index.html'), 'r', encoding='utf-8') as f:
                content = f.read()
            return content, 200, {
                'Content-Type': 'text/html; charset=utf-8',
                'Cache-Control': 'no-store',
            }

        @flask_server.route('/foto.png')
        def _foto():
            return send_from_directory(_PROFILO_DIR, 'foto.png')



        @flask_server.route('/login', methods=['GET', 'POST'])
        def _login():
            msg = ''
            if request.method == 'POST':
                u = request.form.get('email', '').strip().lower()
                p = request.form.get('password', '')
                if check_credentials(u, p):
                    user = get_user(u)
                    if user and user.get('status') == 'suspended':
                        return redirect('/suspended')
                    if user and user.get('status') == 'pending':
                        token = create_verify_token(u)
                        _send_verify_email(u, token)
                        msg = '<div class="error">Devi confermare la tua email prima di accedere. Ti abbiamo inviato un nuovo link di conferma.</div>'
                    else:
                        session['username'] = u
                        next_url = request.args.get('next', '/portafoglio/')
                        return redirect(next_url)
                else:
                    msg = '<div class="error">Email o password non corretti.</div>'
            # Blocco OAuth dinamico (solo se credenziali configurate)
            oauth_html = '<div class="divider">oppure</div>'
            if GOOGLE_CLIENT_ID:
                next_q = urllib.parse.quote(request.args.get('next', '/'))
                oauth_html += f'''
                <a href="/auth/google?next={next_q}" class="btn-oauth btn-google">
                  <svg class="oauth-icon" viewBox="0 0 48 48">
                    <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9 3.2l6.7-6.7C35.7 2.5 30.2 0 24 0 14.7 0 6.7 5.4 2.8 13.3l7.8 6C12.4 13 17.8 9.5 24 9.5z"/>
                    <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v8.5h12.7c-.6 3-2.3 5.5-4.8 7.2l7.5 5.8c4.4-4.1 7.1-10.1 7.1-17z"/>
                    <path fill="#FBBC05" d="M10.6 28.7A14.6 14.6 0 019.5 24c0-1.6.3-3.2.8-4.7l-7.8-6A23.9 23.9 0 000 24c0 3.8.9 7.5 2.5 10.7l8.1-6z"/>
                    <path fill="#34A853" d="M24 48c6.2 0 11.4-2 15.2-5.5l-7.5-5.8c-2 1.4-4.6 2.2-7.7 2.2-6.2 0-11.5-4.2-13.4-9.8l-8 6.1C6.7 42.6 14.7 48 24 48z"/>
                  </svg>
                  Continua con Google
                </a>'''
            if FACEBOOK_APP_ID:
                next_q = urllib.parse.quote(request.args.get('next', '/'))
                oauth_html += f'''
                <a href="/auth/facebook?next={next_q}" class="btn-oauth btn-facebook">
                  <svg class="oauth-icon" viewBox="0 0 24 24" fill="#1877F2">
                    <path d="M24 12.07C24 5.41 18.63 0 12 0S0 5.41 0 12.07C0 18.1 4.39 23.1 10.13 24v-8.44H7.08v-3.49h3.04V9.41c0-3.02 1.8-4.7 4.54-4.7 1.31 0 2.68.24 2.68.24v2.97h-1.51c-1.49 0-1.95.93-1.95 1.88v2.27h3.32l-.53 3.5h-2.79V24C19.61 23.1 24 18.1 24 12.07z"/>
                  </svg>
                  Continua con Facebook
                </a>'''
            if not GOOGLE_CLIENT_ID and not FACEBOOK_APP_ID:
                oauth_html = ''
            return (_LOGIN_HTML
                    .replace('__MSG__', msg)
                    .replace('__OAUTH__', oauth_html))

        @flask_server.route('/logout')
        def _logout():
            session.clear()
            return redirect('/login')

        @flask_server.route('/register', methods=['GET', 'POST'])
        def _register():
            msg = ''
            email_val = ''
            if request.method == 'POST':
                email_val = request.form.get('email', '').strip()
                password  = request.form.get('password', '')
                confirm   = request.form.get('confirm', '')
                if password != confirm:
                    msg = '<div class="error">Le password non coincidono.</div>'
                else:
                    ok, message = register_user(email_val, password)
                    if ok:
                        token = create_verify_token(email_val.strip().lower())
                        _send_verify_email(email_val.strip().lower(), token)
                        return (_REGISTER_HTML
                                .replace('__MSG__', '<div class="success">Registrazione completata! Controlla la tua email per confermare l\'account.</div>')
                                .replace('__EMAIL__', ''))
                    else:
                        msg = f'<div class="error">{message}</div>'
            return (_REGISTER_HTML
                    .replace('__MSG__', msg)
                    .replace('__EMAIL__', email_val))

        @flask_server.route('/verify-email/<token>')
        def _verify_email(token):
            email = consume_verify_token(token)
            if email:
                session['username'] = email
                return redirect('/portafoglio/?verified=1')
            return (_FORGOT_HTML
                    .replace('__MSG__', '<div class="error">Link non valido o scaduto. <a href="/register">Registrati di nuovo</a>.</div>')
                    .replace('__EMAIL__', ''))

        @flask_server.route('/suspended')
        def _suspended():
            return _SUSPENDED_HTML

        @flask_server.route('/setup', methods=['GET', 'POST'])
        def _setup():
            msg = ''
            if request.method == 'POST':
                email      = request.form.get('email', '').strip().lower()
                secret     = request.form.get('secret', '')
                if secret != ADMIN_PASSWORD:
                    msg = '<div class="error">Codice segreto non valido.</div>'
                elif not email:
                    msg = '<div class="error">Inserisci un\'email.</div>'
                else:
                    _add_user(email, ADMIN_PASSWORD, role='admin')
                    _upd_user(email, status='active', plan='admin')
                    session['username'] = email
                    return redirect('/admin')
            return f'''<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8">
<title>Setup Admin</title><style>{_BASE_STYLE}</style></head>
<body><div class="card">
<div class="logo"><h1>A·C Dashboard</h1><p>Promozione account admin</p></div>
{msg}
<form method="post">
  <label>Email account da promuovere</label>
  <input name="email" type="email" placeholder="tua@email.com" autofocus>
  <label>Codice segreto</label>
  <input name="secret" type="password" placeholder="ADMIN_PASSWORD da DigitalOcean">
  <button class="btn" type="submit">Promuovi ad admin</button>
</form>
<p class="footer">Accessibile solo a chi conosce il codice segreto.</p>
</div></body></html>'''

        @flask_server.route('/forgot-password', methods=['GET', 'POST'])
        def _forgot_password():
            msg = ''
            email_val = request.args.get('email', '').strip().lower()
            if request.method == 'POST':
                email_val = request.form.get('email', '').strip().lower()
                token = create_reset_token(email_val)
                if not MAIL_FROM or not MAIL_PASSWORD:
                    msg = '<div class="error">Email non configurata sul server. Contatta l\'amministratore.</div>'
                else:
                    if token:
                        sent = _send_reset_email(email_val, token)
                        if not sent:
                            msg = '<div class="error">Errore nell\'invio email. Riprova tra poco o contatta l\'amministratore.</div>'
                        else:
                            msg = '<div class="success">Se l\'email è registrata riceverai un link entro pochi minuti.</div>'
                            email_val = ''
                    else:
                        # Email non trovata — mostra successo per non rivelare se l'utente esiste
                        msg = '<div class="success">Se l\'email è registrata riceverai un link entro pochi minuti.</div>'
                        email_val = ''
            return (_FORGOT_HTML
                    .replace('__MSG__', msg)
                    .replace('__EMAIL__', email_val))

        @flask_server.route('/reset-password/<token>', methods=['GET', 'POST'])
        def _reset_password(token):
            email = verify_reset_token(token)
            if not email:
                return (_FORGOT_HTML
                        .replace('__MSG__', '<div class="error">Link scaduto o non valido. Richiedi un nuovo link.</div>')
                        .replace('__EMAIL__', ''))
            msg = ''
            if request.method == 'POST':
                password = request.form.get('password', '')
                confirm  = request.form.get('confirm', '')
                if len(password) < 8:
                    msg = '<div class="error">La password deve avere almeno 8 caratteri.</div>'
                elif password != confirm:
                    msg = '<div class="error">Le password non coincidono.</div>'
                else:
                    consume_reset_token(token, password)
                    return redirect('/login?msg=password_reset')
            return _RESET_HTML.replace('__MSG__', msg)

        # ─── Google OAuth ──────────────────────────────────────────────────────
        @flask_server.route('/auth/google')
        def _auth_google():
            if not GOOGLE_CLIENT_ID:
                return redirect('/login')
            session['oauth_next'] = request.args.get('next', '/portafoglio/')
            state = secrets.token_urlsafe(16)
            session['oauth_state'] = state
            params = urllib.parse.urlencode({
                'client_id':     GOOGLE_CLIENT_ID,
                'redirect_uri':  f"{APP_URL}/auth/google/callback",
                'response_type': 'code',
                'scope':         'openid email profile',
                'state':         state,
            })
            return redirect(f"https://accounts.google.com/o/oauth2/auth?{params}")

        @flask_server.route('/auth/google/callback')
        def _auth_google_callback():
            import requests as _req
            if request.args.get('state') != session.pop('oauth_state', None):
                return redirect('/login')
            code = request.args.get('code')
            if not code:
                return redirect('/login')
            token_r = _req.post('https://oauth2.googleapis.com/token', data={
                'code':          code,
                'client_id':     GOOGLE_CLIENT_ID,
                'client_secret': GOOGLE_CLIENT_SECRET,
                'redirect_uri':  f"{APP_URL}/auth/google/callback",
                'grant_type':    'authorization_code',
            }, timeout=10)
            access_token = token_r.json().get('access_token')
            if not access_token:
                return redirect('/login')
            info_r = _req.get('https://www.googleapis.com/oauth2/v2/userinfo',
                              headers={'Authorization': f'Bearer {access_token}'}, timeout=10)
            email = info_r.json().get('email', '').lower()
            if not email:
                return redirect('/login')
            register_oauth_user(email, 'google')
            user = get_user(email)
            if user and user.get('status') == 'suspended':
                return redirect('/suspended')
            session['username'] = email
            return redirect(session.pop('oauth_next', '/portafoglio/'))

        # ─── Facebook OAuth ────────────────────────────────────────────────────
        @flask_server.route('/auth/facebook')
        def _auth_facebook():
            if not FACEBOOK_APP_ID:
                return redirect('/login')
            session['oauth_next'] = request.args.get('next', '/portafoglio/')
            params = urllib.parse.urlencode({
                'client_id':    FACEBOOK_APP_ID,
                'redirect_uri': f"{APP_URL}/auth/facebook/callback",
                'scope':        'email',
                'response_type': 'code',
            })
            return redirect(f"https://www.facebook.com/dialog/oauth?{params}")

        @flask_server.route('/auth/facebook/callback')
        def _auth_facebook_callback():
            import requests as _req
            code = request.args.get('code')
            if not code:
                return redirect('/login')
            token_r = _req.get('https://graph.facebook.com/v18.0/oauth/access_token', params={
                'client_id':     FACEBOOK_APP_ID,
                'redirect_uri':  f"{APP_URL}/auth/facebook/callback",
                'client_secret': FACEBOOK_APP_SECRET,
                'code':          code,
            }, timeout=10)
            access_token = token_r.json().get('access_token')
            if not access_token:
                return redirect('/login')
            info_r = _req.get('https://graph.facebook.com/me',
                              params={'fields': 'email', 'access_token': access_token}, timeout=10)
            email = info_r.json().get('email', '').lower()
            if not email:
                return redirect('/login?error=fb_no_email')
            register_oauth_user(email, 'facebook')
            user = get_user(email)
            if user and user.get('status') == 'suspended':
                return redirect('/suspended')
            session['username'] = email
            return redirect(session.pop('oauth_next', '/portafoglio/'))

        @flask_server.route('/admin', methods=['GET', 'POST'])
        def _admin():
            u = session.get('username')
            user = get_user(u) if u else None
            if not user or user.get('role') != 'admin':
                return redirect('/')

            msg_html = ''

            if request.method == 'POST':
                action = request.form.get('action', '')
                target = request.form.get('target_username', '')
                if action == 'delete' and target and target != u:
                    delete_user(target)
                    msg_html = f'<div class="msg msg-ok">Utente <b>{target}</b> eliminato.</div>'
                elif action == 'update' and target:
                    new_status = request.form.get('status', '')
                    new_plan   = request.form.get('plan', '')
                    new_role   = request.form.get('role', '')
                    new_pw     = request.form.get('new_password', '').strip()
                    kwargs = {}
                    if new_status: kwargs['status'] = new_status
                    if new_plan:   kwargs['plan']   = new_plan
                    if new_role:   kwargs['role']   = new_role
                    if new_pw:
                        if len(new_pw) < 8:
                            msg_html = '<div class="msg msg-err">Password minimo 8 caratteri.</div>'
                        else:
                            import hashlib as _hl
                            kwargs['password_hash'] = _hl.sha256(f"{target}:{new_pw}".encode()).hexdigest()
                    if not msg_html:
                        update_user(target, **kwargs)
                        msg_html = f'<div class="msg msg-ok">Utente <b>{target}</b> aggiornato.</div>'

            users = list_users()
            total     = len(users)
            active    = sum(1 for x in users if x['status'] == 'active')
            pending   = sum(1 for x in users if x['status'] == 'pending')
            suspended = sum(1 for x in users if x['status'] == 'suspended')

            rows = ''
            for usr in users:
                uname   = usr['username']
                status  = usr['status']
                plan    = usr['plan']
                role    = usr['role']
                email   = uname
                created = usr['created_at']

                badge_cls = {
                    'active':    'badge-active',
                    'pending':   'badge-pending',
                    'suspended': 'badge-suspended',
                }.get(status, 'badge-active')

                def _sel(name, options, current):
                    opts = ''.join(
                        f'<option value="{v}" {"selected" if v == current else ""}>{v}</option>'
                        for v in options
                    )
                    return f'<select name="{name}">{opts}</select>'

                protect = 'disabled' if uname == u else ''
                rows += f"""
                <tr>
                  <td><b>{uname}</b></td>
                  <td>{email}</td>
                  <td><span class="badge {badge_cls}">{status}</span></td>
                  <td>{plan}</td>
                  <td>{role}</td>
                  <td>{created}</td>
                  <td>
                    <form method="post" style="display:inline">
                      <input type="hidden" name="action" value="update">
                      <input type="hidden" name="target_username" value="{uname}">
                      {_sel('status', ['active','pending','suspended'], status)}
                      {_sel('plan',   ['free','premium','admin'], plan)}
                      {_sel('role',   ['user','admin'], role)}
                      <input type="password" name="new_password" placeholder="Nuova password (opz.)"
                             style="width:140px;padding:3px 6px;font-size:0.78rem;border:1px solid #ccd9ee;border-radius:4px">
                      <button class="btn-sm btn-save" type="submit" {protect}>Salva</button>
                    </form>
                    <form method="post" style="display:inline"
                          onsubmit="return confirm('Eliminare {uname}?')">
                      <input type="hidden" name="action" value="delete">
                      <input type="hidden" name="target_username" value="{uname}">
                      <button class="btn-sm btn-del" type="submit" {protect}>Elimina</button>
                    </form>
                  </td>
                </tr>"""

            html = _ADMIN_HTML_HEAD + f"""
  <div class="topbar">
    <h1>A·C Dashboard — Admin Panel</h1>
    <div>
      <a href="/">← Dashboard</a>
      &nbsp;&nbsp;
      <a href="/logout">Esci ({u})</a>
    </div>
  </div>
  <div class="container">
    {msg_html}
    <div class="stats">
      <div class="stat-card"><div class="num">{total}</div><div class="lbl">Utenti totali</div></div>
      <div class="stat-card"><div class="num">{active}</div><div class="lbl">Attivi</div></div>
      <div class="stat-card"><div class="num">{pending}</div><div class="lbl">In attesa</div></div>
      <div class="stat-card"><div class="num">{suspended}</div><div class="lbl">Sospesi</div></div>
    </div>
    <h2>Gestione Utenti</h2>
    <table>
      <thead>
        <tr>
          <th>Username</th><th>Email</th><th>Stato</th>
          <th>Piano</th><th>Ruolo</th><th>Registrato</th><th>Azioni</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</body>
</html>"""
            return html

    @flask_server.before_request
    def _require_login():
        if _is_public(request.path):
            return None
        username = session.get('username')
        if not username:
            return redirect(f'/login?next={request.path}')
        user = get_user(username)
        if user and user.get('status') == 'suspended':
            return redirect('/suspended')


if portafoglio_srv:
    _register_auth(portafoglio_srv, add_login_routes=True)
if macro_srv:
    _register_auth(macro_srv)
if frontiera_srv:
    _register_auth(frontiera_srv)
if rendimenti_srv:
    _register_auth(rendimenti_srv)
if opzioni_srv:
    _register_auth(opzioni_srv)

# ─── Bootstrap admin di default ───────────────────────────────────────────────
from auth import add_user as _add_user, update_user as _upd_user, list_users as _list_users, get_user as _get_user
_has_admin = any(u.get('role') == 'admin' for u in _list_users())
if not _has_admin:
    _add_user(ADMIN_EMAIL, ADMIN_PASSWORD, role='admin')
    _upd_user(ADMIN_EMAIL, status='active', plan='admin')
    print(f"[SETUP] Creato admin di default: {ADMIN_EMAIL}", flush=True)

# ─── Routing WSGI ─────────────────────────────────────────────────────────────
from werkzeug.exceptions import NotFound

_ROUTES = [
    *([("/portafoglio", portafoglio_srv)] if portafoglio_srv else []),
    *([("/macro",       macro_srv)]       if macro_srv       else []),
    *([("/frontiera",   frontiera_srv)]   if frontiera_srv   else []),
    *([("/rendimenti",  rendimenti_srv)]  if rendimenti_srv  else []),
    *([("/opzioni",     opzioni_srv)]     if opzioni_srv     else []),
    ("/",  portafoglio_srv or macro_srv or frontiera_srv),  # catch-all
]


def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    for prefix, app in _ROUTES:
        if prefix == "/" or path == prefix or path.startswith(prefix + "/"):
            return app(environ, start_response)
    return NotFound()(environ, start_response)
