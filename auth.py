import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta

ROOT       = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(ROOT, 'users.json')

# Token in memoria (scadono in 1h; persi al riavvio, ma va bene)
_reset_tokens:  dict = {}
_verify_tokens: dict = {}


def _hash(key: str, password: str) -> str:
    return hashlib.sha256(f"{key}:{password}".encode()).hexdigest()


def _load_users() -> dict:
    try:
        with open(USERS_FILE) as f:
            data = json.load(f)
    except (FileNotFoundError, Exception):
        return {}

    migrated = False
    for key, val in list(data.items()):
        if isinstance(val, str):
            data[key] = {
                'password_hash': val,
                'role':          'admin' if key in ('admin', 'andcappe') else 'user',
                'status':        'active',
                'created_at':    datetime.now().isoformat(),
                'plan':          'admin' if key in ('admin', 'andcappe') else 'free',
            }
            migrated = True
    if migrated:
        _save_users(data)
    return data


def _save_users(data: dict) -> None:
    with open(USERS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── Autenticazione base ───────────────────────────────────────────────────────

def check_credentials(key: str, password: str) -> bool:
    users = _load_users()
    u = users.get(key.strip().lower())
    if not u:
        return False
    return u.get('password_hash') == _hash(key.strip().lower(), password)


def register_user(email: str, password: str) -> tuple:
    email = email.strip().lower()
    if not email or not password:
        return False, 'Tutti i campi sono obbligatori.'
    if '@' not in email or '.' not in email.split('@')[-1]:
        return False, 'Inserisci un indirizzo email valido.'
    if len(password) < 8:
        return False, 'La password deve avere almeno 8 caratteri.'
    users = _load_users()
    if email in users:
        return False, 'Email già registrata.'
    users[email] = {
        'password_hash': _hash(email, password),
        'role':          'user',
        'status':        'pending',
        'created_at':    datetime.now().isoformat(),
        'plan':          'free',
    }
    _save_users(users)
    return True, 'Registrazione completata!'


def register_oauth_user(email: str, provider: str) -> None:
    """Crea un utente OAuth se non esiste già. Password casuale (non usata)."""
    email = email.strip().lower()
    users = _load_users()
    if email not in users:
        users[email] = {
            'password_hash': _hash(email, secrets.token_urlsafe(32)),
            'role':          'user',
            'status':        'active',
            'created_at':    datetime.now().isoformat(),
            'plan':          'free',
            'oauth_provider': provider,
        }
        _save_users(users)


def get_user(key: str) -> dict | None:
    return _load_users().get(key.strip().lower())


def update_user(key: str, **kwargs) -> bool:
    users = _load_users()
    if key not in users:
        return False
    users[key].update(kwargs)
    _save_users(users)
    return True


def delete_user(key: str) -> bool:
    users = _load_users()
    if key not in users:
        return False
    del users[key]
    _save_users(users)
    return True


def list_users() -> list:
    users = _load_users()
    result = []
    for key, data in users.items():
        result.append({
            'username':   key,
            'role':       data.get('role', 'user'),
            'status':     data.get('status', 'active'),
            'plan':       data.get('plan', 'free'),
            'created_at': data.get('created_at', '')[:10],
        })
    result.sort(key=lambda x: x['created_at'], reverse=True)
    return result


def add_user(key: str, password: str, role: str = 'user') -> None:
    key = key.strip().lower()
    users = _load_users()
    existing = users.get(key, {})
    users[key] = {
        **existing,
        'password_hash': _hash(key, password),
        'role':          role,
        'status':        existing.get('status', 'active'),
        'created_at':    existing.get('created_at', datetime.now().isoformat()),
        'plan':          existing.get('plan', 'free'),
    }
    _save_users(users)


# ─── Verifica email ────────────────────────────────────────────────────────────

def create_verify_token(email: str) -> str:
    """Genera token di verifica email per un nuovo utente."""
    email = email.strip().lower()
    for t, v in list(_verify_tokens.items()):
        if v['email'] == email:
            del _verify_tokens[t]
    token = secrets.token_urlsafe(32)
    _verify_tokens[token] = {
        'email':      email,
        'expires_at': datetime.now() + timedelta(hours=24),
    }
    return token


def consume_verify_token(token: str) -> str | None:
    """Attiva l'account se il token è valido. Ritorna l'email o None."""
    entry = _verify_tokens.get(token)
    if not entry:
        return None
    if datetime.now() > entry['expires_at']:
        del _verify_tokens[token]
        return None
    email = entry['email']
    users = _load_users()
    if email not in users:
        return None
    users[email]['status'] = 'active'
    _save_users(users)
    _verify_tokens.pop(token, None)
    return email


# ─── Reset password ────────────────────────────────────────────────────────────

def create_reset_token(email: str) -> str | None:
    """Genera token di reset. Ritorna il token se l'email esiste, None altrimenti."""
    email = email.strip().lower()
    if email not in _load_users():
        return None
    # Rimuovi eventuali token precedenti per la stessa email
    for t, v in list(_reset_tokens.items()):
        if v['email'] == email:
            del _reset_tokens[t]
    token = secrets.token_urlsafe(32)
    _reset_tokens[token] = {
        'email':      email,
        'expires_at': datetime.now() + timedelta(hours=1),
    }
    return token


def verify_reset_token(token: str) -> str | None:
    """Ritorna l'email se il token è valido e non scaduto, None altrimenti."""
    entry = _reset_tokens.get(token)
    if not entry:
        return None
    if datetime.now() > entry['expires_at']:
        del _reset_tokens[token]
        return None
    return entry['email']


def consume_reset_token(token: str, new_password: str) -> bool:
    """Reimposta la password e invalida il token. Ritorna True se ok."""
    email = verify_reset_token(token)
    if not email:
        return False
    users = _load_users()
    if email not in users:
        return False
    users[email]['password_hash'] = _hash(email, new_password)
    _save_users(users)
    _reset_tokens.pop(token, None)
    return True
