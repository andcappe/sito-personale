import hashlib
import json
import os

ROOT      = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(ROOT, 'users.json')


def _hash(username: str, password: str) -> str:
    return hashlib.sha256(f"{username}:{password}".encode()).hexdigest()


def check_credentials(username: str, password: str) -> bool:
    try:
        with open(USERS_FILE) as f:
            users = json.load(f)
        return users.get(username.strip()) == _hash(username.strip(), password)
    except FileNotFoundError:
        return False
    except Exception:
        return False


def add_user(username: str, password: str) -> None:
    try:
        with open(USERS_FILE) as f:
            users = json.load(f)
    except Exception:
        users = {}
    users[username.strip()] = _hash(username.strip(), password)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def change_password(username: str, old_password: str, new_password: str) -> bool:
    if not check_credentials(username, old_password):
        return False
    add_user(username, new_password)
    return True


def list_users() -> list:
    try:
        with open(USERS_FILE) as f:
            return list(json.load(f).keys())
    except Exception:
        return []
