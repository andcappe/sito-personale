"""
sessions_manager.py — Gestione sessioni unificata per tutte le dashboard.

Struttura files sul disco:
  sessions/{username}/working.pkl          ← sessione di lavoro corrente (auto)
  sessions/{username}/{nome}_{data}.pkl    ← salvataggi nominati dall'utente

File di default (read-only, versionati in git):
  portafoglio/sessions/market_data.pkl            → chiave 'ETF'
  portafoglio/sessions/market_data_CRIPTO.pkl     → chiave 'CRIPTO'
  portafoglio/sessions/market_data_COMMODITIES.pkl → chiave 'CURRENCIES'

Flusso:
  1. Utente apre dashboard → load_for_user() → working.pkl o default ETF
  2. Carica file / modifica → save_working() → working.pkl aggiornato
  3. Vuole caricare nuovo file → has_unsaved_changes() → warning se True
  4. Salva con nome → save_named() → {nome}_{data}.pkl
  5. Admin → list_all_users() → tutti i file di tutti gli utenti
"""

import os
import json
import pickle
import threading
from datetime import datetime
from pathlib import Path

_ROOT         = Path(os.path.dirname(os.path.abspath(__file__)))
_SESSIONS_DIR = _ROOT / 'sessions'
_PORT_DIR     = _ROOT / 'portafoglio' / 'sessions'

# File di default disponibili nel selettore
DEFAULT_FILES = {
    'ETF':       _PORT_DIR / 'market_data.pkl',
    'CRIPTO':    _PORT_DIR / 'market_data_CRIPTO.pkl',
    'CURRENCIES': _PORT_DIR / 'market_data_COMMODITIES.pkl',
}

_LOCK = threading.Lock()


# ─── Paths ────────────────────────────────────────────────────────────────────

def user_dir(username: str) -> Path:
    d = _SESSIONS_DIR / str(username)
    d.mkdir(parents=True, exist_ok=True)
    return d


def working_path(username: str) -> Path:
    return user_dir(username) / 'working.pkl'


# ─── Lettura / scrittura pkl ──────────────────────────────────────────────────

def _load_pkl(path: Path):
    """Carica un pkl. Ritorna None se non esiste o corrotto."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠ [sessions_manager] load error {path.name}: {e}")
        return None


def _save_pkl(path: Path, data: dict) -> bool:
    """Salva pkl in modo atomico (write-rename)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        tmp.replace(path)
        return True
    except Exception as e:
        print(f"⚠ [sessions_manager] save error {path.name}: {e}")
        return False


# ─── API principale ───────────────────────────────────────────────────────────

def load_default(key: str = 'ETF'):
    """Carica un file di default (ETF / CRIPTO / CURRENCIES)."""
    return _load_pkl(DEFAULT_FILES.get(key, DEFAULT_FILES['ETF']))


def load_working(username: str):
    """Carica la sessione di lavoro corrente dell'utente."""
    return _load_pkl(working_path(username))


def load_for_user(username: str, default_key: str = 'ETF'):
    """
    Carica la sessione per l'utente:
    1. working.pkl se esiste
    2. file di default altrimenti
    Ritorna (data, source) dove source = 'working' | 'ETF' | 'CRIPTO' | 'CURRENCIES'
    """
    data = load_working(username)
    if data is not None:
        return data, 'working'
    data = load_default(default_key)
    return data, default_key


def save_working(username: str, data: dict, source: str = None) -> bool:
    """
    Salva la sessione di lavoro corrente.
    Chiamare ogni volta che l'utente carica un file o modifica i dati.

    Propaga il riferimento all'ultimo salvataggio nominato letto dal working
    esistente e marca lo stato come 'sporco' (_has_unsaved_changes=True): ogni
    nuovo stato di lavoro non ancora salvato con nome va protetto al prossimo
    caricamento. has_unsaved_changes() richiede comunque _last_named_save, quindi
    il warning compare solo se l'utente ha già salvato almeno una volta.
    """
    existing = load_working(username) or {}
    data = dict(data)
    if source:
        data['_source'] = source
    data['_has_unsaved_changes']     = True
    data['_last_named_save']         = existing.get('_last_named_save')
    data['_last_named_save_name']    = existing.get('_last_named_save_name')
    data['_working_saved_at']        = datetime.now().isoformat()
    return _save_pkl(working_path(username), data)


def mark_modified(username: str):
    """Marca la sessione come modificata rispetto all'ultimo salvataggio nominato."""
    wp = working_path(username)
    if not wp.exists():
        return
    try:
        data = _load_pkl(wp)
        if data is not None:
            data['_has_unsaved_changes'] = True
            _save_pkl(wp, data)
    except Exception:
        pass


def has_unsaved_changes(username: str) -> bool:
    """
    True se la sessione ha modifiche non ancora salvate con nome.
    False se non c'è mai stata una sessione di lavoro (prima visita).
    """
    data = load_working(username)
    if data is None:
        return False
    # Se non c'è mai stato un salvataggio nominato → non è una "modifica"
    if not data.get('_last_named_save'):
        return False
    return bool(data.get('_has_unsaved_changes', False))


def save_named(username: str, data: dict, name: str) -> Path:
    """
    Salva la sessione con un nome scelto dall'utente.
    Nome file: {username}_{name}_{DDMMYYYY}.pkl
    Ritorna il Path del file creato.
    """
    safe = ''.join(c for c in name if c.isalnum() or c in ' _-').strip().replace(' ', '_')
    if not safe:
        safe = 'sessione'
    date_str = datetime.now().strftime('%d%m%Y')
    filename = f'{username}_{safe}_{date_str}.pkl'
    path = user_dir(username) / filename

    saved_data = dict(data)
    saved_data['_saved_name']  = name
    saved_data['_saved_at']    = datetime.now().isoformat()
    saved_data['_saved_by']    = username
    _save_pkl(path, saved_data)

    # Aggiorna working: segna come salvato
    wp_data = load_working(username) or saved_data
    wp_data['_last_named_save']     = str(path)
    wp_data['_last_named_save_name'] = name
    wp_data['_has_unsaved_changes'] = False
    _save_pkl(working_path(username), wp_data)

    print(f"✓ [sessions_manager] saved: {path.name}")
    return path


def load_named(username: str, filename: str):
    """Carica un salvataggio nominato dell'utente."""
    path = user_dir(username) / filename
    return _load_pkl(path)


def list_user_files(username: str) -> list:
    """
    Lista i file salvati dell'utente (escluso working.pkl).
    Ritorna [{filename, size_kb, modified, saved_name}]
    """
    d = user_dir(username)
    result = []
    for p in sorted(d.glob('*.pkl'), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.name == 'working.pkl':
            continue
        try:
            stat = p.stat()
            # Leggi solo metadata dal file (leggero)
            try:
                meta = _load_pkl(p)
                saved_name = meta.get('_saved_name', p.stem) if meta else p.stem
                saved_at   = meta.get('_saved_at', '')[:10] if meta else ''
            except Exception:
                saved_name = p.stem
                saved_at   = ''
            result.append({
                'filename':   p.name,
                'saved_name': saved_name,
                'saved_at':   saved_at,
                'size_kb':    round(stat.st_size / 1024, 1),
                'modified':   datetime.fromtimestamp(stat.st_mtime).strftime('%d/%m/%Y %H:%M'),
            })
        except Exception:
            pass
    return result


def delete_user_file(username: str, filename: str) -> bool:
    """Elimina un file salvato dell'utente (non working.pkl)."""
    if filename == 'working.pkl':
        return False
    path = user_dir(username) / filename
    try:
        path.unlink()
        return True
    except Exception:
        return False


# ─── Admin ────────────────────────────────────────────────────────────────────

def list_all_users() -> dict:
    """
    Admin: dizionario {username: {files, has_working, working_info}}
    per la sezione /clienti/.
    """
    result = {}
    if not _SESSIONS_DIR.exists():
        return result
    for udir in sorted(_SESSIONS_DIR.iterdir()):
        if not udir.is_dir():
            continue
        username = udir.name
        wp = working_path(username)
        wp_info = None
        if wp.exists():
            try:
                stat = wp.stat()
                meta = _load_pkl(wp)
                wp_info = {
                    'saved_at':    meta.get('_working_saved_at', '')[:19] if meta else '',
                    'source':      meta.get('_source', '?') if meta else '?',
                    'unsaved':     meta.get('_has_unsaved_changes', False) if meta else False,
                    'last_save':   meta.get('_last_named_save_name', '') if meta else '',
                    'size_kb':     round(stat.st_size / 1024, 1),
                    'modified':    datetime.fromtimestamp(stat.st_mtime).strftime('%d/%m/%Y %H:%M'),
                    'n_assets':    len(meta.get('close_returns', {}).get('columns', []))
                                   if isinstance(meta, dict) else 0,
                }
            except Exception:
                wp_info = {}
        result[username] = {
            'files':       list_user_files(username),
            'has_working': wp.exists(),
            'working':     wp_info,
        }
    return result


def load_user_as_admin(username: str, filename: str = 'working.pkl'):
    """
    Admin: carica la sessione di un utente specifico.
    filename='working.pkl' → sessione corrente, altrimenti salvataggio nominato.
    """
    path = user_dir(username) / filename
    return _load_pkl(path)


# ═════════════════════════════════════════════════════════════════════════════
# PROFILI PORTAFOGLIO — archivio condiviso tra tutte le dashboard
# ═════════════════════════════════════════════════════════════════════════════
# File: sessions/{username}/portfolios.json
# Struttura:
#   { "<nomeProfilo>": {
#        "saved_at": "2026-06-01T...",
#        "portfolios": { "<nomePortafoglio>": {"<asset>": peso, ...}, ... }
#     }, ... }
# Un "portafoglio" è una mappa asset→peso (in %); un "profilo" ne contiene
# quanti se ne vogliono. È il formato scambiato dal dialogo Importa/Esporta.

def _profiles_path(username: str) -> Path:
    return user_dir(username) / 'portfolios.json'


def load_profiles(username: str) -> dict:
    """Tutti i profili dell'utente. {} se non esistono."""
    path = _profiles_path(username)
    if not path.exists():
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ [sessions_manager] load_profiles {username}: {e}")
        return {}


def save_profiles(username: str, data: dict) -> bool:
    path = _profiles_path(username)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        return True
    except Exception as e:
        print(f"⚠ [sessions_manager] save_profiles {username}: {e}")
        return False


def list_profiles(username: str) -> list:
    """Nomi dei profili salvati, ordinati per ultimo salvataggio."""
    data = load_profiles(username)
    return sorted(data.keys(),
                  key=lambda k: data[k].get('saved_at', ''), reverse=True)


def get_profile(username: str, profile_name: str) -> dict:
    """{'saved_at':..., 'portfolios': {nome: {asset: peso}}} o {}."""
    return load_profiles(username).get(profile_name, {})


def export_portfolios(username: str, profile_name: str,
                      portfolios: dict, mode: str = 'replace') -> bool:
    """
    Esporta dei portafogli in un profilo.
    portfolios = {nomePortafoglio: {asset: peso}}
    mode: 'replace' azzera i portafogli del profilo, 'merge' aggiunge/aggiorna.
    """
    profile_name = (profile_name or '').strip()
    if not profile_name or not portfolios:
        return False
    data = load_profiles(username)
    if mode == 'replace' or profile_name not in data:
        data[profile_name] = {'portfolios': {}}
    data[profile_name].setdefault('portfolios', {})
    data[profile_name]['portfolios'].update(portfolios)
    data[profile_name]['saved_at'] = datetime.now().isoformat()
    return save_profiles(username, data)


def delete_profile(username: str, profile_name: str) -> bool:
    data = load_profiles(username)
    if profile_name in data:
        del data[profile_name]
        return save_profiles(username, data)
    return False


# ═════════════════════════════════════════════════════════════════════════════
# ANALISI — modello piatto: un'Analisi = un portafoglio (mappa asset→peso)
# File: sessions/{username}/analyses.json
#   { "<nomeAnalisi>": {"weights": {"<asset>": peso, ...}, "saved_at": iso} }
# ═════════════════════════════════════════════════════════════════════════════

def _analyses_path(username: str) -> Path:
    return user_dir(username) / 'analyses.json'


def load_analyses(username: str) -> dict:
    path = _analyses_path(username)
    if not path.exists():
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ [sessions_manager] load_analyses {username}: {e}")
        return {}


def _save_analyses(username: str, data: dict) -> bool:
    path = _analyses_path(username)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        return True
    except Exception as e:
        print(f"⚠ [sessions_manager] save_analyses {username}: {e}")
        return False


def list_analyses(username: str) -> list:
    """Nomi delle analisi salvate, ultima salvata per prima."""
    data = load_analyses(username)
    return sorted(data.keys(),
                  key=lambda k: data[k].get('saved_at', ''), reverse=True)


def get_analysis(username: str, name: str) -> dict:
    """Pesi {asset: peso} dell'analisi, {} se non esiste."""
    return (load_analyses(username).get(name) or {}).get('weights', {})


def get_analysis_meta(username: str, name: str) -> dict:
    """Metadati {asset: {'ticker':…, 'valuta':…}} per ri-aggiungere asset mancanti."""
    return (load_analyses(username).get(name) or {}).get('meta', {})


def save_analysis(username: str, name: str, weights: dict, meta: dict = None) -> bool:
    """
    Crea o sovrascrive un'analisi (un solo portafoglio per analisi).
    meta: {asset: {'ticker':…, 'valuta':…}} per rendere l'analisi autosufficiente
          (così all'import si possono ri-aggiungere e riscaricare gli asset mancanti).
    """
    name = (name or '').strip()
    if not name or not weights:
        return False
    data = load_analyses(username)
    data[name] = {'weights': dict(weights),
                  'meta': dict(meta or {}),
                  'saved_at': datetime.now().isoformat()}
    return _save_analyses(username, data)


def delete_analysis(username: str, name: str) -> bool:
    data = load_analyses(username)
    if name in data:
        del data[name]
        return _save_analyses(username, data)
    return False


def rename_analysis(username: str, old: str, new: str) -> bool:
    new = (new or '').strip()
    if not old or not new or old == new:
        return False
    data = load_analyses(username)
    if old not in data:
        return False
    data[new] = data.pop(old)
    data[new]['saved_at'] = datetime.now().isoformat()
    return _save_analyses(username, data)
