"""
cloud_storage.py — Persistenza S3-compatibile (Cloudflare R2 / DO Spaces / Backblaze B2).

PROBLEMA: su DigitalOcean App Platform il filesystem è EFFIMERO. Tutto ciò che
l'app scrive a runtime (sessions/<utente>/current.json, working.pkl, analisi,
cache ARIMA, dati di mercato aggiornati di notte) viene cancellato ad ogni
deploy/restart. Solo i file in git sopravvivono.

SOLUZIONE: questo modulo replica le cartelle dati su un bucket S3-compatibile.
  • all'avvio (wsgi.py) → pull_all() scarica il bucket sul disco locale;
  • ad ogni scrittura → push() carica il file sul bucket (write-through);
  • i path/chiavi del bucket = path relativo alla root del progetto, es.
        sessions/andcappe@gmail.com/current.json
        portafoglio/sessions/market_data_arima.pkl

ATTIVAZIONE: solo se sono presenti le 4 env vars qui sotto. In assenza (es. in
sviluppo locale) il modulo è DISATTIVATO e l'app usa solo il disco locale,
quindi in locale e nei test NULLA cambia.

Env vars (S3-compatibili, valgono per R2/Spaces/B2):
  S3_ENDPOINT_URL     es. https://<accountid>.r2.cloudflarestorage.com
  S3_BUCKET           nome del bucket
  S3_ACCESS_KEY_ID
  S3_SECRET_ACCESS_KEY
  S3_REGION           opzionale (default 'auto' — corretto per Cloudflare R2)
"""
import os
import threading
from pathlib import Path

_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Cartelle (relative alla root) i cui file vengono replicati sul bucket.
_SYNC_PREFIXES = ('sessions/', 'portafoglio/sessions/')

_client = None
_client_lock = threading.Lock()


def _cfg():
    return {
        'endpoint': os.environ.get('S3_ENDPOINT_URL', '').strip(),
        'bucket':   os.environ.get('S3_BUCKET', '').strip(),
        'key':      os.environ.get('S3_ACCESS_KEY_ID', '').strip(),
        'secret':   os.environ.get('S3_SECRET_ACCESS_KEY', '').strip(),
        'region':   (os.environ.get('S3_REGION', '').strip() or 'auto'),
    }


def enabled() -> bool:
    """True solo se tutte le credenziali sono presenti."""
    c = _cfg()
    return bool(c['endpoint'] and c['bucket'] and c['key'] and c['secret'])


def _cli():
    """Client boto3 (lazy, riusato). Solleva se boto3 non è installato."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import boto3
                from botocore.config import Config
                c = _cfg()
                _client = boto3.client(
                    's3',
                    endpoint_url=c['endpoint'],
                    aws_access_key_id=c['key'],
                    aws_secret_access_key=c['secret'],
                    region_name=c['region'],
                    config=Config(signature_version='s3v4',
                                  retries={'max_attempts': 3, 'mode': 'standard'}),
                )
    return _client


def _bucket() -> str:
    return _cfg()['bucket']


def _rel_key(local_path) -> str:
    """Chiave bucket = path relativo alla root, in formato POSIX. '' se fuori sync."""
    p = Path(local_path)
    try:
        rel = p.resolve().relative_to(_ROOT.resolve())
    except Exception:
        # path già relativo alla root?
        try:
            rel = p.relative_to(_ROOT)
        except Exception:
            return ''
    key = rel.as_posix()
    if key.endswith('.tmp'):
        return ''
    if not any(key.startswith(pref) for pref in _SYNC_PREFIXES):
        return ''
    return key


# ─── Write-through (upload dopo ogni scrittura) ───────────────────────────────

def _push_blocking(local_path, key):
    try:
        _cli().upload_file(str(local_path), _bucket(), key)
    except Exception as e:
        print(f"⚠ [cloud] upload fallito {key}: {e}", flush=True)


def push(local_path):
    """
    Carica il file sul bucket in un thread daemon (non blocca la callback).
    Best-effort: un errore di rete non interrompe mai l'app.
    No-op se il modulo è disattivato o il path è fuori dalle cartelle sync.
    """
    if not enabled():
        return
    key = _rel_key(local_path)
    if not key:
        return
    p = Path(local_path)
    if not p.exists():
        return
    threading.Thread(target=_push_blocking, args=(str(p), key),
                     daemon=True).start()


# ─── Pull all'avvio (download bucket → disco locale) ──────────────────────────

def pull_all() -> int:
    """
    Scarica dal bucket tutti gli oggetti sotto le cartelle sync, ricreando i file
    sul disco locale. Da chiamare UNA volta all'avvio (wsgi.py), prima di montare
    le app. Best-effort: in caso di errore l'app parte comunque coi default in git.
    Ritorna il numero di file scaricati.
    """
    if not enabled():
        return 0
    n = 0
    try:
        cli = _cli()
        bkt = _bucket()
        paginator = cli.get_paginator('list_objects_v2')
        for pref in _SYNC_PREFIXES:
            for page in paginator.paginate(Bucket=bkt, Prefix=pref):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('/') or key.endswith('.tmp'):
                        continue
                    dest = _ROOT / key
                    try:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        cli.download_file(bkt, key, str(dest))
                        n += 1
                    except Exception as e:
                        print(f"⚠ [cloud] download fallito {key}: {e}", flush=True)
        print(f"✓ [cloud] sincronizzati {n} file dal bucket '{bkt}'", flush=True)
    except Exception as e:
        print(f"⚠ [cloud] pull_all fallito (uso disco locale): {e}", flush=True)
    return n


# ─── Seed (carica i dati locali attuali sul bucket) ───────────────────────────

def seed() -> int:
    """
    Carica sul bucket TUTTI i file presenti in locale sotto le cartelle sync.
    Da lanciare una volta sola in locale per popolare il bucket con i dati
    esistenti (portafogli, analisi, cache ARIMA).  ritorna il n° di file caricati.
    """
    if not enabled():
        print("✗ cloud_storage disattivato: imposta le env vars S3_* prima del seed.")
        return 0
    cli = _cli()
    bkt = _bucket()
    n = 0
    for pref in _SYNC_PREFIXES:
        base = _ROOT / pref
        if not base.exists():
            continue
        for p in base.rglob('*'):
            if not p.is_file() or p.suffix == '.tmp':
                continue
            key = p.relative_to(_ROOT).as_posix()
            try:
                cli.upload_file(str(p), bkt, key)
                n += 1
                print(f"  ↑ {key}")
            except Exception as e:
                print(f"  ✗ {key}: {e}")
    print(f"✓ seed completato: {n} file caricati sul bucket '{bkt}'")
    return n


def _load_dotenv():
    """Carica .env (se presente) per i comandi da riga di comando (seed/pull)."""
    env = _ROOT / '.env'
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, _, v = line.partition('=')
            os.environ.setdefault(k.strip(), v.strip())


if __name__ == '__main__':
    import sys
    _load_dotenv()
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'seed':
        seed()
    elif cmd == 'pull':
        pull_all()
    else:
        print(f"cloud_storage: enabled={enabled()}  bucket={_bucket() or '(nessuno)'}")
        print("Comandi: python cloud_storage.py [seed|pull|status]")
