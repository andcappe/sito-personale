"""
yf_seriale.py — una sola yf.download alla volta in tutto il processo.

yfinance 0.2.x raccoglie i risultati di yf.download in dizionari GLOBALI del
modulo (yfinance.shared._DFS / _ERRORS) e li azzera all'inizio di ogni chiamata.
Due download in thread diversi si pestano i piedi: uno svuota i risultati
dell'altro, i ticker di un dataset finiscono nell'altro, e chi aspetta che
_DFS si riempia può aspettare per sempre. Successo l'11/10/2026 al boot: il
prewarm COMMODITIES e il recupero dei dati utenti sono partiti insieme →
'ZB=F.MC', EIMI.MI/SGLD.MI/YM=F segnati come introvabili, COMMODITIES mai salvato.

Le dashboard girano tutte nello stesso processo (wsgi.py) e chiamano
yf.download da una quindicina di punti: invece di toccarli uno per uno si
avvolge yf.download una volta, qui. Le chiamate a yf.Ticker(...).history non
usano lo stato condiviso e restano libere.
"""
import threading

# Oltre questa attesa si procede comunque, senza lucchetto: meglio un rischio
# di interferenza che una richiesta utente bloccata per sempre dietro un
# download rimasto appeso.
_ATTESA_MAX_S = 180

_LOCK = threading.Lock()


def installa():
    """Avvolge yfinance.download con il lucchetto. Idempotente."""
    try:
        import yfinance as yf
    except Exception:
        return
    if getattr(yf.download, '_seriale', False):
        return
    originale = yf.download

    def download(*args, **kwargs):
        preso = _LOCK.acquire(timeout=_ATTESA_MAX_S)
        if not preso:
            print(f"⚠ [yfinance] lucchetto occupato da oltre {_ATTESA_MAX_S}s — "
                  f"download senza serializzazione", flush=True)
        try:
            return originale(*args, **kwargs)
        finally:
            if preso:
                _LOCK.release()

    download._seriale = True
    download.__wrapped__ = originale
    download.__doc__ = originale.__doc__
    yf.download = download
