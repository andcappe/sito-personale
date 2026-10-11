"""
memoria.py — tiene basso l'RSS del processo su DO (istanza da 512 MB).

Le nove dashboard caricate pesano ~240 MB, ma in produzione il processo stava
fermo a ~485 MB (94,6% della memoria) e l'11/10/2026 è stato ucciso per OOM
(exit 128) mentre si apriva il Portafoglio. La differenza non sono dati vivi:
è memoria che glibc trattiene dopo download e calcoli fatti in thread diversi
(scheduler, prewarm, i 4 thread di gunicorn). Ogni thread si prende la propria
arena di malloc (fino a 8 × CPU) e una volta liberata la memoria resta
frammentata dentro le arene invece di tornare al sistema.

Due interventi, solo su Linux con glibc (altrove è tutto no-op):
  • mallopt(M_ARENA_MAX, 2): al massimo due arene, va fatto PRIMA che partano
    i thread — per questo installa() si chiama in testa a wsgi.py.
    Equivale a MALLOC_ARENA_MAX=2 nell'ambiente, ma non tocca la spec di DO.
  • malloc_trim(0) ogni _PERIODO_S secondi da un thread daemon: restituisce al
    sistema le pagine libere in cima e dentro le arene.
"""
import ctypes
import ctypes.util
import sys
import threading
import time

_PERIODO_S = 60
_M_ARENA_MAX = -8          # costante di <malloc.h>
_ARENE = 2

_installato = False


def _libc():
    if not sys.platform.startswith('linux'):
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library('c') or 'libc.so.6')
        libc.malloc_trim  # solo glibc: musl non ce l'ha
        libc.mallopt
        return libc
    except (OSError, AttributeError):
        return None


def rss_mb():
    """RSS corrente in MB (Linux), None altrove."""
    try:
        with open('/proc/self/statm') as f:
            pagine = int(f.read().split()[1])
        import resource
        return pagine * resource.getpagesize() / 1048576
    except Exception:
        return None


def cgroup_mb():
    """(anon, file, limite) in MB del container, letti da cgroup v2; None se assente.
    'file' è page cache: il sistema la libera da sé sotto pressione, quindi una
    metrica di DO alta per colpa sua non è memoria dell'applicazione."""
    try:
        stat = {}
        with open('/sys/fs/cgroup/memory.stat') as f:
            for riga in f:
                k, v = riga.split()
                stat[k] = int(v)
        with open('/sys/fs/cgroup/memory.max') as f:
            lim = f.read().strip()
        lim = None if lim == 'max' else int(lim) / 1048576
        return stat.get('anon', 0) / 1048576, stat.get('file', 0) / 1048576, lim
    except Exception:
        return None


def _riga_cgroup():
    cg = cgroup_mb()
    if not cg:
        return ''
    anon, file, lim = cg
    return f" | container: anon {anon:.0f} MB, cache {file:.0f} MB" + (
        f", limite {lim:.0f} MB" if lim else '')


def _ciclo(libc):
    ultimo = None
    while True:
        time.sleep(_PERIODO_S)
        try:
            prima = rss_mb()
            libc.malloc_trim(0)
            dopo = rss_mb()
            # Una riga solo quando l'RSS si è mosso di almeno 20 MB: basta per
            # vedere nei log di DO come si comporta, senza riempirli.
            if dopo is not None and (ultimo is None or abs(dopo - ultimo) >= 20):
                print(f"[memoria] RSS {prima:.0f} → {dopo:.0f} MB dopo malloc_trim"
                      f"{_riga_cgroup()}", flush=True)
                ultimo = dopo
        except Exception:
            pass


def installa():
    """Limita le arene e avvia il trim periodico. Idempotente."""
    global _installato
    if _installato:
        return
    _installato = True
    libc = _libc()
    if libc is None:
        return
    try:
        libc.mallopt(_M_ARENA_MAX, _ARENE)
    except Exception:
        pass
    threading.Thread(target=_ciclo, args=(libc,), daemon=True,
                     name='malloc-trim').start()
    print(f"[memoria] arene malloc ≤ {_ARENE}, malloc_trim ogni {_PERIODO_S}s "
          f"(RSS {rss_mb() or 0:.0f} MB){_riga_cgroup()}", flush=True)
