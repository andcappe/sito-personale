"""
Dashboard Analisi Portafoglio — versione client standalone
Replica esatta della Tab 1 di ir_fe_14.py
"""

import io
import json
import base64
import time
import threading
import os
import sys
import uuid
import pickle
import requests
from pathlib import Path
from datetime import datetime

# Assicura che la cartella superiore sia disponibile per l'import di navbar.py
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, html, dcc, dash_table, Input, Output, State, ALL, callback_context, no_update
from dash.exceptions import PreventUpdate
from flask import send_file as flask_send_file

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
_EXTERNAL_STYLESHEETS = [
    'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=_EXTERNAL_STYLESHEETS,
           requests_pathname_prefix='/portafoglio/',
           routes_pathname_prefix='/portafoglio/')
app.server.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Analisi di Portafoglio</title>
{%favicon%}
{%css%}
<style>
  html { font-size: 16px; }
  @keyframes spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>
(function(){
  var _tip=null;
  function _pos(el){
    var r=el.getBoundingClientRect();
    var color=el.getAttribute('data-tooltip-color')||'#1a3a6b';
    var text=el.getAttribute('data-tooltip')||'';
    if(!text)return;
    if(_tip){_tip.remove();_tip=null;}
    _tip=document.createElement('div');
    _tip.textContent=text;
    _tip.style.cssText='position:fixed;background:#fff;color:#1a2a4a;border:2px solid '+color+';border-radius:5px;padding:4px 10px;font-size:11px;font-family:Inter,sans-serif;font-weight:600;z-index:99999;pointer-events:none;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,0.13);';
    document.body.appendChild(_tip);
    var tx=r.right+8, ty=r.top+r.height/2-_tip.offsetHeight/2;
    if(tx+_tip.offsetWidth>window.innerWidth-8) tx=r.left-_tip.offsetWidth-8;
    _tip.style.left=Math.max(4,tx)+'px';
    _tip.style.top=Math.max(4,Math.min(ty,window.innerHeight-_tip.offsetHeight-4))+'px';
  }
  function _find(e){
    var el=e.target;
    while(el&&el!==document.body){if(el.hasAttribute&&el.hasAttribute('data-tooltip'))return el;el=el.parentNode;}
    return null;
  }
  document.addEventListener('mouseover',function(e){var el=_find(e);if(el)_pos(el);},true);
  document.addEventListener('mouseout',function(e){
    var el=_find(e);
    if(el&&!el.contains(e.relatedTarget)){if(_tip){_tip.remove();_tip=null;}}
  },true);
})();
</script>
</body>
</html>
'''

# Percorsi file
_XLSX        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ETF.xlsx')
_PROFILO_HTML = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'profilo.html'))
_FOTO_PNG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'assets', 'foto.png'))

# Cartella condivisa Files/ (un livello sopra rispetto a portafoglio/)
_FILES_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / 'Files'

_FILE_ORDER = ['ETF', 'CRIPTO', 'COMMODITIES']

def _list_files():
    """Restituisce lista di opzioni dcc.Dropdown dai .xlsx in Files/."""
    if not _FILES_DIR.exists():
        return [{'label': 'ETF', 'value': 'ETF.xlsx'}]
    files = list(_FILES_DIR.glob('*.xlsx'))
    files.sort(key=lambda f: _FILE_ORDER.index(f.stem.upper()) if f.stem.upper() in _FILE_ORDER else 99)
    return [{'label': f.stem, 'value': f.name} for f in files] or \
           [{'label': 'ETF', 'value': 'ETF.xlsx'}]

def _xlsx_path(filename='ETF.xlsx'):
    """Percorso assoluto del file xlsx nella cartella Files/."""
    fp = _FILES_DIR / filename
    if fp.exists():
        return str(fp)
    return _XLSX  # fallback al file locale

# Rotta Flask per servire la pagina profilo
@app.server.route('/health')
def health_check():
    return 'OK', 200

@app.server.route('/')
@app.server.route('/sito')
def serve_profilo():
    return flask_send_file(_PROFILO_HTML)

@app.server.route('/foto.png')
def serve_foto():
    return flask_send_file(_FOTO_PNG, mimetype='image/png')

@app.server.route('/clienti')
def pagina_clienti():
    from flask import Response
    files = sorted(SESSIONS_DIR.glob('tickers_*.xlsx'))
    righe = ''
    for i, f in enumerate(files, 1):
        ts_raw = f.stem.replace('tickers_', '')
        try:
            ts_fmt = datetime.strptime(ts_raw, '%Y%m%d_%H%M%S').strftime('%d/%m/%Y %H:%M:%S')
        except Exception:
            ts_fmt = ts_raw
        try:
            df_tmp = pd.read_excel(f)
            n_asset = len(df_tmp)
        except Exception:
            n_asset = '?'
        righe += (
            f'<tr>'
            f'<td style="padding:8px 16px;font-weight:700;color:#1a3a6b">#{i:03d}</td>'
            f'<td style="padding:8px 16px">{ts_fmt}</td>'
            f'<td style="padding:8px 16px">{n_asset} asset</td>'
            f'<td style="padding:8px 16px">'
            f'<a href="/clienti/download/{f.name}" '
            f'style="color:#2554a0;font-weight:600">⬇ Scarica</a>'
            f'</td>'
            f'</tr>\n'
        )
    if not righe:
        righe = '<tr><td colspan="4" style="padding:20px;color:#888;text-align:center">Nessun file caricato ancora.</td></tr>'
    html = f'''<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8"/>
  <title>File Clienti — Andrea Cappelletti</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"/>
  <style>
    body{{font-family:Inter,sans-serif;background:#fff;color:#1a2a4a;margin:0;padding:40px 5%}}
    h1{{font-size:1.6rem;color:#1a3a6b;margin-bottom:4px}}
    p{{color:#5a7099;font-size:0.9rem;margin-bottom:2rem}}
    table{{width:100%;border-collapse:collapse;box-shadow:0 2px 12px rgba(26,58,107,.08)}}
    th{{background:#1a3a6b;color:#fff;padding:10px 16px;text-align:left;font-size:0.82rem;letter-spacing:.05em;text-transform:uppercase}}
    tr:nth-child(even){{background:#f5f8fe}}
    tr:hover{{background:#e8f0fb}}
    a{{text-decoration:none}}
    .back{{display:inline-block;margin-top:2rem;color:#2554a0;font-weight:600;font-size:.85rem}}
  </style>
</head>
<body>
  <h1>File Clienti Caricati</h1>
  <p>Ogni file ticker caricato dal portale è elencato qui con data, ora e numero di asset.</p>
  <table>
    <thead><tr><th>#</th><th>Data/Ora</th><th>Asset</th><th>File</th></tr></thead>
    <tbody>{righe}</tbody>
  </table>
  <a class="back" href="/">← Torna al sito</a>
</body>
</html>'''
    return Response(html, mimetype='text/html')

@app.server.route('/clienti/download/<filename>')
def download_client_file(filename):
    from flask import abort
    if not filename.startswith('tickers_') or not filename.endswith('.xlsx'):
        abort(404)
    filepath = SESSIONS_DIR / filename
    if not filepath.exists():
        abort(404)
    return flask_send_file(str(filepath),
                           mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           as_attachment=True,
                           download_name=filename)

@app.server.route('/test-yf')
def test_yf():
    import json as _json
    done = threading.Event()
    result = [None]
    def _go():
        try:
            df = yf.download('AAPL', start='2024-01-01', auto_adjust=True,
                             progress=False, threads=False)
            result[0] = {'ok': True, 'rows': len(df), 'cols': list(df.columns)}
        except Exception as e:
            result[0] = {'ok': False, 'error': str(e)}
        finally:
            done.set()
    threading.Thread(target=_go, daemon=True).start()
    done.wait(timeout=30)
    return _json.dumps(result[0] or {'ok': False, 'error': 'timeout'})

# ─────────────────────────────────────────────────────────────────────────────
# Colori
# ─────────────────────────────────────────────────────────────────────────────
color_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#a6cee3',
    '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99',
    '#191970', '#006400', '#8B0000', '#4B0082', '#2F4F4F',
    '#8B4513', '#FF1493', '#696969', '#556B2F', '#008B8B',
    '#E9967A', '#90EE90', '#B0C4DE', '#D3D3D3'
]

# ─────────────────────────────────────────────────────────────────────────────
# Calcoli
# ─────────────────────────────────────────────────────────────────────────────
def calculate_rolling_information_ratio(asset_returns, benchmark_returns, window):
    active_return = asset_returns - benchmark_returns.to_numpy()
    rolling_mean  = active_return.rolling(window=window).mean()
    rolling_std   = active_return.rolling(window=window).std()
    return (rolling_mean / rolling_std) * np.sqrt(252)

def calculate_rolling_sharpe_ratio(returns, window):
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std  = returns.rolling(window=window).std()
    return (rolling_mean / rolling_std) * np.sqrt(252)

def calculate_tracking_error_volatility(asset_returns, benchmark_returns, window):
    active_return = asset_returns - benchmark_returns
    return active_return.rolling(window=window).std() * np.sqrt(252)

def calculate_drawdown(returns_series):
    cumulative   = (1 + returns_series).cumprod()
    rolling_max  = cumulative.cummax()
    return (cumulative - rolling_max) / rolling_max

def calculate_historical_cvar(returns_series, window, tail_pct=0.05):
    # Rendimenti composti rolling a N giorni
    n_day_ret = (1 + returns_series).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1
    # CVaR calcolato sulla distribuzione espansa di tutti i rendimenti N-giorni visti fino a t
    def _cvar(w):
        n_tail = max(1, int(len(w) * tail_pct))
        return np.partition(w, n_tail)[:n_tail].mean()
    return n_day_ret.expanding(min_periods=window + 1).apply(_cvar, raw=True)

def _rolling_volatility(returns_series, window):
    return returns_series.rolling(window, min_periods=window // 2).std() * np.sqrt(252)

def _thin(s, max_pts=500):
    """Downsample una serie/DataFrame a max_pts punti per il rendering."""
    if len(s) <= max_pts:
        return s
    step = max(1, len(s) // max_pts)
    return s.iloc[::step]

# ─────────────────────────────────────────────────────────────────────────────
# Cache DataFrame
# ─────────────────────────────────────────────────────────────────────────────
_DF_CACHE: dict = {}

def _df_key(json_str):
    return json_str[:4000]

def _get_df(json_str):
    if not json_str:
        return None
    key = _df_key(json_str)
    if key not in _DF_CACHE:
        df = pd.read_json(io.StringIO(json_str), orient='split')
        df.index = pd.to_datetime(df.index)
        _DF_CACHE[key] = df
        if len(_DF_CACHE) > 20:
            oldest = next(iter(_DF_CACHE))
            del _DF_CACHE[oldest]
    return _DF_CACHE[key].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Cache giornaliera su disco
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Dati di mercato — download + persistenza su disco
# ─────────────────────────────────────────────────────────────────────────────
DOWNLOAD_BATCH_SIZE = 10

_DL_STATE  = {'status': 'idle', 'current': 0, 'total': 0, 'errors': []}
_DL_BUFFER: dict = {}   # dati file attivo — salvati su disco, permanenti
_DL_LOCK   = threading.Lock()

_CL_BUFFERS: dict = {}   # {username: {dati cliente}} — per-utente, solo in memoria
_CL_STATES:  dict = {}   # {username: stato download cliente}
_CL_LOCK   = threading.Lock()

_active_file_store: dict = {'filename': 'ETF.xlsx'}  # file attivo corrente
_PENDING: dict = {}  # ticker in attesa di download da Gestisci — processati da start_refresh


def _get_username() -> str:
    """Username dalla sessione Flask; 'anon' come fallback."""
    try:
        from flask import session as _fs
        return _fs.get('username') or 'anon'
    except Exception:
        return 'anon'


def _cl_buf(username: str) -> dict:
    with _CL_LOCK:
        if username not in _CL_BUFFERS:
            _CL_BUFFERS[username] = {}
        return _CL_BUFFERS[username]


def _cl_state(username: str) -> dict:
    with _CL_LOCK:
        if username not in _CL_STATES:
            _CL_STATES[username] = {'status': 'idle', 'current': 0, 'total': 0, 'errors': []}
        return _CL_STATES[username]


def _cl_clear(username: str):
    with _CL_LOCK:
        _CL_BUFFERS[username] = {}
        _CL_STATES[username]  = {'status': 'idle', 'current': 0, 'total': 0, 'errors': []}


def _build_ticker_list(filename='ETF.xlsx'):
    df   = pd.read_excel(_xlsx_path(filename))
    cols = df.columns.tolist()
    return list(df[cols[0]]), list(df[cols[1]]), list(df[cols[2]])



def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corregge valori fuori scala ×100 o ÷100 (es. Yahoo Finance che alterna GBp↔GBP/EUR).
    Per ogni asset:
    - valori < 2% della mediana → troppo bassi (÷100) → ×100
    - valori > 50× la mediana  → troppo alti  (×100)  → ÷100
    Soglie molto conservative: non tocca nulla che non sia inequivocabilmente sbagliato.
    """
    df = df.copy()
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        med = s.median()
        if med <= 0:
            continue
        # valori troppo bassi (÷100)
        low_mask = (df[col] > 0) & (df[col] < med * 0.02)
        if low_mask.any():
            df.loc[low_mask, col] = df.loc[low_mask, col] * 100
            print(f"  ⚠ {col}: {low_mask.sum()} valori ×100 (erano < 2% mediana)")
        # valori troppo alti (×100) — ricalcola mediana dopo eventuale correzione
        med2 = df[col].dropna().median()
        high_mask = (df[col] > 0) & (df[col] > med2 * 50)
        if high_mask.any():
            df.loc[high_mask, col] = df.loc[high_mask, col] / 100
            print(f"  ⚠ {col}: {high_mask.sum()} valori ÷100 (erano > 50× mediana)")
    return df


def _atomic_pkl_write(path: Path, data: dict):
    """Scrive il pkl in modo atomico: prima su .tmp, poi os.replace."""
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(data, f)
    os.replace(tmp, path)


def _do_add_tickers(new_tickers, new_descr, new_valuta, start_date, cache_file):
    """
    Scarica solo i nuovi ticker (stessa procedura nightly) su file temp,
    fa il merge con il pkl esistente, scrive atomicamente, aggiorna il buffer.
    Lo status rimane 'running' fino a merge completato.
    """
    global _DL_STATE, _DL_BUFFER
    total = len(new_tickers)
    tmp_pkl = cache_file.parent / (cache_file.stem + '_adding_new.pkl')

    try:
        # Scarica i nuovi ticker su file temporaneo (NON tocca il pkl di default)
        _do_download(new_tickers, new_descr, new_valuta, start_date,
                     cache_file=tmp_pkl, update_buffer=False)
        # _do_download lascia _DL_STATE['status'] invariato (update_buffer=False)

        if not tmp_pkl.exists():
            with _DL_LOCK:
                _DL_STATE['status'] = 'error'
            return

        with open(tmp_pkl, 'rb') as f:
            new_data = pickle.load(f)

        # Merge: leggi il pkl esistente, aggiungi le nuove colonne
        merged = dict(new_data)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    ex = pickle.load(f)
                ex_op = ex.get('original_prices')
                ex_cr = ex.get('close_returns')
                ex_tm = dict(ex.get('ticker_map', {}))
                if ex_op is not None and ex_cr is not None:
                    for col in new_data['original_prices'].columns:
                        ex_op[col] = new_data['original_prices'][col].reindex(ex_op.index).ffill()
                        ex_cr[col] = new_data['close_returns'][col].reindex(ex_op.index)
                    ex_tm.update(new_data['ticker_map'])
                    merged['original_prices'] = ex_op
                    merged['close_returns']   = ex_cr
                    merged['ticker_map']      = ex_tm
                    print(f"✓ Merge — {len(ex_op.columns)} asset totali")
            except Exception as e:
                print(f"⚠ Merge fallito, uso solo nuovi: {e}")

        # Scrittura atomica — il pkl di default viene sostituito solo qui
        _atomic_pkl_write(cache_file, merged)

        with _DL_LOCK:
            _DL_BUFFER.update(merged)
            _DL_STATE['status']  = 'done'
            _DL_STATE['current'] = total

    except Exception as e:
        print(f"❌ _do_add_tickers: {e}")
        with _DL_LOCK:
            _DL_STATE['status'] = 'error'
    finally:
        if tmp_pkl.exists():
            tmp_pkl.unlink()


def _do_reload_from_disk(cache_file, active_file='ETF.xlsx'):
    """Ricarica dati + ARIMA dal disco senza scaricare nulla da internet."""
    global _DL_STATE, _DL_BUFFER
    with _DL_LOCK:
        _DL_STATE = {'status': 'running', 'current': 0, 'total': 1, 'errors': []}
    try:
        if not cache_file.exists():
            with _DL_LOCK:
                _DL_STATE['status'] = 'error'
            print(f"⚠ Reload: {cache_file.name} non trovato")
            return
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        # Carica anche ARIMA dal file separato
        arima_pkl = _arima_cache_path(active_file)
        if arima_pkl.exists():
            try:
                with open(arima_pkl, 'rb') as f:
                    ad = pickle.load(f)
                if ad.get('arima'):
                    data['arima']             = ad['arima']
                    data['arima_computed_at'] = ad.get('arima_computed_at', '')
            except Exception:
                pass
        with _DL_LOCK:
            _DL_BUFFER.update(data)
            _DL_STATE['status']  = 'done'
            _DL_STATE['current'] = 1
        print(f"✓ Dati ricaricati da disco — {data.get('saved_at', '?')}")
    except Exception as e:
        print(f"⚠ Reload dal disco fallito: {e}")
        with _DL_LOCK:
            _DL_STATE['status'] = 'error'


def _do_full_update(tickers, descr, valuta, start_date, cache_file, incremental=False):
    """Download incrementale o completo. ARIMA non viene eseguito qui (gira solo a mezzanotte)."""
    if incremental:
        _do_add_tickers(tickers, descr, valuta, start_date, cache_file)
    else:
        _do_download(tickers, descr, valuta, start_date, cache_file=cache_file, update_buffer=True)


def _do_download(tickers, descrizione, valuta, start_date, cache_file=None, update_buffer=True):
    """Scarica da Yahoo Finance e salva nella cache; cache_file=None usa market_data.pkl."""
    global _DL_STATE, _DL_BUFFER
    total = len(tickers)
    with _DL_LOCK:
        _DL_STATE = {'status': 'running', 'current': 0, 'total': total, 'errors': []}

    # Proxy e User-Agent da variabili d'ambiente (configura su Render se Yahoo blocca)
    _proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or None
    _ua    = (os.environ.get('YF_USER_AGENT') or
              'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
              'AppleWebKit/537.36 (KHTML, like Gecko) '
              'Chrome/124.0.0.0 Safari/537.36')

    _dl_kwargs = dict(auto_adjust=True, progress=False)
    if _proxy:
        _dl_kwargs['proxy'] = _proxy
        print(f"▶ Download via proxy: {_proxy.split('@')[-1]}")

    # Patch User-Agent sulla sessione requests usata da yfinance
    try:
        import yfinance.data as _yfd
        if hasattr(_yfd, 'YfData'):
            _yfd.YfData._headers = {'User-Agent': _ua}
    except Exception:
        pass

    # FX
    fx = None
    try:
        fx = yf.download(['EURUSD=X', 'EURGBP=X'], start=start_date,
                         group_by='ticker', **_dl_kwargs)
    except Exception as e:
        print(f"⚠ FX download fallito: {e}")

    def _fx(name):
        if fx is None or fx.empty:
            return None
        try:
            return fx[(name, 'Close')] if isinstance(fx.columns, pd.MultiIndex) else fx['Close']
        except Exception:
            return None

    eurusd, eurgbp = _fx('EURUSD=X'), _fx('EURGBP=X')
    all_prices = {}

    for i in range(0, total, DOWNLOAD_BATCH_SIZE):
        bt = tickers[i:i + DOWNLOAD_BATCH_SIZE]
        bd = descrizione[i:i + DOWNLOAD_BATCH_SIZE]
        bv = valuta[i:i + DOWNLOAD_BATCH_SIZE]
        try:
            raw = yf.download(bt, start=start_date, group_by='ticker', **_dl_kwargs)
            if raw.empty:
                raise ValueError("risposta vuota")
            for j, t in enumerate(bt):
                desc, curr = bd[j], bv[j]
                try:
                    px = (raw[(t, 'Close')].copy() if isinstance(raw.columns, pd.MultiIndex)
                          else raw['Close'].copy())
                    px = px.ffill()
                    if curr == 'USD' and eurusd is not None:
                        px = px / eurusd.reindex(px.index).ffill()
                    elif curr == 'GBP' and eurgbp is not None:
                        px = px / eurgbp.reindex(px.index).ffill()
                    all_prices[desc] = px
                except Exception as e2:
                    with _DL_LOCK:
                        _DL_STATE['errors'].append(f"{t}: {e2}")
        except Exception as e:
            with _DL_LOCK:
                _DL_STATE['errors'].append(f"Batch {i}: {e}")
        with _DL_LOCK:
            _DL_STATE['current'] = min(i + DOWNLOAD_BATCH_SIZE, total)
        time.sleep(0.3)

    if not all_prices:
        with _DL_LOCK:
            _DL_STATE['status'] = 'error'
        print("❌ Download fallito: nessun dato")
        return

    original_prices = pd.DataFrame(all_prices)
    original_prices.index = pd.to_datetime(original_prices.index)
    original_prices = original_prices.ffill()
    original_prices = _clean_prices(original_prices)
    close_returns   = original_prices.pct_change(fill_method=None)
    ticker_map      = {descrizione[i]: tickers[i] for i in range(len(tickers))}
    saved_at        = datetime.now().strftime('%d/%m/%Y %H:%M')

    data = {
        'date':            datetime.now().strftime('%Y-%m-%d'),
        'saved_at':        saved_at,
        'ticker_map':      ticker_map,
        'original_prices': original_prices,
        'close_returns':   close_returns,
    }
    target_pkl = cache_file or _MARKET_DATA_FILE
    try:
        _atomic_pkl_write(target_pkl, data)
        print(f"✓ {target_pkl.name} salvato — {len(all_prices)} asset — {saved_at}")
    except Exception as e:
        print(f"⚠ Salvataggio su disco fallito: {e}")

    with _DL_LOCK:
        if update_buffer:
            _DL_BUFFER.update(data)
            _DL_STATE['status'] = 'done'
        _DL_STATE['current'] = total


def _do_download_client(tickers, descrizione, valuta, start_date, username='anon'):
    """Download per file cliente: dati isolati in _CL_BUFFERS[username]."""
    total = len(tickers)
    with _CL_LOCK:
        _CL_STATES[username]  = {'status': 'running', 'current': 0, 'total': total, 'errors': []}
        _CL_BUFFERS[username] = {}

    _proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or None
    _ua    = (os.environ.get('YF_USER_AGENT') or
              'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
              'AppleWebKit/537.36 (KHTML, like Gecko) '
              'Chrome/124.0.0.0 Safari/537.36')
    _dl_kwargs = dict(auto_adjust=True, progress=False)
    if _proxy:
        _dl_kwargs['proxy'] = _proxy

    try:
        import yfinance.data as _yfd
        if hasattr(_yfd, 'YfData'):
            _yfd.YfData._headers = {'User-Agent': _ua}
    except Exception:
        pass

    fx = None
    try:
        fx = yf.download(['EURUSD=X', 'EURGBP=X'], start=start_date,
                         group_by='ticker', **_dl_kwargs)
    except Exception as e:
        print(f"⚠ FX download cliente fallito: {e}")

    def _fx(name):
        if fx is None or fx.empty:
            return None
        try:
            return fx[(name, 'Close')] if isinstance(fx.columns, pd.MultiIndex) else fx['Close']
        except Exception:
            return None

    eurusd, eurgbp = _fx('EURUSD=X'), _fx('EURGBP=X')
    all_prices = {}

    for i in range(0, total, DOWNLOAD_BATCH_SIZE):
        bt = tickers[i:i + DOWNLOAD_BATCH_SIZE]
        bd = descrizione[i:i + DOWNLOAD_BATCH_SIZE]
        bv = valuta[i:i + DOWNLOAD_BATCH_SIZE]
        try:
            raw = yf.download(bt, start=start_date, group_by='ticker', **_dl_kwargs)
            if raw.empty:
                raise ValueError("risposta vuota")
            for j, t in enumerate(bt):
                desc, curr = bd[j], bv[j]
                try:
                    px = (raw[(t, 'Close')].copy() if isinstance(raw.columns, pd.MultiIndex)
                          else raw['Close'].copy())
                    px = px.ffill()
                    if curr == 'USD' and eurusd is not None:
                        px = px / eurusd.reindex(px.index).ffill()
                    elif curr == 'GBP' and eurgbp is not None:
                        px = px / eurgbp.reindex(px.index).ffill()
                    all_prices[desc] = px
                except Exception as e2:
                    with _CL_LOCK:
                        _CL_STATES[username]['errors'].append(f"{t}: {e2}")
        except Exception as e:
            with _CL_LOCK:
                _CL_STATES[username]['errors'].append(f"Batch {i}: {e}")
        with _CL_LOCK:
            _CL_STATES[username]['current'] = min(i + DOWNLOAD_BATCH_SIZE, total)
        time.sleep(0.3)

    if not all_prices:
        with _CL_LOCK:
            _CL_STATES[username]['status'] = 'error'
        print(f"❌ Download cliente [{username}] fallito: nessun dato")
        return

    original_prices = pd.DataFrame(all_prices)
    original_prices.index = pd.to_datetime(original_prices.index)
    original_prices = original_prices.ffill()
    original_prices = _clean_prices(original_prices)
    close_returns   = original_prices.pct_change(fill_method=None)
    ticker_map      = {descrizione[i]: tickers[i] for i in range(len(tickers))}
    saved_at        = datetime.now().strftime('%d/%m/%Y %H:%M')

    with _CL_LOCK:
        _CL_BUFFERS[username].update({
            'date':            datetime.now().strftime('%Y-%m-%d'),
            'saved_at':        saved_at,
            'ticker_map':      ticker_map,
            'original_prices': original_prices,
            'close_returns':   close_returns,
        })
        _CL_STATES[username]['status']  = 'done'
        _CL_STATES[username]['current'] = total
    print(f"✓ Download cliente [{username}]: {len(all_prices)} asset isolati")


# ─────────────────────────────────────────────────────────────────────────────
# Carica solo nomi (avvio rapido)
# ─────────────────────────────────────────────────────────────────────────────
def load_ticker_names_only(filename='ETF.xlsx'):
    try:
        df        = pd.read_excel(_xlsx_path(filename))
        col_names = df.columns.tolist()
        if len(col_names) < 2:
            return [], {}
        tickers     = list(df[col_names[0]])
        descrizione = list(df[col_names[1]]) if len(col_names) > 1 else [str(t) for t in tickers]
        ticker_map  = {descrizione[i]: tickers[i] for i in range(len(tickers))}
        options     = [{'label': d, 'value': d} for d in descrizione]
        print(f"✓ Nomi caricati: {len(options)} asset da {filename}")
        return options, ticker_map
    except Exception as e:
        print(f"Errore lettura nomi: {e}")
        return [], {}


# ─────────────────────────────────────────────────────────────────────────────
# Gestione sessioni
# ─────────────────────────────────────────────────────────────────────────────
SESSIONS_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions'))
SESSIONS_DIR.mkdir(exist_ok=True)
_INDEX_FILE          = SESSIONS_DIR / 'index.json'
_MARKET_DATA_FILE    = SESSIONS_DIR / 'market_data.pkl'
_ACTIVE_TICKERS_FILE = SESSIONS_DIR / 'active_tickers.xlsx'

def _file_cache_path(filename='ETF.xlsx'):
    """Percorso del pkl di cache per un dato file (ETF usa market_data.pkl per compat)."""
    stem = Path(filename).stem
    if stem == 'ETF':
        return _MARKET_DATA_FILE
    return SESSIONS_DIR / f'market_data_{stem}.pkl'


def _arima_cache_path(filename='ETF.xlsx'):
    """Percorso del pkl ARIMA separato dal pkl principale."""
    stem = Path(filename).stem
    if stem == 'ETF':
        return SESSIONS_DIR / 'market_data_arima.pkl'
    return SESSIONS_DIR / f'market_data_{stem}_arima.pkl'


def _load_xlsx_rows(filename):
    """Carica un file xlsx da Files/ come lista di dict per DataTable."""
    try:
        df = pd.read_excel(_xlsx_path(filename))
        cols = df.columns.tolist()
        rows = []
        for _, row in df.iterrows():
            rows.append({
                'ticker':      str(row[cols[0]]),
                'descrizione': str(row[cols[1]]) if len(cols) > 1 else str(row[cols[0]]),
                'valuta':      str(row[cols[2]]) if len(cols) > 2 else 'EUR',
            })
        return rows
    except Exception as e:
        print(f"⚠ _load_xlsx_rows({filename}): {e}")
        return []


def _save_xlsx_rows(filename, rows):
    """Scrive i dati del DataTable nel file Files/{filename} preservando i nomi colonna."""
    try:
        try:
            existing = pd.read_excel(_xlsx_path(filename))
            cols = existing.columns.tolist()
        except Exception:
            cols = ['Ticker', 'Descrizione', 'Valuta']
        c0 = cols[0] if cols else 'Ticker'
        c1 = cols[1] if len(cols) > 1 else 'Descrizione'
        c2 = cols[2] if len(cols) > 2 else 'Valuta'
        df = pd.DataFrame({
            c0: [r.get('ticker', '') for r in rows],
            c1: [r.get('descrizione', '') for r in rows],
            c2: [r.get('valuta', 'EUR') for r in rows],
        })
        target = _FILES_DIR / filename
        df.to_excel(str(target), index=False)
        return True
    except Exception as e:
        print(f"⚠ _save_xlsx_rows({filename}): {e}")
        return False


CLIENT_SESSION_STORES = [
    "weights-store-P1",
    "weights-store-P2",
    "weights-store-P3",
    "global-assets-selected",
    "stock-data",
    "original-prices-data",
    "asset-checklist",
    "ticker-map-store",
    "insufficient-data-store",
]


def _read_index():
    if not _INDEX_FILE.exists():
        return []
    try:
        return json.loads(_INDEX_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []

def _write_index(records):
    _INDEX_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def list_sessions():
    records = _read_index()
    return sorted(records, key=lambda r: r.get('updated_at', ''), reverse=True)

def save_session(name, description, store_data):
    sid  = str(uuid.uuid4())[:8]
    now  = datetime.now().isoformat(timespec='seconds')
    data = json.dumps(store_data, ensure_ascii=False)
    size_kb = round(len(data.encode()) / 1024, 1)
    (SESSIONS_DIR / f'{sid}.json').write_text(data, encoding='utf-8')
    rec = {'id': sid, 'name': name or sid, 'description': description,
           'size_kb': size_kb, 'created_at': now, 'updated_at': now}
    records = _read_index()
    records.append(rec)
    _write_index(records)
    return rec

def load_session(session_id):
    f = SESSIONS_DIR / f'{session_id}.json'
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        return {}

def delete_session(session_id):
    f = SESSIONS_DIR / f'{session_id}.json'
    if f.exists():
        f.unlink()
    records = [r for r in _read_index() if r['id'] != session_id]
    _write_index(records)


def _format_ts(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime('%d/%m/%Y %H:%M')
    except Exception:
        return iso_str[:16]


def _build_session_row(record):
    sid  = record['id']
    name = record.get('name', sid[:8])
    desc = record.get('description', '')
    ts   = _format_ts(record.get('updated_at', record.get('created_at', '')))
    size = record.get('size_kb', '?')
    btn_s = {'border': 'none', 'border-radius': '3px', 'cursor': 'pointer',
             'font-size': '10px', 'padding': '3px 8px', 'font-weight': 'bold'}
    return html.Div([
        html.Div([
            html.Span(name, style={'font-weight': 'bold', 'font-size': '11px',
                                   'color': '#1a3a5c', 'margin-right': '8px'}),
            html.Span(f'({size} KB)', style={'font-size': '9px', 'color': '#999'}),
            html.Br(),
            html.Span(desc[:60] + ('…' if len(desc) > 60 else ''),
                      style={'font-size': '10px', 'color': '#555'}),
            html.Br(),
            html.Span(f'🕐 {ts}', style={'font-size': '9px', 'color': '#888'}),
        ], style={'flex': '1', 'min-width': '0'}),
        html.Div([
            html.Button('📂 Carica',
                        id={'type': 'session-load-btn',   'index': sid},
                        n_clicks=0,
                        style={**btn_s, 'background-color': '#1a3a5c',
                               'color': 'white', 'margin-right': '4px'}),
            html.Button('🗑 Elimina',
                        id={'type': 'session-delete-btn', 'index': sid},
                        n_clicks=0,
                        style={**btn_s, 'background-color': '#c0392b', 'color': 'white'}),
        ], style={'display': 'flex', 'align-items': 'center',
                  'flex-shrink': '0', 'margin-left': '10px'}),
    ], style={'display': 'flex', 'align-items': 'center', 'padding': '6px 8px',
              'border-bottom': '1px solid #eee', 'background': 'white',
              'border-radius': '3px', 'margin-bottom': '3px'})


def get_session_panel_layout():
    btn_base = {'border': 'none', 'border-radius': '4px', 'cursor': 'pointer',
                'font-size': '11px', 'padding': '4px 10px', 'font-weight': 'bold'}
    return html.Div([
        html.Button('💾 Sessioni', id='session-toggle-btn', n_clicks=0,
                    style={**btn_base, 'background-color': '#1a3a5c',
                           'color': 'white', 'margin-left': '12px',
                           'padding': '6px 14px', 'font-size': '12px'}),
        html.Div(id='session-panel', style={'display': 'none'}, children=[
            html.Div([
                # Colonna sinistra: salva
                html.Div([
                    html.B('💾 Salva sessione corrente',
                           style={'font-size': '12px', 'color': '#1a3a5c',
                                  'display': 'block', 'margin-bottom': '8px'}),
                    dcc.Input(id='session-name-input', type='text',
                              placeholder='Nome sessione…', debounce=False,
                              style={'width': '100%', 'margin-bottom': '6px',
                                     'padding': '5px 8px', 'border': '1px solid #aaa',
                                     'border-radius': '4px', 'font-size': '11px'}),
                    dcc.Textarea(id='session-desc-input',
                                 placeholder='Note / descrizione (opzionale)…', rows=2,
                                 style={'width': '100%', 'margin-bottom': '6px',
                                        'padding': '5px 8px', 'border': '1px solid #aaa',
                                        'border-radius': '4px', 'font-size': '11px',
                                        'resize': 'none'}),
                    html.Button('💾 Salva', id='session-save-btn', n_clicks=0,
                                style={**btn_base, 'background-color': '#1b7a34',
                                       'color': 'white', 'width': '100%'}),
                    html.Div(id='session-save-status',
                             style={'font-size': '10px', 'margin-top': '5px',
                                    'color': '#555', 'min-height': '16px'}),
                ], style={'width': '30%', 'padding-right': '20px',
                          'border-right': '1px solid #ddd'}),
                # Colonna destra: elenco
                html.Div([
                    html.Div([
                        html.B('📂 Sessioni salvate',
                               style={'font-size': '12px', 'color': '#1a3a5c'}),
                        html.Button('🔄', id='session-refresh-btn', n_clicks=0,
                                    title='Aggiorna elenco',
                                    style={**btn_base, 'background-color': '#e8e8e8',
                                           'color': '#333', 'margin-left': '8px',
                                           'padding': '3px 8px'}),
                    ], style={'display': 'flex', 'align-items': 'center',
                              'margin-bottom': '8px'}),
                    html.Div(id='session-list-container',
                             style={'max-height': '240px', 'overflow-y': 'auto'}),
                    dcc.Store(id='session-load-trigger',   data=None),
                    dcc.Store(id='session-delete-trigger', data=None),
                    dcc.Store(id='session-selected-id',    data=None),
                ], style={'flex': '1', 'padding-left': '20px'}),
            ], style={'display': 'flex'}),
        ]),
    ], style={'display': 'inline-block'})


# ─────────────────────────────────────────────────────────────────────────────
# Date range bar
# ─────────────────────────────────────────────────────────────────────────────
def get_date_range_bar(suffix):
    _today         = pd.Timestamp.today().normalize()
    _ten_years_ago = (_today - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    _today_str     = _today.strftime('%Y-%m-%d')
    return html.Div([
        html.Div([
            html.Div('📅 Range temporale:',
                     style={'font-size': '11px', 'font-weight': 'bold',
                            'color': '#1a3a5c', 'white-space': 'nowrap',
                            'margin-right': '10px', 'align-self': 'center'}),
            html.Div('da:', style={'font-size': '11px', 'color': '#555',
                                   'align-self': 'center', 'white-space': 'nowrap',
                                   'margin-right': '4px'}),
            dcc.DatePickerSingle(id=f'dr-start-{suffix}', display_format='DD/MM/YYYY',
                                 first_day_of_week=1, date=_ten_years_ago,
                                 clearable=False, style={'width': '135px'}),
            html.Div('a:', style={'font-size': '11px', 'color': '#555',
                                  'align-self': 'center', 'white-space': 'nowrap',
                                  'margin': '0 4px'}),
            dcc.DatePickerSingle(id=f'dr-end-{suffix}', display_format='DD/MM/YYYY',
                                 first_day_of_week=1, date=_today_str,
                                 clearable=False, style={'width': '135px'}),
            html.Div(id=f'dr-label-{suffix}',
                     style={'font-size': '10px', 'color': '#888', 'white-space': 'nowrap',
                            'align-self': 'center', 'margin-left': '10px',
                            'min-width': '200px'}),
        ], style={'display': 'flex', 'align-items': 'center', 'padding': '6px 12px',
                  'background': '#f0f4fa', 'border': '1px solid #d0d8e8',
                  'border-radius': '6px', 'margin-bottom': '8px', 'gap': '4px'}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Layout Tab 1
# ─────────────────────────────────────────────────────────────────────────────
def get_portfolio_analysis_tab(options_tickers):
    return html.Div([
        html.Div([
            # ── Riga controlli ────────────────────────────────────────────
            html.Div([
                html.Span('Parametri indicatori di rischio:',
                          style={'margin-right': '10px', 'white-space': 'nowrap',
                                 'font-size': '12px', 'font-weight': '700',
                                 'color': '#1a3a5c'}),
                html.Div([
                    html.Div([
                        html.Label('Benchmark:',
                                   style={'margin-right': '4px', 'white-space': 'nowrap',
                                          'font-size': '10px'}),
                        dcc.Dropdown(
                            options=options_tickers or [],
                            value=None,
                            id='benchmark-selector',
                            placeholder='Seleziona…',
                            style={'width': '180px', 'font-size': '10px', 'min-width': '180px'}
                        ),
                    ], style={'margin-right': '10px', 'display': 'flex', 'align-items': 'center'}),
                    html.Div([
                        html.Label('AK-SH-TV W:',
                                   style={'margin-right': '4px', 'white-space': 'nowrap',
                                          'font-size': '10px'}),
                        dcc.Input(id='ir-window-input', type='number', value=30, min=1,
                                  placeholder='30', style={'width': '45px', 'font-size': '10px'}),
                    ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '8px'}),
                    html.Div([
                        html.Label('MA:',
                                   style={'margin-right': '4px', 'white-space': 'nowrap',
                                          'font-size': '10px'}),
                        dcc.Input(id='ak-ma-input', type='number', value=1, min=1,
                                  placeholder='1', style={'width': '40px', 'font-size': '10px'}),
                    ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '8px'}),
                    html.Div([
                        html.Label('VOL-cVAR W:',
                                   style={'margin-right': '4px', 'white-space': 'nowrap',
                                          'font-size': '10px'}),
                        dcc.Input(id='vol-window-input', type='number', value=30, min=1,
                                  placeholder='30', style={'width': '45px', 'font-size': '10px'}),
                    ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '8px'}),
                    html.Button('Update', id='update-portfolio-button', n_clicks=0,
                                style={'background-color': '#c0392b', 'color': 'white',
                                       'border': 'none', 'padding': '4px 10px',
                                       'border-radius': '4px', 'cursor': 'pointer',
                                       'font-weight': 'bold', 'font-size': '10px',
                                       'box-shadow': '0 2px 6px rgba(192,57,43,0.4)'}),
                    html.Div([
                        html.Label('Filtro AKR:',
                                   style={'font-size': '9px', 'font-weight': 'bold',
                                          'color': '#1a3a5c', 'margin-right': '4px',
                                          'white-space': 'nowrap'}),
                        dcc.RadioItems(
                            id='ir-filter-radio',
                            options=[
                                {'label': 'Tutti', 'value': 'all'},
                                {'label': '>−1',   'value': 'gt_minus1'},
                                {'label': '>0',    'value': 'gt_0'},
                            ],
                            value='all', inline=True,
                            inputStyle={'margin-right': '2px'},
                            labelStyle={'margin-right': '6px', 'font-size': '9px',
                                        'cursor': 'pointer'},
                        ),
                    ], style={'display': 'flex', 'align-items': 'center',
                              'margin-left': '8px', 'padding': '2px 6px',
                              'background': '#f0f4fa', 'border': '1px solid #d0d8e8',
                              'border-radius': '5px'}),
                ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap', 'gap': '0px'}),
            ], style={'margin-bottom': '8px', 'display': 'flex', 'align-items': 'center',
                      'flex-wrap': 'wrap', 'padding': '5px 8px',
                      'background': '#f8fafc', 'border': '1px solid #e2e8f0',
                      'border-radius': '6px'}),

            html.Hr(),

            # ── Intestazione full-width: etichette colonne (sx) + range date (dx) ──
            html.Div([
                # Sx 35%: etichette colonne con pulsanti ☑ integrati
                html.Div([
                    html.Div('Asset', **{'data-tooltip': 'Nome dell\'asset'}, style={
                        'width': '22%', 'fontWeight': 'bold', 'fontSize': '6px',
                        'paddingLeft': '4px', 'color': '#1a3a5c',
                        'display': 'flex', 'alignItems': 'center',
                        'overflow': 'hidden',
                    }),
                    *[html.Div([
                        html.Span(lbl, style={
                            'fontWeight': 'bold', 'fontSize': '7px', 'color': col,
                            'lineHeight': '1', 'whiteSpace': 'nowrap',
                        }),
                        html.Button('☑', id=btn_id, n_clicks=0, title=tip,
                            style={'fontSize': '7px', 'border': 'none', 'background': 'none',
                                   'cursor': 'pointer', 'color': col,
                                   'padding': '0', 'margin': '0', 'lineHeight': '1'}),
                    ], className='col-header-cell', style={
                        'width': w, 'display': 'flex', 'flexDirection': 'column',
                        'alignItems': 'center', 'justifyContent': 'center',
                        'overflow': 'hidden', 'gap': '0px',
                    }) for w, lbl, col, tip, btn_id in [
                        ('3%',  'CH',   '#1a3a5c', 'Deseleziona grafici',     'deselect-all-tickers'),
                        ('8%',  'P1',   '#e6194b', 'Azzera pesi P1',          'reset-p1-tab1'),
                        ('8%',  'P2',   '#3cb44b', 'Azzera pesi P2',          'reset-p2-tab1'),
                        ('8%',  'P3',   '#4363d8', 'Azzera pesi P3',          'reset-p3-tab1'),
                        ('5%',  'AKR',  '#1a3a5c', 'Deseleziona AKRatio',     'deselect-all-ir'),
                        ('6%',  'SH',   '#1a3a5c', 'Deseleziona Sharpe',      'deselect-all-sharpe'),
                        ('6%',  'TV',   '#1a3a5c', 'Deseleziona TEV',         'deselect-all-tev'),
                        ('7%',  'DD',   '#1a3a5c', 'Deseleziona DrawDown',    'deselect-all-dd'),
                        ('7%',  'VOL',  '#1a3a5c', 'Deseleziona Volatilità',  'deselect-all-vol'),
                        ('7%',  'VA90', '#1a3a5c', 'Deseleziona VaR 90%',     'deselect-all-var90'),
                        ('8%',  'VA95', '#1a3a5c', 'Deseleziona VaR 95%',     'deselect-all-var95'),
                    ]],
                ], style={
                    'width': '35%', 'display': 'flex', 'alignItems': 'center',
                    'minHeight': '18px', 'padding': '0', 'overflow': 'hidden',
                }),
                # Dx 65%: range temporale — allineato a destra
                html.Div(
                    get_date_range_bar('tab1'),
                    style={'width': '65%', 'display': 'flex', 'align-items': 'center',
                           'justify-content': 'flex-end'},
                ),
            ], style={
                'display': 'flex', 'background': '#eaf4fb',
                'border-top': '2px solid #2e6da4', 'border-bottom': '1px solid #aed6f1',
            }),

            # ── Riga dati: griglia asset (sx 35%) + grafico (dx 65%) ─────
            html.Div([
                # Colonna sinistra: pulsanti Des + righe asset (generati dal callback)
                html.Div([
                    html.Div(id='asset-count-display',
                             style={'font-size': '10px', 'color': '#555',
                                    'padding': '3px 5px 5px 5px', 'margin-bottom': '2px'}),
                    html.Div(
                        '▶ Dati caricati — clicca UPDATE per aggiornare i grafici',
                        id='update-hint',
                        style={'display': 'none', 'font-size': '9px', 'color': '#c0392b',
                               'font-weight': '600', 'padding': '2px 5px 4px 5px',
                               'background': '#fdf2f0', 'border-left': '3px solid #c0392b',
                               'margin-bottom': '4px', 'border-radius': '0 4px 4px 0'}
                    ),
                    html.Div(id='weights-grid-container', style={'display': 'block'}),
                    html.Div([
                        html.Hr(style={'margin': '10px 0'}),
                        html.Div([
                            html.Div('Totale Pesi:',
                                     style={'width': '22%', 'font-weight': 'bold',
                                            'padding-left': '4px', 'font-size': '9px'}),
                            html.Div(id='sum-p1-display', children='0%',
                                     style={'width': '8%', 'text-align': 'center',
                                            'color': '#d62728', 'font-size': '10px'}),
                            html.Div(id='sum-p2-display', children='0%',
                                     style={'width': '8%', 'text-align': 'center',
                                            'color': '#d62728', 'font-size': '10px'}),
                            html.Div(id='sum-p3-display', children='0%',
                                     style={'width': '8%', 'text-align': 'center',
                                            'color': '#d62728', 'font-size': '10px'}),
                            html.Div('', style={'width': '58%'}),
                        ], style={'display': 'flex'}),
                    ], style={'margin-top': '10px'}),
                ], style={'width': '35%', 'vertical-align': 'top'}),

                # Colonna destra: grafico
                html.Div([
                    dcc.Loading(
                        id='graph-loading',
                        target_components={'controls-and-graph': 'figure'},
                        overlay_style={'visibility': 'visible',
                                       'background': 'rgba(248,250,255,0.85)'},
                        custom_spinner=html.Div([
                            html.Div(style={
                                'width': '44px', 'height': '44px',
                                'border': '4px solid #d0ddf0',
                                'borderTop': '4px solid #1a3a6b',
                                'borderRadius': '50%',
                                'animation': 'spin 0.8s linear infinite',
                            }),
                            html.Div('Calcolo in corso…', style={
                                'marginTop': '14px', 'color': '#1a3a6b',
                                'fontWeight': '700', 'fontSize': '13px',
                                'fontFamily': 'Inter, sans-serif',
                                'letterSpacing': '0.02em',
                            }),
                        ], style={'display': 'flex', 'flexDirection': 'column',
                                  'alignItems': 'center', 'padding': '60px'}),
                        children=html.Div(
                            dcc.Graph(id='controls-and-graph',
                                      style={'width': '100%', 'height': '1900px',
                                             'margin': '0', 'padding': '0'},
                                      config={'responsive': True}),
                            style={'overflow-y': 'auto', 'max-height': '82vh',
                                   'width': '100%', 'margin-bottom': '-10px'},
                        ),
                    ),
                    html.Div(id='output', style={'display': 'none'}),
                    html.Div(id='date-values', style={'display': 'none'}),
                    dcc.Textarea(id='selected-column',           value='', style={'display': 'none'}),
                    dcc.Textarea(id='insufficient-data-tickers', value='', style={'display': 'none'}),
                ], style={'width': '65%', 'vertical-align': 'top'}),
            ], style={'display': 'flex', 'border-bottom': '2px solid #2e6da4'}),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Stili modale progresso
# ─────────────────────────────────────────────────────────────────────────────
_MODAL_HIDDEN = {
    'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0',
    'width': '100%', 'height': '100%', 'background': 'rgba(26,58,92,0.45)',
    'zIndex': '2000', 'justifyContent': 'center', 'alignItems': 'center',
}
_MODAL_SHOWN = {**_MODAL_HIDDEN, 'display': 'flex'}

_EDITOR_HIDDEN = {'display': 'none'}
_EDITOR_SHOWN  = {
    'display': 'flex', 'position': 'fixed', 'top': '0', 'left': '0',
    'width': '100%', 'height': '100%', 'background': 'rgba(0,0,0,0.5)',
    'zIndex': '3000', 'justifyContent': 'center', 'alignItems': 'center',
}

_FILL_LOADING = {
    'height': '100%', 'width': '0%',
    'background': 'linear-gradient(90deg,#007755,#00aa77)',
    'borderRadius': '8px', 'transition': 'width 0.5s ease',
}
_STATUS_GREY  = {'fontSize': '0.78rem', 'color': '#6b7a99',
                 'fontFamily': 'Inter, sans-serif', 'textAlign': 'center',
                 'minHeight': '20px', 'marginTop': '6px'}
_STATUS_GREEN = {**_STATUS_GREY, 'color': '#007755', 'fontWeight': '600'}
_STATUS_RED   = {**_STATUS_GREY, 'color': '#c0392b', 'fontWeight': '600'}


# ─────────────────────────────────────────────────────────────────────────────
# Layout principale
# ─────────────────────────────────────────────────────────────────────────────
def _navbar():
    from navbar import make_navbar
    return make_navbar(current='portafoglio')
app.layout = html.Div([
    # ── Navbar ───────────────────────────────────────────────────────────────
    _navbar(),

    # ── Contenuto (margine top per navbar fissa 64px) ────────────────────────
    html.Div([

    # ── Intestazione pagina ───────────────────────────────────────────────────
    html.Div([
        html.H1([
            'Analisi Rischi di Portafoglio',
            html.Span(' - ', style={'color': '#9baabf'}),
            html.Span('Analisi delle metriche dei rischi Finanziari', style={
                'font-size': '1.1rem', 'font-weight': '400', 'color': '#4a5d7a',
            }),
        ], style={
            'margin': '0',
            'font-size': '1.6rem',
            'font-weight': '700',
            'color': '#1a3a6b',
            'font-family': "'Playfair Display', serif",
            'letter-spacing': '0.02em',
        }),
    ], style={
        'padding': '14px 20px 12px',
        'border-bottom': '2px solid #e2e8f0',
        'background': 'linear-gradient(90deg, #f0f4fb 0%, #ffffff 100%)',
        'margin-bottom': '10px',
    }),

    # ── Barra comandi ─────────────────────────────────────────────────────────
    html.Div([
        # 1. Aggiorna dati — nascosto, i callback sono ancora attivi
        html.Div([
            dcc.Loading(type='circle', color='#007755', children=[
                html.Button('⟳ Aggiorna', id='refresh-data-btn', n_clicks=0,
                            title='Forza nuovo download da Yahoo Finance',
                            style={'background-color': '#007755', 'color': 'white',
                                   'border': 'none', 'padding': '7px 16px',
                                   'border-radius': '4px', 'cursor': 'pointer',
                                   'font-weight': 'bold', 'font-size': '12px',
                                   'margin-right': '6px'}),
            ]),
            html.Span(id='data-last-updated', style={'display': 'none'}),
        ], style={'display': 'none'}),
        # 2. Sessioni
        get_session_panel_layout(),
        # 3. Selettore file dataset
        dcc.Dropdown(
            id='file-selector',
            options=_list_files(),
            value='ETF.xlsx',
            clearable=False,
            style={'width': '160px', 'fontSize': '11px', 'display': 'inline-block'},
            optionHeight=28,
        ),
        html.Button('✏️ Gestisci', id='gestisci-btn', n_clicks=0,
                    title='Aggiungi o rimuovi asset dalla lista selezionata',
                    style={'fontSize': '11px', 'padding': '5px 12px', 'borderRadius': '4px',
                           'cursor': 'pointer', 'background': '#fff3e0',
                           'border': '1px solid #ffb74d', 'color': '#e65100',
                           'marginRight': '4px'}),
        # 4. Scarica template ticker
        html.Button('📋 Template', id='btn-download-template', n_clicks=0,
                    title='Scarica il file Excel template da compilare con i tuoi titoli',
                    style={'font-size': '11px', 'padding': '5px 12px',
                           'border-radius': '4px', 'cursor': 'pointer',
                           'background': '#e8f5e9', 'border': '1px solid #a5d6a7',
                           'color': '#1b5e20', 'margin-right': '4px'}),
        # 4. Upload file custom
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Trascina il tuo file']),
            style={'width': '150px', 'height': '32px', 'lineHeight': '32px',
                   'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                   'textAlign': 'center', 'margin': '0 8px', 'font-size': '11px',
                   'color': '#555', 'cursor': 'pointer'},
            multiple=False,
        ),
        # 4. Scarica prezzi
        html.Button('📥 Scarica', id='save-data-button', n_clicks=0,
                    title='Scarica i prezzi correnti come file Excel (date + prezzi per asset)',
                    style={'font-size': '11px', 'padding': '5px 12px',
                           'border-radius': '4px', 'cursor': 'pointer',
                           'background': '#f0f4fb', 'border': '1px solid #c0d0e8',
                           'color': '#1a3a5c', 'margin-right': '8px'}),
        html.Div(id='download-status', style={'font-size': '11px', 'margin-right': '8px'}),
        html.Button('📂 Importa Frontiera', id='import-frontier-btn', n_clicks=0,
                    title='Importa i portafogli F1→P1, F2→P2, F3→P3 calcolati nella Frontiera Efficiente',
                    style={'font-size': '11px', 'padding': '5px 12px',
                           'border-radius': '4px', 'cursor': 'pointer',
                           'background': '#eafaf1', 'border': '1px solid #1a7a4a',
                           'color': '#1a7a4a', 'font-weight': 'bold', 'margin-right': '8px'}),
        html.Div(id='import-frontier-msg', style={'font-size': '11px', 'color': '#1a7a4a',
                                                   'font-weight': '600', 'margin-right': '8px'}),
        dcc.ConfirmDialog(id='import-frontier-confirm',
                          message='Sei sicuro di voler sovrascrivere i pesi di P1, P2, P3 con quelli della Frontiera Efficiente?'),
        html.Button(id='delete-column-button', n_clicks=0, style={'display': 'none'}),
        html.Div(id='upload-status', style={'display': 'none'}),
    ], style={'display': 'flex', 'align-items': 'center',
              'font-size': '10px', 'position': 'relative',
              'padding': '6px 0', 'flex-wrap': 'wrap', 'gap': '2px'}),

    # ── Modal Gestione Lista ──────────────────────────────────────────────────
    html.Div(id='file-editor-overlay', style=_EDITOR_HIDDEN, children=[
        html.Div(style={
            'background': 'white', 'borderRadius': '8px', 'padding': '20px',
            'width': '700px', 'maxHeight': '82vh', 'overflowY': 'auto',
            'boxShadow': '0 4px 24px rgba(0,0,0,0.35)',
        }, children=[
            html.Div([
                html.H4(id='file-editor-title',
                        style={'margin': 0, 'fontSize': '14px', 'fontWeight': '700'}),
                html.Button('✕', id='file-editor-close', n_clicks=0,
                            style={'background': 'none', 'border': 'none', 'fontSize': '20px',
                                   'cursor': 'pointer', 'color': '#666', 'lineHeight': 1}),
            ], style={'display': 'flex', 'justifyContent': 'space-between',
                      'alignItems': 'center', 'marginBottom': '14px'}),
            dash_table.DataTable(
                id='file-editor-table',
                columns=[
                    {'name': 'Ticker',      'id': 'ticker',      'editable': True},
                    {'name': 'Descrizione', 'id': 'descrizione', 'editable': True},
                    {'name': 'Valuta',      'id': 'valuta',      'editable': True},
                ],
                data=[],
                row_deletable=True,
                style_table={'maxHeight': '280px', 'overflowY': 'auto'},
                style_header={'backgroundColor': '#f5f7fa', 'fontWeight': '700',
                              'fontSize': '12px', 'padding': '6px 8px'},
                style_cell={'fontSize': '12px', 'padding': '5px 8px', 'textAlign': 'left',
                            'border': '1px solid #e0e4ec'},
                style_data_conditional=[{'if': {'row_index': 'odd'},
                                         'backgroundColor': '#fafbfd'}],
            ),
            html.Div([
                dcc.Input(id='new-ticker-input', placeholder='Ticker (es. AAPL)',
                          debounce=False,
                          style={'width': '120px', 'fontSize': '12px', 'padding': '5px 8px',
                                 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                dcc.Input(id='new-desc-input', placeholder='Descrizione',
                          debounce=False,
                          style={'width': '210px', 'fontSize': '12px', 'padding': '5px 8px',
                                 'border': '1px solid #ccc', 'borderRadius': '4px'}),
                dcc.Dropdown(
                    id='new-valuta-dropdown',
                    options=[{'label': 'EUR', 'value': 'EUR'},
                             {'label': 'USD', 'value': 'USD'},
                             {'label': 'GBP', 'value': 'GBP'}],
                    value='EUR', clearable=False,
                    style={'width': '85px', 'fontSize': '12px', 'display': 'inline-block'},
                ),
                html.Button('➕ Aggiungi', id='add-ticker-btn', n_clicks=0,
                            style={'fontSize': '12px', 'padding': '5px 14px',
                                   'background': '#e8f5e9', 'border': '1px solid #a5d6a7',
                                   'color': '#1b5e20', 'borderRadius': '4px',
                                   'cursor': 'pointer', 'fontWeight': '600'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px',
                      'marginTop': '12px', 'flexWrap': 'wrap'}),
            html.Div([
                html.Div(id='file-editor-status',
                         style={'fontSize': '12px', 'color': '#555', 'flex': 1}),
                html.Button('💾 Salva e Aggiorna', id='save-file-editor-btn', n_clicks=0,
                            style={'fontSize': '12px', 'padding': '7px 18px',
                                   'background': '#1a3a5c', 'color': 'white',
                                   'border': 'none', 'borderRadius': '4px',
                                   'cursor': 'pointer', 'fontWeight': '600'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between',
                      'alignItems': 'center', 'marginTop': '14px'}),
        ]),
    ]),

    # ── Stores ───────────────────────────────────────────────────────────────
    dcc.Interval(id='refresh-poll-interval', interval=1000, n_intervals=0, disabled=True),
    dcc.Store(id='active-xlsx-file',        data='ETF.xlsx'),
    dcc.Store(id='asset-checklist',         data=[]),
    dcc.Store(id='stock-data',              data=None),
    dcc.Store(id='original-prices-data',    data=None),
    dcc.Store(id='ticker-map-store',        data={}),
    dcc.Store(id='insufficient-data-store', data=[]),
    dcc.Store(id='weights-store-P1',        data={}),
    dcc.Store(id='weights-store-P2',        data={}),
    dcc.Store(id='weights-store-P3',        data={}),
    dcc.Store(id='global-assets-selected',  data=[]),
    dcc.Store(id='tab1-slider-store',       data=None),
    dcc.Store(id='custom-tickers-store',    data=None),
    dcc.Store(id='upload-done-ts',          data=None),
    dcc.Download(id='download-data'),
    dcc.Download(id='download-template'),

    # ── Modale progresso aggiornamento ───────────────────────────────────────
    html.Div([
        html.Div([
            # Intestazione
            html.Div([
                html.Div([
                    html.I(className='fas fa-chart-line',
                           style={'marginRight': '8px', 'color': '#007755'}),
                    html.Span('Aggiornamento Dati di Mercato', style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontSize': '1.05rem', 'fontWeight': '700',
                        'color': '#1a3a5c',
                    }),
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Button('✕', id='progress-modal-close', n_clicks=0, style={
                    'background': 'none', 'border': 'none', 'cursor': 'pointer',
                    'fontSize': '18px', 'color': '#6b7a99', 'padding': '0 4px',
                    'lineHeight': '1', 'marginLeft': '16px',
                }),
            ], style={
                'display': 'flex', 'justifyContent': 'space-between',
                'alignItems': 'center', 'marginBottom': '22px',
                'paddingBottom': '14px', 'borderBottom': '1px solid #e2e8f0',
            }),
            # Barra progresso
            html.Div(
                html.Div(id='modal-progress-fill', style=_FILL_LOADING),
                style={
                    'width': '100%', 'height': '12px', 'background': '#e2e8f0',
                    'borderRadius': '8px', 'overflow': 'hidden', 'marginBottom': '12px',
                },
            ),
            # Testo percentuale
            html.Div(id='modal-pct-text', style={
                'fontSize': '0.9rem', 'color': '#1a3a5c', 'fontWeight': '700',
                'fontFamily': 'Inter, sans-serif', 'textAlign': 'center',
                'marginBottom': '6px',
            }),
            # Messaggio stato
            html.Div(id='modal-status-text', style=_STATUS_GREY),
        ], style={
            'background': '#ffffff', 'borderRadius': '14px',
            'padding': '30px 36px', 'width': '440px', 'maxWidth': '90vw',
            'boxShadow': '0 24px 64px rgba(26,58,92,0.22)',
        }),
    ], id='progress-modal-overlay', style=_MODAL_HIDDEN),

    # ── Overlay scarica dati ──────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div(style={
                'width': '64px', 'height': '64px',
                'border': '6px solid #e2e8f0',
                'borderTop': '6px solid #1a3a5c',
                'borderRadius': '50%',
                'animation': 'spin 0.9s linear infinite',
                'marginBottom': '20px',
            }),
            html.Div('Scarica dati', style={
                'fontFamily': "'Inter', sans-serif",
                'fontSize': '1.05rem', 'fontWeight': '600',
                'color': '#1a3a5c', 'letterSpacing': '0.02em',
            }),
        ], style={
            'display': 'flex', 'flexDirection': 'column',
            'alignItems': 'center', 'justifyContent': 'center',
            'background': '#ffffff', 'borderRadius': '16px',
            'padding': '40px 56px',
            'boxShadow': '0 24px 64px rgba(26,58,92,0.22)',
        }),
    ], id='download-overlay', style=_MODAL_HIDDEN),

    # ── Contenuto Tab 1 ───────────────────────────────────────────────────────
    html.Div(id='tab1-content'),

], style={'marginTop': '106px', 'padding': '0 1%'}),
])


# ─────────────────────────────────────────────────────────────────────────────
# Callback: inizializzazione + upload
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('upload-status',           'children'),
    Output('asset-checklist',         'data'),
    Output('stock-data',              'data'),
    Output('original-prices-data',    'data'),
    Output('insufficient-data-store', 'data'),
    Output('ticker-map-store',        'data'),
    Output('data-last-updated',       'children'),
    Output('custom-tickers-store',    'data'),
    Output('refresh-poll-interval',   'disabled',    allow_duplicate=True),
    Output('refresh-poll-interval',   'n_intervals', allow_duplicate=True),
    Output('refresh-data-btn',        'disabled',    allow_duplicate=True),
    Output('progress-modal-overlay',  'style',       allow_duplicate=True),
    Output('modal-progress-fill',     'style',       allow_duplicate=True),
    Output('modal-pct-text',          'children',    allow_duplicate=True),
    Output('modal-status-text',       'children',    allow_duplicate=True),
    Output('modal-status-text',       'style',       allow_duplicate=True),
    Output('upload-done-ts',          'data',        allow_duplicate=True),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call='initial_duplicate',
)
def update_output(contents, filename):
    import time as _time
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'

    _noup = (no_update,) * 8

    if triggered_id == 'initial_load':
        # Ogni volta che la pagina viene (ri)caricata, svuota il buffer cliente,
        # ripristina il file attivo a ETF (default) e cancella i ticker pendenti.
        _cl_clear(_get_username())
        _active_file_store['filename'] = 'ETF.xlsx'
        _PENDING.clear()
        import pickle as _pickle
        cr, op, tm, saved_at = None, None, {}, ''
        if _MARKET_DATA_FILE.exists():
            try:
                with open(_MARKET_DATA_FILE, 'rb') as _f:
                    _d = _pickle.load(_f)
                cr       = _d.get('close_returns')
                op       = _d.get('original_prices')
                tm       = _d.get('ticker_map', {})
                saved_at = _d.get('saved_at', '')
            except Exception:
                pass
        if cr is not None:
            options  = [{'label': col, 'value': col} for col in cr.columns]
            last_upd = f"Aggiornati: {saved_at}" if saved_at else ''
            with _DL_LOCK:
                _DL_BUFFER.update({'close_returns': cr, 'original_prices': op,
                                   'ticker_map': tm, 'saved_at': saved_at})
            return (
                html.Div(f'✓ {len(options)} asset — da file locale',
                         style={'color': '#007755', 'font-size': '11px'}),
                options,
                cr.to_json(date_format='iso', orient='split'),
                op.to_json(date_format='iso', orient='split'),
                [], tm, last_upd, None,
                *_noup, no_update,
            )
        options, ticker_map = load_ticker_names_only()
        return (
            html.Div('⏳ Download dati in corso…',
                     style={'color': '#e67e22', 'font-size': '11px'}),
            options, None, None, [], ticker_map, '', None,
            *_noup, no_update,
        )

    elif triggered_id == 'upload-data' and contents is not None:
        try:
            _, content_string = contents.split(',')
            decoded  = base64.b64decode(content_string)
            done_ts  = _time.time()

            df        = pd.read_excel(io.BytesIO(decoded))
            col_names = df.columns.tolist()

            is_price_file = False
            try:
                pd.to_datetime(df[col_names[0]], errors='raise')
                num_cols = df.drop(columns=[col_names[0]]).select_dtypes(include='number').columns.tolist()
                is_price_file = len(num_cols) >= 1
            except Exception:
                pass

            if is_price_file:
                df_prices = df.set_index(col_names[0])
                df_prices.index = pd.to_datetime(df_prices.index)
                df_prices = df_prices.select_dtypes(include='number').ffill().dropna(how='all')
                df_prices = _clean_prices(df_prices)
                close_returns = df_prices.pct_change(fill_method=None)
                options    = [{'label': c, 'value': c} for c in df_prices.columns]
                ticker_map = {c: c for c in df_prices.columns}
                saved_at   = datetime.now().strftime('%d/%m/%Y %H:%M')
                with _DL_LOCK:
                    _DL_BUFFER.update({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'saved_at': saved_at,
                        'close_returns': close_returns,
                        'original_prices': df_prices,
                        'ticker_map': ticker_map,
                    })
                return (
                    html.Div(f'✓ {len(options)} asset — prezzi caricati dal file',
                             style={'color': '#007755', 'font-size': '11px'}),
                    options,
                    close_returns.to_json(date_format='iso', orient='split'),
                    df_prices.to_json(date_format='iso', orient='split'),
                    [], ticker_map, f"Caricati: {saved_at}", None,
                    *_noup, done_ts,
                )

            else:
                try:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    copy_path = SESSIONS_DIR / f'tickers_{ts}.xlsx'
                    with open(copy_path, 'wb') as fout:
                        fout.write(decoded)
                    print(f"✓ File cliente salvato: {copy_path.name}")
                except Exception as e:
                    print(f"⚠ Salvataggio file cliente fallito: {e}")

                tickers     = list(df[col_names[0]])
                descrizione = (list(df[col_names[1]]) if len(col_names) > 1
                               else [str(t) for t in tickers])
                valuta      = (list(df[col_names[2]]) if len(col_names) > 2
                               else ['EUR'] * len(tickers))
                ticker_map  = {descrizione[i]: tickers[i] for i in range(len(tickers))}
                options     = [{'label': d, 'value': d} for d in descrizione]
                custom      = {'tickers': tickers, 'descr': descrizione, 'valuta': valuta}

                start_date = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
                _username  = _get_username()
                print(f"▶ Upload file ticker [{_username}]: {len(tickers)} asset da scaricare da {start_date}")
                threading.Thread(
                    target=_do_download_client,
                    args=(tickers, descrizione, valuta, start_date),
                    kwargs={'username': _username},
                    daemon=True,
                ).start()

                return (
                    html.Div(f'⏳ Download avviato — {len(options)} asset da Yahoo Finance…',
                             style={'color': '#e67e22', 'font-size': '11px'}),
                    options, None, None, [], ticker_map, '', custom,
                    False, 0, True, _MODAL_SHOWN, _FILL_LOADING,
                    f'Avvio — {len(tickers)} asset…', '', _STATUS_GREY,
                    done_ts,
                )

        except Exception as e:
            return (
                html.Div(f'Errore: {e}'), [], None, None, [], {}, '', None,
                *_noup, no_update,
            )

    raise PreventUpdate


# Clientside: appena upload-done-ts cambia (ogni upload completato),
# azzera upload-data.contents nel browser → il prossimo upload dello
# stesso file viene rilevato come nuovo (None → base64).
app.clientside_callback(
    """
    function(ts) {
        if (!ts) return window.dash_clientside.no_update;
        return null;
    }
    """,
    Output('upload-data', 'contents', allow_duplicate=True),
    Input('upload-done-ts', 'data'),
    prevent_initial_call=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: renderizza contenuto Tab1
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('tab1-content', 'children'),
    Input('asset-checklist', 'data'),
)
def render_tab1(options_tickers):
    return get_portfolio_analysis_tab(options_tickers)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: cambio file dataset
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('active-xlsx-file',          'data'),
    Output('stock-data',                'data',  allow_duplicate=True),
    Output('original-prices-data',      'data',  allow_duplicate=True),
    Output('asset-checklist',           'data',  allow_duplicate=True),
    Output('ticker-map-store',          'data',  allow_duplicate=True),
    Output('data-last-updated',         'children', allow_duplicate=True),
    Output('upload-status',             'children', allow_duplicate=True),
    Input('file-selector',              'value'),
    prevent_initial_call=True,
)
def on_file_selected(filename):
    if not filename:
        raise PreventUpdate
    _active_file_store['filename'] = filename
    _cl_clear(_get_username())

    cache = _file_cache_path(filename)
    if cache.exists():
        try:
            with open(cache, 'rb') as f:
                data = pickle.load(f)
            cr       = data.get('close_returns')
            op       = data.get('original_prices')
            tm       = data.get('ticker_map', {})
            saved_at = data.get('saved_at', '')
            if cr is not None:
                with _DL_LOCK:
                    _DL_BUFFER.clear()
                    _DL_BUFFER.update(data)
                options = [{'label': c, 'value': c} for c in cr.columns]
                return (filename,
                        cr.to_json(date_format='iso', orient='split'),
                        op.to_json(date_format='iso', orient='split'),
                        options, tm,
                        f'Aggiornati: {saved_at}',
                        html.Div(f'✓ {len(options)} asset — {Path(filename).stem}',
                                 style={'color': '#007755', 'font-size': '11px'}))
        except Exception:
            pass

    # Cache non trovata → scarica da Yahoo Finance
    try:
        tickers, descr, valuta = _build_ticker_list(filename)
    except Exception as e:
        raise PreventUpdate
    options  = [{'label': d, 'value': d} for d in descr]
    tm       = {descr[i]: tickers[i] for i in range(len(tickers))}
    start    = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    with _DL_LOCK:
        _DL_BUFFER.clear()
        _DL_STATE.update({'status': 'running', 'current': 0, 'total': len(tickers), 'errors': []})
    threading.Thread(target=_do_download,
                     args=(tickers, descr, valuta, start),
                     kwargs={'cache_file': cache}, daemon=True).start()
    print(f"▶ Download {filename}: {len(tickers)} ticker")
    return (filename, None, None, options, tm,
            f'Download {Path(filename).stem}…',
            html.Div(f'⏳ Download {Path(filename).stem} — {len(tickers)} asset…',
                     style={'color': '#e67e22', 'font-size': '11px'}))


# ─────────────────────────────────────────────────────────────────────────────
# Callback: apri/chiudi modal editor lista asset
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('file-editor-overlay', 'style'),
    Output('file-editor-table',   'data'),
    Output('file-editor-title',   'children'),
    Output('file-editor-status',  'children', allow_duplicate=True),
    Input('gestisci-btn',         'n_clicks'),
    Input('file-editor-close',    'n_clicks'),
    State('file-selector',        'value'),
    prevent_initial_call=True,
)
def toggle_file_editor(open_n, close_n, filename):
    triggered = callback_context.triggered_id
    if triggered == 'gestisci-btn' and open_n:
        rows = _load_xlsx_rows(filename or 'ETF.xlsx')
        stem = Path(filename or 'ETF.xlsx').stem
        return _EDITOR_SHOWN, rows, f'Gestisci lista: {stem}', ''
    return _EDITOR_HIDDEN, no_update, no_update, no_update


# ─────────────────────────────────────────────────────────────────────────────
# Callback: aggiungi riga al DataTable editor
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('file-editor-table', 'data',         allow_duplicate=True),
    Output('new-ticker-input',  'value'),
    Output('new-desc-input',    'value'),
    Input('add-ticker-btn',     'n_clicks'),
    State('new-ticker-input',   'value'),
    State('new-desc-input',     'value'),
    State('new-valuta-dropdown','value'),
    State('file-editor-table',  'data'),
    prevent_initial_call=True,
)
def add_ticker_row(n, ticker, desc, valuta, rows):
    if not ticker or not ticker.strip():
        raise PreventUpdate
    new_row = {
        'ticker':      ticker.strip().upper(),
        'descrizione': (desc.strip() if desc and desc.strip() else ticker.strip().upper()),
        'valuta':      valuta or 'EUR',
    }
    return (rows or []) + [new_row], '', ''


# ─────────────────────────────────────────────────────────────────────────────
# Callback: salva xlsx e avvia download in background
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('file-editor-status',       'children',    allow_duplicate=True),
    Output('file-editor-overlay',      'style',       allow_duplicate=True),
    Output('file-selector',            'options',     allow_duplicate=True),
    Output('refresh-poll-interval',    'disabled',    allow_duplicate=True),
    Output('refresh-poll-interval',    'n_intervals', allow_duplicate=True),
    Output('refresh-data-btn',         'disabled',    allow_duplicate=True),
    Output('asset-checklist',          'data',        allow_duplicate=True),
    Input('save-file-editor-btn',      'n_clicks'),
    State('file-editor-table',         'data'),
    State('file-selector',             'value'),
    prevent_initial_call=True,
)
def save_file_editor(n, rows, filename):
    if not rows:
        return '⚠ La lista è vuota, nessun file salvato.', no_update, no_update, no_update, no_update, no_update, no_update
    filename = filename or 'ETF.xlsx'
    _cl_clear(_get_username())
    ok = _save_xlsx_rows(filename, rows)
    if not ok:
        return '⚠ Errore nel salvataggio del file.', no_update, no_update, no_update, no_update, no_update, no_update

    _active_file_store['filename'] = filename
    try:
        tickers, descr, valuta_list = _build_ticker_list(filename)
        cache = _file_cache_path(filename)

        # Identifica i ticker nuovi (non presenti nel pkl di default)
        if cache.exists():
            try:
                with open(cache, 'rb') as f:
                    ex = pickle.load(f)
                ex_cols = set(ex.get('original_prices', pd.DataFrame()).columns)
                new_idx = [i for i, d in enumerate(descr) if d not in ex_cols]
                if not new_idx:
                    # Nessun ticker nuovo: chiudi editor, lista asset invariata
                    return ('✓ File salvato — nessun nuovo asset da scaricare.',
                            _EDITOR_HIDDEN, _list_files(),
                            no_update, no_update, no_update, no_update)
                dl_tickers = [tickers[i] for i in new_idx]
                dl_descr   = [descr[i]   for i in new_idx]
                dl_valuta  = [valuta_list[i] for i in new_idx]
            except Exception as e:
                print(f"⚠ Lettura pkl esistente fallita, tratto come nuovo: {e}")
                dl_tickers, dl_descr, dl_valuta = tickers, descr, valuta_list
        else:
            dl_tickers, dl_descr, dl_valuta = tickers, descr, valuta_list

        # Memorizza ticker pendenti — il download parte solo quando l'utente clicca Aggiorna
        start = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
        with _DL_LOCK:
            _PENDING.update({
                'tickers': dl_tickers, 'descr': dl_descr, 'valuta': dl_valuta,
                'cache': cache, 'start': start,
                'incremental': cache.exists(),
            })
        print(f"▶ {len(dl_tickers)} ticker in attesa — clicca Aggiorna per scaricare")
    except Exception as e:
        print(f"⚠ Errore post-salvataggio: {e}")
        return f'⚠ {e}', _EDITOR_HIDDEN, _list_files(), no_update, no_update, no_update, no_update

    # Chiude l'editor, SVUOTA la lista asset, chiede all'utente di cliccare Aggiorna
    return ('✓ File salvato — clicca Aggiorna per caricare i nuovi asset.',
            _EDITOR_HIDDEN, _list_files(),
            no_update, no_update, no_update, [])


# ─────────────────────────────────────────────────────────────────────────────
# Callback: avvia aggiornamento manuale
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('refresh-poll-interval',   'disabled',    allow_duplicate=True),
    Output('refresh-poll-interval',   'n_intervals', allow_duplicate=True),
    Output('refresh-data-btn',        'disabled',    allow_duplicate=True),
    Output('progress-modal-overlay',  'style',       allow_duplicate=True),
    Output('modal-progress-fill',     'style',       allow_duplicate=True),
    Output('modal-pct-text',          'children',    allow_duplicate=True),
    Output('modal-status-text',       'children',    allow_duplicate=True),
    Output('modal-status-text',       'style',       allow_duplicate=True),
    Input('refresh-data-btn', 'n_clicks'),
    State('dr-start-tab1',    'date'),
    prevent_initial_call=True,
)
def start_refresh(n_clicks, start_date_picker):
    if not n_clicks:
        raise PreventUpdate

    with _DL_LOCK:
        if _DL_STATE.get('status') == 'running':
            current = _DL_STATE.get('current', 0)
            total   = _DL_STATE.get('total', 1) or 1
            pct     = int(current / total * 100)
            return (False, 0, True, _MODAL_SHOWN,
                    {**_FILL_LOADING, 'width': f'{pct}%'},
                    f'Aggiornamento in corso: {current}/{total}…',
                    'Un aggiornamento è già in corso…', _STATUS_GREY)

        # Prendi i ticker pendenti da Gestisci (se presenti)
        pending = dict(_PENDING)
        _PENDING.clear()

    active_file = _active_file_store.get('filename', 'ETF.xlsx')
    cache = _file_cache_path(active_file)
    _cl_clear(_get_username())

    if pending:
        # Nuovi ticker da Gestisci: download solo quelli + merge nel pkl
        dl_tickers  = pending['tickers']
        dl_descr    = pending['descr']
        dl_valuta   = pending['valuta']
        incremental = pending.get('incremental', True)
        start_date  = pending['start']
        with _DL_LOCK:
            _DL_STATE.update({'status': 'running', 'current': 0,
                              'total': len(dl_tickers), 'errors': []})
        threading.Thread(
            target=_do_full_update,
            args=(dl_tickers, dl_descr, dl_valuta, start_date, cache),
            kwargs={'incremental': incremental},
            daemon=True,
        ).start()
        print(f"▶ Download {len(dl_tickers)} nuovi ticker per {active_file}")
        label = f'{len(dl_tickers)} nuovi asset'
    else:
        # Nessun ticker pendente: ricarica il file di default dal disco (nessun download)
        threading.Thread(
            target=_do_reload_from_disk,
            args=(cache, active_file),
            daemon=True,
        ).start()
        print(f"▶ Ricaricamento {active_file} da disco")
        label = f'Caricamento {Path(active_file).stem}'

    return (False, 0, True, _MODAL_SHOWN, _FILL_LOADING,
            f'{label}…', '', _STATUS_GREY)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: polling aggiornamento
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('stock-data',              'data',     allow_duplicate=True),
    Output('original-prices-data',    'data',     allow_duplicate=True),
    Output('asset-checklist',         'data',     allow_duplicate=True),
    Output('ticker-map-store',        'data',     allow_duplicate=True),
    Output('data-last-updated',       'children', allow_duplicate=True),
    Output('refresh-poll-interval',   'disabled', allow_duplicate=True),
    Output('refresh-data-btn',        'disabled', allow_duplicate=True),
    Output('modal-progress-fill',     'style',    allow_duplicate=True),
    Output('modal-pct-text',          'children', allow_duplicate=True),
    Output('modal-status-text',       'children', allow_duplicate=True),
    Output('modal-status-text',       'style',    allow_duplicate=True),
    Output('progress-modal-overlay',  'style',    allow_duplicate=True),
    Input('refresh-poll-interval', 'n_intervals'),
    prevent_initial_call=True,
)
def poll_refresh_progress(n):
    # Legge dal buffer cliente (per-utente) se attivo, altrimenti da ETF
    _u = _get_username()
    with _CL_LOCK:
        cl_state  = dict(_cl_state(_u))
        cl_buffer = dict(_cl_buf(_u))
    with _DL_LOCK:
        dl_state  = dict(_DL_STATE)
        dl_buffer = dict(_DL_BUFFER)

    client_active = cl_state.get('status') in ('running', 'done') and cl_state.get('status') != 'idle'
    state  = cl_state  if client_active else dl_state
    buffer = cl_buffer if client_active else dl_buffer

    status  = state.get('status', 'idle')
    current = state.get('current', 0)
    total   = state.get('total', 1) or 1
    pct     = int(current / total * 100)
    modal_fill = {**_FILL_LOADING, 'width': f'{pct}%'}

    if status == 'idle':
        raise PreventUpdate

    if status == 'running':
        return (no_update, no_update, no_update, no_update, no_update,
                False, True,
                modal_fill, f'{current} / {total}  ({pct}%)',
                'Download in corso…', _STATUS_GREY, no_update)

    if status == 'error':
        err_fill = {**_FILL_LOADING, 'width': '100%', 'background': '#c0392b'}
        return (no_update, no_update, no_update, no_update, no_update,
                True, False,
                err_fill, '❌ Download fallito',
                'Si è verificato un errore.', _STATUS_RED, no_update)

    close_returns   = buffer.get('close_returns')
    original_prices = buffer.get('original_prices')
    if close_returns is None or close_returns.empty:
        err_fill = {**_FILL_LOADING, 'width': '100%', 'background': '#c0392b'}
        return (no_update, no_update, no_update, no_update, no_update,
                True, False,
                err_fill, '❌ Nessun dato ricevuto',
                'Il download è terminato senza dati.', _STATUS_RED, no_update)

    options      = [{'label': col, 'value': col} for col in close_returns.columns]
    ticker_map   = buffer.get('ticker_map', {})
    saved_at     = buffer.get('saved_at', '')
    returns_json = close_returns.to_json(date_format='iso', orient='split')
    prices_json  = original_prices.to_json(date_format='iso', orient='split')
    ok_fill      = {**_FILL_LOADING, 'width': '100%'}

    errors    = state.get('errors', [])
    n_ok      = len(options)
    n_err     = len(errors)
    err_note  = f' — ⚠ {n_err} non trovati su Yahoo' if n_err else ''
    status_msg = (f'Dati pronti — {n_ok} asset scaricati{err_note}.\n'
                  + ('\n'.join(errors[:5]) if errors else ''))

    return (
        returns_json, prices_json, options, ticker_map,
        f"Aggiornati: {saved_at}",
        True, False,
        ok_fill, f'✓ {n_ok} asset{err_note}',
        status_msg, _STATUS_GREEN if not n_err else {**_STATUS_GREEN, 'color': '#b8860b'},
        _MODAL_HIDDEN,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Callback: chiudi modale aggiornamento
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('progress-modal-overlay', 'style', allow_duplicate=True),
    Input('progress-modal-close', 'n_clicks'),
    prevent_initial_call=True,
)
def close_progress_modal(n):
    if not n:
        raise PreventUpdate
    return _MODAL_HIDDEN


# ─────────────────────────────────────────────────────────────────────────────
# Callback: salva dati (download Excel)
# ─────────────────────────────────────────────────────────────────────────────
# Clientside: mostra overlay immediatamente al click (callback primario per download-overlay.style)
app.clientside_callback(
    """
    function(n) {
        if (!n || n === 0) return {display: 'none'};
        return {
            display: 'flex', position: 'fixed', top: '0', left: '0',
            width: '100%', height: '100%',
            background: 'rgba(26,58,92,0.45)',
            zIndex: '2000', justifyContent: 'center', alignItems: 'center'
        };
    }
    """,
    Output('download-overlay', 'style'),
    Input('save-data-button',  'n_clicks'),
    prevent_initial_call=True,
)

@app.callback(
    Output('download-data',    'data'),
    Output('download-status',  'children'),
    Output('download-overlay', 'style', allow_duplicate=True),
    Input('save-data-button',  'n_clicks'),
    State('original-prices-data', 'data'),
    prevent_initial_call=True,
)
def salva_dati(n_clicks, original_prices_data):
    if not n_clicks or n_clicks == 0:
        raise PreventUpdate

    # Legge dal buffer cliente (per-utente) se attivo, altrimenti da ETF
    with _CL_LOCK:
        cl_prices = _cl_buf(_get_username()).get('original_prices')
    if cl_prices is not None and not cl_prices.empty:
        df_prices = cl_prices
    else:
        with _DL_LOCK:
            df_prices = _DL_BUFFER.get('original_prices')
    if df_prices is None or df_prices.empty:
        if original_prices_data:
            df_prices = pd.read_json(io.StringIO(original_prices_data), orient='split')
            df_prices.index = pd.to_datetime(df_prices.index)

    if df_prices is None or df_prices.empty:
        return (no_update,
                html.Div('⚠ Nessun dato disponibile — clicca prima ⟳ Aggiorna',
                         style={'color': '#e67e22', 'font-size': '11px'}),
                _MODAL_HIDDEN)
    try:
        df_prices.index = pd.to_datetime(df_prices.index).strftime('%Y-%m-%d')
        df_prices.index.name = 'Data'
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            df_prices.to_excel(writer, sheet_name='Prezzi')
        out.seek(0)
        return (
            dcc.send_bytes(out.read(), 'prezzi_asset.xlsx'),
            html.Div('✓ File scaricato', style={'color': 'green', 'font-size': '11px'}),
            _MODAL_HIDDEN,
        )
    except Exception as e:
        return no_update, html.Div(f'Errore: {e}', style={'color': 'red'}), _MODAL_HIDDEN


# ─────────────────────────────────────────────────────────────────────────────
# Callback: importa portafogli dalla Frontiera Efficiente
# ─────────────────────────────────────────────────────────────────────────────
_FE_PORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'sessions', 'fe_portfolio.json')

@app.callback(
    Output('import-frontier-confirm', 'displayed'),
    Output('import-frontier-msg', 'children'),
    Input('import-frontier-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def ask_import_frontier(n):
    if not n:
        raise PreventUpdate
    if not os.path.exists(_FE_PORT_FILE):
        return False, '⚠ Nessun portafoglio salvato dalla Frontiera'
    try:
        with open(_FE_PORT_FILE) as f:
            data = json.load(f)
        saved_at = data.get('saved_at', '')
        ports = list(data.get('portfolios', {}).keys())
        return True, f'Trovato: {", ".join(ports)} del {saved_at}'
    except Exception as e:
        return False, f'⚠ Errore lettura: {e}'


@app.callback(
    Output('weights-store-P1', 'data', allow_duplicate=True),
    Output('weights-store-P2', 'data', allow_duplicate=True),
    Output('weights-store-P3', 'data', allow_duplicate=True),
    Output({'type': 'weight-input', 'index': ALL}, 'value', allow_duplicate=True),
    Output('import-frontier-msg', 'children', allow_duplicate=True),
    Input('import-frontier-confirm', 'submit_n_clicks'),
    State({'type': 'weight-input', 'index': ALL}, 'id'),
    State({'type': 'weight-input', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def do_import_frontier(submit, all_ids, all_vals):
    if not submit:
        raise PreventUpdate
    try:
        with open(_FE_PORT_FILE) as f:
            data = json.load(f)
        ports = data.get('portfolios', {})
        w1 = ports.get('F1', {})
        w2 = ports.get('F2', {})
        w3 = ports.get('F3', {})

        new_vals = []
        for inp_id, cur_val in zip(all_ids, all_vals):
            idx = inp_id['index']  # es. 'P1-Az. ACWI'
            if idx.startswith('P1-'):
                asset = idx[3:]
                new_vals.append(w1.get(asset, 0) if w1 else cur_val)
            elif idx.startswith('P2-'):
                asset = idx[3:]
                new_vals.append(w2.get(asset, 0) if w2 else cur_val)
            elif idx.startswith('P3-'):
                asset = idx[3:]
                new_vals.append(w3.get(asset, 0) if w3 else cur_val)
            else:
                new_vals.append(cur_val)

        imported = [k for k in ['F1','F2','F3'] if k in ports]
        return w1 or no_update, w2 or no_update, w3 or no_update, new_vals, f'✓ Importato: {", ".join(imported)}'
    except Exception as e:
        return no_update, no_update, no_update, all_vals, f'⚠ Errore: {e}'


# ─────────────────────────────────────────────────────────────────────────────
# Callback: scarica template ticker Excel
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('download-template', 'data'),
    Input('btn-download-template', 'n_clicks'),
    prevent_initial_call=True,
)
def download_template(n_clicks):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Template Ticker'

    headers = ['TICKER', 'DESCRIZIONE', 'VALUTA', 'MERCATO']
    examples = [
        ['ISAC.L',   'Az. ACWI',   'USD', 'Azionario ACWI'],
        ['SWDA.MI',  'Az. World',  'EUR', 'Azionario World'],
    ]

    header_fill = PatternFill('solid', fgColor='1A3A6B')
    header_font = Font(bold=True, color='FFFFFF', size=11)
    example_fill = PatternFill('solid', fgColor='EBF3FF')
    thin = Side(style='thin', color='C0D0E8')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    col_widths = [14, 30, 10, 25]

    for col_idx, (h, w) in enumerate(zip(headers, col_widths), start=1):
        cell = ws.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = border
        ws.column_dimensions[cell.column_letter].width = w

    for row_idx, row_data in enumerate(examples, start=2):
        for col_idx, val in enumerate(row_data, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.fill = example_fill
            cell.alignment = Alignment(horizontal='left', vertical='center')
            cell.border = border

    ws.row_dimensions[1].height = 20

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return dcc.send_bytes(out.read(), 'template_ticker.xlsx')


# ─────────────────────────────────────────────────────────────────────────────
# Callback: benchmark dropdown popolato dopo caricamento dati
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('benchmark-selector', 'options'),
    Output('benchmark-selector', 'value'),
    Input('asset-checklist', 'data'),
    State('benchmark-selector', 'value'),
)
def update_benchmark_options(options_tickers, current_value):
    if not options_tickers:
        return [], None
    portfolio_opts = [
        {'label': '── Portafoglio P1', 'value': 'Port1'},
        {'label': '── Portafoglio P2', 'value': 'Port2'},
        {'label': '── Portafoglio P3', 'value': 'Port3'},
    ]
    all_options = portfolio_opts + options_tickers
    if current_value and any(opt['value'] == current_value for opt in all_options):
        return all_options, current_value
    return all_options, options_tickers[0]['value'] if options_tickers else None


# ─────────────────────────────────────────────────────────────────────────────
# Callback: mostra/nascondi hint Update
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('update-hint', 'style'),
    Input('stock-data',              'data'),
    Input('update-portfolio-button', 'n_clicks'),
)
def toggle_update_hint(stock_data, update_clicks):
    _shown  = {'display': 'block', 'font-size': '9px', 'color': '#c0392b',
                'font-weight': '600', 'padding': '2px 5px 4px 5px',
                'background': '#fdf2f0', 'border-left': '3px solid #c0392b',
                'margin-bottom': '4px', 'border-radius': '0 4px 4px 0'}
    _hidden = {**_shown, 'display': 'none'}
    ctx = callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
    if triggered == 'update-portfolio-button' and update_clicks:
        return _hidden
    if triggered == 'stock-data' and stock_data:
        return _shown
    return _hidden


# ─────────────────────────────────────────────────────────────────────────────
# Callback: griglia pesi e asset
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('weights-grid-container', 'children'),
    Output('asset-count-display',    'children'),
    Input('update-portfolio-button', 'n_clicks'),
    State('stock-data',    'data'),
    State('asset-checklist', 'data'),
    State({'type': 'graph-select-checkbox',  'index': ALL}, 'value'),
    State({'type': 'ir-select-checkbox',     'index': ALL}, 'value'),
    State({'type': 'sharpe-select-checkbox', 'index': ALL}, 'value'),
    State({'type': 'tev-select-checkbox',    'index': ALL}, 'value'),
    State({'type': 'dd-select-checkbox',     'index': ALL}, 'value'),
    State({'type': 'vol-select-checkbox',    'index': ALL}, 'value'),
    State({'type': 'var90-select-checkbox',  'index': ALL}, 'value'),
    State({'type': 'var95-select-checkbox',  'index': ALL}, 'value'),
    State('weights-store-P1', 'data'),
    State('weights-store-P2', 'data'),
    State('weights-store-P3', 'data'),
    State('benchmark-selector', 'value'),
    State('ir-window-input',    'value'),
    State('ir-filter-radio',    'value'),
    prevent_initial_call=False,
)
def generate_asset_and_weight_inputs(update_clicks, stock_data_json, options_tickers,
                                      graph_vals, ir_vals, sharpe_vals, tev_vals,
                                      dd_vals, vol_vals, var90_vals, var95_vals,
                                      saved_p1, saved_p2, saved_p3,
                                      benchmark_value, ir_window, ir_filter):
    _placeholder = html.Div(
        'Carica i dati per visualizzare gli asset',
        style={'color': '#888', 'font-style': 'italic', 'font-size': '11px', 'padding': '12px 8px'}
    )
    # Se stock-data non è nello store, prova direttamente dal buffer in memoria
    if not stock_data_json:
        with _DL_LOCK:
            buf = dict(_DL_BUFFER)
        cr = buf.get('close_returns')
        if cr is not None and not cr.empty:
            stock_data_json = cr.to_json(date_format='iso', orient='split')
            if not options_tickers:
                options_tickers = [{'label': c, 'value': c} for c in cr.columns]
    if not stock_data_json or not options_tickers:
        return [_placeholder], ''

    # Ripristina selezioni precedenti
    saved_selected = []
    for grp in (graph_vals or []) + (ir_vals or []) + (sharpe_vals or []) + \
               (tev_vals or []) + (dd_vals or []) + (vol_vals or []) + \
               (var90_vals or []) + (var95_vals or []):
        saved_selected.extend(grp)

    asset_names = [option['value'] for option in options_tickers]
    saved_p1 = saved_p1 or {}
    saved_p2 = saved_p2 or {}
    saved_p3 = saved_p3 or {}

    # Calcola AKRatio per colorazione etichette
    assets_above_threshold = set()
    if ir_filter and ir_filter != 'all' and benchmark_value:
        df_all = _get_df(stock_data_json)
        window = ir_window if (ir_window and ir_window > 0) else 30
        if df_all is not None and benchmark_value in df_all.columns:
            threshold = 0.0 if ir_filter == 'gt_0' else -1.0
            benchmark_returns = df_all[benchmark_value].dropna()
            for asset in asset_names:
                if asset == benchmark_value or asset not in df_all.columns:
                    continue
                tail = pd.concat([df_all[asset].dropna(), benchmark_returns], axis=1).dropna().iloc[-window:]
                if len(tail) < window:
                    continue
                ir_val = calculate_rolling_information_ratio(
                    tail.iloc[:, 0], tail.iloc[:, 1], window=window).iloc[-1]
                if pd.notna(ir_val) and ir_val > threshold:
                    assets_above_threshold.add(asset)

    n_total = len(asset_names)
    n_above = len(assets_above_threshold)
    if ir_filter and ir_filter != 'all':
        label = '> 0' if ir_filter == 'gt_0' else '> −1'
        count_text = [
            html.Span(f'{n_above}', style={'font-weight': 'bold', 'color': '#c0392b', 'font-size': '12px'}),
            html.Span(f' / {n_total} asset', style={'color': '#555'}),
            html.Span(f'  in rosso (AKRatio {label})', style={'color': '#c0392b', 'font-style': 'italic'}),
        ]
    else:
        count_text = [
            html.Span(f'{n_total}', style={'font-weight': 'bold', 'color': '#1a3a5c', 'font-size': '12px'}),
            html.Span(f' / {n_total} asset', style={'color': '#555'}),
        ]

    rows = []

    def _chk(chk_id, opt_val, sel_val, w):
        return html.Div(
            dcc.Checklist(id=chk_id,
                          options=[{'label': '', 'value': opt_val}],
                          value=sel_val,
                          inputStyle={'width': '10px', 'height': '10px',
                                      'cursor': 'pointer', 'margin': '0'},
                          style={'display': 'flex', 'justify-content': 'center',
                                 'align-items': 'center', 'width': '100%'},
                          className='asset-checkbox'),
            style={'width': w, 'height': '22px', 'display': 'flex',
                   'align-items': 'center', 'justify-content': 'center'})

    for asset in asset_names:
        def create_weight_input(portfolio_index, a=asset):
            port_key  = f'P{portfolio_index}'
            saved_val = {1: saved_p1, 2: saved_p2, 3: saved_p3}[portfolio_index].get(a, 0)
            return dcc.Input(
                id={'type': 'weight-input', 'index': f'{port_key}-{a}'},
                type='number', value=saved_val, min=0, max=100, step=0.1, placeholder='0',
                style={'width': '90%', 'text-align': 'right', 'font-size': '9px',
                       'height': '18px', 'padding': '1px 2px', 'box-sizing': 'border-box'}
            )

        asset_val   = [asset]                       if asset in saved_selected else []
        ir_val      = [f'{asset}_InformationRatio'] if f'{asset}_InformationRatio' in saved_selected else []
        sharpe_val  = [f'{asset}_Sharpe']           if f'{asset}_Sharpe'           in saved_selected else []
        tev_val     = [f'{asset}_TEV']              if f'{asset}_TEV'              in saved_selected else []
        dd_val      = [f'{asset}_DD']               if f'{asset}_DD'               in saved_selected else []
        vol_val     = [f'{asset}_Vol']              if f'{asset}_Vol'              in saved_selected else []
        var90_val   = [f'{asset}_VaR90']            if f'{asset}_VaR90'            in saved_selected else []
        var95_val   = [f'{asset}_VaR95']            if f'{asset}_VaR95'            in saved_selected else []
        _label_color = '#c0392b' if asset in assets_above_threshold else '#1a3a5c'

        row_content = html.Div([
            html.Div(
                html.Span(asset, style={
                    'color': _label_color, 'fontWeight': 'bold',
                    'overflow': 'hidden', 'whiteSpace': 'nowrap',
                    'textOverflow': 'ellipsis', 'maxWidth': '100%', 'fontSize': '9px',
                }),
                **{'data-tooltip': asset, 'data-tooltip-color': _label_color},
                style={'width': '22%', 'height': '28px', 'display': 'flex',
                       'alignItems': 'center', 'paddingLeft': '4px',
                       'overflow': 'hidden', 'position': 'relative', 'cursor': 'default'},
            ),
            _chk({'type': 'graph-select-checkbox',  'index': asset}, asset,                       asset_val,  '3%'),
            html.Div(create_weight_input(1), className='weight-input-cell', style={'width': '8%'}),
            html.Div(create_weight_input(2), className='weight-input-cell', style={'width': '8%'}),
            html.Div(create_weight_input(3), className='weight-input-cell', style={'width': '8%'}),
            _chk({'type': 'ir-select-checkbox',     'index': asset}, f'{asset}_InformationRatio', ir_val,     '5%'),
            _chk({'type': 'sharpe-select-checkbox', 'index': asset}, f'{asset}_Sharpe',           sharpe_val, '6%'),
            _chk({'type': 'tev-select-checkbox',    'index': asset}, f'{asset}_TEV',              tev_val,    '6%'),
            _chk({'type': 'dd-select-checkbox',     'index': asset}, f'{asset}_DD',               dd_val,     '7%'),
            _chk({'type': 'vol-select-checkbox',    'index': asset}, f'{asset}_Vol',              vol_val,    '7%'),
            _chk({'type': 'var90-select-checkbox',  'index': asset}, f'{asset}_VaR90',            var90_val,  '7%'),
            _chk({'type': 'var95-select-checkbox',  'index': asset}, f'{asset}_VaR95',            var95_val,  '8%'),
        ], style={'display': 'flex', 'align-items': 'center', 'border-bottom': '1px dotted #eee'})
        rows.append(row_content)

    # Righe portafogli P1/P2/P3
    def _pchk(chk_id, opt_val, sel_val, w):
        return html.Div(
            dcc.Checklist(id=chk_id,
                          options=[{'label': '', 'value': opt_val}],
                          value=sel_val,
                          inputStyle={'width': '10px', 'height': '10px',
                                      'cursor': 'pointer', 'margin': '0'},
                          style={'display': 'flex', 'justify-content': 'center',
                                 'align-items': 'center', 'width': '100%'}),
            style={'width': w, 'height': '22px', 'display': 'flex',
                   'align-items': 'center', 'justify-content': 'center'})

    for portfolio_num in [1, 2, 3]:
        portfolio_name  = f'Port{portfolio_num}'
        port_val        = [portfolio_name]                              if portfolio_name                              in saved_selected else []
        ir_port_val     = [f'{portfolio_name}_InformationRatio']       if f'{portfolio_name}_InformationRatio'       in saved_selected else []
        sharpe_port_val = [f'{portfolio_name}_Sharpe']                 if f'{portfolio_name}_Sharpe'                 in saved_selected else []
        tev_port_val    = [f'{portfolio_name}_TEV']                    if f'{portfolio_name}_TEV'                    in saved_selected else []
        dd_port_val     = [f'{portfolio_name}_DD']                     if f'{portfolio_name}_DD'                     in saved_selected else []
        vol_port_val    = [f'{portfolio_name}_Vol']                    if f'{portfolio_name}_Vol'                    in saved_selected else []
        var90_port_val  = [f'{portfolio_name}_VaR90']                  if f'{portfolio_name}_VaR90'                  in saved_selected else []
        var95_port_val  = [f'{portfolio_name}_VaR95']                  if f'{portfolio_name}_VaR95'                  in saved_selected else []

        portfolio_row = html.Div([
            html.Div(
                html.Span(portfolio_name, style={
                    'color': '#0066cc', 'fontWeight': 'bold',
                    'overflow': 'hidden', 'whiteSpace': 'nowrap',
                    'textOverflow': 'ellipsis', 'maxWidth': '100%', 'fontSize': '9px',
                }),
                **{'data-tooltip': portfolio_name, 'data-tooltip-color': '#0066cc'},
                style={'width': '22%', 'height': '28px', 'display': 'flex',
                       'alignItems': 'center', 'paddingLeft': '4px',
                       'overflow': 'hidden', 'position': 'relative', 'cursor': 'default'}),
            _pchk({'type': 'graph-select-checkbox',  'index': portfolio_name}, portfolio_name,                       port_val,        '3%'),
            html.Div('', style={'width': '8%'}),
            html.Div('', style={'width': '8%'}),
            html.Div('', style={'width': '8%'}),
            _pchk({'type': 'ir-select-checkbox',     'index': portfolio_name}, f'{portfolio_name}_InformationRatio', ir_port_val,     '5%'),
            _pchk({'type': 'sharpe-select-checkbox', 'index': portfolio_name}, f'{portfolio_name}_Sharpe',           sharpe_port_val, '6%'),
            _pchk({'type': 'tev-select-checkbox',    'index': portfolio_name}, f'{portfolio_name}_TEV',              tev_port_val,    '6%'),
            _pchk({'type': 'dd-select-checkbox',     'index': portfolio_name}, f'{portfolio_name}_DD',               dd_port_val,     '7%'),
            _pchk({'type': 'vol-select-checkbox',    'index': portfolio_name}, f'{portfolio_name}_Vol',              vol_port_val,    '7%'),
            _pchk({'type': 'var90-select-checkbox',  'index': portfolio_name}, f'{portfolio_name}_VaR90',            var90_port_val,  '7%'),
            _pchk({'type': 'var95-select-checkbox',  'index': portfolio_name}, f'{portfolio_name}_VaR95',            var95_port_val,  '8%'),
        ], style={'display': 'flex', 'align-items': 'center', 'border-bottom': '1px dotted #eee',
                  'background-color': '#f0f0f0'})
        rows.append(portfolio_row)

    return rows, count_text


# ─────────────────────────────────────────────────────────────────────────────
# Callback: raccoglie asset selezionati nello store globale
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('global-assets-selected', 'data'),
    Input({'type': 'graph-select-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def collect_selected_assets(all_values):
    return [v[0] for v in all_values if v]


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks: pulsanti cancella colonna
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output({'type': 'graph-select-checkbox', 'index': ALL}, 'value'),
    Input('deselect-all-tickers', 'n_clicks'),
    State({'type': 'graph-select-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def deselect_all_tickers(n, vals):
    if not n: raise PreventUpdate
    return [[] for _ in vals]

@app.callback(
    Output({'type': 'ir-select-checkbox', 'index': ALL}, 'value'),
    Input('deselect-all-ir', 'n_clicks'),
    State({'type': 'ir-select-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def deselect_all_ir(n, vals):
    if not n: raise PreventUpdate
    return [[] for _ in vals]

@app.callback(
    Output({'type': 'sharpe-select-checkbox', 'index': ALL}, 'value'),
    Input('deselect-all-sharpe', 'n_clicks'),
    State({'type': 'sharpe-select-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def deselect_all_sharpe(n, vals):
    if not n: raise PreventUpdate
    return [[] for _ in vals]

@app.callback(
    Output({'type': 'tev-select-checkbox', 'index': ALL}, 'value'),
    Input('deselect-all-tev', 'n_clicks'),
    State({'type': 'tev-select-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def deselect_all_tev(n, vals):
    if not n: raise PreventUpdate
    return [[] for _ in vals]

@app.callback(
    Output({'type': 'dd-select-checkbox', 'index': ALL}, 'value'),
    Input('deselect-all-dd', 'n_clicks'),
    State({'type': 'dd-select-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def deselect_all_dd(n, vals):
    if not n: raise PreventUpdate
    return [[] for _ in vals]

@app.callback(
    Output({'type': 'vol-select-checkbox', 'index': ALL}, 'value'),
    Input('deselect-all-vol', 'n_clicks'),
    State({'type': 'vol-select-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def deselect_all_vol(n, vals):
    if not n: raise PreventUpdate
    return [[] for _ in vals]

@app.callback(
    Output({'type': 'var90-select-checkbox', 'index': ALL}, 'value'),
    Input('deselect-all-var90', 'n_clicks'),
    State({'type': 'var90-select-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def deselect_all_var90(n, vals):
    if not n: raise PreventUpdate
    return [[] for _ in vals]

@app.callback(
    Output({'type': 'var95-select-checkbox', 'index': ALL}, 'value'),
    Input('deselect-all-var95', 'n_clicks'),
    State({'type': 'var95-select-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def deselect_all_var95(n, vals):
    if not n: raise PreventUpdate
    return [[] for _ in vals]

@app.callback(
    Output({'type': 'weight-input', 'index': ALL}, 'value'),
    Input('reset-p1-tab1', 'n_clicks'),
    Input('reset-p2-tab1', 'n_clicks'),
    Input('reset-p3-tab1', 'n_clicks'),
    State({'type': 'weight-input', 'index': ALL}, 'id'),
    State({'type': 'weight-input', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def reset_portfolio_weights(n1, n2, n3, all_ids, all_vals):
    ctx = callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
    prefix = {'reset-p1-tab1': 'P1-', 'reset-p2-tab1': 'P2-', 'reset-p3-tab1': 'P3-'}.get(triggered)
    if not prefix:
        raise PreventUpdate
    return [0 if inp_id['index'].startswith(prefix) else val
            for inp_id, val in zip(all_ids, all_vals)]


# ─────────────────────────────────────────────────────────────────────────────
# Callback: somme pesi
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('sum-p1-display', 'children'),
    Output('sum-p2-display', 'children'),
    Output('sum-p3-display', 'children'),
    Output('sum-p1-display', 'style'),
    Output('sum-p2-display', 'style'),
    Output('sum-p3-display', 'style'),
    Output('weights-store-P1', 'data'),
    Output('weights-store-P2', 'data'),
    Output('weights-store-P3', 'data'),
    Input({'type': 'weight-input', 'index': ALL}, 'value'),
    State({'type': 'weight-input', 'index': ALL}, 'id'),
    State('weights-store-P1', 'data'),
    State('weights-store-P2', 'data'),
    State('weights-store-P3', 'data'),
    prevent_initial_call=True,
)
def update_portfolio_weights(all_input_values, all_input_ids, p1_data, p2_data, p3_data):
    p1 = dict(p1_data or {})
    p2 = dict(p2_data or {})
    p3 = dict(p3_data or {})

    for val, inp_id in zip(all_input_values, all_input_ids):
        idx = inp_id['index']
        if idx.startswith('P1-'):
            p1[idx[3:]] = val or 0
        elif idx.startswith('P2-'):
            p2[idx[3:]] = val or 0
        elif idx.startswith('P3-'):
            p3[idx[3:]] = val or 0

    sum1 = sum(v for v in p1.values() if v)
    sum2 = sum(v for v in p2.values() if v)
    sum3 = sum(v for v in p3.values() if v)

    def _style(s):
        base = {'width': '8%', 'text-align': 'center', 'font-size': '10px'}
        base['color'] = '#1b7a34' if abs(s - 100) < 0.01 else '#d62728'
        return base

    return (f'{sum1:.1f}%', f'{sum2:.1f}%', f'{sum3:.1f}%',
            _style(sum1), _style(sum2), _style(sum3),
            p1, p2, p3)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: seleziona colonna dal click sul grafico
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('selected-column', 'value'),
    Input('update-portfolio-button', 'n_clicks'),
    Input('delete-column-button',    'n_clicks'),
    Input('controls-and-graph',      'clickData'),
    State('controls-and-graph',      'figure'),
    prevent_initial_call=True,
)
def update_selected_column(update_clicks, delete_clicks, clickData, figure):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
    if triggered_id in ('update-portfolio-button', 'delete-column-button'):
        return ''
    if triggered_id == 'controls-and-graph' and clickData and clickData['points']:
        curve_number = clickData['points'][0]['curveNumber']
        if figure and 'data' in figure and curve_number < len(figure['data']):
            return figure['data'][curve_number].get('name', '')
    return ''


# ─────────────────────────────────────────────────────────────────────────────
# Callback: date picker → tab1-slider-store (grafico reagisce al cambio date)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('dr-start-tab1',     'date'),
    Output('dr-end-tab1',       'date'),
    Output('dr-label-tab1',     'children'),
    Output('tab1-slider-store', 'data'),
    Input('stock-data',         'data'),
    Input('dr-start-tab1',      'date'),
    Input('dr-end-tab1',        'date'),
    prevent_initial_call=False,
)
def sync_date_range(stock_data, start_date, end_date):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

    if not stock_data:
        raise PreventUpdate

    df = _get_df(stock_data)
    if df is None or df.empty:
        raise PreventUpdate

    ts_min = int(df.index.min().timestamp())
    ts_max = int(df.index.max().timestamp())

    if triggered_id == 'stock-data':
        # Inizializzazione: 10 anni fa → oggi, clampati al range dei dati
        today         = pd.Timestamp.today().normalize()
        ten_years_ago = today - pd.DateOffset(years=10)
        s = max(ts_min, int(ten_years_ago.timestamp()))
        e = min(ts_max, int(today.timestamp()))
        d_start = pd.Timestamp(s, unit='s').strftime('%Y-%m-%d')
        d_end   = pd.Timestamp(e, unit='s').strftime('%Y-%m-%d')
        label   = (f"{pd.Timestamp(s, unit='s').strftime('%d/%m/%Y')} — "
                   f"{pd.Timestamp(e, unit='s').strftime('%d/%m/%Y')}")
        return d_start, d_end, label, [s, e]

    # L'utente ha cambiato una delle due date: non riscrivere i picker (evita loop)
    try:
        s = int(pd.Timestamp(start_date).timestamp()) if start_date else ts_min
        e = int(pd.Timestamp(end_date).timestamp())   if end_date   else ts_max
    except Exception:
        s, e = ts_min, ts_max

    s = max(ts_min, min(s, ts_max))
    e = max(ts_min, min(e, ts_max))

    label = (f"{pd.Timestamp(s, unit='s').strftime('%d/%m/%Y')} — "
             f"{pd.Timestamp(e, unit='s').strftime('%d/%m/%Y')}")

    return no_update, no_update, label, [s, e]


# ─────────────────────────────────────────────────────────────────────────────
# Callback: aggiorna grafico principale
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('controls-and-graph',        'figure'),
    Output('date-values',               'children'),
    Output('insufficient-data-tickers', 'value'),

    Input('update-portfolio-button', 'n_clicks'),
    Input('delete-column-button',    'n_clicks'),
    Input('controls-and-graph',      'clickData'),

    State('dr-start-tab1',           'date'),
    State('dr-end-tab1',             'date'),
    State('tab1-slider-store',       'data'),
    State('global-assets-selected',  'data'),
    State('benchmark-selector',      'value'),
    State('ir-window-input',         'value'),
    State('ak-ma-input',             'value'),
    State('weights-store-P1',        'data'),
    State('weights-store-P2',        'data'),
    State('weights-store-P3',        'data'),
    State({'type': 'ir-select-checkbox',     'index': ALL}, 'value'),
    State({'type': 'sharpe-select-checkbox', 'index': ALL}, 'value'),
    State({'type': 'tev-select-checkbox',    'index': ALL}, 'value'),
    State({'type': 'dd-select-checkbox',     'index': ALL}, 'value'),
    State({'type': 'vol-select-checkbox',    'index': ALL}, 'value'),
    State({'type': 'var90-select-checkbox',  'index': ALL}, 'value'),
    State({'type': 'var95-select-checkbox',  'index': ALL}, 'value'),
    State('vol-window-input',        'value'),
    State('insufficient-data-store', 'data'),
    State('selected-column',         'value'),
    State('stock-data',              'data'),
    prevent_initial_call=True,
)
def update_graph(update_clicks, delete_clicks, clickData, picker_start, picker_end, date_range, selected_assets,
                 benchmark_value, ir_window, ak_ma_window,
                 weights_p1_data, weights_p2_data, weights_p3_data,
                 all_ir_checkbox_values, all_sharpe_checkbox_values, all_tev_checkbox_values,
                 all_dd_checkbox_values, all_vol_checkbox_values,
                 all_var90_checkbox_values, all_var95_checkbox_values, vol_window,
                 insufficient_data, selected_column, stock_data):

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not update_clicks and not delete_clicks and not clickData:
        raise PreventUpdate

    if not stock_data:
        with _DL_LOCK:
            buf = dict(_DL_BUFFER)
        cr = buf.get('close_returns')
        if cr is not None and not cr.empty:
            stock_data = cr.to_json(date_format='iso', orient='split')
        else:
            return {}, 'Nessun dato caricato — clicca ⟳ Aggiorna', ''

    def _collect(vals):
        result = []
        for v in (vals or []):
            if v:
                result.extend(v)
        return result

    selected_irs    = _collect(all_ir_checkbox_values)
    selected_sharpe = _collect(all_sharpe_checkbox_values)
    selected_tev    = _collect(all_tev_checkbox_values)
    selected_dd     = _collect(all_dd_checkbox_values)
    selected_vol    = _collect(all_vol_checkbox_values)
    selected_var90  = _collect(all_var90_checkbox_values)
    selected_var95  = _collect(all_var95_checkbox_values)

    vol_window = vol_window if (vol_window and vol_window > 0) else 30
    ir_window  = ir_window  if (ir_window  and ir_window  > 0) else 30

    close_returns = _get_df(stock_data)

    # Priorità ai valori diretti dei DatePicker (evita race condition con tab1-slider-store)
    if picker_start and picker_end:
        start_date  = pd.to_datetime(picker_start)
        end_date    = pd.to_datetime(picker_end)
        filtered_df = close_returns.loc[start_date:end_date]
    elif date_range and len(date_range) == 2:
        start_date  = pd.to_datetime(date_range[0], unit='s')
        end_date    = pd.to_datetime(date_range[1], unit='s')
        filtered_df = close_returns.loc[start_date:end_date]
    else:
        filtered_df = close_returns
        start_date  = close_returns.index.min()
        end_date    = close_returns.index.max()

    asset_columns      = list(filtered_df.columns)
    df_with_portfolios = filtered_df.copy()

    weights_data_map = {
        'Port1': weights_p1_data,
        'Port2': weights_p2_data,
        'Port3': weights_p3_data,
    }

    def calc_portfolio(df, name, weights_dict, asset_cols):
        weights_dict = weights_dict or {}
        total_w = sum(weights_dict.values())
        normalized = {}
        if total_w > 0:
            for a, w in weights_dict.items():
                if a in asset_cols and w and w > 0:
                    normalized[a] = w / 100.0
        port = pd.Series(0.0, index=df.index, name=name)
        if normalized:
            for a, w in normalized.items():
                if a in df.columns:
                    port += df[a] * w
        return port

    for p_name, w_dict in weights_data_map.items():
        df_with_portfolios[p_name] = calc_portfolio(
            df_with_portfolios, p_name, w_dict, asset_columns)

    if benchmark_value is None or benchmark_value not in df_with_portfolios.columns:
        benchmark_col = df_with_portfolios.columns[0] if not df_with_portfolios.empty else None
    else:
        benchmark_col = benchmark_value

    if benchmark_col is None:
        return {}, 'Nessun dato caricato', ''

    benchmark_returns = df_with_portfolios[benchmark_col]
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1

    # Calcola IR/Sharpe/TEV solo per gli asset effettivamente selezionati
    _need_ir     = {s.replace('_InformationRatio', '') for s in selected_irs    if s}
    _need_sharpe = {si.replace('_Sharpe', '')          for si in selected_sharpe if si}
    _need_tev    = {ti.replace('_TEV', '')             for ti in selected_tev    if ti}

    _extra = {}
    for col in _need_ir:
        if col in df_with_portfolios.columns and col != benchmark_col:
            _extra[f'{col}_InformationRatio'] = calculate_rolling_information_ratio(
                df_with_portfolios[col], benchmark_returns, window=ir_window)
    for col in _need_sharpe:
        if col in df_with_portfolios.columns:
            _extra[f'{col}_SharpeRatio'] = calculate_rolling_sharpe_ratio(
                df_with_portfolios[col], window=ir_window)
    for col in _need_tev:
        if col in df_with_portfolios.columns and col != benchmark_col:
            _extra[f'{col}_TEV'] = calculate_tracking_error_volatility(
                df_with_portfolios[col], benchmark_returns, window=ir_window)

    if _extra:
        df_final = pd.concat([df_with_portfolios,
                               pd.DataFrame(_extra, index=df_with_portfolios.index)], axis=1)
    else:
        df_final = df_with_portfolios

    # Versione ridotta per il rendering (max 500 punti per trace)
    df_plot = _thin(df_final)
    benchmark_cumulative_returns = _thin(benchmark_cumulative_returns)

    fig = make_subplots(
        rows=7, cols=1, shared_xaxes=False, vertical_spacing=0.03,
        row_heights=[0.26, 0.12, 0.12, 0.12, 0.12, 0.13, 0.13],
        specs=[[{"secondary_y": False}]] * 7,
        subplot_titles=('Cumulative Returns', 'AKRatio', 'Sharpe Ratio',
                        'TEV', 'DrawDown', 'Volatilità', 'VaR (90% / 95%)')
    )

    for row in range(1, 8):
        fig.add_trace(
            go.Scatter(x=[df_final.index[0], df_final.index[-1]], y=[0, 0],
                       mode='lines', line=dict(color='rgba(0,0,0,0)'),
                       showlegend=False, hoverinfo='skip'),
            row=row, col=1)

    selected_assets = selected_assets or []

    # Subplot 1: Cumulative Returns
    if benchmark_col in selected_assets:
        bench_name = f'{benchmark_col} Cum. Returns'
        is_sel = (selected_column == bench_name)
        fig.add_trace(go.Scatter(x=benchmark_cumulative_returns.index,
                                  y=benchmark_cumulative_returns,
                                  name=bench_name,
                                  line=dict(color='red', width=8 if is_sel else 2),
                                  legend='legend'), row=1, col=1)
        color_index = 1
    else:
        color_index = 0

    for col in selected_assets:
        if col in df_with_portfolios.columns and col != benchmark_col:
            series = df_with_portfolios[col]
            if col.startswith('Port'):
                pnum    = int(col[-1])
                w_dict  = {1: weights_p1_data, 2: weights_p2_data, 3: weights_p3_data}.get(pnum, {}) or {}
                active  = [a for a, w in w_dict.items() if w and w > 0 and a in filtered_df.columns]
                if active:
                    first_valid = filtered_df[active].dropna(how='any').index.min()
                    if pd.notna(first_valid):
                        series = series.loc[first_valid:]
            cum_ret    = _thin((1 + series).cumprod() - 1)
            trace_name = f'{col} Cum. Returns'
            is_sel     = (selected_column == trace_name)
            if is_sel:
                line_dict = dict(color='red', width=8)
            else:
                tc = color_palette[color_index % len(color_palette)]
                line_dict = dict(color=tc)
                if col.startswith('Port'):
                    line_dict.update({'width': 4, 'dash': 'solid'})
            fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name=trace_name,
                                      line=line_dict, legend='legend'), row=1, col=1)
            color_index += 1

    # Mappa colore per asset (evita scan O(n²) in ogni subplot successivo)
    _color_map = {}
    for _t in fig.data:
        _nm = _t.name or ''
        if ' Cum. Returns' in _nm:
            _asset = _nm.replace(' Cum. Returns', '')
            try:
                _c = _t.line.color
                if _c:
                    _color_map[_asset] = _c
            except AttributeError:
                pass

    # Subplot 2: AKRatio
    _ak_ma = int(ak_ma_window) if ak_ma_window and int(ak_ma_window) > 1 else 1
    color_index = 0
    for col_ir in selected_irs:
        if col_ir in df_plot.columns:
            orig = col_ir.replace('_InformationRatio', '')
            is_sel = (selected_column == col_ir)
            if is_sel:
                line_dict = dict(color='red', width=8)
            else:
                tc = _color_map.get(orig, color_palette[color_index % len(color_palette)])
                line_dict = dict(color=tc, width=4 if orig.startswith('Port') else 2.5)
            y_vals = df_plot[col_ir].rolling(_ak_ma, min_periods=1).mean() if _ak_ma > 1 else df_plot[col_ir]
            fig.add_trace(go.Scatter(x=df_plot.index, y=y_vals,
                                      name=col_ir.replace('_InformationRatio', '_AKRatio'),
                                      line=line_dict, legend='legend2'), row=2, col=1)
            color_index += 1
    if selected_irs:
        fig.add_trace(go.Scatter(x=df_plot.index, y=[0]*len(df_plot), name='Zero Line (IR)',
                                  line=dict(color='red', dash='solid', width=2),
                                  showlegend=True, legend='legend2'), row=2, col=1)

    # Subplot 3: Sharpe
    for si in selected_sharpe:
        if si:
            an = si.replace('_Sharpe', '')
            sc = f'{an}_SharpeRatio'
            if sc in df_plot.columns:
                tn    = f'{an} Sharpe Ratio'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, dash='solid',
                                     width=4 if an.startswith('Port') else 2.5)
                y_vals = df_plot[sc].rolling(_ak_ma, min_periods=1).mean() if _ak_ma > 1 else df_plot[sc]
                fig.add_trace(go.Scatter(x=df_plot.index, y=y_vals, name=tn,
                                          line=line_dict, legend='legend3'), row=3, col=1)

    # Subplot 4: TEV
    for ti in selected_tev:
        if ti:
            an = ti.replace('_TEV', '')
            tc_nm = f'{an}_TEV'
            if tc_nm in df_plot.columns:
                tn    = f'{an} TEV'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, dash='dash',
                                     width=4 if an.startswith('Port') else 2.5)
                y_vals = df_plot[tc_nm].rolling(_ak_ma, min_periods=1).mean() if _ak_ma > 1 else df_plot[tc_nm]
                fig.add_trace(go.Scatter(x=df_plot.index, y=y_vals, name=tn,
                                          line=line_dict, legend='legend4'), row=4, col=1)

    # Subplot 5: DrawDown
    for di in selected_dd:
        if di:
            an = di.replace('_DD', '')
            if an in df_with_portfolios.columns:
                dds   = _thin(calculate_drawdown(df_with_portfolios[an]))
                tn    = f'{an} DrawDown'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                    fillcolor = 'rgba(200,0,0,0.08)'
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, dash='dot',
                                     width=4 if an.startswith('Port') else 2.5)
                    fillcolor = (tc.replace('rgb', 'rgba').replace(')', ',0.10)')
                                 if tc and tc.startswith('rgb') else 'rgba(200,0,0,0.08)')
                fig.add_trace(go.Scatter(x=dds.index, y=dds, name=tn,
                                          line=line_dict, legend='legend5',
                                          fill='tozeroy', fillcolor=fillcolor), row=5, col=1)

    # Subplot 6: Volatilità
    for vi in selected_vol:
        if vi:
            an = vi.replace('_Vol', '')
            if an in df_with_portfolios.columns:
                vs    = _thin(_rolling_volatility(df_with_portfolios[an], vol_window))
                tn    = f'{an} Volatilità'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, width=4 if an.startswith('Port') else 2.5)
                fig.add_trace(go.Scatter(x=vs.index, y=vs, name=tn,
                                          line=line_dict, legend='legend6'), row=6, col=1)

    # Subplot 7: VaR
    for v90 in selected_var90:
        if v90:
            an = v90.replace('_VaR90', '')
            if an in df_with_portfolios.columns:
                vs    = _thin(calculate_historical_cvar(df_with_portfolios[an], vol_window, 0.10))
                tn    = f'{an} VaR90'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, width=4 if an.startswith('Port') else 2.5,
                                     dash='solid')
                fig.add_trace(go.Scatter(x=vs.index, y=vs, name=tn,
                                          line=line_dict, legend='legend7'), row=7, col=1)

    for v95 in selected_var95:
        if v95:
            an = v95.replace('_VaR95', '')
            if an in df_with_portfolios.columns:
                vs    = _thin(calculate_historical_cvar(df_with_portfolios[an], vol_window, 0.05))
                tn    = f'{an} VaR95'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, width=4 if an.startswith('Port') else 2.5,
                                     dash='dot')
                fig.add_trace(go.Scatter(x=vs.index, y=vs, name=tn,
                                          line=line_dict, legend='legend7'), row=7, col=1)

    # ── Tooltip personalizzato per ogni linea ────────────────────────────────
    # Riquadro con bordo del colore della linea, sfondo bianco, testo nero.
    # Riga 1: data + valore  |  Riga 2: nome serie per esteso.
    for _tr in fig.data:
        try:
            _lc = _tr.line.color
        except AttributeError:
            continue
        if not _lc or _lc == 'rgba(0,0,0,0)':
            continue
        _nm = _tr.name or ''
        if not _nm:
            continue
        _fmt = '.1%' if any(k in _nm for k in ('Cum. Returns', 'DrawDown', 'VaR')) else '.4f'
        _tr.update(
            hovertemplate=(
                f'%{{x|%d/%m/%Y}}   %{{y:{_fmt}}}<br>'
                f'{_nm}'
                '<extra></extra>'
            ),
            hoverlabel=dict(
                bgcolor='white',
                bordercolor=_lc,
                font=dict(color='black', size=11),
                namelength=0,
            ),
        )

    for row in range(1, 8):
        fig.update_xaxes(title_text='', row=row, col=1)
    fig.update_yaxes(title_text='Cumulative Returns', row=1, col=1)
    fig.update_yaxes(title_text='AKRatio',            row=2, col=1)
    fig.update_yaxes(title_text='Sharpe Ratio',       row=3, col=1)
    fig.update_yaxes(title_text='TEV',                row=4, col=1)
    fig.update_yaxes(title_text='DrawDown',           row=5, col=1)
    fig.update_yaxes(title_text='Volatilità (ann.)',  row=6, col=1)
    fig.update_yaxes(title_text='VaR',                row=7, col=1)

    fig.update_layout(
        uirevision='graph',
        hovermode='closest',
        height=1900, showlegend=True,
        legend=dict(title=dict(text='<b>Asset</b>', font=dict(size=11)),
                    orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.01,
                    bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend2=dict(title=dict(text='<b>AKRatio</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.72, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend3=dict(title=dict(text='<b>Sharpe Ratio</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.58, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend4=dict(title=dict(text='<b>TEV</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.44, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend5=dict(title=dict(text='<b>DrawDown</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.30, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend6=dict(title=dict(text='<b>Volatilità</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.16, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend7=dict(title=dict(text='<b>VaR — (solido 90%, punteg. 95%)</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.02, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        margin=dict(b=20, t=60, r=220),
        autosize=True,
        xaxis=dict(range=[start_date, end_date]),
        xaxis2=dict(range=[start_date, end_date]),
        xaxis3=dict(range=[start_date, end_date]),
        xaxis4=dict(range=[start_date, end_date]),
        xaxis5=dict(range=[start_date, end_date]),
        xaxis6=dict(range=[start_date, end_date]),
        xaxis7=dict(range=[start_date, end_date]),
    )

    date_values = (f"Intervallo di date: {start_date.strftime('%d-%m-%Y')} — "
                   f"{end_date.strftime('%d-%m-%Y')}")
    insuff_text = (f"Ticker con dati insufficienti: {', '.join(insufficient_data)}"
                   if insufficient_data else "")

    return fig, date_values, insuff_text


# ─────────────────────────────────────────────────────────────────────────────
# Callback: sessione — toggle pannello
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('session-panel', 'style'),
    Input('session-toggle-btn', 'n_clicks'),
    State('session-panel', 'style'),
    prevent_initial_call=True,
)
def toggle_session_panel(n, current_style):
    if current_style and current_style.get('display') == 'none':
        return {'display': 'block', 'position': 'absolute', 'top': '70px', 'right': '10px',
                'z-index': '1000', 'background': 'white', 'border': '1px solid #ccc',
                'border-radius': '8px', 'box-shadow': '0 4px 20px rgba(0,0,0,0.15)',
                'padding': '16px 20px', 'width': '720px', 'max-width': '95vw'}
    return {'display': 'none'}


# ─────────────────────────────────────────────────────────────────────────────
# Callback: sessione — aggiorna lista
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('session-list-container', 'children'),
    Input('session-refresh-btn',    'n_clicks'),
    Input('session-save-btn',       'n_clicks'),
    Input('session-delete-trigger', 'data'),
    Input('session-toggle-btn',     'n_clicks'),
    prevent_initial_call=False,
)
def refresh_session_list(*_):
    sessions = list_sessions()
    if not sessions:
        return html.Div('Nessuna sessione salvata.',
                        style={'font-size': '11px', 'color': '#aaa',
                               'padding': '10px', 'text-align': 'center'})
    return [_build_session_row(r) for r in sessions]


# ─────────────────────────────────────────────────────────────────────────────
# Callback: sessione — salva
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('session-save-status', 'children'),
    Output('session-save-status', 'style'),
    Input('session-save-btn', 'n_clicks'),
    State('session-name-input', 'value'),
    State('session-desc-input', 'value'),
    *[State(sid, 'data') for sid in CLIENT_SESSION_STORES],
    prevent_initial_call=True,
)
def save_session_cb(n_clicks, name, desc, *store_values):
    if not n_clicks:
        raise PreventUpdate
    store_data = {sid: val for sid, val in zip(CLIENT_SESSION_STORES, store_values)
                  if val is not None}
    if not store_data:
        return ('⚠ Nessun dato da salvare. Carica prima i dati.',
                {'color': '#e65100', 'font-size': '10px', 'margin-top': '5px'})
    rec = save_session(name=name or '', description=desc or '', store_data=store_data)
    return (f"✅ Salvata: \"{rec['name']}\" ({rec['size_kb']} KB)",
            {'color': '#1b7a34', 'font-size': '10px', 'margin-top': '5px'})


# ─────────────────────────────────────────────────────────────────────────────
# Callback: sessione — elimina
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('session-delete-trigger', 'data'),
    Input({'type': 'session-delete-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def delete_session_cb(all_clicks):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered = ctx.triggered[0]
    if not triggered['value']:
        raise PreventUpdate
    try:
        id_dict    = json.loads(triggered['prop_id'].split('.')[0])
        session_id = id_dict['index']
    except Exception:
        raise PreventUpdate
    delete_session(session_id)
    return str(session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: sessione — seleziona per il caricamento
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('session-selected-id',  'data'),
    Output('session-load-trigger', 'data'),
    Input({'type': 'session-load-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def select_session(all_clicks):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered = ctx.triggered[0]
    if not triggered['value']:
        raise PreventUpdate
    try:
        id_dict    = json.loads(triggered['prop_id'].split('.')[0])
        session_id = id_dict['index']
    except Exception:
        raise PreventUpdate
    return session_id, session_id


@app.callback(
    *[Output(sid, 'data', allow_duplicate=True) for sid in CLIENT_SESSION_STORES],
    Input('session-load-trigger', 'data'),
    prevent_initial_call=True,
)
def load_session_cb(session_id):
    if not session_id:
        raise PreventUpdate
    store_data = load_session(session_id)
    return tuple(store_data.get(sid, no_update) for sid in CLIENT_SESSION_STORES)


# ─────────────────────────────────────────────────────────────────────────────
# Startup: carica tutti i pkl esistenti, scarica in background quelli mancanti
# ─────────────────────────────────────────────────────────────────────────────
def _startup_load():
    global _DL_STATE, _DL_BUFFER

    # Carica ETF (file attivo di default) nel buffer principale
    if _MARKET_DATA_FILE.exists():
        try:
            with open(_MARKET_DATA_FILE, 'rb') as f:
                data = pickle.load(f)
            with _DL_LOCK:
                _DL_BUFFER.update(data)
                _DL_STATE['status']  = 'done'
                _DL_STATE['current'] = 1
                _DL_STATE['total']   = 1
            print(f"✓ Dati ETF caricati da disco — {data.get('saved_at', '?')}")
        except Exception as e:
            print(f"⚠ Lettura market_data.pkl fallita: {e}")

    # Carica ARIMA dal file separato (se esiste)
    _arima_pkl = _arima_cache_path('ETF.xlsx')
    if _arima_pkl.exists():
        try:
            with open(_arima_pkl, 'rb') as f:
                arima_data = pickle.load(f)
            with _DL_LOCK:
                if arima_data.get('arima'):
                    _DL_BUFFER['arima']             = arima_data['arima']
                    _DL_BUFFER['arima_computed_at'] = arima_data.get('arima_computed_at', '')
            print(f"✓ ARIMA caricato da {_arima_pkl.name}")
        except Exception as e:
            print(f"⚠ Lettura {_arima_pkl.name} fallita: {e}")

    # Scarica in background tutti i file xlsx per cui manca il pkl
    def _bg_all():
        start = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
        xlsx_files = sorted(_FILES_DIR.glob('*.xlsx')) if _FILES_DIR.exists() else [Path(_XLSX)]
        for xlsx_path in xlsx_files:
            filename  = xlsx_path.name
            cache_pkl = _file_cache_path(filename)
            if Path(cache_pkl).exists():
                print(f"✓ Cache {filename} già presente — skip download")
                continue
            try:
                print(f"▶ Download iniziale {filename}…")
                tickers, descr, valuta = _build_ticker_list(filename)
                is_etf = (filename == 'ETF.xlsx')
                _do_download(tickers, descr, valuta, start,
                             cache_file=cache_pkl, update_buffer=is_etf)
                print(f"✓ Download completato: {filename}")
            except Exception as e:
                print(f"⚠ Download iniziale {filename} fallito: {e}")

    threading.Thread(target=_bg_all, daemon=True).start()

_startup_load()



# ─────────────────────────────────────────────────────────────────────────────
# ARIMA+GARCH notturno: calcola mu/cov e li scrive in market_data.pkl['arima']
# Il formato è quello letto da frontiera-efficiente/app.py → _read_arima_cache()
# ─────────────────────────────────────────────────────────────────────────────
def _compute_arima_garch(returns_df, window=250):
    """ARIMA(p,0,q) best-AIC + GARCH(1,1). Restituisce (mu_series, cov_df) annualizzati."""
    import warnings
    try:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    except ImportError:
        print("⚠ statsmodels non installato — ARIMA saltato")
        return None, None
    try:
        from arch import arch_model as _arch
        _has_arch = True
    except ImportError:
        _has_arch = False

    cols = [c for c in returns_df.columns if returns_df[c].dropna().shape[0] >= 60]
    mu_d, sig_d, resid_d = {}, {}, {}

    for i, col in enumerate(cols):
        # Conversione a rendimenti logaritmici: più stazionari e normali
        s_arith = returns_df[col].dropna().tail(window)
        s = np.log1p(s_arith)  # ln(1 + r_aritm) = rendimento logaritmico giornaliero

        best_aic  = np.inf
        best_mu   = float(s.mean())
        best_resid = (s - s.mean()).values

        for p in range(3):
            for q in range(3):
                if p == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        m = _ARIMA(s, order=(p, 0, q)).fit()
                    if m.aic < best_aic:
                        best_aic   = m.aic
                        best_resid = m.resid.values
                        # Usa la previsione a 252 passi (come il cono di previsione ARIMA):
                        # evita l'instabilità di const/(1-sum(AR)) per processi near unit root
                        fc = m.get_forecast(steps=252).predicted_mean
                        best_mu = float(fc.mean())
                except Exception:
                    pass
        mu_d[col]    = best_mu   # media logaritmica giornaliera incondizionata
        resid_d[col] = best_resid

        sig_fallback = float(np.std(best_resid)) or float(s.std())
        if _has_arch and len(best_resid) > 20:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    gm = _arch(best_resid * 100, vol='Garch', p=1, q=1, rescale=False)
                    gf = gm.fit(disp='off')
                    pr = gf.params
                    o, a, b = pr['omega'], pr['alpha[1]'], pr['beta[1]']
                    if a + b < 1.0:
                        sig_garch = float(np.sqrt(o / (1 - a - b))) / 100
                        # safeguard: rifiuta stime GARCH fuori range (< 1/10 o > 10x lo std storico)
                        sig_d[col] = sig_garch if sig_fallback / 10 < sig_garch < sig_fallback * 10 else sig_fallback
                    else:
                        sig_d[col] = sig_fallback
            except Exception:
                sig_d[col] = sig_fallback
        else:
            sig_d[col] = sig_fallback
        print(f"  [{i+1}/{len(cols)}] {col}: mu_log={mu_d[col]:.5f} vol_log={sig_d[col]:.5f}")

    if not mu_d:
        return None, None

    min_len = min(len(resid_d[c]) for c in cols)
    r_mat   = np.column_stack([resid_d[c][:min_len] for c in cols])
    corr    = np.nan_to_num(np.corrcoef(r_mat.T), nan=0.0)
    np.fill_diagonal(corr, 1.0)
    vols    = np.array([sig_d[c] for c in cols])
    D       = np.diag(vols)
    cov_d   = D @ corr @ D + 1e-8 * np.eye(len(cols))

    # Correzione di Jensen: converte rendimento log atteso in rendimento aritmetico atteso
    # E[R_aritm] = exp(mu_log * 252 + 0.5 * sigma_log^2 * 252) - 1
    def _jensen(mu_log, sig_log, s_arith_fallback):
        exp_arg = mu_log * 252 + 0.5 * sig_log**2 * 252
        if not np.isfinite(exp_arg) or exp_arg > 3.0:  # cap a ~1900% — oltre è numericamente rotto
            return float(s_arith_fallback.mean() * 252)
        return float(np.exp(exp_arg) - 1)

    mu_annual  = pd.Series(
        {c: _jensen(mu_d[c], sig_d[c], returns_df[c].dropna().tail(window)) for c in cols}
    )
    cov_annual = pd.DataFrame(cov_d * 252, index=cols, columns=cols)
    return mu_annual, cov_annual


def _save_arima_to_pkl(mu_series, cov_df, target_pkl=None):
    """Salva ARIMA nel file separato (NON dentro il pkl principale)."""
    ts = datetime.now().strftime('%d/%m/%Y %H:%M')
    arima_data = {
        'mu':          mu_series.to_dict(),
        'cov':         cov_df.to_dict(),
        'computed_at': ts,
    }
    # target_pkl qui è il percorso del FILE ARIMA separato (non il pkl principale)
    arima_pkl = Path(target_pkl) if target_pkl else _arima_cache_path('ETF.xlsx')
    try:
        existing = {}
        if arima_pkl.exists():
            with open(arima_pkl, 'rb') as f:
                existing = pickle.load(f)
        existing['arima']             = arima_data
        existing['arima_computed_at'] = ts
        _atomic_pkl_write(arima_pkl, existing)
        print(f"✓ ARIMA salvato in {arima_pkl.name} — {ts}")
    except Exception as e:
        print(f"⚠ Salvataggio ARIMA fallito ({arima_pkl.name}): {e}")
    # Aggiorna buffer in memoria (solo per il file ETF principale)
    if target_pkl is None or 'market_data_arima' in str(arima_pkl):
        with _DL_LOCK:
            _DL_BUFFER['arima']             = arima_data
            _DL_BUFFER['arima_computed_at'] = ts


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler: dati alle 18:30 (lun-ven), ARIMA+GARCH a mezzanotte (lun-sab)
# ─────────────────────────────────────────────────────────────────────────────
def _scheduled_update():
    """Aggiornamento notturno: scarica dati per tutti i file in Files/."""
    start = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    active_file = _active_file_store.get('filename', 'ETF.xlsx')
    xlsx_files = sorted(_FILES_DIR.glob('*.xlsx')) if _FILES_DIR.exists() else []
    if not xlsx_files:
        xlsx_files = [Path(_XLSX)]
    for xlsx_path in xlsx_files:
        filename = xlsx_path.name
        try:
            tickers, descr, valuta = _build_ticker_list(filename)
            cache = _file_cache_path(filename)
            is_active = (filename == active_file)
            print(f"⏰ Aggiornamento {filename}: {len(tickers)} ticker")
            _do_download(tickers, descr, valuta, start,
                         cache_file=cache, update_buffer=is_active)
        except Exception as e:
            print(f"⚠ Aggiornamento {filename} fallito: {e}")


def _scheduled_arima():
    """Calcolo ARIMA+GARCH a mezzanotte su tutti i file in Files/."""
    xlsx_files = sorted(_FILES_DIR.glob('*.xlsx')) if _FILES_DIR.exists() else [Path(_XLSX)]
    for xlsx_path in xlsx_files:
        filename = xlsx_path.name
        cache_pkl = _file_cache_path(filename)
        arima_pkl = _arima_cache_path(filename)
        try:
            if not Path(cache_pkl).exists():
                print(f"⚠ ARIMA {filename}: cache non trovata, skip")
                continue
            with open(cache_pkl, 'rb') as f:
                cached = pickle.load(f)
            ret = cached.get('close_returns')
            if ret is None or (hasattr(ret, 'empty') and ret.empty):
                print(f"⚠ ARIMA {filename}: nessun dato disponibile")
                continue
            print(f"🌙 ARIMA+GARCH {filename} — {len(ret.columns)} asset…")
            mu, cov = _compute_arima_garch(ret, window=250)
            if mu is not None:
                _save_arima_to_pkl(mu, cov, target_pkl=arima_pkl)
                print(f"✓ ARIMA+GARCH completato: {filename}")
            else:
                print(f"⚠ ARIMA {filename}: calcolo non riuscito")
        except Exception as e:
            print(f"⚠ ARIMA notturno {filename} fallito: {e}")


try:
    from apscheduler.schedulers.background import BackgroundScheduler
    _scheduler = BackgroundScheduler(timezone='Europe/Rome')
    _scheduler.add_job(_scheduled_update, 'cron', hour=0, minute=0,
                       day_of_week='mon-sat', misfire_grace_time=3600)
    _scheduler.add_job(_scheduled_arima,  'cron', hour=0, minute=30,
                       day_of_week='mon-sat', misfire_grace_time=3600)
    _scheduler.start()
    print("✓ Scheduler avviato — dati mezzanotte, ARIMA 00:30 (lun-sab)")
except ImportError:
    print("⚠ apscheduler non installato — aggiornamento automatico disabilitato")
    print("  pip install apscheduler")

# ─────────────────────────────────────────────────────────────────────────────
server = app.server   # esposto per gunicorn

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8051))
    app.run(debug=False, port=port, host='0.0.0.0')
