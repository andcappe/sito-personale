"""
Frontiera Efficiente — App standalone
Ottimizzazione di portafoglio alla Markowitz con visualizzazione interattiva.
"""

import io
import json
import pickle
import sys
import threading
import os
import base64
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update, ALL, ctx, Patch
from dash.exceptions import PreventUpdate

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.browser_css import BROWSER_RESET_CSS, FONT
from settings.caratteri import NOME_ASSET
import sessions_manager as _sm
import arima_utenti
import time

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
_EXT = [
    'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]
app  = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=_EXT,
            requests_pathname_prefix='/frontiera/',
            routes_pathname_prefix='/frontiera/')
app.server.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

_ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

def _cloud_push(path):
    """Replica il file su storage persistente (S3/R2) se configurato. Best-effort."""
    try:
        import sys as _sys
        if str(_ROOT_DIR) not in _sys.path:
            _sys.path.insert(0, str(_ROOT_DIR))
        import cloud_storage
        cloud_storage.push(path)
    except Exception:
        pass

def _get_username():
    try:
        from flask import session as _fs
        u = _fs.get('username')
        print(f"[fe _get_username] flask session username={u!r}", flush=True)
        return u or 'anon'
    except Exception as e:
        print(f"[fe _get_username] exception: {e}", flush=True)
        return 'anon'

def _read_user_json():
    """Solo voci-asset di current.json (chiavi meta tipo _tipo escluse)."""
    try:
        u = _get_username()
        raw = json.load(open(_ROOT_DIR / 'sessions' / u / 'current.json'))
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    except Exception:
        return {}


def _fe_consistency(data):
    """Coerenza profilo: ogni asset ha dates+returns della stessa lunghezza.
    Le chiavi meta (non-dict, es. _tipo) vengono ignorate."""
    if not isinstance(data, dict):
        return False
    for v in data.values():
        if not isinstance(v, dict):
            continue
        d, r = v.get('dates'), v.get('returns')
        if not d or not r or len(d) != len(r):
            return False
    return True


_FE_JSON_WRITE_LOCK = threading.Lock()


def _fe_atomic_json_write(path, data):
    """Scrittura atomica (temp+rename) + validazione: mai file a metà o incoerente.
    Temp UNIVOCO (pid+thread): la Frontiera gira nello stesso processo del
    Portafoglio; un .tmp condiviso causerebbe scritture concorrenti sovrapposte
    e JSON corrotto su current.json."""
    if not _fe_consistency(data):
        print("⚠ [frontiera] profilo incoerente — non salvato")
        return False
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
    try:
        with _FE_JSON_WRITE_LOCK:
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, path)
        _cloud_push(path)       # replica su storage persistente (S3/R2)
        return True
    except Exception as e:
        print(f"⚠ [frontiera] scrittura current.json fallita: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def _update_user_json_fe(checked=None, weights=None):
    try:
        u = _get_username()
        path = _ROOT_DIR / 'sessions' / u / 'current.json'
        data = json.load(open(path))
    except Exception:
        return
    changed = False
    if checked is not None:
        s = set(checked)
        for desc in data:
            if not isinstance(data[desc], dict):
                continue
            v = desc in s
            if data[desc].get('checked') != v:
                data[desc]['checked'] = v
                changed = True
    if weights is not None:
        for desc in data:
            if not isinstance(data[desc], dict):
                continue
            for k in ('P1', 'P2', 'P3'):
                v = float(weights.get(k, {}).get(desc, 0) or 0)
                if data[desc].get(k) != v:
                    data[desc][k] = v
                    changed = True
    if changed:
        _fe_atomic_json_write(_ROOT_DIR / 'sessions' / _get_username() / 'current.json', data)

app.index_string = '''
<!DOCTYPE html><html>
<head>{%metas%}<title>Frontiera Efficiente — Andrea Cappelletti</title>{%favicon%}{%css%}
<style>
''' + BROWSER_RESET_CSS + '''
  @keyframes fe-spin { to { transform: rotate(360deg); } }
  .fe-box .SingleDatePickerInput { border: 1px solid #ccd9ee; border-radius: 4px; }
  .fe-box .DateInput { width: 104px; }
  .fe-box .DateInput_input { font-size: 13px; padding: 4px 8px; height: 28px; line-height: 20px; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
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
<script>
(function(){
  if (location.search.indexOf('embed=1') === -1) return;
  function apply(){
    var nb = document.getElementById('site-navbar');
    if (nb) nb.style.display = 'none';
    document.querySelectorAll('[style*="margin-top: 106"],[style*="margin-top:106"]')
      .forEach(function(e){ e.style.marginTop = '8px'; });
  }
  var n = 0, iv = setInterval(function(){ apply(); if (++n > 30) clearInterval(iv); }, 100);
  document.addEventListener('DOMContentLoaded', apply);
})();
</script>
</body></html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Configurazione ottimizzatore
# 0 = singolo start (veloce, originale)
# 1 = doppio start: warm + uniforme, tiene il migliore (lieve overhead)
# 2 = multi-start: warm + uniforme + MRP + mu-proporzionale (più preciso, ~4x più lento)
FRONTIER_MULTISTART = 0
# ─────────────────────────────────────────────────────────────────────────────
# Funzioni matematiche
# ─────────────────────────────────────────────────────────────────────────────
def _port_perf(w, mu, cov):
    ret = float(np.sum(mu * w) * 252)
    vol = float(np.sqrt(np.dot(w.T, np.dot(cov * 252, w))))
    return ret, vol

def _port_cvar(w, returns_df, pct):
    """CVaR: annualised mean of the worst-pct% returns."""
    pr = returns_df.values @ w
    threshold = np.percentile(pr, pct)
    tail = pr[pr <= threshold]
    mean_tail = float(np.mean(tail)) if len(tail) > 0 else float(threshold)
    return float(-mean_tail * np.sqrt(252))

_ARIMA_LOCK  = threading.Lock()
# Calcoli ARIMA in background, uno per richiesta: req_id → {pct, total, done, error}.
# Prima lo stato era uno solo per tutto il processo (due utenti si rubavano la barra).
_ARIMA_JOBS = {}
_ARIMA_FALLITI = {}      # utente → istante dell'ultimo calcolo fallito (niente giri a vuoto)
_ARIMA_RIPROVA_S = 900
_ARIMA_CACHE_INFO = {'available': False, 'ts': ''}

# ── Stili barra di avanzamento ARIMA (riusati da più callback) ───────────────
_ARIMA_PROG_HIDDEN = {'display': 'none'}
_ARIMA_PROG_SHOWN  = {'display': 'flex', 'flexDirection': 'column', 'gap': '5px',
                      'padding': '8px 16px', 'background': '#eef4ff',
                      'borderRadius': '8px', 'margin': '4px 0',
                      'flexShrink': '0', 'minWidth': '300px'}


def _arima_bar_style(pct, color='#1a3a6b'):
    """Stile del riempimento della barra di avanzamento (0–100%)."""
    return {'width': f'{max(0, min(100, int(pct)))}%', 'height': '100%',
            'background': color, 'borderRadius': '6px',
            'transition': 'width 0.3s ease'}


def _run_arima_thread(req_id, username, returns_all, assets, tickers):
    """Stima ARIMA+GARCH degli asset mancanti e li aggiunge al file dell'utente
    (arima_utenti): il job notturno poi li rifà con i dati nuovi."""
    def _avanza(fatti, totale):
        with _ARIMA_LOCK:
            job = _ARIMA_JOBS.get(req_id)
            if job is not None:
                job['pct'], job['total'] = fatti, totale
    try:
        arima_utenti.aggiorna(username, returns_all, assets, tickers=tickers, avanzamento=_avanza)
        with _ARIMA_LOCK:
            _ARIMA_JOBS[req_id].update({'done': True, 'fine': time.time()})
    except Exception as e:
        print(f"⚠ [frontiera] ARIMA {username} fallito: {e}", flush=True)
        _ARIMA_FALLITI[username] = time.time()
        with _ARIMA_LOCK:
            _ARIMA_JOBS[req_id].update({'done': True, 'error': str(e), 'fine': time.time()})


def calc_frontier(returns_df, n=20, wmin=0.0, wmax=1.0, rf=0.02, risk='vol', mu_override=None, cov_override=None):
    mu  = mu_override if mu_override is not None else returns_df.mean()
    cov = cov_override if cov_override is not None else returns_df.cov()
    # Sanifica NaN nella matrice di covarianza (asset senza dati nella finestra)
    if hasattr(cov, 'values'):
        cov_arr = np.nan_to_num(cov.values, nan=0.0)
        np.fill_diagonal(cov_arr, np.where(np.isnan(np.diag(cov.values)), 0.0, np.diag(cov.values)))
        cov = pd.DataFrame(cov_arr, index=cov.index, columns=cov.columns)
    na  = len(returns_df.columns)
    bounds = tuple((wmin, wmax) for _ in range(na))
    eq     = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    w0     = np.array([1/na] * na)
    opts   = {'ftol': 1e-10, 'maxiter': 2000}

    def _opt(obj, w_start, extra_cs=(), extra_starts=()):
        cs = (eq,) + tuple(extra_cs)
        if FRONTIER_MULTISTART == 0:
            # singolo start
            r = minimize(obj, w_start, method='SLSQP', bounds=bounds, constraints=cs, options=opts)
            if not r.success:
                r = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cs, options=opts)
            return r
        elif FRONTIER_MULTISTART == 1:
            # doppio start: warm + uniforme
            candidates = (w_start, w0)
        else:
            # multi-start completo
            candidates = (w_start, w0) + tuple(extra_starts)
        best = None
        for w_init in candidates:
            r = minimize(obj, w_init, method='SLSQP', bounds=bounds, constraints=cs, options=opts)
            if r.success and (best is None or r.fun < best.fun):
                best = r
        return best if best is not None else minimize(obj, w0, method='SLSQP',
                                                      bounds=bounds, constraints=cs, options=opts)

    # ── Portafoglio a massimo rendimento (estremo destro della frontiera) ──────
    # Risolto esplicitamente: per un universo senza vincoli superiori = 100% nell'asset
    # con rendimento più alto; con vincoli wmax < 1 l'ottimizzatore trova l'optimum reale.
    def neg_ret(w): return -_port_perf(w, mu, cov)[0]
    max_res = _opt(neg_ret, w0)
    max_ret, _ = _port_perf(max_res.x, mu, cov)

    mu_arr = mu.values if hasattr(mu, 'values') else np.array(mu)
    mu_w   = np.clip(mu_arr - mu_arr.min() + 1e-8, 0, None)
    mu_w   = np.clip(mu_w, wmin, wmax); mu_w /= mu_w.sum()

    if risk in ('standard', 'vol', 'arima_garch'):
        def obj_risk(w): return _port_perf(w, mu, cov)[1]

        # Portafoglio a minimo rischio (estremo sinistro)
        min_res = _opt(obj_risk, w0)
        min_ret, min_risk = _port_perf(min_res.x, mu, cov)

        rows = [{'Return': min_ret, 'Volatility': min_risk,
                 'Sharpe': (min_ret-rf)/min_risk if min_risk>0 else 0,
                 'Weights': min_res.x}]

        # Portafogli intermedi: target uniformi tra MVP e MRP
        targets = np.linspace(min_ret, max_ret, n)[1:-1]
        prev_w  = min_res.x.copy()
        for t in targets:
            ret_cs = {'type':'eq','fun': lambda x,t=t: _port_perf(x,mu,cov)[0]-t}
            r = _opt(obj_risk, prev_w, extra_cs=(ret_cs,), extra_starts=(max_res.x, mu_w))
            if r.success:
                ret, vol_p = _port_perf(r.x, mu, cov)
                rows.append({'Return':ret,'Volatility':vol_p,
                             'Sharpe':(ret-rf)/vol_p if vol_p>0 else 0,'Weights':r.x})
                prev_w = r.x.copy()

        # Portafoglio a massimo rendimento (estremo esplicito)
        max_risk = _port_perf(max_res.x, mu, cov)[1]
        rows.append({'Return': max_ret, 'Volatility': max_risk,
                     'Sharpe': (max_ret-rf)/max_risk if max_risk>0 else 0,
                     'Weights': max_res.x})

    else:  # CVaR
        pct = 10 if risk == 'cvar90' else 5
        def obj_risk(w): return _port_cvar(w, returns_df, pct)

        min_res  = _opt(obj_risk, w0)
        min_ret, _ = _port_perf(min_res.x, mu, cov)
        min_cvar = obj_risk(min_res.x)

        rows = [{'Return': min_ret, 'Volatility': min_cvar,
                 'Sharpe': (min_ret-rf)/min_cvar if min_cvar>0 else 0,
                 'Weights': min_res.x}]

        targets = np.linspace(min_ret, max_ret, n)[1:-1]
        prev_w  = min_res.x.copy()
        for t in targets:
            ret_cs = {'type':'eq','fun': lambda x,t=t: _port_perf(x,mu,cov)[0]-t}
            r = _opt(obj_risk, prev_w, extra_cs=(ret_cs,), extra_starts=(max_res.x, mu_w))
            if r.success:
                ret, _ = _port_perf(r.x, mu, cov)
                v      = obj_risk(r.x)
                rows.append({'Return':ret,'Volatility':v,
                             'Sharpe':(ret-rf)/v if v>0 else 0,'Weights':r.x})
                prev_w = r.x.copy()

        max_cvar = obj_risk(max_res.x)
        rows.append({'Return': max_ret, 'Volatility': max_cvar,
                     'Sharpe': (max_ret-rf)/max_cvar if max_cvar>0 else 0,
                     'Weights': max_res.x})

    df_f = pd.DataFrame(rows)
    if not df_f.empty:
        idx_ms     = df_f['Sharpe'].idxmax()
        max_sharpe = df_f.loc[idx_ms]
        idx_mv     = df_f['Volatility'].idxmin()
        min_vol    = df_f.loc[idx_mv]
    else:
        max_sharpe = None
        min_vol    = None

    return df_f, max_sharpe, min_vol, returns_df.columns.tolist()


def calc_single_portfolio(weights_dict, returns_df, rf=0.02):
    names = returns_df.columns.tolist()
    w = np.array([weights_dict.get(n, 0)/100 for n in names], dtype=float)
    s = w.sum()
    if s > 0:
        w = w / s
    mu  = returns_df.mean()
    cov = returns_df.cov()
    ret, vol = _port_perf(w, mu, cov)
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return ret, vol, sharpe, w

_PORT_PKL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'portafoglio', 'sessions', 'market_data.pkl',
))
_PORT_SESSIONS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'portafoglio', 'sessions',
))

_FE_FILES_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / 'Files'

_FE_FILE_ORDER = ['ETF', 'CRIPTO', 'COMMODITIES']



def _fe_port_cache_path(filename='ETF.xlsx'):
    """Percorso del pkl di portafoglio corrispondente a filename."""
    stem = Path(filename).stem
    if stem == 'ETF':
        return _PORT_PKL
    return os.path.join(_PORT_SESSIONS, f'market_data_{stem}.pkl')

_FE_USER_PKL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'sessions', 'user_data.pkl',
)



_FE_DEFAULT_ARIMA_PKL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'sessions', 'default_arima.pkl',
)


def _save_arima(mu_series, cov_df, source='user'):
    computed_at = datetime.now().strftime('%d/%m/%Y %H:%M')
    arima_block = {
        'mu': mu_series.to_dict(),
        'cov': cov_df.to_dict(),
        'computed_at': computed_at,
    }
    if source == 'default':
        target = _FE_DEFAULT_ARIMA_PKL
        try:
            existing = {}
            if os.path.exists(target):
                with open(target, 'rb') as f:
                    existing = pickle.load(f)
            existing['arima'] = arima_block
            with open(target, 'wb') as f:
                pickle.dump(existing, f)
        except Exception:
            pass
    else:
        try:
            existing = {}
            if os.path.exists(_FE_USER_PKL):
                with open(_FE_USER_PKL, 'rb') as f:
                    existing = pickle.load(f)
            existing['arima'] = arima_block
            with open(_FE_USER_PKL, 'wb') as f:
                pickle.dump(existing, f)
        except Exception:
            pass


def _read_default_arima():
    try:
        if os.path.exists(_FE_DEFAULT_ARIMA_PKL):
            with open(_FE_DEFAULT_ARIMA_PKL, 'rb') as f:
                d = pickle.load(f)
            return d.get('arima')
    except Exception:
        pass
    return None


def _read_user_data():
    try:
        if os.path.exists(_FE_USER_PKL):
            with open(_FE_USER_PKL, 'rb') as f:
                d = pickle.load(f)
            prices  = d.get('prices')
            returns = d.get('returns')
            arima   = d.get('arima')
            if prices is not None and returns is not None:
                return prices, returns, d.get('saved_at', ''), arima
    except Exception:
        pass
    return None, None, '', None


def _reconstruct_from_json_fe(ns):
    """Ricostruisce (prices_df, returns_df) dal JSON utente (source of truth)."""
    try:
        first = next(iter(ns.values()))
        dates = pd.to_datetime(first['dates'])
        pr, ret = {}, {}
        for desc, v in ns.items():
            p = v.get('prices') or []
            r = v.get('returns') or []
            if p:
                pr[desc]  = [float(x) if x is not None else float('nan') for x in p]
            if r:
                ret[desc] = [float(x) if x is not None else float('nan') for x in r]
        op = pd.DataFrame(pr,  index=dates) if pr  else None
        cr = pd.DataFrame(ret, index=dates) if ret else None
        return op, cr
    except Exception:
        return None, None


def _read_shared_data():
    """
    Legge i dati di portafoglio condivisi.
    0. JSON utente (source of truth — sempre coerente con lista asset).
    1. Buffer live di _app_portafoglio nel processo wsgi.
    2. Fallback: market_data.pkl per esecuzione standalone.
    Ritorna (prices_df, returns_df, saved_at) o (None, None, None).
    """
    # — JSON utente (source of truth) —
    try:
        ns = _read_user_json()
        if ns:
            op, cr = _reconstruct_from_json_fe(ns)
            if op is not None and cr is not None:
                return op, cr, ''
    except Exception:
        pass

    # — buffer live (wsgi.py carica portafoglio prima di frontiera) —
    try:
        port = sys.modules.get('_app_portafoglio')
        if port is not None:
            with port._DL_LOCK:
                buf = dict(port._DL_BUFFER)
            prices  = buf.get('original_prices')
            returns = buf.get('close_returns')
            if prices is not None and returns is not None:
                return prices, returns, buf.get('saved_at', '')
    except Exception:
        pass

    # — fallback: pkl salvato da portafoglio —
    try:
        if os.path.exists(_PORT_PKL):
            with open(_PORT_PKL, 'rb') as f:
                data = pickle.load(f)
            prices  = data.get('original_prices')
            returns = data.get('close_returns')
            if prices is not None and returns is not None:
                return prices, returns, data.get('saved_at', '')
    except Exception:
        pass

    return None, None, None


def _read_arima_from_pkl(pkl_path):
    """Legge mu/cov ARIMA da un pkl specifico. Restituisce (mu, cov, ts) o (None, None, None)."""
    try:
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            arima = data.get('arima')
            if arima and isinstance(arima, dict):
                mu_raw  = arima.get('mu')
                cov_raw = arima.get('cov')
                ts      = arima.get('computed_at', '')
                if mu_raw and cov_raw:
                    return pd.Series(mu_raw), pd.DataFrame(cov_raw), ts
    except Exception:
        pass
    return None, None, None


def _read_arima_cache():
    """Legge mu/cov ARIMA pre-calcolati dal buffer live o dal pkl.
    Preferisce il dato più recente tra buffer e pkl (confronto timestamp).
    Restituisce (mu_series, cov_df, computed_at) o (None, None, None)."""
    def _parse(arima_data):
        if not arima_data or not isinstance(arima_data, dict):
            return None, None, None
        mu_raw  = arima_data.get('mu')
        cov_raw = arima_data.get('cov')
        ts      = arima_data.get('computed_at', '')
        if not mu_raw or not cov_raw:
            return None, None, None
        mu_series = pd.Series(mu_raw)
        cov_df    = pd.DataFrame(cov_raw)
        return mu_series, cov_df, ts

    def _parse_ts(ts_str):
        from datetime import datetime
        for fmt in ('%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S'):
            try:
                return datetime.strptime(ts_str, fmt)
            except Exception:
                pass
        return None

    buf_mu = buf_cov = buf_ts = None
    buf_raw = None
    try:
        port = sys.modules.get('_app_portafoglio')
        if port is not None:
            with port._DL_LOCK:
                buf_raw = port._DL_BUFFER.get('arima')
            buf_mu, buf_cov, buf_ts = _parse(buf_raw)
    except Exception:
        pass

    pkl_mu = pkl_cov = pkl_ts = None
    pkl_raw = None
    try:
        # Prima cerca nel file ARIMA separato (nuovo formato)
        _arima_sep = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'portafoglio', 'sessions', 'market_data_arima.pkl',
        ))
        _src = _arima_sep if os.path.exists(_arima_sep) else _PORT_PKL
        if os.path.exists(_src):
            with open(_src, 'rb') as f:
                pkl_data = pickle.load(f)
            pkl_raw = pkl_data.get('arima')
            pkl_mu, pkl_cov, pkl_ts = _parse(pkl_raw)
    except Exception:
        pass

    # Usa il dato più recente: se il pkl è più nuovo, aggiorna anche il buffer
    if buf_mu is not None and pkl_mu is not None:
        buf_dt = _parse_ts(buf_ts or '')
        pkl_dt = _parse_ts(pkl_ts or '')
        if pkl_dt and buf_dt and pkl_dt > buf_dt:
            try:
                port = sys.modules.get('_app_portafoglio')
                if port is not None:
                    with port._DL_LOCK:
                        port._DL_BUFFER['arima']             = pkl_raw
                        port._DL_BUFFER['arima_computed_at'] = pkl_ts
            except Exception:
                pass
            return pkl_mu, pkl_cov, pkl_ts
        return buf_mu, buf_cov, buf_ts

    if buf_mu is not None:
        return buf_mu, buf_cov, buf_ts
    if pkl_mu is not None:
        return pkl_mu, pkl_cov, pkl_ts

    # Fallback: default_arima.pkl (calcolato localmente sulla frontiera per asset di default)
    def_raw = _read_default_arima()
    if def_raw:
        def_mu, def_cov, def_ts = _parse(def_raw)
        if def_mu is not None:
            return def_mu, def_cov, def_ts

    return None, None, None


def _get_arima_cache(src):
    """(mu, cov) della cache ARIMA per la sorgente ('user' | 'default').
    (None, None) se assente. Stessa logica di lettura usata da calc_and_render."""
    try:
        if src == 'user':
            arima = _read_user_data()[3]
            if arima and isinstance(arima, dict) and arima.get('mu') and arima.get('cov'):
                return pd.Series(arima['mu']), pd.DataFrame(arima['cov'])
            return None, None
        mu, cov, _ = _read_arima_from_pkl(_PORT_PKL)
        if mu is None:
            mu, cov, _ = _read_arima_cache()
        return mu, cov
    except Exception:
        return None, None


def _refresh_arima_cache_info():
    """Aggiorna _ARIMA_CACHE_INFO leggendo il pkl una volta sola (chiamata ad avvio e post-calcolo)."""
    mu, _, ts = _read_arima_cache()
    _ARIMA_CACHE_INFO['available'] = mu is not None
    _ARIMA_CACHE_INFO['ts'] = ts or ''


# Leggi la cache ARIMA una volta all'avvio (in un thread per non bloccare l'import)
threading.Thread(target=_refresh_arima_cache_info, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers dati
# ─────────────────────────────────────────────────────────────────────────────
_DF_CACHE: dict = {}

# Tetto ai rendimenti giornalieri: oltre ±50%/giorno è quasi sempre un tick
# corrotto che fa esplodere volatilità/covarianze (anche in ARIMA+GARCH).
_MAX_DAILY_RET = 0.5

def _get_returns(data_json):
    if not data_json:
        return None
    key = hash(data_json[:200])
    if key not in _DF_CACHE:
        df = pd.read_json(io.StringIO(data_json), orient='split')
        df.index = pd.to_datetime(df.index)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.clip(-_MAX_DAILY_RET, _MAX_DAILY_RET)   # neutralizza tick corrotti
        _DF_CACHE[key] = df
    return _DF_CACHE[key].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Stili
# ─────────────────────────────────────────────────────────────────────────────
# Colori frontiere
_FC = {'F1': '#0066cc', 'F2': '#2ca02c', 'F3': '#d62728'}   # blu, verde, rosso = P1, P2, P3
_CML_C = {'F1': '#004c99', 'F2': '#1e7a1e', 'F3': '#a01b1c'}
_PALETTE = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
    '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5',
]


def _short_history(returns_df):
    """Restituisce un dict {asset: 'YYYY-MM-DD'} per gli asset che iniziano
    dopo la prima data disponibile nel dataset (storia incompleta)."""
    first_dates = {col: returns_df[col].first_valid_index() for col in returns_df.columns}
    valid_starts = [d for d in first_dates.values() if d is not None]
    if not valid_starts:
        return {}
    dataset_start = min(valid_starts)
    return {asset: d.strftime('%d/%m/%Y')
            for asset, d in first_dates.items()
            if d is not None and d > dataset_start}

# Filo blu intorno al nome = serie storica più corta del dataset: stesso filo
# dell'Analisi Portafoglio (_PF_BLU_STORIA / _pf_fili), testo sempre del colore base.
_FE_BLU_STORIA = '#1d6fe0'


def _asset_name_div(asset, short_map):
    corta   = asset in short_map
    color   = _FE_BLU_STORIA if corta else '#1a3a5c'
    tooltip = f'{asset} — serie storica dal {short_map[asset]}' if corta else asset
    return html.Div(
        html.Span(asset, style={'overflow':'hidden','whiteSpace':'nowrap',
                                'textOverflow':'ellipsis','maxWidth':'100%',
                                **NOME_ASSET,'color':'#1a3a5c',
                                'padding':'2px 6px','borderRadius':'5px',
                                'boxSizing':'border-box',
                                'boxShadow': f'inset 0 0 0 1.5px {_FE_BLU_STORIA}' if corta else 'none'}),
        **{'data-tooltip': tooltip, 'data-tooltip-color': color},
        style={'width':'25%','height':'24px','display':'flex','alignItems':'center',
               'paddingLeft':'0px','overflow':'hidden','position':'relative','cursor':'default'}
    )

def _w_cell(w, color):
    if w is None or w < 0.05:
        return html.Span('—', style={'fontSize':'10px','color':'#bbb'})
    return html.Span(f'{w:.1f}%', style={'fontSize':'10px','fontWeight':'700','color': color})

# ─────────────────────────────────────────────────────────────────────────────
# Helper: grafici andamento (performance cumulativa) e drawdown
# ─────────────────────────────────────────────────────────────────────────────
# Stessa impostazione dell'Analisi Portafoglio: un colore per asset (palette
# _PF_COLORI_CELL del Portafoglio), uguale nei due grafici, e una sola legenda in
# cima all'andamento. Il drawdown non ha legenda: le serie hanno gli stessi nomi e
# il clientside callback _fe_sincronizza_legenda gli riporta ogni clic (anche il
# doppio clic che isola un titolo). Le due figure hanno gli stessi margini
# sinistro/destro, così gli assi delle date sono allineati.
_FE_COLORI_ASSET = ['#0047ff', '#ff8c00', '#00a651', '#9400d3', '#000000', '#00bcd4',
                    '#ff1493', '#8b4513', '#c8a800', '#708090', '#006400', '#000080',
                    '#ff6f61', '#7fff00', '#b03060', '#40e0d0']
_FE_MARGINE = dict(l=60, r=20, b=40)


def _fe_grafico_vuoto(testo='Seleziona asset (📊) o calcola le frontiere'):
    return go.Figure().update_layout(
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=testo, xref='paper', yref='paper', x=0.5, y=0.5,
                          showarrow=False, font=dict(size=13, color='#6b7a99'))])


def _fe_prezzi(prices_data, date_start, date_end):
    prices_df = pd.read_json(io.StringIO(prices_data), orient='split')
    prices_df.index = pd.to_datetime(prices_df.index)
    if hasattr(prices_df.index, 'tz') and prices_df.index.tz is not None:
        prices_df.index = prices_df.index.tz_localize(None)
    if date_start: prices_df = prices_df.loc[date_start:]
    if date_end:   prices_df = prices_df.loc[:date_end]
    return prices_df


def _fe_serie(prices_data, chart_assets, frontier_weights, show_frontiers,
              date_start, date_end, trasforma):
    """[(nome, colore, larghezza, serie)] per gli asset 📊 e i portafogli F1/F2/F3
    accesi. `trasforma` riceve l'indice cumulato (base 1) e restituisce la serie da
    disegnare. I colori dipendono solo dall'ordine degli asset, quindi sono uguali
    nell'andamento e nel drawdown."""
    prices_df = _fe_prezzi(prices_data, date_start, date_end)
    ret_df = prices_df.pct_change().clip(-_MAX_DAILY_RET, _MAX_DAILY_RET)
    out = []
    i = 0
    for asset in chart_assets or []:
        if asset not in prices_df.columns:
            continue
        s = prices_df[asset].dropna()
        if len(s) <= 1:
            continue
        cum = (1 + s.pct_change().clip(-_MAX_DAILY_RET, _MAX_DAILY_RET).fillna(0)).cumprod()
        out.append((asset, _FE_COLORI_ASSET[i % len(_FE_COLORI_ASSET)], 1.5, trasforma(cum), None))
        i += 1
    # linee continue: F1/F2/F3 a 3 px, i portafogli dell'Analisi a 2 px
    voci = [(f, c, f'Portafoglio {f}', 3) for f, c in _FC.items()]
    voci += [(p, c, f'Portafoglio {p} (Analisi)', 2) for p, c in _P_COLORS.items()]
    for fname, fcolor, nome, larg in voci:
        if not (show_frontiers or {}).get(fname, False):
            continue
        fw = (frontier_weights or {}).get(fname, {})
        port_cols = [c for c in fw if fw[c] > 0 and c in prices_df.columns]
        if not port_cols:
            continue
        w_raw = np.array([fw[c] for c in port_cols], dtype=float)
        w_raw /= w_raw.sum()
        common_start = max(prices_df[c].first_valid_index() for c in port_cols)
        sub_ret = ret_df.loc[common_start:, port_cols].dropna(how='any')
        if len(sub_ret) < 2:
            continue
        cum = (1 + pd.Series(sub_ret.values @ w_raw, index=sub_ret.index)).cumprod()
        out.append((nome, fcolor, larg, trasforma(cum), None))
    return out


def _fe_figura(serie, titolo, asse_y, con_legenda):
    fig = go.Figure()
    for nome, colore, larg, s, *tratto in serie:
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode='lines', name=nome, legendgroup=nome,
            showlegend=con_legenda,
            line=dict(width=larg, color=colore, dash=(tratto[0] if tratto else None)),
            hoverlabel=dict(bgcolor='white', bordercolor=colore,
                            font=dict(color='black', size=11)),
            hovertemplate=f'<b>{nome}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.2f}}%<extra></extra>',
        ))
    fig.update_layout(
        xaxis=dict(gridcolor='#e8eef8', zeroline=False, tickformat='%Y', dtick='M12',
                   tickangle=0, tickfont=dict(size=10)),
        yaxis=dict(title=asse_y, gridcolor='#e8eef8', zeroline=True, zerolinecolor='#aaa',
                   ticksuffix='%'),
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
        hovermode='closest',
        showlegend=con_legenda,
    )
    if con_legenda:
        # In cima, orizzontale, due voci per riga come nell'Analisi Portafoglio;
        # sopra la legenda il titolo, a sinistra.
        righe = (len(serie) + 1) // 2
        fig.update_layout(
            legend=dict(orientation='h', entrywidthmode='fraction', entrywidth=0.5,
                        traceorder='normal', x=0, xanchor='left', y=1.02, yanchor='bottom',
                        font=dict(size=11), bgcolor='rgba(255,255,255,0.85)',
                        bordercolor='#aed6f1', borderwidth=1),
            title=dict(text=titolo, font=dict(size=13, color='#1a3a6b'), x=0.01,
                       y=1, yanchor='top', yref='container', pad=dict(t=8)),
            margin=dict(**_FE_MARGINE, t=36 + 21 * righe + 12),
        )
    else:
        fig.update_layout(
            title=dict(text=titolo, font=dict(size=13, color='#1a3a6b'), x=0.01),
            margin=dict(**_FE_MARGINE, t=40),
        )
    return fig


def _build_perf_chart(prices_data, chart_assets, frontier_weights, show_frontiers, date_start, date_end):
    if not prices_data:
        return _fe_grafico_vuoto()
    try:
        serie = _fe_serie(prices_data, chart_assets, frontier_weights, show_frontiers,
                          date_start, date_end, lambda cum: (cum / cum.iloc[0] - 1) * 100)
        if not serie:
            return _fe_grafico_vuoto()
        return _fe_figura(serie, 'Andamento — performance cumulativa (%)',
                          'Rendimento cumulativo (%)', con_legenda=True)
    except Exception:
        return _fe_grafico_vuoto()


def _build_drawdown_chart(prices_data, chart_assets, frontier_weights, show_frontiers, date_start, date_end):
    if not prices_data:
        return _fe_grafico_vuoto()
    try:
        serie = _fe_serie(prices_data, chart_assets, frontier_weights, show_frontiers,
                          date_start, date_end, lambda cum: (cum / cum.cummax() - 1) * 100)
        if not serie:
            return _fe_grafico_vuoto()
        return _fe_figura(serie, 'Drawdown (%)', 'Drawdown (%)', con_legenda=False)
    except Exception:
        return _fe_grafico_vuoto()

# ─────────────────────────────────────────────────────────────────────────────
# Navbar
# ─────────────────────────────────────────────────────────────────────────────
def _navbar():
    from navbar import make_navbar
    return make_navbar(current='frontiera')

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
# Riquadri dei comandi (📅 Periodo, 📈 Frontiere)
_FE_BOX = {'background':'white','border':'1px solid #ccd9ee','borderRadius':'6px',
           'padding':'6px 10px','display':'flex','flexDirection':'column','gap':'4px'}
_FE_TITOLO_BOX = {'fontSize':'12px','fontWeight':'700','color':'#1a3a6b','marginBottom':'2px'}
_FE_ETICHETTA = {'fontSize':FONT['xs'],'color':'#6b7a99','fontWeight':'600'}
_FE_CAMPO = {'display':'flex','flexDirection':'column','gap':'2px'}
_FE_INPUT = {'width':'64px','fontSize':FONT['sm'],'padding':'3px 5px','height':'30px',
             'border':'1px solid #ccd9ee','borderRadius':'4px','boxSizing':'border-box'}

# Misure di rischio delle tendine P1/P2/P3 e colori dei portafogli
_MISURE = [('standard', 'Standard'), ('cvar90', 'CVaR 90%'), ('cvar95', 'CVaR 95%'),
           ('arima_garch', 'ARIMA+GARCH'), ('vol', 'Vol (fin.)')]
_P_COLORS = {'P1': '#0066cc', 'P2': '#2ca02c', 'P3': '#d62728'}   # = _FC

_FE_CALC_STILE = {'background':'#0066cc','color':'white','border':'none',
                  'padding':'8px 16px','borderRadius':'4px','cursor':'pointer',
                  'fontWeight':'bold','fontSize':'12px','marginTop':'auto',
                  'boxShadow':'0 2px 6px rgba(0,102,204,0.35)'}
_FE_CALC_DA_RIFARE = {**_FE_CALC_STILE, 'background':'#e67e22',
                      'boxShadow':'0 2px 6px rgba(230,126,34,0.45)'}


app.layout = html.Div([
    _navbar(),

    html.Div([
        # ── Comandi: due riquadri, 📈 Frontiere e 📅 Periodo (con Calcola Frontiera) ─────────────────
        html.Div([
            html.Div([
                html.Div([
                    html.Div('📈 Frontiere', style={**_FE_TITOLO_BOX, 'marginBottom':'0'}),
                    html.Span('Una frontiera per ogni portafoglio P1/P2/P3 con almeno 2 asset spuntati',
                              style={'fontSize':FONT['xs'],'color':'#6b7a99','fontStyle':'italic'}),
                ], style={'display':'flex','alignItems':'baseline','gap':'10px','flexWrap':'wrap',
                          'marginBottom':'6px'}),
                html.Div([
                    html.Div([
                        html.Span('Punti sulla curva', style=_FE_ETICHETTA),
                        dcc.Input(id='fe-n-portfolios', type='number', value=15, min=5, max=100,
                                  style=_FE_INPUT),
                    ], style=_FE_CAMPO),
                    html.Div([
                        html.Span('Peso min %', style=_FE_ETICHETTA),
                        dcc.Input(id='fe-min-weight', type='number', value=0, min=0, max=100,
                                  style=_FE_INPUT),
                    ], style=_FE_CAMPO),
                    html.Div([
                        html.Span('Peso max %', style=_FE_ETICHETTA),
                        dcc.Input(id='fe-max-weight', type='number', value=100, min=0, max=100,
                                  style=_FE_INPUT),
                    ], style=_FE_CAMPO),
                    html.Div([
                        html.Span('Risk free %', style=_FE_ETICHETTA),
                        dcc.Input(id='fe-rf', type='number', value=2.0, min=0, max=20, step=0.1,
                                  style=_FE_INPUT),
                    ], style=_FE_CAMPO),
                    # Una misura di rischio per ogni portafoglio (F1 usa quella di P1, …)
                    *[html.Div([
                        html.Span(f'Rischio {p}', style={**_FE_ETICHETTA, 'color': _P_COLORS[p]}),
                        dcc.Dropdown(id=f'fe-risk-{p.lower()}',
                                     options=[{'label': n, 'value': v} for v, n in _MISURE],
                                     value='standard', clearable=False, searchable=False,
                                     className='fe-rischio',
                                     style={'width': '128px', 'fontSize': FONT['sm'],
                                            'borderLeft': f'3px solid {_P_COLORS[p]}',
                                            'borderRadius': '4px'}),
                    ], style=_FE_CAMPO) for p in ('P1', 'P2', 'P3')],
                    html.Div([
                        html.Span('Finestra gg', style=_FE_ETICHETTA),
                        dcc.Input(id='fe-arima-window', type='number', value=250, min=20, max=1260,
                                  style=_FE_INPUT),
                    ], id='fe-arima-window-div',
                       style={'display':'none','flexDirection':'column','gap':'2px'}),
                    html.Div(id='fe-arima-cache-status',
                             style={'display':'none','fontSize':'9px','padding':'2px 6px',
                                    'borderRadius':'3px','fontWeight':'600'}),
                    # Tutto passa dall'unico pulsante generale del Portafoglio (sopra i tab).
                    html.Button('🔄 Importa/Esporta Portafoglio', id='fpio-btn', n_clicks=0,
                                style={'display': 'none'}),
                    html.Button('💾 Salva in Portafoglio', id='fe-save-to-port-btn', n_clicks=0,
                                title='Salva F1→P1, F2→P2, F3→P3 nella dashboard Analisi Portafoglio',
                                style={'display':'none'}),
                ], style={'display':'flex','alignItems':'flex-end','gap':'14px','flexWrap':'wrap'}),
                # Riepilogo delle frontiere calcolate (Max Sharpe F1/F2/F3, metodo)
                html.Div(id='fe-stats-panel',
                         style={'marginTop':'6px','fontSize':'11px','color':'#1a3a5c'}),
                html.Div(id='fe-save-port-msg', style={'fontSize':'11px','color':'#1a7a4a','fontWeight':'600'}),
            ], className='fe-box', style={**_FE_BOX, 'flex':'1 1 520px'}),
            html.Div([
                html.Div('📅 Periodo', style=_FE_TITOLO_BOX),
                html.Div([
                    html.Div([
                        html.Span('Da', style=_FE_ETICHETTA),
                        dcc.DatePickerSingle(id='fe-date-start',
                            date=(pd.Timestamp.today()-pd.DateOffset(years=10)).strftime('%Y-%m-%d'),
                            display_format='DD/MM/YYYY'),
                    ], style=_FE_CAMPO),
                    html.Div([
                        html.Span('A', style=_FE_ETICHETTA),
                        dcc.DatePickerSingle(id='fe-date-end',
                            date=pd.Timestamp.today().strftime('%Y-%m-%d'),
                            display_format='DD/MM/YYYY'),
                    ], style=_FE_CAMPO),
                ], style={'display':'flex','gap':'10px','flexWrap':'wrap'}),
                html.Span(id='fe-last-updated',
                          style={'fontSize':FONT['xs'],'color':'#6b7a99','fontStyle':'italic'}),
                html.Button('Calcola Frontiera', id='fe-calc-btn', n_clicks=0,
                            title='Calcola le frontiere coi parametri impostati',
                            style=_FE_CALC_STILE),
            ], className='fe-box', style={**_FE_BOX, 'flex':'0 0 auto'}),

        ], style={'display':'flex','gap':'8px','flexWrap':'wrap','alignItems':'stretch',
                  'padding':'8px','background':'#f0f4fb','borderBottom':'2px solid #0066cc'}),

        # ── Griglia + Grafici ────────────────────────────────────────────────
        html.Div([
            # Sinistra: asset
            html.Div([
                html.Div([
                    html.Div('Asset',
                             style={'width':'25%','fontWeight':'bold','fontSize':FONT['xs'],
                                    'paddingLeft':'4px','color':'#1a3a5c'}),
                    html.Div([
                        html.Span('📊', style={'fontSize':FONT['icon']}),
                        html.Button('☑', id='fe-selall-chart', n_clicks=0, title='Seleziona / Deseleziona tutto grafico',
                                    style={'fontSize':FONT['icon'],'border':'none','background':'none','cursor':'pointer',
                                           'color':'#555','padding':'0 2px','lineHeight':'1'}),
                    ], style={'width':'7%','textAlign':'center','display':'flex','alignItems':'center',
                              'justifyContent':'center','gap':'2px','position':'relative'}),
                    html.Div([
                        html.Span('P1', style={'fontWeight':'bold','fontSize':FONT['xs'],'color':'#0066cc'}),
                        html.Button('☑', id='fe-selall-p1', n_clicks=0, title='Seleziona / Deseleziona tutto P1',
                                    style={'fontSize':FONT['icon'],'border':'none','background':'none','cursor':'pointer',
                                           'color':'#0066cc','padding':'0 2px','lineHeight':'1'}),
                    ], style={'width':'8%','textAlign':'center','display':'flex','alignItems':'center',
                              'justifyContent':'center','gap':'2px','position':'relative'}),
                    html.Div([
                        html.Span('P2', style={'fontWeight':'bold','fontSize':FONT['xs'],'color':'#2ca02c'}),
                        html.Button('☑', id='fe-selall-p2', n_clicks=0, title='Seleziona / Deseleziona tutto P2',
                                    style={'fontSize':FONT['icon'],'border':'none','background':'none','cursor':'pointer',
                                           'color':'#2ca02c','padding':'0 2px','lineHeight':'1'}),
                    ], style={'width':'8%','textAlign':'center','display':'flex','alignItems':'center',
                              'justifyContent':'center','gap':'2px','position':'relative'}),
                    html.Div([
                        html.Span('P3', style={'fontWeight':'bold','fontSize':FONT['xs'],'color':'#d62728'}),
                        html.Button('☑', id='fe-selall-p3', n_clicks=0, title='Seleziona / Deseleziona tutto P3',
                                    style={'fontSize':FONT['icon'],'border':'none','background':'none','cursor':'pointer',
                                           'color':'#d62728','padding':'0 2px','lineHeight':'1'}),
                    ], style={'width':'8%','textAlign':'center','display':'flex','alignItems':'center',
                              'justifyContent':'center','gap':'2px','position':'relative'}),
                    html.Div('F1 %', **{'data-tooltip':'Peso Max-Sharpe Frontiera 1'},
                             style={'width':'15%','textAlign':'center','fontWeight':'bold',
                                    'fontSize':FONT['xs'],'color':'#0066cc','position':'relative','cursor':'default'}),
                    html.Div('F2 %', **{'data-tooltip':'Peso Max-Sharpe Frontiera 2'},
                             style={'width':'15%','textAlign':'center','fontWeight':'bold',
                                    'fontSize':FONT['xs'],'color':'#2ca02c','position':'relative','cursor':'default'}),
                    html.Div('F3 %', **{'data-tooltip':'Peso Max-Sharpe Frontiera 3'},
                             style={'width':'14%','textAlign':'center','fontWeight':'bold',
                                    'fontSize':FONT['xs'],'color':'#d62728','position':'relative','cursor':'default'}),
                ], style={'display':'flex','alignItems':'center','minHeight':'28px',
                          'background':'#e8f0fb','borderBottom':'1px solid #aed6f1',
                          'position':'sticky','top':'0','zIndex':'2'}),
                html.Div(id='fe-asset-count', style={'fontSize':FONT['sm'],'color':'#555',
                                                      'padding':'3px 5px'}),
                html.Div(id='fe-grid', children=[
                    html.Div('Carica i dati e clicca Calcola Frontiera',
                             style={'color':'#888','fontStyle':'italic',
                                    'fontSize':'11px','padding':'12px 8px'})
                ]),
            ], style={'width':'35%','overflowY':'auto','borderRight':'1px solid #ccd9ee',
                      'background':'white','maxHeight':'870px'}),

            # Destra: grafici — flex-column per riempire esattamente lo spazio disponibile
            html.Div([
                html.Div(id='fe-hint',
                         style={'display':'none','fontSize':'9px','color':'#0066cc',
                                'fontWeight':'600','padding':'2px 5px 4px',
                                'background':'#e8f4ff','borderLeft':'3px solid #0066cc',
                                'marginBottom':'4px','borderRadius':'0 4px 4px 0',
                                'flexShrink':'0'}),

                # ── Toast ARIMA background: appare al primo avvio, si chiude da solo ──
                html.Div(id='fe-arima-toast', style={'display': 'none'},
                         children=html.Div([
                    html.Div([
                        html.Span('⏳ Calcolo ARIMA+GARCH avviato in background',
                                  style={'fontWeight': '700', 'fontSize': '12px',
                                         'color': '#1a3a5c'}),
                        html.Button('✕', id='fe-arima-toast-close', n_clicks=0,
                                    style={'background': 'none', 'border': 'none',
                                           'fontSize': '14px', 'cursor': 'pointer',
                                           'color': '#888', 'marginLeft': 'auto',
                                           'padding': '0 4px'}),
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px',
                              'marginBottom': '6px'}),
                    html.P([
                        'Il modello ARIMA+GARCH viene calcolato in background per tutti gli asset. ',
                        html.Strong('Nel frattempo puoi usare gli altri metodi di rischio: '),
                        'Varianza storica, EWMA o Varianza uguale.',
                    ], style={'fontSize': '11px', 'color': '#444', 'margin': '0',
                              'lineHeight': '1.5'}),
                    html.P('Quando il calcolo sarà completato, la barra qui sotto scomparirà '
                           'e potrai ricalcolare la frontiera con ARIMA+GARCH.',
                           style={'fontSize': '10px', 'color': '#666', 'margin': '6px 0 0',
                                  'fontStyle': 'italic'}),
                ], style={
                    'background': '#f0f6ff',
                    'border': '1px solid #a8c4e8',
                    'borderLeft': '4px solid #1a3a5c',
                    'borderRadius': '6px',
                    'padding': '10px 14px',
                    'marginBottom': '8px',
                    'boxShadow': '0 2px 8px rgba(26,58,92,0.1)',
                })),

                html.Div(id='fe-arima-progress-div',
                         style=_ARIMA_PROG_HIDDEN,
                         children=[
                    html.Div([
                        html.Div(style={'width':'14px','height':'14px','border':'2px solid #ccd9ee',
                                        'borderTop':'2px solid #1a3a6b','borderRadius':'50%',
                                        'animation':'fe-spin 0.9s linear infinite','flexShrink':'0'}),
                        html.Span(id='fe-arima-progress-text', children='ARIMA in corso...',
                                  style={'fontSize':'11px','color':'#1a3a6b','fontWeight':'600'}),
                    ], style={'display':'flex','alignItems':'center','gap':'8px'}),
                    # Barra di avanzamento (track + riempimento)
                    html.Div(
                        html.Div(id='fe-arima-progress-bar', style=_arima_bar_style(0)),
                        style={'width':'100%','height':'9px','background':'#ccd9ee',
                               'borderRadius':'6px','overflow':'hidden'}),
                ]),
                dcc.Loading(
                    type='circle',
                    color='#1a3a6b',
                    fullscreen=True,
                    overlay_style={
                        'visibility':'visible','opacity':1,
                        'backgroundColor':'rgba(240,244,251,0.82)',
                        'zIndex':9999,
                    },
                    custom_spinner=html.Div([
                        html.Div(style={
                            'width':'52px','height':'52px','border':'5px solid #ccd9ee',
                            'borderTop':'5px solid #1a3a6b','borderRadius':'50%',
                            'animation':'fe-spin 0.9s linear infinite','margin':'0 auto',
                        }),
                        html.P('Calcolo frontiera in corso…',
                               style={'color':'#1a3a6b','marginTop':'14px','fontSize':'13px',
                                      'fontWeight':'600','fontFamily':'Inter, sans-serif',
                                      'textAlign':'center'}),
                    ], style={'textAlign':'center','padding':'32px 40px',
                              'background':'white','borderRadius':'12px',
                              'boxShadow':'0 4px 24px rgba(26,58,107,0.15)'}),
                    children=dcc.Graph(id='fe-frontier-chart',
                                       style={'height':'100%'},
                                       config={'displayModeBar':True}),
                    style={'height':'420px'},
                ),
                dcc.Graph(id='fe-perf-chart',
                          style={'height':'460px','marginTop':'6px'},
                          config={'displayModeBar':True}),
                dcc.Graph(id='fe-drawdown-chart',
                          style={'height':'320px','marginTop':'6px'},
                          config={'displayModeBar':True}),
                dcc.Store(id='fe-legenda-sync'),
            ], style={
                'width':'65%','padding':'6px','background':'white',
                'display':'flex','flexDirection':'column',
            }),
        ], style={'display':'flex','alignItems':'flex-start'}),

    ], style={'marginTop':'106px'}),

    # ── Stores ───────────────────────────────────────────────────────────────
    dcc.Store(id='_fe-page-load',    data=1),
    dcc.Store(id='fe-stock-data',    data=None),
    dcc.Store(id='fe-prices-data',   data=None),
    dcc.Store(id='fe-loaded-flag',   data=False),
    dcc.Store(id='fe-data-source',   data='default'),
    dcc.Store(id='fe-f1-weights',      data=None),
    dcc.Store(id='fe-f2-weights',      data=None),
    dcc.Store(id='fe-f3-weights',      data=None),
    dcc.Store(id='fe-frontier-rawdata',data=None),
    dcc.Store(id='fe-param-calcolo',  data=None),   # parametri dell'ultima frontiera disegnata
    dcc.Store(id='fe-selected-pt',     data=None),
    dcc.Store(id='fe-arima-reqid',     data=None),
    dcc.Interval(id='fe-arima-poll',   interval=600, n_intervals=0, disabled=True),
    dcc.Store(id='fe-sync-sig',        data=''),
    dcc.Store(id='fe-auto-calc-flag',  data=None),   # frontiera già auto-calcolata all'apertura?
    dcc.Interval(id='fe-live-sync',    interval=2000, n_intervals=0, disabled=False),

    # ── Modal Importa/Esporta Portafoglio ─────────────────────────────────────
    html.Div(id='fpio-overlay',
             style={'display':'none','position':'fixed','top':'0','left':'0',
                    'width':'100%','height':'100%','zIndex':'9000',
                    'background':'rgba(0,0,0,0.45)','alignItems':'center',
                    'justifyContent':'center'},
             children=[
        html.Div([
            html.Div([
                html.Span('🔄 Importa / Esporta Portafoglio',
                          style={'fontWeight':'700','fontSize':'14px','color':'#1a3a5c'}),
                html.Button('✕', id='fpio-close', n_clicks=0,
                            style={'background':'none','border':'none','fontSize':'18px',
                                   'cursor':'pointer','color':'#666','float':'right'}),
            ], style={'display':'flex','justifyContent':'space-between',
                      'alignItems':'center','marginBottom':'12px'}),

            dcc.RadioItems(id='fpio-mode',
                           options=[{'label':' 📤 Esporta F1/F2/F3','value':'export'},
                                    {'label':' 📥 Importa','value':'import'}],
                           value='export', inline=True,
                           inputStyle={'marginRight':'4px'},
                           labelStyle={'marginRight':'18px','fontSize':'12px','fontWeight':'600'},
                           style={'marginBottom':'14px','paddingBottom':'10px',
                                  'borderBottom':'1px solid #eee'}),

            # Esporta
            html.Div(id='fpio-export-view', children=[
                html.Div('1. Colonna da esportare:',
                         style={'fontSize':'11px','fontWeight':'600','color':'#1a3a5c',
                                'marginBottom':'6px'}),
                dcc.RadioItems(id='fpio-exp-col',
                               options=[{'label':' F1','value':'F1'},
                                        {'label':' F2','value':'F2'},
                                        {'label':' F3','value':'F3'}],
                               value='F1', inline=True,
                               inputStyle={'marginRight':'4px'},
                               labelStyle={'marginRight':'16px','fontSize':'12px','fontWeight':'600'},
                               style={'marginBottom':'12px'}),
                html.Div('2. Salva come Analisi (nuova o sovrascrivi esistente):',
                         style={'fontSize':'11px','fontWeight':'600',
                                'color':'#1a3a5c','marginBottom':'6px'}),
                dcc.Dropdown(id='fpio-exp-profile', placeholder="Sovrascrivi un'analisi esistente…",
                             style={'fontSize':'11px','marginBottom':'6px'}),
                dcc.Input(id='fpio-exp-new', placeholder='…oppure scrivi una nuova analisi',
                          style={'width':'100%','fontSize':'11px','marginBottom':'12px',
                                 'padding':'5px 8px','border':'1px solid #aaa','borderRadius':'4px'}),
                html.Button('📤 Esporta', id='fpio-exp-btn', n_clicks=0,
                            style={'background':'#1b7a34','color':'white','border':'none',
                                   'padding':'7px 16px','borderRadius':'4px','cursor':'pointer',
                                   'fontSize':'12px','fontWeight':'bold'}),
                html.Div(id='fpio-exp-status', style={'fontSize':'11px','marginTop':'8px','minHeight':'16px'}),
            ]),

            # Importa
            html.Div(id='fpio-import-view', style={'display':'none'}, children=[
                html.Div('1. Analisi da importare:', style={'fontSize':'11px','fontWeight':'600',
                         'color':'#1a3a5c','marginBottom':'6px'}),
                dcc.Dropdown(id='fpio-imp-profile', placeholder="Scegli un'analisi…",
                             style={'fontSize':'11px','marginBottom':'12px'}),
                html.Div('2. Metti nella colonna:', style={'fontSize':'11px','fontWeight':'600',
                         'color':'#1a3a5c','marginBottom':'6px'}),
                dcc.RadioItems(id='fpio-imp-target',
                               options=[{'label':' F1','value':'F1'},
                                        {'label':' F2','value':'F2'},
                                        {'label':' F3','value':'F3'}],
                               value='F1', inline=True,
                               inputStyle={'marginRight':'4px'},
                               labelStyle={'marginRight':'16px','fontSize':'12px','fontWeight':'600'},
                               style={'marginBottom':'12px'}),
                html.Button('📥 Importa', id='fpio-imp-btn', n_clicks=0,
                            style={'background':'#1a3a5c','color':'white','border':'none',
                                   'padding':'7px 16px','borderRadius':'4px','cursor':'pointer',
                                   'fontSize':'12px','fontWeight':'bold'}),
                html.Div(id='fpio-imp-status', style={'fontSize':'11px','marginTop':'8px','minHeight':'16px'}),
            ]),
        ], style={'background':'white','borderRadius':'10px','padding':'20px 24px',
                  'width':'460px','boxShadow':'0 4px 24px rgba(0,0,0,0.18)','position':'relative'}),
    ]),
], style={'minHeight':'100vh'})

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def _fe_current_sig(ns):
    """Firma di current.json usata per il sync live: cambia se cambiano asset, spunte
    o pesi P1/P2/P3. Stessa formula in on_page_load e fe_live_sync, così il primo giro
    del sync all'apertura riconosce 'nessun cambiamento' e non ri-scrive i dati
    (evita il doppio/triplo caricamento di griglia/anteprima all'apertura)."""
    if not ns:
        return ''
    return repr((
        tuple(sorted(ns.keys())),
        tuple(sorted(d for d, v in ns.items() if v.get('checked'))),
        tuple(sorted((d, round(float(v.get('P1') or 0), 4),
                          round(float(v.get('P2') or 0), 4),
                          round(float(v.get('P3') or 0), 4))
                     for d, v in ns.items()
                     if (v.get('P1') or v.get('P2') or v.get('P3')))),
    ))


@app.callback(
    Output('fe-stock-data',    'data',     allow_duplicate=True),
    Output('fe-prices-data',   'data',     allow_duplicate=True),
    Output('fe-loaded-flag',   'data',     allow_duplicate=True),
    Output('fe-last-updated',  'children', allow_duplicate=True),
    Output('fe-data-source',   'data',     allow_duplicate=True),
    Output('fe-sync-sig',      'data',     allow_duplicate=True),
    Input('_fe-page-load',     'data'),
    prevent_initial_call='initial_duplicate',
)
def on_page_load(_):
    import uuid as _uuid
    # Firma iniziale coerente col sync live: impedisce a fe_live_sync di ri-scrivere
    # subito i dati appena caricati (causa del doppio/triplo caricamento all'apertura).
    _sig0 = _fe_current_sig(_read_user_json())

    # 1. Dati di portafoglio condivisi (portafoglio app connessa)
    prices, returns, saved_at = _read_shared_data()
    if prices is not None:
        n = len(prices.columns)
        label = f'Da analisi di portafoglio ({n} asset)' + (f' — {saved_at}' if saved_at else '')
        return (returns.to_json(orient='split', date_format='iso'),
                prices.to_json(orient='split', date_format='iso'),
                True, label, 'default', _sig0)

    # 2. Dati personali utente salvati in sessione precedente
    u_prices, u_returns, u_saved_at, u_arima = _read_user_data()
    if u_prices is not None:
        n = len(u_prices.columns)
        label = f'Dati personali ({n} asset) — {u_saved_at}'
        return (u_returns.to_json(orient='split', date_format='iso'),
                u_prices.to_json(orient='split', date_format='iso'),
                True, label, 'user', _sig0)

    raise PreventUpdate


_ARIMA_TOAST_HIDDEN = {'display': 'none'}

@app.callback(
    Output('fe-arima-toast', 'style', allow_duplicate=True),
    Input('fe-arima-toast-close', 'n_clicks'),
    prevent_initial_call=True,
)
def close_arima_toast(n):
    if not n:
        raise PreventUpdate
    return _ARIMA_TOAST_HIDDEN


def _pesi_analisi():
    """Pesi P1/P2/P3 dell'Analisi Portafoglio (current.json): {P: {asset: peso}}."""
    ns = _read_user_json() or {}
    out = {}
    for p in ('P1', 'P2', 'P3'):
        d = {k: float(v.get(p) or 0) for k, v in ns.items()
             if isinstance(v, dict) and float(v.get(p) or 0) > 0}
        if d:
            out[p] = d
    return out


def _righe_portafogli_analisi(pesi, spuntati=()):
    """Righe «Portafoglio P1/P2/P3 · Analisi» in fondo alla griglia, prima e dopo il
    calcolo: i portafogli dell'Analisi Portafoglio, da confrontare (spunta 📊) con
    gli asset e con i portafogli F1/F2/F3 della frontiera in andamento e drawdown.
    Checkbox `fe-chart-port` con indice P1/P2/P3 (le frontiere usano F1/F2/F3)."""
    pesi = pesi or {}
    righe = []
    for pkey, fidx in (('P1', 'F1'), ('P2', 'F2'), ('P3', 'F3')):
        pw = pesi.get(pkey)
        if not pw:
            continue
        colore = _P_COLORS[pkey]
        tot = round(sum(float(x or 0) for x in pw.values()), 1)
        celle = [html.Div(_w_cell(tot if f == fidx else None, _FC[f]),
                          style={'width': w, 'display':'flex','alignItems':'center',
                                 'justifyContent':'center'})
                 for f, w in (('F1', '15%'), ('F2', '15%'), ('F3', '14%'))]
        righe.append(html.Div([
            html.Div(
                html.Span(f'Portafoglio {pkey} · Analisi',
                          title=f'Portafoglio {pkey} dell\'Analisi Portafoglio ({len(pw)} asset): '
                                'linea continua sottile in andamento e drawdown',
                          style={'fontSize':'8px','color':colore,'fontWeight':'700',
                                 'overflow':'hidden','whiteSpace':'nowrap','textOverflow':'ellipsis'}),
                style={'width':'25%','height':'28px','display':'flex','alignItems':'center','paddingLeft':'4px'}
            ),
            html.Div(
                dcc.Checklist(
                    id={'type':'fe-chart-port','index':pkey},
                    options=[{'label':'','value':pkey}],
                    value=[pkey] if pkey in spuntati else [],
                    inputStyle={'accentColor': colore},
                    style={'display':'flex','justifyContent':'center'},
                ),
                style={'width':'7%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            *[html.Div('—', style={'width':'8%','textAlign':'center','fontSize':'8px','color':'#ccc'})
              for _ in range(3)],
            *celle,
        ], style={'display':'flex','alignItems':'center','height':'28px',
                  'borderBottom':'1px solid #e0e8f8', 'background':'#fbfbf7'}))
    if not righe:
        return []
    return [html.Div(style={'height':'3px','background':'#ccd9ee','margin':'2px 0'}),
            html.Div(html.Span("Portafogli dell'Analisi Portafoglio — spunta per confrontarne "
                               "andamento e drawdown con asset e frontiere",
                               style={'fontSize':'7px','color':'#8899bb','fontStyle':'italic'}),
                     style={'padding':'2px 4px','background':'#f8faff',
                            'borderBottom':'1px solid #e0e8f8'}),
            *righe]


def _build_grid_rows(returns_df, pre_select=None, pre_weights=None):
    pre_select = set(pre_select or [])
    pre_p1 = (pre_weights or {}).get('P1', {})
    pre_p2 = (pre_weights or {}).get('P2', {})
    pre_p3 = (pre_weights or {}).get('P3', {})
    assets    = returns_df.dropna(how='all', axis=1).columns.tolist()
    short_map = _short_history(returns_df)
    rows = []
    for i, asset in enumerate(assets):
        row = html.Div([
            _asset_name_div(asset, short_map),
            html.Div(
                dcc.Checklist(id={'type':'fe-chart','index':asset},
                              options=[{'label':'','value':asset}],
                              value=[asset] if asset in pre_select else [],
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'7%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(
                dcc.Checklist(id={'type':'fe-p1','index':asset},
                              options=[{'label':'','value':asset}], value=[asset] if pre_p1.get(asset, 0) else [],
                              inputStyle={'accentColor':'#0066cc'},
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(
                dcc.Checklist(id={'type':'fe-p2','index':asset},
                              options=[{'label':'','value':asset}], value=[asset] if pre_p2.get(asset, 0) else [],
                              inputStyle={'accentColor':'#2ca02c'},
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(
                dcc.Checklist(id={'type':'fe-p3','index':asset},
                              options=[{'label':'','value':asset}], value=[asset] if pre_p3.get(asset, 0) else [],
                              inputStyle={'accentColor':'#d62728'},
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(id={'type':'fe-wgt-f1','index':asset}, children=_w_cell(pre_p1.get(asset),'#0066cc'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wgt-f2','index':asset}, children=_w_cell(pre_p2.get(asset),'#2ca02c'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wgt-f3','index':asset}, children=_w_cell(pre_p3.get(asset),'#d62728'),
                     style={'width':'14%','display':'flex','alignItems':'center','justifyContent':'center'}),
        ], style={'display':'flex','alignItems':'center','height':'24px',
                  'borderBottom':'1px solid #f0f4fb',
                  'background':'white' if i % 2 == 0 else '#fafcff'})
        rows.append(row)

    # all'ingresso i portafogli dell'Analisi sono già spuntati: si vedono subito
    rows.extend(_righe_portafogli_analisi(pre_weights, spuntati=tuple((pre_weights or {}).keys())))

    return rows, f'{len(assets)} asset'


@app.callback(
    Output('fe-grid',        'children', allow_duplicate=True),
    Output('fe-asset-count', 'children', allow_duplicate=True),
    Input('fe-loaded-flag',  'data'),
    Input('fe-stock-data',   'data'),
    Input('fe-sync-sig',     'data'),   # ricostruisci anche su cambio selezione/pesi
    State('fe-frontier-rawdata', 'data'),
    prevent_initial_call=True,
)
def build_grid_on_load(loaded, stock_data, _sync_sig, rawdata):
    if not stock_data:
        raise PreventUpdate
    # Se una frontiera è già calcolata e il trigger è solo un cambio di
    # selezione/pesi (fe-sync-sig dall'Interval di sync), NON ricostruire la
    # griglia: distruggerebbe le righe-portafoglio (F1/F2/F3 + checkbox di
    # confronto + riga "Portfolio selezionato") create da calc_and_render,
    # e con esse i pesi del punto cliccato sulla frontiera. La griglia va
    # rifatta solo se cambia la LISTA asset (fe-stock-data) o al caricamento.
    _trig = (callback_context.triggered[0]['prop_id'].split('.')[0]
             if callback_context.triggered else '')
    if _trig == 'fe-sync-sig' and rawdata:
        raise PreventUpdate
    returns_df = _get_returns(stock_data)
    if returns_df is None or returns_df.empty:
        raise PreventUpdate
    ns = _read_user_json()
    pre_weights = {
        'P1': {d: v['P1'] for d, v in ns.items() if v.get('P1', 0)},
        'P2': {d: v['P2'] for d, v in ns.items() if v.get('P2', 0)},
        'P3': {d: v['P3'] for d, v in ns.items() if v.get('P3', 0)},
    } if ns else None
    # Spunta l'asset (📊) se è 'checked' OPPURE se ha un peso in P1/P2/P3:
    # passando dal Portafoglio alla Frontiera gli asset con pesi risultano già selezionati.
    pre_select = [d for d, v in ns.items()
                  if v.get('checked') or v.get('P1', 0) or v.get('P2', 0) or v.get('P3', 0)] if ns else []
    return _build_grid_rows(returns_df, pre_select=pre_select, pre_weights=pre_weights)


# Niente calcolo automatico all'apertura (scelta dell'utente, 23/11/2026): si vedono
# i pesi P1/P2/P3 dell'Analisi e il loro andamento; la frontiera parte solo dal clic
# su Calcola Frontiera. (Era auto_calc_on_open; fe-auto-calc-flag resta nel layout.)


# Cambio di parametri (date, «Finestra gg», rischio P1/P2/P3, punti, pesi, risk free)
# con una frontiera già disegnata: NON si ricalcola (scelta dell'utente, più calcoli di
# fila appesantivano il server). Il pulsante diventa arancione finché non si ripreme
# Calcola Frontiera; torna blu quando arriva la frontiera nuova (fe-frontier-rawdata).
_FE_PARAMETRI = [('fe-date-start', 'date'), ('fe-date-end', 'date'), ('fe-arima-window', 'value'),
                 ('fe-risk-p1', 'value'), ('fe-risk-p2', 'value'), ('fe-risk-p3', 'value'),
                 ('fe-n-portfolios', 'value'), ('fe-min-weight', 'value'), ('fe-max-weight', 'value'),
                 ('fe-rf', 'value'), ({'type':'fe-p1','index':ALL}, 'value'),
                 ({'type':'fe-p2','index':ALL}, 'value'), ({'type':'fe-p3','index':ALL}, 'value')]


def _firma_parametri(valori):
    return json.dumps(valori, sort_keys=True, default=str)


# Arriva una frontiera nuova: si annotano i parametri con cui è stata disegnata.
@app.callback(
    Output('fe-param-calcolo', 'data'),
    Input('fe-frontier-rawdata', 'data'),
    *[State(i, pr) for i, pr in _FE_PARAMETRI],
    prevent_initial_call=True,
)
def _annota_parametri(rawdata, *valori):
    return _firma_parametri(list(valori)) if rawdata else None


@app.callback(
    Output('fe-calc-btn', 'children'),
    Output('fe-calc-btn', 'style'),
    Input('fe-param-calcolo', 'data'),
    *[Input(i, pr) for i, pr in _FE_PARAMETRI],
    prevent_initial_call=True,
)
def _calcola_da_rifare(firma, *valori):
    if not firma or _firma_parametri(list(valori)) == firma:
        return 'Calcola Frontiera', _FE_CALC_STILE
    return '⟳ Ricalcola Frontiera', _FE_CALC_DA_RIFARE


# Quando cambia la LISTA asset (es. cancelli un asset in Analisi Portafoglio),
# la frontiera GIÀ CALCOLATA è stale (può contenere l'asset rimosso nel grafico
# e nei pesi F1/F2/F3). La azzeriamo: così l'asset cancellato non resta appeso.
# L'utente ricalcola con "Calcola Frontiera" sulla nuova lista.
@app.callback(
    Output('fe-frontier-chart',   'figure',   allow_duplicate=True),
    Output('fe-perf-chart',       'figure',   allow_duplicate=True),
    Output('fe-drawdown-chart',   'figure',   allow_duplicate=True),
    Output('fe-stats-panel',      'children', allow_duplicate=True),
    Output('fe-f1-weights',       'data',     allow_duplicate=True),
    Output('fe-f2-weights',       'data',     allow_duplicate=True),
    Output('fe-f3-weights',       'data',     allow_duplicate=True),
    Output('fe-frontier-rawdata', 'data',     allow_duplicate=True),
    Output('fe-selected-pt',      'data',     allow_duplicate=True),
    Input('fe-stock-data',        'data'),
    prevent_initial_call=True,
)
def _clear_stale_frontier(_stock_data):
    _empty = go.Figure().update_layout(
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
        annotations=[dict(text='Lista asset aggiornata — clicca "Calcola Frontiera"',
                          xref='paper', yref='paper', x=0.5, y=0.5,
                          showarrow=False, font=dict(size=13, color='#6b7a99'))])
    # Ripristina i pesi F1/F2/F3 con i valori P1/P2/P3 da current.json:
    # così i portafogli costruiti in Analisi Portafoglio rimangono visibili
    # anche dopo che la frontiera viene resettata per una lista asset diversa.
    ns = _read_user_json()
    def _w_json(key):
        if not ns:
            return None
        d = {desc: v[key] for desc, v in ns.items() if v.get(key, 0)}
        return json.dumps(d) if d else None
    return _empty, _empty, _empty, '', _w_json('P1'), _w_json('P2'), _w_json('P3'), None, None


# ── Anteprima: asset spuntati come punti (rend. medio / rischio) PRIMA del calcolo ──
# Appena i dati sono caricati, mostra sul grafico principale gli asset spuntati
# in P1/P2/P3 (non quelli 📊, che valgono per i grafici di performance) come punti rendimento medio annuo vs rischio (misura scelta: Standard / CVaR /
# Vol), senza dover calcolare la frontiera. Reagisce alle spunte P1/P2/P3 e alla
# misura di rischio. Con una frontiera già calcolata (fe-frontier-rawdata) non
# ridisegna: accende/spegne soltanto i punti degli asset.
def _asset_nei_portafogli(*liste):
    """Asset spuntati in almeno una delle colonne P1/P2/P3 della griglia."""
    return {a for vals in liste for v in (vals or []) if v for a in v}


# ── Misura di rischio per portafoglio (tendine P1/P2/P3, 23/11/2026) ────────
# Ogni portafoglio ha la sua misura; la frontiera F1 usa quella di P1, ecc.
#   Standard / CVaR 90-95: giorni comuni a tutti gli asset del portafoglio dentro le
#     date scelte (se un asset è più giovane la frontiera parte da lui; se tutti
#     coprono la finestra, si usa la finestra intera).
#   Vol (fin.): stessi giorni comuni, ultimi «Finestra gg» (250 di default).
#   ARIMA+GARCH: μ e σ dal file personale arima_garch.json (ultimi 250 rendimenti di
#     ogni asset, date ignorate), correlazioni sugli ultimi 250 giorni.
_NOME_MISURA = dict(_MISURE)
_ASSE_MISURA = {'standard': 'Volatilità', 'vol': 'Volatilità [fin.]', 'cvar90': 'CVaR 90%',
                'cvar95': 'CVaR 95%', 'arima_garch': 'Vol GARCH'}
_P_DI = {'F1': 'P1', 'F2': 'P2', 'F3': 'P3'}


def _cvar_di(misura):
    return 10 if misura == 'cvar90' else (5 if misura == 'cvar95' else None)


def _stima(returns_df, assets, misura, finestra_vol, arima=None, returns_all=None):
    """(df, mu, cov) giornalieri per un gruppo di asset con la misura scelta, o None.
    df = campione dei rendimenti (scenari del CVaR); returns_df è già filtrato sulle date."""
    valid = [a for a in assets if a in returns_df.columns]
    if not valid:
        return None
    if misura == 'arima_garch':
        ra = returns_all if returns_all is not None else returns_df
        valid = [a for a in valid if a in ra.columns]
        if not valid:
            return None
        mu, cov = arima_utenti.mu_cov(arima or {}, ra, valid)
        return ra[valid].tail(arima_utenti.FINESTRA), mu, cov
    df = returns_df[valid].dropna()
    win = int(finestra_vol or 250) if misura == 'vol' else None
    if win and win < len(df):
        df = df.tail(win)
    if len(df) < 10:
        return None
    return df, df.mean(), df.cov()


def _rend_rischio(w, stima, misura):
    """(rendimento, rischio) annui del portafoglio w sulla stima."""
    df, mu, cov = stima
    ret, vol = _port_perf(np.asarray(w, dtype=float), mu.values, cov.values)
    pct = _cvar_di(misura)
    if pct is not None:
        vol = _port_cvar(np.asarray(w, dtype=float), df.dropna(), pct)
    return ret, vol


def _etichetta_rischio(misura, rischio):
    pct = _cvar_di(misura)
    nome = f'CVaR {100 - pct}%' if pct else ('Vol GARCH' if misura == 'arima_garch' else 'Vol')
    return f'{nome}: {rischio * 100:.2f}%'


def _etichetta_asse(usate):
    """Titolo dell'asse x: una misura sola → il suo nome; più misure → elenco per P."""
    uniche = list(dict.fromkeys(m for _, m in usate))
    if not uniche:
        return 'Rischio Ann. (%)'
    if len(uniche) == 1:
        return f'{_ASSE_MISURA[uniche[0]]} Ann. (%)'
    return 'Rischio Ann. (%) — ' + ' · '.join(f'{p}: {_ASSE_MISURA[m]}' for p, m in usate)


def _punti_asset(fig, returns_df, assets, gruppi, misure, finestra_vol, arima, returns_all,
                 visibili, grande=False):
    """Punti degli asset, uno per ogni portafoglio P1/P2/P3, nel colore del portafoglio
    (blu, verde, rosso), con la misura di rischio e il campione di quel portafoglio: un
    asset in due portafogli compare due volte, e le due posizioni mostrano come cambia
    il rischio. `visibili` = {P: asset spuntati}; tutti gli asset entrano comunque nella
    figura (spenti), così spuntarne uno dopo il calcolo è solo un cambio di visibilità.
    Una voce di legenda per portafoglio («Asset P1»…) li mostra/nasconde tutti insieme."""
    for p in ('P1', 'P2', 'P3'):
        sel = list(gruppi.get(p) or [])
        misura = misure[p]
        colore = _P_COLORS[p]
        gruppo = f'asset-{p}'
        st_p = (_stima(returns_df, sel, misura, finestra_vol, arima, returns_all)
                if len(sel) >= 2 else None)
        visti = set(visibili.get(p) or ())
        if visti:
            # voce di legenda del gruppo (segnaposto senza punti: resta anche se
            # gli asset vengono spenti uno per uno dalla griglia)
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers', name=f'Asset {p}',
                legendgroup=gruppo, showlegend=True, hoverinfo='skip',
                marker=dict(size=8, symbol='circle', color=colore, opacity=0.8)))
        for asset in assets:
            st = st_p if (st_p is not None and asset in st_p[0].columns) else None
            if st is None:
                st = _stima(returns_df, [asset], misura, finestra_vol, arima, returns_all)
            if st is None:
                continue
            cols = list(st[0].columns)
            w = np.zeros(len(cols)); w[cols.index(asset)] = 1.0
            ret_a, risk_a = _rend_rischio(w, st, misura)
            fig.add_trace(go.Scatter(
                x=[risk_a * 100], y=[ret_a * 100],
                mode='markers+text', name=asset,
                meta={'punto_asset': asset, 'portafoglio': p},
                legendgroup=gruppo, showlegend=False,
                visible=(asset in visti),
                marker=dict(size=7 if not grande else 8, symbol='circle',
                            opacity=0.8, color=colore,
                            line=dict(color='white', width=0.5)),
                text=[asset[:9]], textposition='top center',
                textfont=dict(size=7 if not grande else 8, color=colore),
                hoverlabel=dict(bgcolor='white', bordercolor=colore,
                                font=dict(color='black', size=11)),
                hovertemplate=(f'<b>{asset}</b> · {p}<br>{_etichetta_rischio(misura, risk_a)}'
                               f'<br>Rendimento: {ret_a*100:.2f}%'
                               f'<br><i>{_NOME_MISURA[misura]}</i><extra></extra>'),
            ))


def _fig_senza_portafogli():
    return go.Figure().update_layout(
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
        annotations=[dict(text='Nessun portafoglio: spunta almeno 2 asset in P1, P2 o P3 '
                               'e premi Calcola Frontiera',
                          xref='paper', yref='paper', x=0.5, y=0.5,
                          showarrow=False, font=dict(size=13, color='#6b7a99'))])


def _punti_portafogli(fig, returns_df, misure, finestra_vol, arima=None, returns_all=None):
    """P1/P2/P3 dell'Analisi Portafoglio (pesi di current.json) come rombi, ognuno con
    la misura di rischio e il campione del proprio portafoglio. Restituisce i P disegnati."""
    ns = _read_user_json()
    disegnati = []
    for pname, pcolor in _P_COLORS.items():
        raw = {d: float(v.get(pname) or 0) for d, v in ns.items() if float(v.get(pname) or 0) > 0}
        misura = misure[pname]
        st = _stima(returns_df, list(raw), misura, finestra_vol, arima, returns_all)
        if st is None:
            continue
        assets = list(st[0].columns)
        tot = sum(raw[a] for a in assets)
        if tot <= 0:
            continue
        w = np.array([raw[a] / tot for a in assets])
        r_p, v_p = _rend_rischio(w, st, misura)
        disegnati.append(pname)
        hover_lines = '<br>'.join(f'{a}: {wi*100:.1f}%' for a, wi in zip(assets, w))
        fig.add_trace(go.Scatter(
            x=[v_p * 100], y=[r_p * 100],
            mode='markers+text', name=pname,
            marker=dict(size=14, symbol='diamond', color=pcolor,
                        line=dict(color='white', width=1.5)),
            text=[pname], textposition='top center',
            textfont=dict(size=9, color=pcolor, family='Inter, sans-serif'),
            showlegend=True,
            hovertemplate=(f'<b>{pname} (portafoglio) · {_NOME_MISURA[misura]}</b><br>'
                           f'{_etichetta_rischio(misura, v_p)}<br>'
                           f'Rendimento: {r_p*100:.2f}%<br>'
                           f'{hover_lines}<extra></extra>'),
        ))
    return disegnati


def _gruppi_spuntati(p1_vals, p2_vals, p3_vals):
    """Asset spuntati in ogni colonna P1/P2/P3 della griglia."""
    return {p: [a for v in (vals or []) if v for a in v]
            for p, vals in (('P1', p1_vals), ('P2', p2_vals), ('P3', p3_vals))}


def _misure_scelte(r1, r2, r3):
    return {'P1': r1 or 'standard', 'P2': r2 or 'standard', 'P3': r3 or 'standard'}


_TENDINE_RISCHIO = ('fe-risk-p1', 'fe-risk-p2', 'fe-risk-p3')


@app.callback(
    Output('fe-frontier-chart', 'figure', allow_duplicate=True),
    Input({'type':'fe-p1','index':ALL}, 'value'),
    Input({'type':'fe-p2','index':ALL}, 'value'),
    Input({'type':'fe-p3','index':ALL}, 'value'),
    Input('fe-risk-p1',          'value'),
    Input('fe-risk-p2',          'value'),
    Input('fe-risk-p3',          'value'),
    State('fe-stock-data',       'data'),
    State('fe-frontier-rawdata', 'data'),
    State('fe-date-start',       'date'),
    State('fe-date-end',         'date'),
    State('fe-arima-window',     'value'),
    State('fe-frontier-chart',   'figure'),
    prevent_initial_call=True,
)
def preview_asset_points(p1_vals, p2_vals, p3_vals, r1, r2, r3, stock_data, rawdata,
                         date_start, date_end, arima_window, fig_attuale=None):
    # Frontiera già calcolata → il grafico lo disegna calc_and_render (un cambio di
    # misura non ricalcola: si ripreme Calcola); qui si accendono/spengono soltanto
    # i punti degli asset secondo le spunte P1/P2/P3.
    scelti = _asset_nei_portafogli(p1_vals, p2_vals, p3_vals)
    gruppi_ora = {p: set(v) for p, v in _gruppi_spuntati(p1_vals, p2_vals, p3_vals).items()}
    if rawdata:
        if ctx.triggered_id in _TENDINE_RISCHIO or not fig_attuale:
            raise PreventUpdate
        patch, cambiati = Patch(), 0
        for i, tr in enumerate(fig_attuale.get('data') or []):
            meta = tr.get('meta')
            if isinstance(meta, dict) and 'punto_asset' in meta:
                vis = meta['punto_asset'] in gruppi_ora.get(meta.get('portafoglio'), ())
                if (tr.get('visible', True) is not False) != vis:
                    patch['data'][i]['visible'] = vis
                    cambiati += 1
        if not cambiati:
            raise PreventUpdate
        return patch
    if not stock_data:
        raise PreventUpdate
    returns_all = _get_returns(stock_data)
    if returns_all is None or returns_all.empty:
        raise PreventUpdate
    returns_df = returns_all
    if date_start: returns_df = returns_df.loc[date_start:]
    if date_end:   returns_df = returns_df.loc[:date_end]
    returns_df = returns_df.dropna(how='all', axis=1).dropna(how='all', axis=0)
    misure = _misure_scelte(r1, r2, r3)
    gruppi = _gruppi_spuntati(p1_vals, p2_vals, p3_vals)
    # Nell'anteprima l'ARIMA non si calcola: si usa quello che c'è nel file
    # (gli asset senza stima prendono media e deviazione degli ultimi 250 giorni).
    arima = (arima_utenti.leggi(_get_username())
             if 'arima_garch' in misure.values() else None)

    fig = go.Figure()
    chart_assets = [a for a in returns_df.columns if a in scelti]
    _punti_asset(fig, returns_df, chart_assets, gruppi, misure, arima_window, arima,
                 returns_all, gruppi_ora, grande=True)
    # ── P1/P2/P3 dell'Analisi Portafoglio come punti, anche PRIMA del calcolo ──
    disegnati = _punti_portafogli(fig, returns_df, misure, arima_window, arima, returns_all)

    if not fig.data:
        return _fig_senza_portafogli()

    usati = [p for p in ('P1', 'P2', 'P3') if p in disegnati or gruppi.get(p)]
    fig.update_layout(
        title=dict(text='Rendimento / Rischio — asset selezionati',
                   font=dict(size=14, color='#1a3a6b'), x=0.02),
        xaxis=dict(title=_etichetta_asse([(p, misure[p]) for p in usati]),
                   gridcolor='#e8eef8', zeroline=False),
        yaxis=dict(title='Rendimento Atteso Ann. (%)', gridcolor='#e8eef8', zeroline=False),
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
        margin=dict(l=50, r=30, t=80, b=40), hovermode='closest',
    )
    return fig


@app.callback(
    Output('fe-arima-window-div',    'style'),
    Output('fe-arima-cache-status',  'children'),
    Output('fe-arima-cache-status',  'style'),
    Input('fe-risk-p1', 'value'),
    Input('fe-risk-p2', 'value'),
    Input('fe-risk-p3', 'value'),
    Input('fe-arima-reqid', 'data'),
)
def toggle_window_div(r1, r2, r3, _req):
    base = {'flexDirection':'column','gap':'2px'}   # come gli altri campi del riquadro Frontiere
    misure = _misure_scelte(r1, r2, r3).values()
    # «Finestra gg» serve solo se almeno un portafoglio usa Vol (fin.)
    win_style = {'display':'flex', **base} if 'vol' in misure else {'display':'none', **base}
    if 'arima_garch' not in misure:
        return win_style, '', {'display':'none'}
    dati = arima_utenti.leggi(_get_username())
    n = sum(1 for v in dati.get('asset', {}).values() if v.get('mu') is not None)
    stile = {'display':'inline-block','fontSize':'9px','padding':'2px 6px',
             'borderRadius':'3px','fontWeight':'600','alignSelf':'flex-end'}
    if n:
        txt = f'⚡ ARIMA {n} asset · {dati.get("aggiornato", "")}'
        stile.update({'background':'#d4edda','color':'#155724','border':'1px solid #c3e6cb'})
    else:
        txt = '⏳ ARIMA da calcolare (al primo Calcola Frontiera)'
        stile.update({'background':'#fff3cd','color':'#856404','border':'1px solid #ffc107'})
    return win_style, txt, stile


@app.callback(
    Output('fe-hint', 'style'),
    Input('fe-loaded-flag', 'data'),
    Input('fe-calc-btn',    'n_clicks'),
)
def toggle_hint(loaded, calc):
    shown  = {'display':'block','fontSize':'9px','color':'#0066cc','fontWeight':'600',
               'padding':'2px 5px 4px','background':'#e8f4ff',
               'borderLeft':'3px solid #0066cc','marginBottom':'4px','borderRadius':'0 4px 4px 0'}
    hidden = {**shown,'display':'none'}
    ctx = callback_context
    trig = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
    if trig == 'fe-calc-btn':
        return hidden
    if trig == 'fe-loaded-flag' and loaded:
        return {**shown, 'children': '▶ Dati caricati — clicca CALCOLA FRONTIERA per procedere'}
    return hidden


# ── Legenda unica: un clic sulla legenda dell'andamento vale anche per il drawdown ──
# Il drawdown non ha legenda; le serie hanno gli stessi nomi. Dopo ogni clic (o doppio
# clic che isola un titolo) si copia la visibilità per nome, tutto nel browser.
app.clientside_callback(
    """
    function(_rd) {
        var nu = window.dash_clientside.no_update;
        var gd = document.querySelector('#fe-perf-chart .js-plotly-plot');
        var dd = document.querySelector('#fe-drawdown-chart .js-plotly-plot');
        if (!gd || !dd || !gd.data || !dd.data || !window.Plotly) { return nu; }
        var vis = {};
        gd.data.forEach(function(t) { vis[t.name] = (t.visible === undefined) ? true : t.visible; });
        var idx = [], val = [];
        dd.data.forEach(function(t, i) {
            if (t.name in vis) { idx.push(i); val.push(vis[t.name]); }
        });
        if (idx.length) { window.Plotly.restyle(dd, {visible: val}, idx); }
        return nu;
    }
    """,
    Output('fe-legenda-sync', 'data'),
    Input('fe-perf-chart', 'restyleData'),
    prevent_initial_call=True,
)


# ── Aggiorna performance e drawdown chart al click di 📊, cambio date o cambio pesi ──────
@app.callback(
    Output('fe-perf-chart',      'figure', allow_duplicate=True),
    Output('fe-drawdown-chart',  'figure', allow_duplicate=True),
    Input({'type':'fe-chart','index':ALL},      'value'),
    Input({'type':'fe-chart-port','index':ALL}, 'value'),
    Input('fe-date-start',   'date'),
    Input('fe-date-end',     'date'),
    Input('fe-f1-weights',   'data'),
    Input('fe-f2-weights',   'data'),
    Input('fe-f3-weights',   'data'),
    State('fe-prices-data',  'data'),
    prevent_initial_call=True,
)
def update_perf_chart(chart_vals, port_chart_vals, date_start, date_end, f1j, f2j, f3j, prices_data):
    chart_assets = [a for v in (chart_vals or []) if v for a in v]
    ctx = callback_context
    port_inputs = ctx.inputs_list[1] if len(ctx.inputs_list) > 1 else []
    show_frontiers = {inp['id']['index']: bool(inp.get('value')) for inp in (port_inputs or [])}
    fw = {}
    for fname, jdata in [('F1', f1j), ('F2', f2j), ('F3', f3j)]:
        if jdata:
            try:
                fw[fname] = json.loads(jdata)
            except Exception:
                pass
    if any(show_frontiers.get(p) for p in ('P1', 'P2', 'P3')):
        fw.update(_pesi_analisi())
    return (
        _build_perf_chart(prices_data, chart_assets, fw, show_frontiers, date_start, date_end),
        _build_drawdown_chart(prices_data, chart_assets, fw, show_frontiers, date_start, date_end),
    )


# ── Calcola le 3 frontiere e ricostruisce la griglia ─────────────────────────
@app.callback(
    Output('fe-grid',            'children'),
    Output('fe-asset-count',     'children'),
    Output('fe-frontier-chart',  'figure'),
    Output('fe-perf-chart',      'figure'),
    Output('fe-drawdown-chart',  'figure'),
    Output('fe-stats-panel',     'children'),
    Output('fe-f1-weights',        'data'),
    Output('fe-f2-weights',        'data'),
    Output('fe-f3-weights',        'data'),
    Output('fe-frontier-rawdata',  'data'),
    Output('fe-selected-pt',       'data'),
    Output('fe-arima-poll',        'disabled'),
    Output('fe-arima-reqid',       'data'),
    Output('fe-arima-progress-div','style'),
    Input('fe-calc-btn',           'n_clicks'),
    State('fe-stock-data',       'data'),
    State('fe-prices-data',      'data'),
    State({'type':'fe-p1','index':ALL}, 'value'),
    State({'type':'fe-p2','index':ALL}, 'value'),
    State({'type':'fe-p3','index':ALL}, 'value'),
    State({'type':'fe-p1','index':ALL}, 'id'),
    State({'type':'fe-chart','index':ALL}, 'value'),
    State('fe-n-portfolios',     'value'),
    State('fe-min-weight',       'value'),
    State('fe-max-weight',       'value'),
    State('fe-rf',               'value'),
    State('fe-risk-p1',          'value'),
    State('fe-risk-p2',          'value'),
    State('fe-risk-p3',          'value'),
    State('fe-date-start',       'date'),
    State('fe-date-end',         'date'),
    State({'type':'fe-chart-port','index':ALL}, 'value'),
    State({'type':'fe-chart-port','index':ALL}, 'id'),
    State('fe-selected-pt',       'data'),
    State('fe-arima-window',      'value'),
    State('fe-data-source',       'data'),
    prevent_initial_call=True,
)
def calc_and_render(n, stock_data, prices_data,
                    p1_vals, p2_vals, p3_vals, p_ids, chart_vals,
                    n_port, wmin, wmax, rf, r1, r2, r3,
                    date_start, date_end, port_chart_vals, port_chart_ids, sel_pt_cur,
                    arima_window, data_source):
    _EMPTY_FIG = go.Figure().update_layout(
        paper_bgcolor='white', plot_bgcolor='#f8faff', font_color='#1a3a5c',
        annotations=[dict(text='Carica dati e clicca Calcola Frontiera',
                          xref='paper', yref='paper', x=0.5, y=0.5,
                          showarrow=False, font=dict(size=14, color='#6b7a99'))])
    _PH = html.Div('Carica i dati e clicca Calcola Frontiera',
                   style={'color':'#888','fontStyle':'italic','fontSize':'11px','padding':'12px 8px'})

    if not n or not stock_data:
        return [_PH], '', _EMPTY_FIG, _EMPTY_FIG, _EMPTY_FIG, '', None, None, None, None, None, True, None, {'display':'none'}

    returns_df = _get_returns(stock_data)
    if returns_df is None or returns_df.empty:
        return [_PH], '', _EMPTY_FIG, _EMPTY_FIG, _EMPTY_FIG, '', None, None, None, None, None, True, None, {'display':'none'}

    returns_all = returns_df   # tutto current.json: l'ARIMA ignora le date scelte
    if date_start: returns_df = returns_df.loc[date_start:]
    if date_end:   returns_df = returns_df.loc[:date_end]
    # Rimuove solo colonne/righe interamente NaN: il periodo comune si sceglie per
    # ogni portafoglio in _stima (dropna sui suoi asset).
    returns_df = returns_df.dropna(how='all', axis=1).dropna(how='all', axis=0)
    all_assets = returns_df.columns.tolist()
    misure = _misure_scelte(r1, r2, r3)

    # P1: default tutti gli asset; P2/P3: solo se l'utente ha spuntato almeno 2 asset
    def _p_assets(vals, ids, default_all=False):
        if not ids:
            return all_assets if default_all else []
        sel = [pid['index'] for v, pid in zip(vals, ids) if v]
        if len(sel) == 0:
            return all_assets if default_all else []
        return sel if len(sel) >= 2 else []

    # Una frontiera solo per i portafogli con almeno 2 asset spuntati: niente più
    # F1 «su tutto il file» quando P1 è vuoto (scelta dell'utente, 20/11/2026).
    p1_sel = _p_assets(p1_vals, p_ids, default_all=False)
    p2_sel = _p_assets(p2_vals, p_ids, default_all=False)
    p3_sel = _p_assets(p3_vals, p_ids, default_all=False)
    gruppi = {'P1': p1_sel, 'P2': p2_sel, 'P3': p3_sel}

    chart_assets = [a for v in (chart_vals or []) if v for a in v]
    port_chart_checked = {
        pid['index'] for v, pid in zip(port_chart_vals or [], port_chart_ids or []) if v
    }

    if not p1_sel and not p2_sel and not p3_sel:
        # Nessun portafoglio: niente frontiera e un avviso; i grafici di andamento
        # e drawdown (asset 📊) restano come sono.
        return (no_update, no_update, _fig_senza_portafogli(), no_update, no_update, '',
                None, None, None, None, None, True, None, {'display':'none'})

    wmin_f = (wmin or 0) / 100
    wmax_f = (wmax or 100) / 100
    rf_f   = (rf or 2.0) / 100
    n_f    = int(n_port or 15)

    # ── ARIMA+GARCH: stime dal file personale; gli asset che mancano si calcolano
    # in background (barra di avanzamento), poi on_arima_done ripreme Calcola.
    arima, senza_arima = None, []
    if 'arima_garch' in misure.values():
        username = _get_username()
        arima = arima_utenti.leggi(username)
        ns = _read_user_json()
        pesi_json = {p: [d for d, v in ns.items() if float(v.get(p) or 0) > 0] for p in gruppi}
        servono = sorted({a for p in gruppi if misure[p] == 'arima_garch'
                          for a in list(gruppi[p]) + pesi_json[p]
                          if a in returns_all.columns and returns_all[a].notna().any()})
        tickers = {d: v.get('ticker') for d, v in ns.items()}
        manc = arima_utenti.mancanti(arima, servono, tickers)
        recente = time.time() - _ARIMA_FALLITI.get(username, 0) < _ARIMA_RIPROVA_S
        if manc and not recente:
            import uuid as _uuid
            with _ARIMA_LOCK:
                # un calcolo già in corso per questo utente (doppio clic, due schede):
                # si segue quello invece di farne partire un secondo
                req_id = next((r for r, j in _ARIMA_JOBS.items()
                               if j.get('utente') == username and not j.get('done')), None)
                nuovo = req_id is None
                if nuovo:
                    req_id = str(_uuid.uuid4())[:8]
                    _ARIMA_JOBS[req_id] = {'pct': 0, 'total': len(manc), 'done': False,
                                           'error': None, 'utente': username}
            if nuovo:
                print(f"[calc_and_render] ARIMA {username}: {len(servono)} asset, "
                      f"{len(manc)} da calcolare in background")
                threading.Thread(target=_run_arima_thread,
                                 args=(req_id, username, returns_all, manc, tickers),
                                 daemon=True).start()
            _nu = no_update
            return (_nu, _nu, _nu, _nu, _nu, _nu,
                    _nu, _nu, _nu, _nu, _nu,
                    False, req_id, _ARIMA_PROG_SHOWN)
        senza_arima = [a for a in servono if arima_utenti.mu_vol(arima, a) is None]

    # ── Calcola le 3 frontiere, ognuna con la misura del suo portafoglio ────────
    frontier_res  = {}   # fname → (df_f, max_sharpe, min_vol, names)
    frontier_wgts = {}   # fname → {asset: weight%}
    # ogni calcolo riparte senza portafogli F scelti: F1/F2/F3 compaiono solo
    # cliccando un punto della frontiera (on_frontier_click)
    _existing_sel = {}

    for fname, assets_sel in [('F1', p1_sel), ('F2', p2_sel), ('F3', p3_sel)]:
        misura = misure[_P_DI[fname]]
        st = _stima(returns_df, assets_sel, misura, arima_window, arima, returns_all)
        if st is None or len(st[0].columns) < 2:
            continue
        df_sub, mu_sub, cov_sub = st
        if misura == 'arima_garch':
            # asset a varianza nulla fuori; μ estremi winsorizzati (±3σ fra gli
            # asset) per non avere frontiere verticali
            ok = [a for a in df_sub.columns if cov_sub.loc[a, a] > 1e-10]
            if len(ok) < 2:
                continue
            df_sub, mu_sub, cov_sub = df_sub[ok], mu_sub[ok], cov_sub.loc[ok, ok]
            if mu_sub.std() > 0:
                mu_sub = mu_sub.clip(mu_sub.mean() - 3 * mu_sub.std(),
                                     mu_sub.mean() + 3 * mu_sub.std())
        try:
            df_f, ms, mv, names = calc_frontier(
                df_sub, n=n_f, wmin=wmin_f, wmax=wmax_f,
                rf=rf_f, risk=misura, mu_override=mu_sub, cov_override=cov_sub)
            frontier_res[fname] = (df_f, ms, mv, names)
            ex = _existing_sel.get(fname, {})
            pt_idx = ex.get('pt_idx', -1)
            if pt_idx >= 0 and pt_idx < len(df_f):
                row  = df_f.iloc[pt_idx]
                wvec = row['Weights']
            elif ms is not None:
                wvec  = ms['Weights']
                pt_idx = -1
            else:
                continue
            frontier_wgts[fname] = {names[i]: round(wvec[i] * 100, 2) for i in range(len(names))}
        except Exception as e:
            print(f"[calc_and_render] frontiera {fname} ({misura}) fallita: {e}")

    # ── Grafico frontiera ────────────────────────────────────────────────────
    fig = go.Figure()
    # Tutti gli asset entrano nella figura, ma si vedono solo quelli spuntati in
    # P1/P2/P3: così togliere/rimettere una spunta dopo il calcolo è solo un cambio
    # di visibilità (preview_asset_points), con i valori già calcolati (anche ARIMA).
    _punti_asset(fig, returns_df, all_assets, gruppi, misure, arima_window, arima,
                 returns_all, {p: set(v) for p, v in gruppi.items()})

    for fname, (df_f, ms, mv, names) in frontier_res.items():
        fcolor  = _FC[fname]
        cml_col = _CML_C[fname]
        nome_m  = _NOME_MISURA[misure[_P_DI[fname]]]
        if not df_f.empty:
            cd = [[fname, i, i + 1] for i in range(len(df_f))]
            fig.add_trace(go.Scatter(
                x=df_f['Volatility'] * 100, y=df_f['Return'] * 100,
                mode='lines+markers', name=f'Frontiera {fname} · {nome_m}',
                line=dict(color=fcolor, width=2),
                marker=dict(size=6),
                customdata=cd,
                hovertemplate=f'<b>{fname} ({nome_m}) · Portafoglio %{{customdata[2]}}</b>'
                              f'<br>Rischio: %{{x:.2f}}%<br>Rendimento: %{{y:.2f}}%<extra></extra>',
            ))
        if ms is not None and ms['Sharpe'] > 0:
            vr = np.linspace(0, ms['Volatility'] * 1.8, 100)
            fig.add_trace(go.Scatter(
                x=vr * 100, y=(rf_f + ms['Sharpe'] * vr) * 100,
                mode='lines', name=f'CML {fname}',
                line=dict(color=cml_col, dash='dash', width=1.5), showlegend=False,
                hovertemplate=f'<b>CML {fname}</b><br>%{{x:.2f}}% → %{{y:.2f}}%<extra></extra>',
            ))
        if ms is not None:
            idx_ms = int(df_f['Sharpe'].idxmax())
            fig.add_trace(go.Scatter(
                x=[ms['Volatility'] * 100], y=[ms['Return'] * 100],
                mode='markers', name=f'Max Sharpe {fname}',
                marker=dict(symbol='circle', size=14, color=fcolor,
                            line=dict(color='#1a1a1a', width=1.2)),
                customdata=[[fname, idx_ms, 'sharpe']],
                hovertemplate=(f'<b>Max Sharpe {fname}: {ms["Sharpe"]:.2f}</b>'
                               f'<br>Rischio: {ms["Volatility"]*100:.2f}%'
                               f'<br>Rendimento: {ms["Return"]*100:.2f}%<extra></extra>'),
            ))

    # ── P1/P2/P3 dell'Analisi Portafoglio come punti di confronto ─────────────
    disegnati = _punti_portafogli(fig, returns_df, misure, arima_window, arima, returns_all)

    usati = [p for p in ('P1', 'P2', 'P3')
             if p in disegnati or {v: k for k, v in _P_DI.items()}[p] in frontier_res]
    fig.update_layout(
        title=dict(text='Frontiera Efficiente',
                   font=dict(size=14, color='#1a3a6b'), x=0.02),
        xaxis=dict(title=_etichetta_asse([(p, misure[p]) for p in usati]),
                   gridcolor='#e8eef8', zeroline=False),
        yaxis=dict(title='Rendimento Atteso Ann. (%)', gridcolor='#e8eef8', zeroline=False),
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=50, r=30, t=80, b=40), hovermode='closest',
    )

    # ── Performance e Drawdown chart ─────────────────────────────────────────
    # F1 spuntato di default solo se non c'era ancora nessuna riga di frontiera
    # (le righe P1/P2/P3 dell'Analisi ci sono anche prima del calcolo).
    # I portafogli F non si disegnano finché non si clicca la frontiera; restano
    # quelli dell'Analisi con la spunta che avevano.
    port_chart_checked = {k for k in port_chart_checked if not str(k).startswith('F')}
    pesi_analisi = _pesi_analisi()
    show_frontiers = {k: (k in port_chart_checked) for k in pesi_analisi}
    pesi_grafici = dict(pesi_analisi)
    # colonne F1/F2/F3 % della griglia: pesi dell'Analisi finché non si clicca un punto
    pesi_colonne = {f: pesi_analisi.get(p, {}) for f, p in (('F1', 'P1'), ('F2', 'P2'), ('F3', 'P3'))}
    fig2   = _build_perf_chart(prices_data, chart_assets, pesi_grafici, show_frontiers, date_start, date_end)
    fig_dd = _build_drawdown_chart(prices_data, chart_assets, pesi_grafici, show_frontiers, date_start, date_end)

    # ── Ricostruisci griglia con F1/F2/F3 weights ────────────────────────────
    p1_set    = set(p1_sel)
    p2_set    = set(p2_sel)
    p3_set    = set(p3_sel)
    chart_set = set(chart_assets)


    short_map = _short_history(returns_df)
    rows = []
    for i, asset in enumerate(all_assets):
        f1w = pesi_colonne['F1'].get(asset)
        f2w = pesi_colonne['F2'].get(asset)
        f3w = pesi_colonne['F3'].get(asset)
        row = html.Div([
            _asset_name_div(asset, short_map),
            html.Div(
                dcc.Checklist(id={'type':'fe-chart','index':asset},
                              options=[{'label':'','value':asset}],
                              value=[asset] if asset in chart_set else [],
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'7%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(
                dcc.Checklist(id={'type':'fe-p1','index':asset},
                              options=[{'label':'','value':asset}],
                              value=[asset] if asset in p1_set else [],
                              inputStyle={'accentColor':'#0066cc'},
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(
                dcc.Checklist(id={'type':'fe-p2','index':asset},
                              options=[{'label':'','value':asset}],
                              value=[asset] if asset in p2_set else [],
                              inputStyle={'accentColor':'#2ca02c'},
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(
                dcc.Checklist(id={'type':'fe-p3','index':asset},
                              options=[{'label':'','value':asset}],
                              value=[asset] if asset in p3_set else [],
                              inputStyle={'accentColor':'#d62728'},
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(id={'type':'fe-wgt-f1','index':asset}, children=_w_cell(f1w, '#0066cc'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wgt-f2','index':asset}, children=_w_cell(f2w, '#2ca02c'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wgt-f3','index':asset}, children=_w_cell(f3w, '#d62728'),
                     style={'width':'14%','display':'flex','alignItems':'center','justifyContent':'center'}),
        ], style={'display':'flex','alignItems':'center','height':'24px',
                  'borderBottom':'1px solid #f0f4fb',
                  'background':'white' if i % 2 == 0 else '#fafcff'})
        rows.append(row)

    # ── Righe portafoglio (F1/F2/F3) ─────────────────────────────────────────
    _PORT_BG = {'F1': '#f0f6ff', 'F2': '#f0fff4', 'F3': '#fff3f3'}

    # Costruisci sel_pt (label semplice per riga riepilogo)
    sel_pt = {}
    for fname, (df_f, ms, _mv, _names) in frontier_res.items():
        ex = _existing_sel.get(fname, {})
        pt_idx = ex.get('pt_idx', -1)
        ex_label = ex.get('label', '')
        if pt_idx >= 0 and pt_idx < len(df_f):
            sel_pt[fname] = {'label': ex_label or f'P{pt_idx + 1}', 'pt_idx': pt_idx}

    def _sel_cell(fname):
        info = sel_pt.get(fname, {})
        if not info:
            return html.Span('—', style={'fontSize':'8px','color':'#bbb'})
        return html.Span(info['label'],
                         style={'fontSize':'9px','fontWeight':'700','color':_FC[fname]})

    if frontier_wgts:
        rows.append(html.Div(style={'height':'3px','background':'#ccd9ee','margin':'2px 0'}))
        # ── Riga riepilogo portafoglio selezionato ────────────────────────────
        rows.append(html.Div([
            html.Div(html.Span('Portfolio selezionato',
                               style={'fontSize':'7px','color':'#8899bb','fontStyle':'italic'}),
                     style={'width':'25%','height':'36px','display':'flex','alignItems':'center','paddingLeft':'4px'}),
            html.Div(style={'width':'7%'}),
            html.Div('', style={'width':'8%'}),
            html.Div('', style={'width':'8%'}),
            html.Div('', style={'width':'8%'}),
            html.Div(id={'type':'fe-sel-info','index':'F1'}, children=_sel_cell('F1'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center','flexDirection':'column'}),
            html.Div(id={'type':'fe-sel-info','index':'F2'}, children=_sel_cell('F2'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center','flexDirection':'column'}),
            html.Div(id={'type':'fe-sel-info','index':'F3'}, children=_sel_cell('F3'),
                     style={'width':'14%','display':'flex','alignItems':'center','justifyContent':'center','flexDirection':'column'}),
        ], style={'display':'flex','alignItems':'center','height':'36px',
                  'borderBottom':'1px solid #e0e8f8','background':'#f8faff'}))
        for fname, fcolor in _FC.items():
            if fname not in frontier_wgts:
                continue
            # nascosta finché non si clicca un punto della frontiera fname
            rows.append(html.Div(id={'type':'fe-riga-f','index':fname}, children=[
                html.Div(
                    html.Span(f'Portafoglio {fname}',
                              style={'fontSize':'8px','color':fcolor,'fontWeight':'700',
                                     'overflow':'hidden','whiteSpace':'nowrap','textOverflow':'ellipsis'}),
                    style={'width':'25%','height':'28px','display':'flex','alignItems':'center','paddingLeft':'4px'}
                ),
                html.Div(
                    dcc.Checklist(
                        id={'type':'fe-chart-port','index':fname},
                        options=[{'label':'','value':fname}],
                        value=[fname] if fname in port_chart_checked else [],
                        inputStyle={'accentColor': fcolor},
                        style={'display':'flex','justifyContent':'center'},
                    ),
                    style={'width':'7%','display':'flex','justifyContent':'center','alignItems':'center'}
                ),
                html.Div('—', style={'width':'8%','textAlign':'center','fontSize':'8px','color':'#ccc'}),
                html.Div('—', style={'width':'8%','textAlign':'center','fontSize':'8px','color':'#ccc'}),
                html.Div('—', style={'width':'8%','textAlign':'center','fontSize':'8px','color':'#ccc'}),
                html.Div(_w_cell(100.0 if fname == 'F1' else None, '#0066cc'),
                         style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
                html.Div(_w_cell(100.0 if fname == 'F2' else None, '#2ca02c'),
                         style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
                html.Div(_w_cell(100.0 if fname == 'F3' else None, '#d62728'),
                         style={'width':'14%','display':'flex','alignItems':'center','justifyContent':'center'}),
            ], style={'display':'none','alignItems':'center','height':'28px',
                      'borderBottom':'1px solid #e0e8f8',
                      'background': _PORT_BG.get(fname, 'white')}))

    # Portafogli dell'Analisi sotto le frontiere, con la spunta che avevano
    rows.extend(_righe_portafogli_analisi(pesi_analisi, port_chart_checked))

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = []
    for fname, (df_f, ms, mv, names) in frontier_res.items():
        if ms is not None:
            misura = misure[_P_DI[fname]]
            dettaglio = {'vol': f' {int(arima_window or 250)} gg',
                         'arima_garch': f' {arima_utenti.FINESTRA} gg'}.get(misura, '')
            stats.append(html.Span([
                html.B(f'Max Sharpe {fname} ', style={'color': _FC[fname]}),
                html.Span(f'({_NOME_MISURA[misura]}{dettaglio}): ',
                          style={'color': '#6b7a99'}),
                f"{ms['Return']*100:.1f}% rend · {ms['Volatility']*100:.1f}% rischio · "
                f"Sharpe {ms['Sharpe']:.2f}   ",
            ], style={'marginRight':'8px', 'fontSize':'11px'}))
    if senza_arima:
        stats.append(html.Span(
            f'ARIMA non disponibile per {", ".join(senza_arima[:4])}'
            f'{"…" if len(senza_arima) > 4 else ""}: media e volatilità degli ultimi '
            f'{arima_utenti.FINESTRA} giorni',
            style={'fontSize':FONT['xs'],'color':'#856404','fontStyle':'italic'}))

    f1j = json.dumps(pesi_colonne['F1']) if pesi_colonne['F1'] else None
    f2j = json.dumps(pesi_colonne['F2']) if pesi_colonne['F2'] else None
    f3j = json.dumps(pesi_colonne['F3']) if pesi_colonne['F3'] else None
    count = f'{len(all_assets)} asset · {len(frontier_res)} frontiere calcolate'

    raw_data = {}
    for fname, (df_f, _ms, _mv, names) in frontier_res.items():
        raw_data[fname] = {
            'assets': names,
            'points': [
                {'vol': float(r['Volatility']), 'ret': float(r['Return']),
                 'weights': list(r['Weights'])}
                for _, r in df_f.iterrows()
            ],
        }
    rawdata_json = json.dumps(raw_data)

    return (rows, count, fig, fig2, fig_dd,
            html.Div(stats, style={'display':'flex','flexWrap':'wrap','gap':'4px'}),
            f1j, f2j, f3j, rawdata_json, json.dumps(sel_pt),
            True, None, {'display':'none'})


# ── Click su frontiera → aggiorna label + pesi della frontiera cliccata ───────
@app.callback(
    Output('fe-selected-pt', 'data',             allow_duplicate=True),
    Output('fe-f1-weights',  'data',             allow_duplicate=True),
    Output('fe-f2-weights',  'data',             allow_duplicate=True),
    Output('fe-f3-weights',  'data',             allow_duplicate=True),
    Output({'type':'fe-riga-f','index':ALL},     'style', allow_duplicate=True),
    Output({'type':'fe-chart-port','index':ALL}, 'value', allow_duplicate=True),
    Input('fe-frontier-chart',   'clickData'),
    State('fe-selected-pt',      'data'),
    State('fe-frontier-rawdata', 'data'),
    State({'type':'fe-riga-f','index':ALL},     'style'),
    State({'type':'fe-riga-f','index':ALL},     'id'),
    State({'type':'fe-chart-port','index':ALL}, 'value'),
    State({'type':'fe-chart-port','index':ALL}, 'id'),
    prevent_initial_call=True,
)
def on_frontier_click(click_data, sel_pt_json, rawdata_json,
                      stili_f, ids_f, val_port, ids_port):
    if not click_data:
        raise PreventUpdate
    pt = click_data['points'][0]
    cd = pt.get('customdata')
    if not cd or len(cd) < 2:
        raise PreventUpdate
    fname, pt_idx = cd[0], int(cd[1])
    is_sharpe = len(cd) >= 3 and cd[2] == 'sharpe'
    sel_pt = json.loads(sel_pt_json) if sel_pt_json else {}
    # Mostra sempre il numero del portafoglio; segna quello Max-Sharpe con ★
    label = f'P{pt_idx + 1}' + (' ★Sharpe' if is_sharpe else '')
    sel_pt[fname] = {'label': label, 'pt_idx': pt_idx}

    # Aggiorna i pesi della frontiera cliccata
    w_out = {k: no_update for k in ('F1', 'F2', 'F3')}
    try:
        raw = json.loads(rawdata_json) if rawdata_json else {}
        fdata = raw.get(fname)
        if fdata and pt_idx < len(fdata['points']):
            assets  = fdata['assets']
            weights = fdata['points'][pt_idx]['weights']
            w_dict  = json.dumps({assets[i]: round(weights[i] * 100, 2)
                                  for i in range(len(assets))})
            w_out[fname] = w_dict
    except Exception:
        pass
    if w_out.get(fname) is no_update:
        raise PreventUpdate

    # Il portafoglio Fn compare in griglia (riga visibile) e nei grafici (spuntato)
    # solo ora, al primo clic sulla sua frontiera; ai clic successivi la spunta
    # resta quella scelta dall'utente.
    stili_out, val_out = [], []
    for st, d in zip(stili_f or [], ids_f or []):
        stili_out.append({**(st or {}), 'display': 'flex'} if d['index'] == fname else no_update)
    nascosta = next((((st or {}).get('display') == 'none') for st, d in zip(stili_f or [], ids_f or [])
                     if d['index'] == fname), False)
    for v, d in zip(val_port or [], ids_port or []):
        val_out.append([fname] if (d['index'] == fname and nascosta) else no_update)

    return json.dumps(sel_pt), w_out['F1'], w_out['F2'], w_out['F3'], stili_out, val_out


# ── Aggiorna riga riepilogo portafoglio selezionato ──────────────────────────
@app.callback(
    Output({'type':'fe-sel-info','index':ALL}, 'children'),
    Input('fe-selected-pt', 'data'),
    State({'type':'fe-sel-info','index':ALL}, 'id'),
    prevent_initial_call=True,
)
def update_sel_info(sel_pt_json, ids):
    sel_pt = json.loads(sel_pt_json) if sel_pt_json else {}
    result = []
    for d in (ids or []):
        fname = d['index']
        info  = sel_pt.get(fname, {})
        label = info.get('label', '') if info else ''
        color = _FC.get(fname, '#888') if label else '#bbb'
        result.append(
            html.Span(label or '—',
                      style={'fontSize':'9px','fontWeight':'700', 'color': color})
        )
    return result


# ── Aggiorna celle peso griglia quando i pesi cambiano ───────────────────────
def _make_wgt_cell_cb(store_id, col_type, color):
    @app.callback(
        Output({'type': col_type, 'index': ALL}, 'children'),
        Input(store_id, 'data'),
        State({'type': col_type, 'index': ALL}, 'id'),
        prevent_initial_call=True,
    )
    def _upd(wj, ids):
        fw = json.loads(wj) if wj else {}
        return [_w_cell(fw.get(d['index']), color) for d in (ids or [])]
    return _upd

_wgt_f1_cb = _make_wgt_cell_cb('fe-f1-weights', 'fe-wgt-f1', '#0066cc')
_wgt_f2_cb = _make_wgt_cell_cb('fe-f2-weights', 'fe-wgt-f2', '#2ca02c')
_wgt_f3_cb = _make_wgt_cell_cb('fe-f3-weights', 'fe-wgt-f3', '#d62728')


# ── ARIMA: polling progresso ──────────────────────────────────────────────────
@app.callback(
    Output('fe-arima-reqid',         'data',     allow_duplicate=True),
    Output('fe-arima-progress-text', 'children', allow_duplicate=True),
    Output('fe-arima-poll',          'disabled', allow_duplicate=True),
    Output('fe-arima-progress-bar',  'style',    allow_duplicate=True),
    Input('fe-arima-poll',           'n_intervals'),
    State('fe-arima-reqid',          'data'),
    prevent_initial_call=True,
)
def arima_poll(n_int, cur_req_id):
    if not cur_req_id or ':' in cur_req_id:
        raise PreventUpdate
    with _ARIMA_LOCK:
        s = dict(_ARIMA_JOBS.get(cur_req_id) or {})
    if not s:
        # processo riavviato a calcolo in corso: si ricalcola da capo col pulsante
        return (cur_req_id + ':error', '❌ Calcolo ARIMA interrotto', True,
                _arima_bar_style(100, '#c0392b'))
    if s.get('error'):
        return (cur_req_id + ':error', f"❌ Errore ARIMA: {s['error'][:50]}", True,
                _arima_bar_style(100, '#c0392b'))
    if s.get('done'):
        return cur_req_id + ':done', '✓ ARIMA completato', True, _arima_bar_style(100)
    total = s['total'] or 1
    pct   = int(s['pct'] / total * 100)
    return (no_update,
            f"ARIMA: {s['pct']}/{total} asset nuovi ({pct}%)",
            False,
            _arima_bar_style(pct))


# ── ARIMA finito → il file personale ha tutti gli asset: si ripreme Calcola ──
# (con un errore pure: calc_and_render non riparte per 15 minuti e usa media e
# volatilità degli ultimi 250 giorni per gli asset senza stima).
@app.callback(
    Output('fe-calc-btn',           'n_clicks',   allow_duplicate=True),
    Output('fe-arima-progress-div', 'style',      allow_duplicate=True),
    Input('fe-arima-reqid',         'data'),
    State('fe-calc-btn',            'n_clicks'),
    prevent_initial_call=True,
)
def on_arima_done(req_id, cur_clicks):
    if not req_id or not (req_id.endswith(':done') or req_id.endswith(':error')):
        raise PreventUpdate
    with _ARIMA_LOCK:
        # i job finiti restano 10 minuti: un'altra scheda può star seguendo lo stesso
        for r in [r for r, j in _ARIMA_JOBS.items()
                  if j.get('done') and time.time() - j.get('fine', 0) > 600]:
            _ARIMA_JOBS.pop(r, None)
    return (cur_clicks or 0) + 1, (_ARIMA_PROG_HIDDEN if req_id.endswith(':done') else no_update)


# ── Salva portafogli F1/F2/F3 per importazione in Analisi Portafoglio ────────
_FE_PORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'portafoglio', 'sessions', 'fe_portfolio.json')

@app.callback(
    Output('fe-save-port-msg', 'children'),
    Input('fe-save-to-port-btn', 'n_clicks'),
    State('fe-f1-weights', 'data'),
    State('fe-f2-weights', 'data'),
    State('fe-f3-weights', 'data'),
    prevent_initial_call=True,
)
def save_to_portafoglio(n, f1j, f2j, f3j):
    if not n:
        raise PreventUpdate
    portfolios = {}
    for fname, jdata in [('F1', f1j), ('F2', f2j), ('F3', f3j)]:
        if jdata:
            try:
                portfolios[fname] = json.loads(jdata) if isinstance(jdata, str) else jdata
            except Exception:
                pass
    if not portfolios:
        return '⚠ Calcola prima la frontiera'
    try:
        data = {'portfolios': portfolios, 'saved_at': datetime.now().strftime('%d/%m/%Y %H:%M')}
        with open(_FE_PORT_FILE, 'w') as f:
            json.dump(data, f)
        labels = ', '.join(portfolios.keys())
        return f'✓ Salvato ({labels}) — {data["saved_at"]}'
    except Exception as e:
        return f'⚠ Errore: {e}'


# ═════════════════════════════════════════════════════════════════════════════
# IMPORTA / ESPORTA PORTAFOGLIO — profili condivisi (sessions_manager)
# ═════════════════════════════════════════════════════════════════════════════
_FPIO_OVERLAY = {'position':'fixed','top':'0','left':'0','width':'100%','height':'100%',
                 'zIndex':'9000','background':'rgba(0,0,0,0.45)',
                 'alignItems':'center','justifyContent':'center'}

@app.callback(
    Output('fpio-overlay',     'style'),
    Output('fpio-exp-profile', 'options'),
    Output('fpio-imp-profile', 'options'),
    Input('fpio-btn',   'n_clicks'),
    Input('fpio-close', 'n_clicks'),
    prevent_initial_call=True,
)
def fpio_toggle(open_n, close_n):
    if callback_context.triggered_id == 'fpio-btn':
        opts = [{'label': a, 'value': a} for a in _sm.list_analyses(_get_username())]
        return {**_FPIO_OVERLAY, 'display': 'flex'}, opts, opts
    return {**_FPIO_OVERLAY, 'display': 'none'}, no_update, no_update


# Reset dei campi del dialogo ad ogni apertura (evita stato vecchio)
@app.callback(
    Output('fpio-mode',        'value',    allow_duplicate=True),
    Output('fpio-imp-profile', 'value',    allow_duplicate=True),
    Output('fpio-imp-status',  'children', allow_duplicate=True),
    Output('fpio-exp-status',  'children', allow_duplicate=True),
    Output('fpio-exp-new',     'value',    allow_duplicate=True),
    Output('fpio-exp-profile', 'value',    allow_duplicate=True),
    Input('fpio-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def fpio_reset(n):
    if not n:
        raise PreventUpdate
    return 'export', None, '', '', '', None


@app.callback(
    Output('fpio-export-view', 'style'),
    Output('fpio-import-view', 'style'),
    Input('fpio-mode', 'value'),
)
def fpio_switch_view(mode):
    if mode == 'import':
        return {'display': 'none'}, {'display': 'block'}
    return {'display': 'block'}, {'display': 'none'}


def _fpio_load_w(j):
    if not j:
        return {}
    try:
        return json.loads(j) if isinstance(j, str) else dict(j)
    except Exception:
        return {}


@app.callback(
    Output('fpio-exp-status',  'children'),
    Output('fpio-exp-profile', 'options', allow_duplicate=True),
    Output('fpio-imp-profile', 'options', allow_duplicate=True),
    Output('fpio-exp-new',     'value'),
    Input('fpio-exp-btn', 'n_clicks'),
    State('fpio-exp-col',  'value'),
    State('fe-f1-weights', 'data'),
    State('fe-f2-weights', 'data'),
    State('fe-f3-weights', 'data'),
    State('fpio-exp-profile', 'value'),
    State('fpio-exp-new',     'value'),
    prevent_initial_call=True,
)
def fpio_export(n, col, f1j, f2j, f3j, ana_sel, ana_new):
    if not n:
        raise PreventUpdate
    u = _get_username()
    name = (ana_new or '').strip() or (ana_sel or '').strip()
    if not name:
        return "⚠ Scrivi un nome nuovo o scegli un'analisi da sovrascrivere", no_update, no_update, no_update
    col = col if col in ('F1', 'F2', 'F3') else 'F1'

    # 1) Pesi dallo store del browser (immediati ma a volte desincronizzati)
    wmap = {'F1': _fpio_load_w(f1j), 'F2': _fpio_load_w(f2j), 'F3': _fpio_load_w(f3j)}
    weights = {a: float(v) for a, v in (wmap[col] or {}).items() if v}

    # Niente ripiego su current.json: lì P1/P2/P3 sono i portafogli dell'Analisi,
    # non i pesi della frontiera (la copia automatica è stata tolta il 20/11/2026).

    print(f"[FPIO-EXPORT] col={col} | store={len(wmap[col])} | "
          f"finale={len(weights)} asset {list(weights.items())[:4]}", flush=True)
    if not weights:
        return f'⚠ La colonna {col} non ha pesi (calcola prima le frontiere)', no_update, no_update, no_update
    # Meta autosufficiente: ticker+valuta da current.json (per ri-aggiungere i mancanti all'import)
    ns = _read_user_json() or {}
    meta = {a: {'ticker': (ns.get(a, {}) or {}).get('ticker', ''),
                'valuta': (ns.get(a, {}) or {}).get('currency', 'EUR')}
            for a in weights}
    ok = _sm.save_analysis(u, name, weights, meta=meta)
    opts = [{'label': a, 'value': a} for a in _sm.list_analyses(u)]
    if ok:
        return (f'✅ Colonna {col} salvata nell\'analisi "{name}" ({len(weights)} asset)',
                opts, opts, '')
    return '⚠ Errore durante l\'esportazione', no_update, no_update, no_update


@app.callback(
    Output('fe-f1-weights', 'data', allow_duplicate=True),
    Output('fe-f2-weights', 'data', allow_duplicate=True),
    Output('fe-f3-weights', 'data', allow_duplicate=True),
    Output({'type': 'fe-p1', 'index': ALL}, 'value', allow_duplicate=True),
    Output({'type': 'fe-p2', 'index': ALL}, 'value', allow_duplicate=True),
    Output({'type': 'fe-p3', 'index': ALL}, 'value', allow_duplicate=True),
    Output({'type': 'fe-chart-port', 'index': ALL}, 'value', allow_duplicate=True),
    Output('fpio-imp-status', 'children'),
    Output('fpio-overlay',    'style', allow_duplicate=True),
    Input('fpio-imp-btn', 'n_clicks'),
    State('fpio-imp-profile', 'value'),
    State('fpio-imp-target',  'value'),
    State({'type': 'fe-p1', 'index': ALL}, 'id'),
    State({'type': 'fe-p2', 'index': ALL}, 'id'),
    State({'type': 'fe-p3', 'index': ALL}, 'id'),
    State({'type': 'fe-chart-port', 'index': ALL}, 'id'),
    prevent_initial_call=True,
)
def fpio_import(n, analysis, target, ids1, ids2, ids3, ids_port):
    if not n:
        raise PreventUpdate
    ids1, ids2, ids3 = ids1 or [], ids2 or [], ids3 or []
    ids_port = ids_port or []
    nu = (no_update, no_update, no_update,
          [no_update] * len(ids1), [no_update] * len(ids2), [no_update] * len(ids3),
          [no_update] * len(ids_port))
    if not analysis:
        return (*nu, "⚠ Scegli un'analisi", no_update)
    target = target if target in ('F1', 'F2', 'F3') else 'F1'
    imported = {a: float(v) for a, v in (_sm.get_analysis(_get_username(), analysis) or {}).items()}
    if not imported:
        return (*nu, '⚠ Analisi vuota', no_update)

    grid_assets = {d['index'] for d in ids1}
    if not grid_assets:
        return (*nu, '⚠ Nessun asset in griglia: attendi il caricamento dati e riprova', no_update)
    matched = len(set(imported) & grid_assets)
    if matched == 0:
        return (*nu, f'⚠ Gli asset dell\'analisi non coincidono con la griglia '
                     f'({len(imported)} attesi, 0 trovati)', no_update)

    def _checks(ids):
        return [[d['index']] if imported.get(d['index'], 0) else [] for d in ids]

    # Aggiorna SOLO la colonna target (le altre restano invariate)
    outs = {'F1': no_update, 'F2': no_update, 'F3': no_update}
    outs[target] = json.dumps(imported)
    chk = {'F1': [no_update] * len(ids1), 'F2': [no_update] * len(ids2), 'F3': [no_update] * len(ids3)}
    chk[target] = _checks({'F1': ids1, 'F2': ids2, 'F3': ids3}[target])
    chk_port = [[d['index']] if d['index'] == target else no_update for d in ids_port]

    msg = f'✓ Analisi "{analysis}" importata in {target} — {matched}/{len(imported)} asset abbinati'
    return (outs['F1'], outs['F2'], outs['F3'],
            chk['F1'], chk['F2'], chk['F3'], chk_port,
            msg, {**_FPIO_OVERLAY, 'display': 'none'})


@app.callback(
    Output('fe-f1-weights', 'data', allow_duplicate=True),
    Input({'type': 'fe-chart', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def _fe_sync_checked(chart_vals):
    selected = [v[0] for v in (chart_vals or []) if v]
    _update_user_json_fe(checked=selected)
    return no_update


# Niente più copia automatica dei pesi di frontiera in P1/P2/P3 di current.json
# (`_fe_sync_weights`, tolto il 20/11/2026): a ogni calcolo il Max Sharpe di F1/F2/F3
# sovrascriveva i portafogli dell'Analisi Portafoglio, e al giro dopo la Frontiera
# partiva da quelli (P1 = un asset al 100% → F1 sparita). I pesi dell'Analisi si
# cambiano solo nel Portafoglio; per riportare un punto di frontiera c'è
# «💾 Salva in Portafoglio».


# ── Select-all / deselect-all for P1, P2, P3 ────────────────────────────────
def _make_selall_cb(btn_id, col_type):
    @app.callback(
        Output({'type': col_type, 'index': ALL}, 'value'),
        Input(btn_id, 'n_clicks'),
        State({'type': col_type, 'index': ALL}, 'value'),
        State({'type': col_type, 'index': ALL}, 'options'),
        prevent_initial_call=True,
    )
    def _toggle(n_clicks, cur_vals, opts_list):
        # if every checkbox is already fully selected → deselect all; else select all
        all_selected = all(
            set(v) == {o['value'] for o in ops}
            for v, ops in zip(cur_vals, opts_list)
        )
        if all_selected:
            return [[] for _ in cur_vals]
        return [[ops[0]['value']] for ops in opts_list]
    return _toggle

_cb_chart = _make_selall_cb('fe-selall-chart', 'fe-chart')
_cb_p1    = _make_selall_cb('fe-selall-p1',   'fe-p1')
_cb_p2    = _make_selall_cb('fe-selall-p2',   'fe-p2')
_cb_p3    = _make_selall_cb('fe-selall-p3',   'fe-p3')


@app.callback(
    Output('fe-stock-data',   'data',     allow_duplicate=True),
    Output('fe-prices-data',  'data',     allow_duplicate=True),
    Output('fe-loaded-flag',  'data',     allow_duplicate=True),
    Output('fe-last-updated', 'children', allow_duplicate=True),
    Output('fe-data-source',  'data',     allow_duplicate=True),
    Output('fe-sync-sig',     'data'),
    Input('fe-live-sync',     'n_intervals'),
    State('fe-sync-sig',      'data'),
    prevent_initial_call=True,
)
def fe_live_sync(_, sig):
    ns = _read_user_json()
    if not ns:
        raise PreventUpdate

    # Firma su due livelli:
    # 1. asset_sig: solo nomi asset → cambia se si aggiunge/rimuove un titolo
    # 2. full_sig: asset + spunte + pesi → cambia anche se si modifica solo un peso
    #    (stessa formula di on_page_load via _fe_current_sig)
    asset_sig = repr(tuple(sorted(ns.keys())))
    full_sig  = _fe_current_sig(ns)
    old_full = (sig or '')
    if full_sig == old_full:
        raise PreventUpdate

    op, cr = _reconstruct_from_json_fe(ns)
    if op is None or cr is None:
        raise PreventUpdate

    # Estrai asset_sig dall'ultima firma salvata per capire se la lista è cambiata
    try:
        old_asset_sig = repr(eval(old_full)[0]) if old_full else ''
    except Exception:
        old_asset_sig = ''

    n = len(op.columns)
    from navbar import ETICHETTA_PERSONALE
    lbl = f'{ETICHETTA_PERSONALE} ({n} asset)'

    if asset_sig != old_asset_sig:
        # Lista asset cambiata: aggiorna fe-stock-data → _clear_stale_frontier fa il reset
        return (
            cr.to_json(orient='split', date_format='iso'),
            op.to_json(orient='split', date_format='iso'),
            True, lbl, 'default', full_sig,
        )
    else:
        # Solo pesi/spunte cambiati: aggiorna solo fe-sync-sig → ricostruisce la griglia
        # senza toccare fe-stock-data → la frontiera calcolata NON viene cancellata
        return (
            no_update, no_update, no_update, lbl, no_update, full_sig,
        )


# ─────────────────────────────────────────────────────────────────────────────
server = app.server

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8052))
    app.run(debug=False, port=port, host='0.0.0.0')
