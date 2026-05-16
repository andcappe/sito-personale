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

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update, ALL
from dash.exceptions import PreventUpdate

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.browser_css import BROWSER_RESET_CSS, FONT

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

app.index_string = '''
<!DOCTYPE html><html>
<head>{%metas%}<title>Frontiera Efficiente — Andrea Cappelletti</title>{%favicon%}{%css%}
<style>
''' + BROWSER_RESET_CSS + '''
  @keyframes fe-spin { to { transform: rotate(360deg); } }
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
_ARIMA_STATE = {
    'req_id': None, 'running': False, 'done': False,
    'pct': 0, 'total': 0, 'error': None, 'mu': None, 'cov': None,
}
# Stato cache ARIMA pre-calcolata — letto una volta all'avvio, aggiornato dopo ogni calcolo
_ARIMA_CACHE_INFO = {'available': False, 'ts': ''}

def _arima_garch_mu_vol(returns_df, window=250, req_id=None):
    """ARIMA(p,0,q) best-AIC + GARCH(1,1) sui residui.
    Restituisce (mu_series_daily, cov_df_daily).

    μ  = media incondizionale ARIMA = const/(1-Σφᵢ)
         → aspettativa a lungo orizzonte, stabile e coerente con Markowitz.
         La previsione condizionale a 1 passo (forecast) NON va usata:
         dipende dai residui recenti e annualizzata ×252 produce μ caotici.
    σ  = volatilità condizionale GARCH 1-step → cattura il regime di rischio corrente.
    Σ  = D_garch @ corr_resid @ D_garch   (correlazioni sui residui ARIMA).
    """
    import warnings
    try:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA_CLS
    except ImportError:
        return returns_df.mean(), returns_df.cov()
    try:
        from arch import arch_model as _arch_model
        _has_arch = True
    except ImportError:
        _has_arch = False

    mu_dict        = {}
    vol_daily_dict = {}
    resid_dict     = {}   # residui ARIMA allineati per calcolo correlazioni
    cols = returns_df.columns.tolist()

    with _ARIMA_LOCK:
        if req_id and _ARIMA_STATE.get('req_id') == req_id:
            _ARIMA_STATE['total'] = len(cols)
            _ARIMA_STATE['pct']   = 0

    for i, col in enumerate(cols):
        s = returns_df[col].dropna().tail(window)
        if len(s) < 30:
            mu_dict[col]        = float(s.mean())
            vol_daily_dict[col] = float(s.std())
            resid_dict[col]     = (s - s.mean()).values
        else:
            best_aic   = np.inf
            best_resid = (s - s.mean()).values
            best_model = None
            for p in range(3):
                for q in range(3):
                    if p == 0 and q == 0:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            m = _ARIMA_CLS(s, order=(p, 0, q)).fit()
                        if m.aic < best_aic:
                            best_aic   = m.aic
                            best_resid = m.resid.values
                            best_model = m
                    except Exception:
                        pass

            # ── μ incondizionale: const / (1 - Σ φᵢ) ───────────────────────
            # Aspettativa a lungo orizzonte del processo ARIMA.
            # Guard: se Σφᵢ > 0.85 (near-unit-root), denom < 0.15 è mal
            # identificato e amplifica la costante → fallback a media campionaria.
            # Sanity cap: se il risultato supera 5× la media storica, è un
            # artefatto numerico → fallback.
            best_mu   = float(s.mean())   # default: media campionaria
            hist_mean = float(s.mean())
            if best_model is not None:
                try:
                    par = best_model.params
                    ar_coefs = [par[k] for k in par.index if k.startswith('ar.')]
                    ar_sum   = sum(ar_coefs)
                    denom    = 1.0 - ar_sum
                    if 'const' in par.index and abs(denom) > 0.15:
                        mu_candidate = float(par['const'] / denom)
                        # Sanity: scarta se > 5× la media storica in valore assoluto
                        if hist_mean != 0 and abs(mu_candidate) > 5 * abs(hist_mean) + 1e-5:
                            best_mu = hist_mean
                        else:
                            best_mu = mu_candidate
                    else:
                        best_mu = float(best_model.fittedvalues.mean())
                except Exception:
                    best_mu = float(s.mean())

            mu_dict[col]    = best_mu
            resid_dict[col] = best_resid

            # ── σ condizionale: GARCH(1,1) 1-step ───────────────────────────
            if _has_arch and len(best_resid) > 20:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        scaled = best_resid * 100
                        gm = _arch_model(scaled, vol='Garch', p=1, q=1, rescale=False)
                        gf = gm.fit(disp='off')
                        fc = gf.forecast(horizon=1)
                    vol_daily_dict[col] = float(np.sqrt(fc.variance.values[-1, 0])) / 100
                except Exception:
                    vol_daily_dict[col] = float(np.std(best_resid))
            else:
                vol_daily_dict[col] = float(np.std(best_resid))

        with _ARIMA_LOCK:
            if req_id and _ARIMA_STATE.get('req_id') == req_id:
                _ARIMA_STATE['pct'] = i + 1

    mu_series = pd.Series(mu_dict)

    # ── Σ = D_garch @ corr @ D_garch ─────────────────────────────────────────
    # Correlazioni sui rendimenti grezzi: allineate per data, robuste a serie
    # di lunghezza diversa (pandas corr usa osservazioni pairwise complete).
    r_df = returns_df[cols].tail(window)
    corr = np.nan_to_num(r_df.corr().values, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    vols  = np.array([vol_daily_dict[c] for c in cols])
    D     = np.diag(vols)
    cov_matrix = D @ corr @ D + 1e-8 * np.eye(len(cols))   # ensure positive-definite
    cov_df = pd.DataFrame(cov_matrix, index=cols, columns=cols)

    return mu_series, cov_df


def _run_arima_thread(req_id, returns_df, window):
    try:
        mu_series, cov_df = _arima_garch_mu_vol(returns_df, window, req_id)
        with _ARIMA_LOCK:
            if _ARIMA_STATE['req_id'] == req_id:
                _ARIMA_STATE['mu']      = mu_series
                _ARIMA_STATE['cov']     = cov_df
                _ARIMA_STATE['done']    = True
                _ARIMA_STATE['running'] = False
    except Exception as e:
        with _ARIMA_LOCK:
            if _ARIMA_STATE['req_id'] == req_id:
                _ARIMA_STATE['error']   = str(e)
                _ARIMA_STATE['done']    = True
                _ARIMA_STATE['running'] = False

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


def _read_shared_data():
    """
    Legge i dati di portafoglio condivisi.
    1. Buffer live di _app_portafoglio nel processo wsgi (sempre aggiornato).
    2. Fallback: market_data.pkl per esecuzione standalone.
    Ritorna (prices_df, returns_df, saved_at) o (None, None, None).
    """
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
        if os.path.exists(_PORT_PKL):
            with open(_PORT_PKL, 'rb') as f:
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
    return None, None, None


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

def _get_returns(data_json):
    if not data_json:
        return None
    key = hash(data_json[:200])
    if key not in _DF_CACHE:
        df = pd.read_json(io.StringIO(data_json), orient='split')
        df.index = pd.to_datetime(df.index)
        _DF_CACHE[key] = df
    return _DF_CACHE[key].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Stili
# ─────────────────────────────────────────────────────────────────────────────
# Colori frontiere
_FC = {'F1': '#0066cc', 'F2': '#2ca02c', 'F3': '#e6550d'}
_CML_C = {'F1': '#6633cc', 'F2': '#007700', 'F3': '#cc4400'}
_PALETTE = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
    '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5',
]


def _short_history(returns_df):
    """Restituisce un dict {asset: 'YYYY-MM-DD'} per gli asset che iniziano
    dopo la prima data disponibile nel dataset (storia incompleta)."""
    first_dates = {col: returns_df[col].first_valid_index() for col in returns_df.columns}
    dataset_start = min(d for d in first_dates.values() if d is not None)
    return {asset: d.strftime('%d/%m/%Y')
            for asset, d in first_dates.items()
            if d is not None and d > dataset_start}

def _asset_name_div(asset, short_map):
    color   = '#cc2200' if asset in short_map else '#1a3a5c'
    tooltip = f'{asset} — dati dal {short_map[asset]}' if asset in short_map else asset
    return html.Div(
        html.Span(asset, style={'overflow':'hidden','whiteSpace':'nowrap',
                                'textOverflow':'ellipsis','maxWidth':'100%',
                                'fontSize':FONT['sm'],'color':color,'fontWeight':'600'}),
        **{'data-tooltip': tooltip, 'data-tooltip-color': color},
        style={'width':'25%','height':'24px','display':'flex','alignItems':'center',
               'paddingLeft':'4px','overflow':'hidden','position':'relative','cursor':'default'}
    )

def _w_cell(w, color):
    if w is None or w < 0.05:
        return html.Span('—', style={'fontSize':FONT['sm'],'color':'#bbb'})
    return html.Span(f'{w:.1f}%', style={'fontSize':FONT['sm'],'fontWeight':'700','color': color})

# ─────────────────────────────────────────────────────────────────────────────
# Helper: grafico performance cumulativa
# ─────────────────────────────────────────────────────────────────────────────
def _build_perf_chart(prices_data, chart_assets, frontier_weights, show_frontiers, date_start, date_end):
    empty = go.Figure().update_layout(
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        annotations=[dict(text='Seleziona asset (📊) o calcola le frontiere',
                          xref='paper', yref='paper', x=0.5, y=0.5,
                          showarrow=False, font=dict(size=13, color='#6b7a99'))])
    if not prices_data:
        return empty
    try:
        prices_df = pd.read_json(io.StringIO(prices_data), orient='split')
        prices_df.index = pd.to_datetime(prices_df.index)
        if date_start: prices_df = prices_df.loc[date_start:]
        if date_end:   prices_df = prices_df.loc[:date_end]
        fig2 = go.Figure()
        for i, asset in enumerate(chart_assets or []):
            if asset not in prices_df.columns:
                continue
            s = prices_df[asset].dropna()
            if len(s) <= 1:
                continue
            cum   = (s / s.iloc[0] - 1) * 100
            color = _PALETTE[i % len(_PALETTE)]
            fig2.add_trace(go.Scatter(
                x=cum.index, y=cum.values, mode='lines', name=asset,
                line=dict(width=1.5, color=color), opacity=0.85,
                hoverlabel=dict(bgcolor='white', bordercolor=color,
                                font=dict(color='black', size=11)),
                hovertemplate=f'<b>{asset}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.1f}}%<extra></extra>',
            ))
        ret_df = prices_df.pct_change()
        for fname, fcolor in _FC.items():
            if not (show_frontiers or {}).get(fname, False):
                continue
            fw = frontier_weights.get(fname, {})
            if not fw:
                continue
            port_cols = [c for c in fw if fw[c] > 0 and c in prices_df.columns]
            if not port_cols:
                continue
            w_raw = np.array([fw[c] for c in port_cols], dtype=float)
            w_raw /= w_raw.sum()
            common_start = max(prices_df[c].first_valid_index() for c in port_cols)
            sub_ret = ret_df.loc[common_start:, port_cols].dropna(how='any')
            if len(sub_ret) < 2:
                continue
            port_ret = sub_ret.values @ w_raw
            cum_p = (np.cumprod(1 + port_ret) - 1) * 100
            fig2.add_trace(go.Scatter(
                x=sub_ret.index, y=cum_p, mode='lines',
                name=f'Portafoglio {fname}',
                line=dict(width=3, color=fcolor),
                hoverlabel=dict(bgcolor='white', bordercolor=fcolor,
                                font=dict(color='black', size=11)),
                hovertemplate=f'<b>Portafoglio {fname}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.1f}}%<extra></extra>',
            ))
        if not fig2.data:
            return empty
        fig2.update_layout(
            title=dict(text='Performance Cumulativa (%)', font=dict(size=13, color='#1a3a6b'), x=0.02),
            xaxis=dict(gridcolor='#e8eef8', zeroline=False,
                       tickformat='%Y', dtick='M12',
                       tickangle=0, tickfont=dict(size=10)),
            yaxis=dict(title='Rendimento cumulativo (%)', gridcolor='#e8eef8',
                       zeroline=True, zerolinecolor='#aaa'),
            paper_bgcolor='white', plot_bgcolor='#f8faff',
            font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.01,
                        font=dict(size=10)),
            margin=dict(l=50, r=150, t=40, b=50),
            hovermode='closest',
        )
        return fig2
    except Exception:
        return empty

def _build_drawdown_chart(prices_data, chart_assets, frontier_weights, show_frontiers, date_start, date_end):
    empty = go.Figure().update_layout(
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        annotations=[dict(text='Seleziona asset (📊) o calcola le frontiere',
                          xref='paper', yref='paper', x=0.5, y=0.5,
                          showarrow=False, font=dict(size=13, color='#6b7a99'))])
    if not prices_data:
        return empty
    try:
        prices_df = pd.read_json(io.StringIO(prices_data), orient='split')
        prices_df.index = pd.to_datetime(prices_df.index)
        if date_start: prices_df = prices_df.loc[date_start:]
        if date_end:   prices_df = prices_df.loc[:date_end]
        fig_dd = go.Figure()

        def _drawdown(s):
            cum = (1 + s.pct_change().fillna(0)).cumprod()
            roll_max = cum.cummax()
            return (cum / roll_max - 1) * 100

        for i, asset in enumerate(chart_assets or []):
            if asset not in prices_df.columns:
                continue
            s = prices_df[asset].dropna()
            if len(s) <= 1:
                continue
            dd = _drawdown(s)
            color = _PALETTE[i % len(_PALETTE)]
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values, mode='lines', name=asset,
                line=dict(width=1.5, color=color), opacity=0.85,
                fill='tozeroy', fillcolor=color.replace(')', ',0.08)').replace('rgb', 'rgba') if color.startswith('rgb') else color,
                hoverlabel=dict(bgcolor='white', bordercolor=color,
                                font=dict(color='black', size=11)),
                hovertemplate=f'<b>{asset}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.2f}}%<extra></extra>',
            ))

        ret_df = prices_df.pct_change()
        for fname, fcolor in _FC.items():
            if not (show_frontiers or {}).get(fname, False):
                continue
            fw = frontier_weights.get(fname, {})
            if not fw:
                continue
            port_cols = [c for c in fw if fw[c] > 0 and c in prices_df.columns]
            if not port_cols:
                continue
            w_raw = np.array([fw[c] for c in port_cols], dtype=float)
            w_raw /= w_raw.sum()
            common_start = max(prices_df[c].first_valid_index() for c in port_cols)
            sub_ret = ret_df.loc[common_start:, port_cols].dropna(how='any')
            if len(sub_ret) < 2:
                continue
            port_ret = pd.Series(sub_ret.values @ w_raw, index=sub_ret.index)
            cum_p = (1 + port_ret).cumprod()
            roll_max = cum_p.cummax()
            dd_p = (cum_p / roll_max - 1) * 100
            fig_dd.add_trace(go.Scatter(
                x=dd_p.index, y=dd_p.values, mode='lines',
                name=f'Portafoglio {fname}',
                line=dict(width=3, color=fcolor),
                hoverlabel=dict(bgcolor='white', bordercolor=fcolor,
                                font=dict(color='black', size=11)),
                hovertemplate=f'<b>Portafoglio {fname}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.2f}}%<extra></extra>',
            ))

        if not fig_dd.data:
            return empty
        fig_dd.update_layout(
            title=dict(text='Drawdown (%)', font=dict(size=13, color='#1a3a6b'), x=0.02),
            xaxis=dict(gridcolor='#e8eef8', zeroline=False,
                       tickformat='%Y', dtick='M12', tickangle=0, tickfont=dict(size=10)),
            yaxis=dict(title='Drawdown (%)', gridcolor='#e8eef8',
                       zeroline=True, zerolinecolor='#aaa'),
            paper_bgcolor='white', plot_bgcolor='#f8faff',
            font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.01,
                        font=dict(size=10)),
            margin=dict(l=50, r=150, t=40, b=50),
            hovermode='closest',
        )
        return fig_dd
    except Exception:
        return empty

# ─────────────────────────────────────────────────────────────────────────────
# Navbar
# ─────────────────────────────────────────────────────────────────────────────
def _navbar():
    from navbar import make_navbar
    return make_navbar()

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    _navbar(),

    html.Div([
        # ── Barra comandi ────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span(id='fe-last-updated',
                          style={'fontSize':FONT['sm'],'color':'#6b7a99','fontStyle':'italic',
                                 'alignSelf':'center','whiteSpace':'nowrap'}),
                html.Div([
                    html.Div('da:',style={'fontSize':FONT['sm'],'marginRight':'4px'}),
                    dcc.DatePickerSingle(id='fe-date-start',
                        date=(pd.Timestamp.today()-pd.DateOffset(years=10)).strftime('%Y-%m-%d'),
                        display_format='DD/MM/YYYY',
                        style={'fontSize':FONT['sm']}),
                    html.Div('a:',style={'fontSize':FONT['sm'],'margin':'0 4px'}),
                    dcc.DatePickerSingle(id='fe-date-end',
                        date=pd.Timestamp.today().strftime('%Y-%m-%d'),
                        display_format='DD/MM/YYYY',
                        style={'fontSize':FONT['sm']}),
                ], style={'display':'flex','alignItems':'center'}),
                html.Button('📋 Template', id='fe-btn-template', n_clicks=0,
                            title='Scarica il template Excel per i tuoi titoli',
                            style={'fontSize':'11px','padding':'5px 12px','borderRadius':'4px',
                                   'cursor':'pointer','background':'#e8f5e9',
                                   'border':'1px solid #a5d6a7','color':'#1b5e20'}),
                dcc.Upload(id='fe-upload-data',
                           children=html.Div('Trascina il tuo file'),
                           style={'width':'150px','height':'32px','lineHeight':'32px',
                                  'borderWidth':'1px','borderStyle':'dashed','borderRadius':'5px',
                                  'textAlign':'center','fontSize':'11px',
                                  'color':'#555','cursor':'pointer'},
                           multiple=False),
                html.Button('📥 Scarica', id='fe-btn-scarica', n_clicks=0,
                            title='Scarica i prezzi correnti come file Excel',
                            style={'fontSize':'11px','padding':'5px 12px','borderRadius':'4px',
                                   'cursor':'pointer','background':'#f0f4fb',
                                   'border':'1px solid #c0d0e8','color':'#1a3a5c'}),
                html.Div(id='fe-upload-status',
                         style={'fontSize':FONT['sm'],'color':'#555','alignSelf':'center'}),
            ], style={'display':'flex','alignItems':'center','gap':'8px','flexWrap':'wrap'}),
        ], style={'padding':'8px 10px','background':'#f0f4fb',
                  'borderBottom':'1px solid #ccd9ee','display':'flex',
                  'alignItems':'center','flexWrap':'wrap','gap':'8px'}),

        # ── Intestazione colonne ─────────────────────────────────────────────
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
                    html.Span('P3', style={'fontWeight':'bold','fontSize':FONT['xs'],'color':'#e6550d'}),
                    html.Button('☑', id='fe-selall-p3', n_clicks=0, title='Seleziona / Deseleziona tutto P3',
                                style={'fontSize':FONT['icon'],'border':'none','background':'none','cursor':'pointer',
                                       'color':'#e6550d','padding':'0 2px','lineHeight':'1'}),
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
                                'fontSize':FONT['xs'],'color':'#e6550d','position':'relative','cursor':'default'}),
            ], style={'width':'35%','display':'flex','alignItems':'center','minHeight':'28px'}),
            html.Div([
                html.Div([
                    html.Label('N. Port:', style={'fontSize':FONT['sm'],'marginRight':'4px'}),
                    dcc.Input(id='fe-n-portfolios', type='number', value=15, min=5, max=100,
                              style={'width':'50px','fontSize':FONT['sm']}),
                ], style={'display':'flex','alignItems':'center','marginRight':'8px'}),
                html.Div([
                    html.Label('Min %:', style={'fontSize':FONT['sm'],'marginRight':'4px'}),
                    dcc.Input(id='fe-min-weight', type='number', value=0, min=0, max=100,
                              style={'width':'45px','fontSize':FONT['sm']}),
                ], style={'display':'flex','alignItems':'center','marginRight':'8px'}),
                html.Div([
                    html.Label('Max %:', style={'fontSize':FONT['sm'],'marginRight':'4px'}),
                    dcc.Input(id='fe-max-weight', type='number', value=100, min=0, max=100,
                              style={'width':'45px','fontSize':FONT['sm']}),
                ], style={'display':'flex','alignItems':'center','marginRight':'8px'}),
                html.Div([
                    html.Label('Risk Free %:', style={'fontSize':FONT['sm'],'marginRight':'4px'}),
                    dcc.Input(id='fe-rf', type='number', value=2.0, min=0, max=20, step=0.1,
                              style={'width':'50px','fontSize':FONT['sm']}),
                ], style={'display':'flex','alignItems':'center','marginRight':'8px'}),
                dcc.RadioItems(id='fe-risk-measure',
                    options=[
                        {'label': ' Standard',    'value': 'standard'},
                        {'label': ' CVaR 90%',    'value': 'cvar90'},
                        {'label': ' CVaR 95%',    'value': 'cvar95'},
                        {'label': ' ARIMA+GARCH', 'value': 'arima_garch'},
                        {'label': ' Vol (fin.)',   'value': 'vol'},
                    ],
                    value='standard', inline=True,
                    inputStyle={'marginRight':'3px','cursor':'pointer'},
                    labelStyle={'marginRight':'8px','fontSize':FONT['sm'],'cursor':'pointer'}),
                html.Div([
                    html.Label('Finestra gg:', style={'fontSize':FONT['sm'],'marginRight':'4px'}),
                    dcc.Input(id='fe-arima-window', type='number', value=250, min=20, max=1260,
                              style={'width':'55px','fontSize':FONT['sm']}),
                ], id='fe-arima-window-div',
                   style={'display':'none','alignItems':'center','marginRight':'8px'}),
                html.Div(id='fe-arima-cache-status',
                         style={'display':'none','fontSize':'9px','padding':'2px 6px',
                                'borderRadius':'3px','fontWeight':'600'}),
                html.Button('Calcola Frontiera', id='fe-calc-btn', n_clicks=0,
                            style={'background':'#0066cc','color':'white','border':'none',
                                   'padding':'6px 14px','borderRadius':'4px','cursor':'pointer',
                                   'fontWeight':'bold','fontSize':'11px',
                                   'boxShadow':'0 2px 6px rgba(0,102,204,0.35)'}),
            ], style={'width':'65%','display':'flex','alignItems':'center',
                      'flexWrap':'wrap','gap':'4px'}),
        ], style={'display':'flex','background':'#e8f0fb',
                  'borderTop':'2px solid #0066cc','borderBottom':'1px solid #aed6f1',
                  'padding':'4px 6px'}),

        # ── Griglia + Grafici ────────────────────────────────────────────────
        html.Div([
            # Sinistra: asset
            html.Div([
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
                html.Div(id='fe-arima-progress-div',
                         style={'display':'none','alignItems':'center','justifyContent':'center',
                                'gap':'10px','padding':'8px 16px','background':'#eef4ff',
                                'borderRadius':'8px','margin':'4px 0','flexShrink':'0'},
                         children=[
                    html.Div(style={'width':'16px','height':'16px','border':'3px solid #ccd9ee',
                                    'borderTop':'3px solid #1a3a6b','borderRadius':'50%',
                                    'animation':'fe-spin 0.9s linear infinite','flexShrink':'0'}),
                    html.Span(id='fe-arima-progress-text', children='ARIMA in corso...',
                              style={'fontSize':'11px','color':'#1a3a6b','fontWeight':'600'}),
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
                          style={'height':'420px','marginTop':'6px'},
                          config={'displayModeBar':True}),
                dcc.Graph(id='fe-drawdown-chart',
                          style={'height':'300px','marginTop':'6px'},
                          config={'displayModeBar':True}),
                html.Div(id='fe-stats-panel',
                         style={'padding':'4px 10px','fontSize':'11px','color':'#1a3a5c'}),
            ], style={
                'width':'65%','padding':'6px','background':'white',
                'display':'flex','flexDirection':'column',
            }),
        ], style={'display':'flex','alignItems':'flex-start'}),

    ], style={'marginTop':'64px'}),

    # ── Stores ───────────────────────────────────────────────────────────────
    dcc.Store(id='_fe-page-load',    data=1),
    dcc.Store(id='fe-stock-data',    data=None),
    dcc.Store(id='fe-prices-data',   data=None),
    dcc.Store(id='fe-loaded-flag',   data=False),
    dcc.Store(id='fe-f1-weights',      data=None),
    dcc.Store(id='fe-f2-weights',      data=None),
    dcc.Store(id='fe-f3-weights',      data=None),
    dcc.Store(id='fe-frontier-rawdata',data=None),
    dcc.Store(id='fe-selected-pt',     data=None),
    dcc.Store(id='fe-arima-reqid',     data=None),
    dcc.Interval(id='fe-arima-poll', interval=600,  n_intervals=0, disabled=True),
    dcc.Download(id='fe-dl-template'),
    dcc.Download(id='fe-dl-prices'),

], style={'minHeight':'100vh'})

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output('fe-stock-data',   'data',     allow_duplicate=True),
    Output('fe-prices-data',  'data',     allow_duplicate=True),
    Output('fe-loaded-flag',  'data',     allow_duplicate=True),
    Output('fe-last-updated', 'children', allow_duplicate=True),
    Input('_fe-page-load',    'data'),
    prevent_initial_call='initial_duplicate',
)
def on_page_load(_):
    prices, returns, saved_at = _read_shared_data()
    if prices is not None:
        n = len(prices.columns)
        label = f'Da analisi di portafoglio ({n} asset)' + (f' — {saved_at}' if saved_at else '')
        return (returns.to_json(orient='split', date_format='iso'),
                prices.to_json(orient='split', date_format='iso'),
                True, label)

    raise PreventUpdate


@app.callback(
    Output('fe-grid',        'children', allow_duplicate=True),
    Output('fe-asset-count', 'children', allow_duplicate=True),
    Input('fe-loaded-flag',  'data'),
    State('fe-stock-data',   'data'),
    prevent_initial_call=True,
)
def build_grid_on_load(loaded, stock_data):
    if not loaded or not stock_data:
        raise PreventUpdate
    returns_df = _get_returns(stock_data)
    if returns_df is None or returns_df.empty:
        raise PreventUpdate
    assets    = returns_df.dropna(how='all', axis=1).columns.tolist()
    short_map = _short_history(returns_df)
    rows = []
    for i, asset in enumerate(assets):
        row = html.Div([
            _asset_name_div(asset, short_map),
            html.Div(
                dcc.Checklist(id={'type':'fe-chart','index':asset},
                              options=[{'label':'','value':asset}], value=[],
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'7%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(
                dcc.Checklist(id={'type':'fe-p1','index':asset},
                              options=[{'label':'','value':asset}], value=[],
                              inputStyle={'accentColor':'#0066cc'},
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(
                dcc.Checklist(id={'type':'fe-p2','index':asset},
                              options=[{'label':'','value':asset}], value=[],
                              inputStyle={'accentColor':'#2ca02c'},
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(
                dcc.Checklist(id={'type':'fe-p3','index':asset},
                              options=[{'label':'','value':asset}], value=[],
                              inputStyle={'accentColor':'#e6550d'},
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(id={'type':'fe-wgt-f1','index':asset}, children=_w_cell(None,'#0066cc'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wgt-f2','index':asset}, children=_w_cell(None,'#2ca02c'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wgt-f3','index':asset}, children=_w_cell(None,'#e6550d'),
                     style={'width':'14%','display':'flex','alignItems':'center','justifyContent':'center'}),
        ], style={'display':'flex','alignItems':'center','height':'24px',
                  'borderBottom':'1px solid #f0f4fb',
                  'background':'white' if i % 2 == 0 else '#fafcff'})
        rows.append(row)
    return rows, f'{len(assets)} asset'


@app.callback(
    Output('fe-arima-window-div',    'style'),
    Output('fe-arima-cache-status',  'children'),
    Output('fe-arima-cache-status',  'style'),
    Input('fe-risk-measure', 'value'),
)
def toggle_window_div(risk):
    base = {'alignItems':'center','marginRight':'8px'}
    # Finestra configurabile solo per Vol (fin.)
    win_style = {'display':'flex', **base} if risk == 'vol' else {'display':'none', **base}

    if risk != 'arima_garch':
        return win_style, '', {'display':'none'}

    if _ARIMA_CACHE_INFO['available']:
        txt   = f'⚡ cache {_ARIMA_CACHE_INFO["ts"]}'
        style = {'display':'inline-block','fontSize':'9px','padding':'2px 6px',
                 'borderRadius':'3px','fontWeight':'600',
                 'background':'#d4edda','color':'#155724','border':'1px solid #c3e6cb'}
    else:
        txt   = '⏳ nessuna cache — calcolo in background'
        style = {'display':'inline-block','fontSize':'9px','padding':'2px 6px',
                 'borderRadius':'3px','fontWeight':'600',
                 'background':'#fff3cd','color':'#856404','border':'1px solid #ffc107'}
    return win_style, txt, style


@app.callback(
    Output('fe-stock-data',      'data',            allow_duplicate=True),
    Output('fe-prices-data',     'data',            allow_duplicate=True),
    Output('fe-loaded-flag',     'data',            allow_duplicate=True),
    Output('fe-upload-status',   'children'),
    Output('fe-last-updated',    'children'),
    Input('fe-upload-data',      'contents'),
    State('fe-upload-data',      'filename'),
    prevent_initial_call=True,
)
def upload_file(contents, filename):
    if not contents:
        raise PreventUpdate
    try:
        _, cs   = contents.split(',')
        decoded = base64.b64decode(cs)
        df      = pd.read_excel(io.BytesIO(decoded))
        cols    = df.columns.tolist()

        pd.to_datetime(df[cols[0]], errors='raise')
        df_prices = df.set_index(cols[0])
        df_prices.index = pd.to_datetime(df_prices.index)
        df_prices = df_prices.select_dtypes(include='number').ffill().dropna(how='all')
        returns_df = df_prices.pct_change(fill_method=None)
        saved_at   = datetime.now().strftime('%d/%m/%Y %H:%M')
        return (
            returns_df.to_json(orient='split', date_format='iso'),
            df_prices.to_json(orient='split', date_format='iso'),
            True,
            f'✓ {len(df_prices.columns)} asset — prezzi dal file',
            f'Caricati: {saved_at}',
        )
    except Exception as e:
        return no_update, no_update, no_update, f'Errore: {e}', ''


@app.callback(
    Output('fe-dl-template', 'data'),
    Input('fe-btn-template', 'n_clicks'),
    prevent_initial_call=True,
)
def download_template(n):
    if not n:
        raise PreventUpdate
    df = pd.DataFrame({
        'Ticker':      ['SPY', 'TLT', 'GLD', 'VEA', 'EEM'],
        'Descrizione': ['S&P 500 ETF', 'Bond USA 20yr', 'Oro', 'Europa Sviluppata', 'Mercati Emergenti'],
        'Valuta':      ['USD', 'USD', 'USD', 'USD', 'USD'],
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return dcc.send_bytes(buf.read(), 'template_frontiera.xlsx')


@app.callback(
    Output('fe-dl-prices', 'data'),
    Input('fe-btn-scarica', 'n_clicks'),
    State('fe-prices-data', 'data'),
    prevent_initial_call=True,
)
def download_prices(n, prices_data):
    if not n or not prices_data:
        raise PreventUpdate
    try:
        prices_df = pd.read_json(io.StringIO(prices_data), orient='split')
        prices_df.index = pd.to_datetime(prices_df.index)
        buf = io.BytesIO()
        prices_df.reset_index().to_excel(buf, index=False)
        buf.seek(0)
        return dcc.send_bytes(buf.read(), 'prezzi_frontiera.xlsx')
    except Exception:
        raise PreventUpdate


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
    State('fe-risk-measure',     'value'),
    State('fe-date-start',       'date'),
    State('fe-date-end',         'date'),
    State({'type':'fe-chart-port','index':ALL}, 'value'),
    State({'type':'fe-chart-port','index':ALL}, 'id'),
    State('fe-selected-pt',       'data'),
    State('fe-arima-window',      'value'),
    prevent_initial_call=True,
)
def calc_and_render(n, stock_data, prices_data,
                    p1_vals, p2_vals, p3_vals, p_ids, chart_vals,
                    n_port, wmin, wmax, rf, risk,
                    date_start, date_end, port_chart_vals, port_chart_ids, sel_pt_cur,
                    arima_window):
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

    if date_start: returns_df = returns_df.loc[date_start:]
    if date_end:   returns_df = returns_df.loc[:date_end]
    # Rimuove solo colonne/righe interamente NaN — NON il dropna globale che
    # taglierebbe il dataset al periodo del titolo più recente (es. crypto/futures).
    returns_df = returns_df.dropna(how='all', axis=1).dropna(how='all', axis=0)
    # Standard/CVaR: tutta la storia. ARIMA: 250 fissi. Vol: finestra configurabile.
    if risk == 'vol':
        win = int(arima_window or 250)
    elif risk == 'arima_garch':
        win = 250
    else:
        win = None  # standard, cvar90, cvar95 → tutta la storia
    all_assets = returns_df.columns.tolist()

    # P1: default tutti gli asset; P2/P3: solo se l'utente ha spuntato almeno 2 asset
    def _p_assets(vals, ids, default_all=False):
        if not ids:
            return all_assets if default_all else []
        sel = [pid['index'] for v, pid in zip(vals, ids) if v]
        if len(sel) == 0:
            return all_assets if default_all else []
        return sel if len(sel) >= 2 else []

    p1_sel = _p_assets(p1_vals, p_ids, default_all=False)
    p2_sel = _p_assets(p2_vals, p_ids, default_all=False)
    p3_sel = _p_assets(p3_vals, p_ids, default_all=False)

    chart_assets = [a for v in (chart_vals or []) if v for a in v]
    port_chart_checked = {
        pid['index'] for v, pid in zip(port_chart_vals or [], port_chart_ids or []) if v
    }

    if not p1_sel and not p2_sel and not p3_sel:
        return (no_update, no_update, _EMPTY_FIG, _EMPTY_FIG, _EMPTY_FIG, '',
                None, None, None, None, None, True, None, {'display':'none'})

    wmin_f = (wmin or 0) / 100
    wmax_f = (wmax or 100) / 100
    rf_f   = (rf or 2.0) / 100
    n_f    = int(n_port or 15)

    arima_label = ''
    def _mu(df_sub):
        return None

    # ── Calcola le 3 frontiere ───────────────────────────────────────────────
    frontier_res  = {}   # fname → (df_f, max_sharpe, min_vol, names)
    frontier_wgts = {}   # fname → {asset: weight%}

    _existing_sel = json.loads(sel_pt_cur) if sel_pt_cur else {}

    # ── ARIMA+GARCH: prova cache pre-calcolata, altrimenti avvia thread ────────
    if risk == 'arima_garch':
        arima_win = int(arima_window or 250)
        mu_cached, cov_cached, arima_ts = _read_arima_cache()
        print(f"[calc_and_render] ARIMA cache: {'trovata' if mu_cached is not None else 'VUOTA'}")
        if mu_cached is not None:
            # Cache disponibile → inietta nel _ARIMA_STATE e scatta on_arima_done subito
            import uuid as _uuid
            req_id = str(_uuid.uuid4())[:8]
            with _ARIMA_LOCK:
                _ARIMA_STATE.update({
                    'req_id': req_id, 'running': False, 'done': True,
                    'pct': len(all_assets), 'total': len(all_assets),
                    'error': None, 'mu': mu_cached, 'cov': cov_cached,
                })
            _prog_style = {'display':'flex','alignItems':'center','justifyContent':'center',
                           'gap':'10px','padding':'8px 16px','background':'#eef4ff',
                           'borderRadius':'8px','margin':'4px 0','flexShrink':'0'}
            _nu = no_update
            # Restituisce req_id + ':done' così on_arima_done scatta immediatamente
            return (_nu, _nu, _nu, _nu, _nu,
                    _nu, _nu, _nu, _nu, _nu,
                    True, req_id + ':done', _prog_style)
        else:
            # Nessuna cache → calcolo in background
            import uuid as _uuid
            req_id = str(_uuid.uuid4())[:8]
            with _ARIMA_LOCK:
                _ARIMA_STATE.update({
                    'req_id': req_id, 'running': True, 'done': False,
                    'pct': 0, 'total': len(all_assets), 'error': None, 'mu': None, 'cov': None,
                })
            t = threading.Thread(
                target=_run_arima_thread,
                args=(req_id, returns_df, arima_win),
                daemon=True,
            )
            t.start()
            _prog_style = {'display':'flex','alignItems':'center','justifyContent':'center',
                           'gap':'10px','padding':'8px 16px','background':'#eef4ff',
                           'borderRadius':'8px','margin':'4px 0','flexShrink':'0'}
            _nu = no_update
            return (_nu, _nu, _nu, _nu, _nu,
                    _nu, _nu, _nu, _nu, _nu,
                    False, req_id, _prog_style)

    for fname, assets_sel in [('F1', p1_sel), ('F2', p2_sel), ('F3', p3_sel)]:
        valid = [a for a in assets_sel if a in returns_df.columns]
        if len(valid) < 2:
            continue
        # dropna solo sugli asset selezionati per questa frontiera,
        # poi applica la finestra temporale
        df_sub = returns_df[valid].dropna()
        if win and win < len(df_sub):
            df_sub = df_sub.tail(win)
        if len(df_sub) < 10:
            continue
        try:
            df_f, ms, mv, names = calc_frontier(
                df_sub, n=n_f, wmin=wmin_f, wmax=wmax_f,
                rf=rf_f, risk=risk, mu_override=_mu(df_sub))
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
        except Exception:
            pass

    # ── Grafico frontiera ────────────────────────────────────────────────────
    fig = go.Figure()
    _cvar_pct = 10 if risk == 'cvar90' else (5 if risk == 'cvar95' else None)

    # Singoli asset: media e rischio calcolati sulla serie propria di ogni asset
    # (finestra applicata individualmente, no dropna globale su tutti gli asset)
    for ai, asset in enumerate(all_assets):
        s = returns_df[asset].dropna()
        if win and win < len(s):
            s = s.tail(win)
        if len(s) < 10:
            continue
        ret_a = float(s.mean() * 252)
        vol_a = float(s.std() * np.sqrt(252))
        if _cvar_pct is not None:
            risk_a = _port_cvar(np.array([1.0]), s.to_frame(), _cvar_pct)
            risk_lbl = f'CVaR {100-_cvar_pct}%: {risk_a*100:.2f}%'
        else:
            risk_a = vol_a
            risk_lbl = f'Vol: {vol_a*100:.2f}%'
        color = _PALETTE[ai % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=[risk_a * 100], y=[ret_a * 100],
            mode='markers+text', name=asset,
            marker=dict(size=6, symbol='circle', opacity=0.75, color=color),
            text=[asset[:9]], textposition='top center',
            textfont=dict(size=7, color=color), showlegend=False,
            hoverlabel=dict(bgcolor='white', bordercolor=color,
                            font=dict(color='black', size=11)),
            hovertemplate=f'<b>{asset}</b><br>{risk_lbl}<br>Rendimento: {ret_a*100:.2f}%<extra></extra>',
        ))

    for fname, (df_f, ms, mv, names) in frontier_res.items():
        fcolor  = _FC[fname]
        cml_col = _CML_C[fname]
        if not df_f.empty:
            cd = [[fname, i] for i in range(len(df_f))]
            fig.add_trace(go.Scatter(
                x=df_f['Volatility'] * 100, y=df_f['Return'] * 100,
                mode='lines+markers', name=f'Frontiera {fname}',
                line=dict(color=fcolor, width=2),
                marker=dict(size=6),
                customdata=cd,
                hovertemplate=f'<b>{fname}</b><br>Rischio: %{{x:.2f}}%<br>Rendimento: %{{y:.2f}}%<extra></extra>',
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
                marker=dict(symbol='circle', size=12, color='red',
                            line=dict(color='#880000', width=1.5)),
                customdata=[[fname, idx_ms, 'sharpe']],
                hovertemplate=(f'<b>Max Sharpe {fname}: {ms["Sharpe"]:.2f}</b>'
                               f'<br>Rischio: {ms["Volatility"]*100:.2f}%'
                               f'<br>Rendimento: {ms["Return"]*100:.2f}%<extra></extra>'),
            ))

    risk_label = {
        'standard':    'Volatilità Ann. (%)',
        'vol':         'Volatilità Ann. (%) [fin.]',
        'cvar90':      'CVaR 90% Ann. (%)',
        'cvar95':      'CVaR 95% Ann. (%)',
        'arima_garch': 'Vol GARCH Ann. (%)',
    }[risk]
    fig.update_layout(
        title=dict(text=f'Frontiera Efficiente{arima_label}',
                   font=dict(size=14, color='#1a3a6b'), x=0.02),
        xaxis=dict(title=risk_label, gridcolor='#e8eef8', zeroline=False),
        yaxis=dict(title='Rendimento Atteso Ann. (%)', gridcolor='#e8eef8', zeroline=False),
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=50, r=30, t=80, b=40), hovermode='closest',
    )

    # ── Performance e Drawdown chart ─────────────────────────────────────────
    show_frontiers = {fname: (fname in port_chart_checked) for fname in frontier_wgts}
    fig2   = _build_perf_chart(prices_data, chart_assets, frontier_wgts, show_frontiers, date_start, date_end)
    fig_dd = _build_drawdown_chart(prices_data, chart_assets, frontier_wgts, show_frontiers, date_start, date_end)

    # ── Ricostruisci griglia con F1/F2/F3 weights ────────────────────────────
    p1_set    = set(p1_sel)
    p2_set    = set(p2_sel)
    p3_set    = set(p3_sel)
    chart_set = set(chart_assets)


    short_map = _short_history(returns_df)
    rows = []
    for i, asset in enumerate(all_assets):
        f1w = frontier_wgts.get('F1', {}).get(asset)
        f2w = frontier_wgts.get('F2', {}).get(asset)
        f3w = frontier_wgts.get('F3', {}).get(asset)
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
                              inputStyle={'accentColor':'#e6550d'},
                              style={'display':'flex','justifyContent':'center'}),
                style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}
            ),
            html.Div(id={'type':'fe-wgt-f1','index':asset}, children=_w_cell(f1w, '#0066cc'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wgt-f2','index':asset}, children=_w_cell(f2w, '#2ca02c'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wgt-f3','index':asset}, children=_w_cell(f3w, '#e6550d'),
                     style={'width':'14%','display':'flex','alignItems':'center','justifyContent':'center'}),
        ], style={'display':'flex','alignItems':'center','height':'24px',
                  'borderBottom':'1px solid #f0f4fb',
                  'background':'white' if i % 2 == 0 else '#fafcff'})
        rows.append(row)

    # ── Righe portafoglio (F1/F2/F3) ─────────────────────────────────────────
    _PORT_BG = {'F1': '#f0f6ff', 'F2': '#f0fff4', 'F3': '#fff5f0'}

    # Costruisci sel_pt (label semplice per riga riepilogo)
    sel_pt = {}
    for fname, (df_f, ms, _mv, _names) in frontier_res.items():
        ex = _existing_sel.get(fname, {})
        pt_idx = ex.get('pt_idx', -1)
        ex_label = ex.get('label', '')
        if pt_idx >= 0 and pt_idx < len(df_f):
            label = ex_label if ex_label == 'Sharpe' else f'P({pt_idx + 1})'
        else:
            label = 'Sharpe'
        sel_pt[fname] = {'label': label, 'pt_idx': pt_idx}

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
            rows.append(html.Div([
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
                        value=[fname] if (fname in port_chart_checked or
                                          (fname == 'F1' and not port_chart_ids)) else [],
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
                html.Div(_w_cell(100.0 if fname == 'F3' else None, '#e6550d'),
                         style={'width':'14%','display':'flex','alignItems':'center','justifyContent':'center'}),
            ], style={'display':'flex','alignItems':'center','height':'28px',
                      'borderBottom':'1px solid #e0e8f8',
                      'background': _PORT_BG.get(fname, 'white')}))

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = []
    for fname, (df_f, ms, mv, names) in frontier_res.items():
        if ms is not None:
            stats.append(html.Span([
                html.B(f'Max Sharpe {fname}: ', style={'color': _FC[fname]}),
                f"{ms['Return']*100:.1f}% rend · {ms['Volatility']*100:.1f}% rischio · "
                f"Sharpe {ms['Sharpe']:.2f}   ",
            ], style={'marginRight':'8px', 'fontSize':'11px'}))
    if arima_label:
        stats.append(html.Span(
            f'Metodo: ARIMA+GARCH finestra {arima_window or 250}gg',
            style={'fontSize':FONT['sm'],'color':'#6b7a99','fontStyle':'italic'}))

    f1j = json.dumps(frontier_wgts.get('F1', {}))
    f2j = json.dumps(frontier_wgts.get('F2', {}))
    f3j = json.dumps(frontier_wgts.get('F3', {}))
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
    Input('fe-frontier-chart',   'clickData'),
    State('fe-selected-pt',      'data'),
    State('fe-frontier-rawdata', 'data'),
    prevent_initial_call=True,
)
def on_frontier_click(click_data, sel_pt_json, rawdata_json):
    if not click_data:
        raise PreventUpdate
    pt = click_data['points'][0]
    cd = pt.get('customdata')
    if not cd or len(cd) < 2:
        raise PreventUpdate
    fname, pt_idx = cd[0], int(cd[1])
    is_sharpe = len(cd) >= 3 and cd[2] == 'sharpe'
    sel_pt = json.loads(sel_pt_json) if sel_pt_json else {}
    sel_pt[fname] = {'label': 'Sharpe' if is_sharpe else f'P({pt_idx + 1})', 'pt_idx': pt_idx}

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

    return json.dumps(sel_pt), w_out['F1'], w_out['F2'], w_out['F3']


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
        color = 'red' if label == 'Sharpe' else (_FC.get(fname, '#888') if label else '#bbb')
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
_wgt_f3_cb = _make_wgt_cell_cb('fe-f3-weights', 'fe-wgt-f3', '#e6550d')


# ── ARIMA: polling progresso ──────────────────────────────────────────────────
@app.callback(
    Output('fe-arima-reqid',         'data',     allow_duplicate=True),
    Output('fe-arima-progress-text', 'children', allow_duplicate=True),
    Output('fe-arima-poll',          'disabled', allow_duplicate=True),
    Input('fe-arima-poll',           'n_intervals'),
    State('fe-arima-reqid',          'data'),
    prevent_initial_call=True,
)
def arima_poll(n_int, cur_req_id):
    if not cur_req_id:
        raise PreventUpdate
    raw_id = cur_req_id.replace(':done', '').replace(':error', '')
    with _ARIMA_LOCK:
        s = dict(_ARIMA_STATE)
    if s.get('req_id') != raw_id:
        raise PreventUpdate
    if s.get('error'):
        return cur_req_id + ':error', f"❌ Errore: {s['error'][:50]}", True
    if s.get('done'):
        return cur_req_id + ':done', '✓ Completato', True
    total = s['total'] or 1
    pct   = int(s['pct'] / total * 100)
    return (no_update,
            f"ARIMA: {s['pct']}/{total} titoli ({pct}%)",
            False)


# ── ARIMA: quando completato → calcola frontiere e aggiorna ──────────────────
@app.callback(
    Output('fe-grid',               'children',   allow_duplicate=True),
    Output('fe-asset-count',        'children',   allow_duplicate=True),
    Output('fe-frontier-chart',     'figure',     allow_duplicate=True),
    Output('fe-perf-chart',         'figure',     allow_duplicate=True),
    Output('fe-drawdown-chart',     'figure',     allow_duplicate=True),
    Output('fe-stats-panel',        'children',   allow_duplicate=True),
    Output('fe-f1-weights',         'data',       allow_duplicate=True),
    Output('fe-f2-weights',         'data',       allow_duplicate=True),
    Output('fe-f3-weights',         'data',       allow_duplicate=True),
    Output('fe-frontier-rawdata',   'data',       allow_duplicate=True),
    Output('fe-selected-pt',        'data',       allow_duplicate=True),
    Output('fe-arima-progress-div', 'style',      allow_duplicate=True),
    Input('fe-arima-reqid',         'data'),
    State('fe-stock-data',          'data'),
    State('fe-prices-data',         'data'),
    State({'type':'fe-p1','index':ALL},       'value'),
    State({'type':'fe-p2','index':ALL},       'value'),
    State({'type':'fe-p3','index':ALL},       'value'),
    State({'type':'fe-p1','index':ALL},       'id'),
    State({'type':'fe-chart','index':ALL},    'value'),
    State('fe-n-portfolios',        'value'),
    State('fe-min-weight',          'value'),
    State('fe-max-weight',          'value'),
    State('fe-rf',                  'value'),
    State('fe-risk-measure',        'value'),
    State('fe-arima-window',        'value'),
    State('fe-date-start',          'date'),
    State('fe-date-end',            'date'),
    State({'type':'fe-chart-port','index':ALL}, 'value'),
    State({'type':'fe-chart-port','index':ALL}, 'id'),
    State('fe-selected-pt',         'data'),
    prevent_initial_call=True,
)
def on_arima_done(req_id, stock_data, prices_data,
                  p1_vals, p2_vals, p3_vals, p_ids, chart_vals,
                  n_port, wmin, wmax, rf, risk, arima_window,
                  date_start, date_end, port_chart_vals, port_chart_ids, sel_pt_cur):
    if not req_id:
        raise PreventUpdate
    raw_id = req_id.replace(':done', '').replace(':error', '')
    with _ARIMA_LOCK:
        s = dict(_ARIMA_STATE)
    if s.get('req_id') != raw_id or not s.get('done') or s.get('error'):
        raise PreventUpdate
    mu_series = s.get('mu')
    cov_df = s.get('cov')
    if mu_series is None:
        raise PreventUpdate
    if not stock_data:
        raise PreventUpdate

    returns_df = _get_returns(stock_data)
    if returns_df is None or returns_df.empty:
        raise PreventUpdate

    if date_start: returns_df = returns_df.loc[date_start:]
    if date_end:   returns_df = returns_df.loc[:date_end]
    returns_df = returns_df.dropna(how='all', axis=1).dropna(how='all', axis=0)
    win_ag = int(arima_window or 250)
    all_assets = returns_df.columns.tolist()

    def _p_assets(vals, ids, default_all=False):
        if not ids:
            return all_assets if default_all else []
        sel = [pid['index'] for v, pid in zip(vals, ids) if v]
        if len(sel) == 0:
            return all_assets if default_all else []
        return sel if len(sel) >= 2 else []

    p1_sel = _p_assets(p1_vals, p_ids, default_all=False)
    p2_sel = _p_assets(p2_vals, p_ids, default_all=False)
    p3_sel = _p_assets(p3_vals, p_ids, default_all=False)

    chart_assets = [a for v in (chart_vals or []) if v for a in v]
    port_chart_checked = {
        pid['index'] for v, pid in zip(port_chart_vals or [], port_chart_ids or []) if v
    }

    wmin_f = (wmin or 0) / 100
    wmax_f = (wmax or 100) / 100
    rf_f   = (rf or 2.0) / 100
    n_f    = int(n_port or 15)
    arima_label = f' [ARIMA+GARCH finestra {arima_window or 250}gg]'
    _existing_sel = json.loads(sel_pt_cur) if sel_pt_cur else {}

    print(f"[on_arima_done] p1={len(p1_sel)} p2={len(p2_sel)} p3={len(p3_sel)} mu={len(mu_series) if mu_series is not None else 'None'}")
    frontier_res  = {}
    frontier_wgts = {}
    for fname, assets_sel in [('F1', p1_sel), ('F2', p2_sel), ('F3', p3_sel)]:
        valid = [a for a in assets_sel if a in returns_df.columns]
        if len(valid) < 2:
            print(f"[on_arima_done] {fname}: valid={len(valid)} — skip")
            continue
        df_sub = returns_df[valid].dropna()
        if win_ag < len(df_sub):
            df_sub = df_sub.tail(win_ag)
        if len(df_sub) < 10:
            continue
        mu_sub  = mu_series.reindex(valid).fillna(mu_series.mean()) if mu_series is not None else None
        cov_sub = cov_df.reindex(index=valid, columns=valid) if cov_df is not None else None

        # Filtra asset con diagonale della cov nulla/NaN (es. Russell2000 senza dati)
        if cov_sub is not None:
            cov_arr_chk = np.nan_to_num(cov_sub.values, nan=0.0)
            diag_chk    = np.diag(cov_arr_chk)
            bad_assets  = [valid[i] for i, d in enumerate(diag_chk) if d <= 1e-10]
            if bad_assets:
                print(f"[on_arima_done] {fname}: escludo asset cov-zero: {bad_assets}")
                valid   = [a for a in valid if a not in bad_assets]
                if len(valid) < 2:
                    continue
                df_sub  = returns_df[valid].dropna()
                if win_ag < len(df_sub):
                    df_sub = df_sub.tail(win_ag)
                mu_sub  = mu_series.reindex(valid).fillna(mu_series.mean())
                cov_sub = cov_df.reindex(index=valid, columns=valid)

        # Winsorizza mu estremi (±3σ cross-sectionale) per evitare frontiere verticali
        if mu_sub is not None and len(mu_sub) > 1:
            _m_mean = mu_sub.mean()
            _m_std  = mu_sub.std()
            if _m_std > 0:
                _lo, _hi = _m_mean - 3 * _m_std, _m_mean + 3 * _m_std
                _clipped = mu_sub.clip(_lo, _hi)
                _outliers = mu_sub[mu_sub != _clipped]
                if not _outliers.empty:
                    print(f"[on_arima_done] {fname}: winsorize mu outliers: {_outliers.to_dict()}")
                mu_sub = _clipped

        print(f"[on_arima_done] {fname}: valid={len(valid)} df_sub={df_sub.shape} mu_sub={mu_sub is not None} cov_sub={cov_sub is not None}")
        try:
            df_f, ms, mv, names = calc_frontier(
                df_sub, n=n_f, wmin=wmin_f, wmax=wmax_f,
                rf=rf_f, risk=risk, mu_override=mu_sub, cov_override=cov_sub)
            frontier_res[fname] = (df_f, ms, mv, names)
            ex = _existing_sel.get(fname, {})
            pt_idx = ex.get('pt_idx', -1)
            if pt_idx >= 0 and pt_idx < len(df_f):
                row  = df_f.iloc[pt_idx]
                wvec = row['Weights']
            elif ms is not None:
                wvec   = ms['Weights']
                pt_idx = -1
            else:
                continue
            frontier_wgts[fname] = {names[i]: round(wvec[i] * 100, 2) for i in range(len(names))}
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f"[on_arima_done] ERRORE {fname}: {_e}")

    # Re-use the same chart/grid/stats building as calc_and_render
    # (frontier_res, frontier_wgts, all_assets, etc. are all set)
    # Build frontier chart
    _EMPTY_FIG = go.Figure().update_layout(paper_bgcolor='white', plot_bgcolor='#f8faff')
    fig = go.Figure()
    _cvar_pct_done = 10 if risk == 'cvar90' else (5 if risk == 'cvar95' else None)
    for ai, asset in enumerate(all_assets):
        s = returns_df[asset].dropna()
        if win_ag < len(s):
            s = s.tail(win_ag)
        if len(s) < 10:
            continue
        # Per ARIMA+GARCH usa mu/vol stimati, altrimenti usa media/std storici
        if mu_series is not None and asset in mu_series.index:
            ret_a = float(mu_series[asset] * 252)
        else:
            ret_a = float(s.mean() * 252)
        if cov_df is not None and asset in cov_df.columns:
            vol_a = float(np.sqrt(cov_df.loc[asset, asset] * 252))
        else:
            vol_a = float(s.std() * np.sqrt(252))
        if _cvar_pct_done is not None:
            risk_a = _port_cvar(np.array([1.0]), s.to_frame(), _cvar_pct_done)
            risk_lbl = f'CVaR {100-_cvar_pct_done}%: {risk_a*100:.2f}%'
        else:
            risk_a = vol_a
            risk_lbl = f'Vol: {vol_a*100:.2f}%'
        color = _PALETTE[ai % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=[risk_a*100], y=[ret_a*100], mode='markers+text', name=asset,
            marker=dict(size=6, opacity=0.75, color=color),
            text=[asset[:9]], textposition='top center',
            textfont=dict(size=7, color=color), showlegend=False,
            hoverlabel=dict(bgcolor='white', bordercolor=color,
                            font=dict(color='black', size=11)),
            hovertemplate=f'<b>{asset}</b><br>{risk_lbl}<br>Ren:{ret_a*100:.2f}%<extra></extra>',
        ))
    for fname, (df_f, ms, mv, names) in frontier_res.items():
        fcolor  = _FC[fname]
        cml_col = _CML_C[fname]
        if not df_f.empty:
            cd = [[fname, i] for i in range(len(df_f))]
            fig.add_trace(go.Scatter(
                x=df_f['Volatility']*100, y=df_f['Return']*100,
                mode='lines+markers', name=f'Frontiera {fname}',
                line=dict(color=fcolor, width=2), marker=dict(size=6),
                customdata=cd,
                hovertemplate=f'<b>{fname}</b><br>Rischio:%{{x:.2f}}%<br>Ren:%{{y:.2f}}%<extra></extra>',
            ))
        if ms is not None and ms['Sharpe'] > 0:
            vr = np.linspace(0, ms['Volatility']*1.8, 100)
            fig.add_trace(go.Scatter(
                x=vr*100, y=(rf_f + ms['Sharpe']*vr)*100,
                mode='lines', name=f'CML {fname}',
                line=dict(color=cml_col, dash='dash', width=1.5), showlegend=False,
                hovertemplate=f'<b>CML {fname}</b><br>%{{x:.2f}}%→%{{y:.2f}}%<extra></extra>',
            ))
        if ms is not None:
            fig.add_trace(go.Scatter(
                x=[ms['Volatility']*100], y=[ms['Return']*100], mode='markers',
                name=f'Max Sharpe {fname}',
                marker=dict(symbol='circle', size=12, color='red',
                            line=dict(color='#880000', width=1.5)),
                hovertemplate=(f'<b>Max Sharpe {fname}:{ms["Sharpe"]:.2f}</b>'
                               f'<br>Rischio:{ms["Volatility"]*100:.2f}%'
                               f'<br>Ren:{ms["Return"]*100:.2f}%<extra></extra>'),
            ))
    risk_label = {
        'standard':    'Volatilità Ann. (%)',
        'vol':         'Volatilità Ann. (%) [fin.]',
        'cvar90':      'CVaR 90% Ann. (%)',
        'cvar95':      'CVaR 95% Ann. (%)',
        'arima_garch': 'Vol GARCH Ann. (%)',
    }[risk]
    fig.update_layout(
        title=dict(text=f'Frontiera Efficiente{arima_label}', font=dict(size=14,color='#1a3a6b'), x=0.02),
        xaxis=dict(title=risk_label, gridcolor='#e8eef8', zeroline=False),
        yaxis=dict(title='Rendimento Atteso Ann. (%)', gridcolor='#e8eef8', zeroline=False),
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=50, r=30, t=80, b=40), hovermode='closest',
    )

    show_frontiers = {fname: (fname in port_chart_checked) for fname in frontier_wgts}
    fig2   = _build_perf_chart(prices_data, chart_assets, frontier_wgts, show_frontiers, date_start, date_end)
    fig_dd = _build_drawdown_chart(prices_data, chart_assets, frontier_wgts, show_frontiers, date_start, date_end)

    # Build grid rows
    chart_set = set(chart_assets)
    p1_set    = set(p1_sel)
    p2_set    = set(p2_sel)
    p3_set    = set(p3_sel)
    short_map = _short_history(returns_df)
    rows = []
    for i, asset in enumerate(all_assets):
        f1w = frontier_wgts.get('F1', {}).get(asset)
        f2w = frontier_wgts.get('F2', {}).get(asset)
        f3w = frontier_wgts.get('F3', {}).get(asset)
        rows.append(html.Div([
            _asset_name_div(asset, short_map),
            html.Div(dcc.Checklist(id={'type':'fe-chart','index':asset},
                                   options=[{'label':'','value':asset}],
                                   value=[asset] if asset in chart_set else [],
                                   style={'display':'flex','justifyContent':'center'}),
                     style={'width':'7%','display':'flex','justifyContent':'center','alignItems':'center'}),
            html.Div(dcc.Checklist(id={'type':'fe-p1','index':asset},
                                   options=[{'label':'','value':asset}],
                                   value=[asset] if asset in p1_set else [],
                                   inputStyle={'accentColor':'#0066cc'},
                                   style={'display':'flex','justifyContent':'center'}),
                     style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}),
            html.Div(dcc.Checklist(id={'type':'fe-p2','index':asset},
                                   options=[{'label':'','value':asset}],
                                   value=[asset] if asset in p2_set else [],
                                   inputStyle={'accentColor':'#2ca02c'},
                                   style={'display':'flex','justifyContent':'center'}),
                     style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}),
            html.Div(dcc.Checklist(id={'type':'fe-p3','index':asset},
                                   options=[{'label':'','value':asset}],
                                   value=[asset] if asset in p3_set else [],
                                   inputStyle={'accentColor':'#e6550d'},
                                   style={'display':'flex','justifyContent':'center'}),
                     style={'width':'8%','display':'flex','justifyContent':'center','alignItems':'center'}),
            html.Div(id={'type':'fe-wgt-f1','index':asset}, children=_w_cell(f1w,'#0066cc'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wgt-f2','index':asset}, children=_w_cell(f2w,'#2ca02c'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wgt-f3','index':asset}, children=_w_cell(f3w,'#e6550d'),
                     style={'width':'14%','display':'flex','alignItems':'center','justifyContent':'center'}),
        ], style={'display':'flex','alignItems':'center','height':'24px',
                  'borderBottom':'1px solid #f0f4fb',
                  'background':'white' if i % 2 == 0 else '#fafcff'}))

    # Portfolio summary + portfolio rows
    sel_pt = {}
    _PORT_BG = {'F1':'#f0f6ff','F2':'#f0fff4','F3':'#fff5f0'}
    for fname, (df_f, ms, _mv, _names) in frontier_res.items():
        ex = _existing_sel.get(fname, {})
        pt_idx = ex.get('pt_idx', -1)
        label = f'P({pt_idx+1})' if pt_idx >= 0 and pt_idx < len(df_f) else 'Sharpe'
        sel_pt[fname] = {'label': label, 'pt_idx': pt_idx}

    def _sel_cell_local(fname):
        info = sel_pt.get(fname, {})
        return html.Span(info.get('label','—') if info else '—',
                         style={'fontSize':'9px','fontWeight':'700',
                                'color':_FC.get(fname,'#888') if info else '#bbb'})

    if frontier_wgts:
        rows.append(html.Div(style={'height':'3px','background':'#ccd9ee','margin':'2px 0'}))
        rows.append(html.Div([
            html.Div(html.Span('Portfolio selezionato',
                               style={'fontSize':'7px','color':'#8899bb','fontStyle':'italic'}),
                     style={'width':'25%','height':'36px','display':'flex','alignItems':'center','paddingLeft':'4px'}),
            html.Div(style={'width':'7%'}),
            html.Div('',style={'width':'8%'}), html.Div('',style={'width':'8%'}), html.Div('',style={'width':'8%'}),
            html.Div(id={'type':'fe-sel-info','index':'F1'}, children=_sel_cell_local('F1'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center','flexDirection':'column'}),
            html.Div(id={'type':'fe-sel-info','index':'F2'}, children=_sel_cell_local('F2'),
                     style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center','flexDirection':'column'}),
            html.Div(id={'type':'fe-sel-info','index':'F3'}, children=_sel_cell_local('F3'),
                     style={'width':'14%','display':'flex','alignItems':'center','justifyContent':'center','flexDirection':'column'}),
        ], style={'display':'flex','alignItems':'center','height':'36px','borderBottom':'1px solid #e0e8f8','background':'#f8faff'}))
        for fname, fcolor in _FC.items():
            if fname not in frontier_wgts:
                continue
            rows.append(html.Div([
                html.Div(html.Span(f'Portafoglio {fname}',
                                   style={'fontSize':'8px','color':fcolor,'fontWeight':'700',
                                          'overflow':'hidden','whiteSpace':'nowrap','textOverflow':'ellipsis'}),
                         style={'width':'25%','height':'28px','display':'flex','alignItems':'center','paddingLeft':'4px'}),
                html.Div(dcc.Checklist(id={'type':'fe-chart-port','index':fname},
                                       options=[{'label':'','value':fname}],
                                       value=[fname] if (fname in port_chart_checked or
                                                         (fname=='F1' and not port_chart_ids)) else [],
                                       inputStyle={'accentColor':fcolor},
                                       style={'display':'flex','justifyContent':'center'}),
                         style={'width':'7%','display':'flex','justifyContent':'center','alignItems':'center'}),
                html.Div('—',style={'width':'8%','textAlign':'center','fontSize':'8px','color':'#ccc'}),
                html.Div('—',style={'width':'8%','textAlign':'center','fontSize':'8px','color':'#ccc'}),
                html.Div('—',style={'width':'8%','textAlign':'center','fontSize':'8px','color':'#ccc'}),
                html.Div(_w_cell(100.0 if fname=='F1' else None,'#0066cc'),
                         style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
                html.Div(_w_cell(100.0 if fname=='F2' else None,'#2ca02c'),
                         style={'width':'15%','display':'flex','alignItems':'center','justifyContent':'center'}),
                html.Div(_w_cell(100.0 if fname=='F3' else None,'#e6550d'),
                         style={'width':'14%','display':'flex','alignItems':'center','justifyContent':'center'}),
            ], style={'display':'flex','alignItems':'center','height':'28px',
                      'borderBottom':'1px solid #e0e8f8','background':_PORT_BG.get(fname,'white')}))

    # Stats
    stats = []
    for fname, (df_f, ms, mv, names) in frontier_res.items():
        if ms is not None:
            stats.append(html.Span([
                html.B(f'Max Sharpe {fname}: ', style={'color':_FC[fname]}),
                f"{ms['Return']*100:.1f}% rend · {ms['Volatility']*100:.1f}% rischio · Sharpe {ms['Sharpe']:.2f}   ",
            ], style={'marginRight':'8px','fontSize':'11px'}))
    stats.append(html.Span(arima_label.strip(' []'),
                           style={'fontSize':FONT['sm'],'color':'#6b7a99','fontStyle':'italic'}))

    f1j = json.dumps(frontier_wgts.get('F1', {}))
    f2j = json.dumps(frontier_wgts.get('F2', {}))
    f3j = json.dumps(frontier_wgts.get('F3', {}))
    count = f'{len(all_assets)} asset · {len(frontier_res)} frontiere calcolate'

    raw_data = {}
    for fname, (df_f, _ms, _mv, names) in frontier_res.items():
        raw_data[fname] = {
            'assets': names,
            'points': [{'vol':float(r['Volatility']),'ret':float(r['Return']),'weights':list(r['Weights'])}
                       for _, r in df_f.iterrows()],
        }

    # Aggiorna la cache info in memoria — ora disponibile per toggle_window_div
    _ARIMA_CACHE_INFO['available'] = True
    _ARIMA_CACHE_INFO['ts'] = datetime.now().strftime('%d/%m/%Y %H:%M')

    return (rows, count, fig, fig2, fig_dd,
            html.Div(stats, style={'display':'flex','flexWrap':'wrap','gap':'4px'}),
            f1j, f2j, f3j, json.dumps(raw_data), json.dumps(sel_pt),
            {'display':'none'})


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

# ─────────────────────────────────────────────────────────────────────────────
server = app.server

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8052))
    app.run(debug=False, port=port, host='0.0.0.0')
