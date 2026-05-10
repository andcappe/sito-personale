"""
Frontiera Efficiente — App standalone
Ottimizzazione di portafoglio alla Markowitz con visualizzazione interattiva.
"""

import io
import json
import pickle
import sys
import threading
import concurrent.futures
import os
import base64
import requests
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update, ALL
from dash.exceptions import PreventUpdate

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
  body { margin:0; font-family:'Inter',sans-serif; background:#f5f8fe; }
  @keyframes fe-spin { to { transform: rotate(360deg); } }
  [data-tooltip]{ position:relative; }
  [data-tooltip]::after{
    content:attr(data-tooltip); position:absolute; left:100%; top:50%;
    transform:translateY(-50%); background:#1a3a5c; color:#fff;
    padding:4px 8px; border-radius:4px; font-size:11px;
    white-space:nowrap; z-index:9999; pointer-events:none;
    opacity:0; transition:opacity 0.15s; margin-left:6px;
  }
  [data-tooltip]:hover::after{ opacity:1; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────────────────────
_XLSX           = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TARBIUTH.xlsx')
DOWNLOAD_BATCH  = 5
DOWNLOAD_TIMEOUT= 40
_DL_STATE  = {'status': 'idle', 'current': 0, 'total': 0, 'errors': []}
_DL_BUFFER = {}   # buffer locale per upload personalizzati e Aggiorna
_DL_LOCK   = threading.Lock()

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
    'pct': 0, 'total': 0, 'error': None, 'mu': None,
}

def _auto_arima_mu(returns_df, window=250, horizon=20, req_id=None):
    """Auto-select best ARIMA(p,d,q) by AIC on last `window` days."""
    import warnings
    try:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA_CLS
    except ImportError:
        return returns_df.mean() * 252
    mu   = {}
    cols = returns_df.columns.tolist()
    with _ARIMA_LOCK:
        if req_id and _ARIMA_STATE.get('req_id') == req_id:
            _ARIMA_STATE['total'] = len(cols)
            _ARIMA_STATE['pct']   = 0
    for i, col in enumerate(cols):
        s = returns_df[col].dropna().tail(window)
        if len(s) < 20:
            mu[col] = float(s.mean()) * 252
        else:
            best_aic, best_fc = np.inf, float(s.mean()) * 252
            for p in range(3):
                for d in [0, 1]:
                    for q in range(3):
                        if p == 0 and q == 0:
                            continue
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                m = _ARIMA_CLS(s, order=(p, d, q)).fit()
                            if m.aic < best_aic:
                                best_aic = m.aic
                                best_fc  = float(m.forecast(steps=horizon).mean()) * 252
                        except Exception:
                            pass
            mu[col] = best_fc
        with _ARIMA_LOCK:
            if req_id and _ARIMA_STATE.get('req_id') == req_id:
                _ARIMA_STATE['pct'] = i + 1
    return pd.Series(mu)


def _run_arima_thread(req_id, returns_df, window, horizon):
    try:
        mu_series = _auto_arima_mu(returns_df, window, horizon, req_id)
        with _ARIMA_LOCK:
            if _ARIMA_STATE['req_id'] == req_id:
                _ARIMA_STATE['mu']      = mu_series
                _ARIMA_STATE['done']    = True
                _ARIMA_STATE['running'] = False
    except Exception as e:
        with _ARIMA_LOCK:
            if _ARIMA_STATE['req_id'] == req_id:
                _ARIMA_STATE['error']   = str(e)
                _ARIMA_STATE['done']    = True
                _ARIMA_STATE['running'] = False

def calc_frontier(returns_df, n=20, wmin=0.0, wmax=1.0, rf=0.02, risk='vol', mu_override=None):
    mu  = mu_override if mu_override is not None else returns_df.mean()
    cov = returns_df.cov()
    na  = len(returns_df.columns)
    bounds = tuple((wmin, wmax) for _ in range(na))
    eq     = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    w0     = [1/na] * na

    if risk == 'vol':
        def obj_vol(w): return _port_perf(w, mu, cov)[1]
        min_res = minimize(obj_vol, w0, method='SLSQP', bounds=bounds, constraints=eq)
        min_ret, _ = _port_perf(min_res.x, mu, cov)
        targets = np.linspace(min_ret, mu.max()*252*0.95, n)
        rows = []
        for t in targets:
            cs = (eq, {'type':'eq','fun': lambda x,t=t: _port_perf(x,mu,cov)[0]-t})
            r  = minimize(obj_vol, w0, method='SLSQP', bounds=bounds, constraints=cs)
            if r.success:
                ret, vol = _port_perf(r.x, mu, cov)
                rows.append({'Return':ret,'Volatility':vol,
                             'Sharpe':(ret-rf)/vol if vol>0 else 0,'Weights':r.x})
    else:
        pct = 10 if risk == 'cvar10' else 5
        def obj_var(w): return _port_cvar(w, returns_df, pct)
        min_res = minimize(obj_var, w0, method='SLSQP', bounds=bounds, constraints=eq)
        min_ret, _ = _port_perf(min_res.x, mu, cov)
        targets = np.linspace(min_ret, mu.max()*252*0.95, n)
        rows = []
        for t in targets:
            cs = (eq, {'type':'eq','fun': lambda x,t=t: _port_perf(x,mu,cov)[0]-t})
            r  = minimize(obj_var, w0, method='SLSQP', bounds=bounds, constraints=cs)
            if r.success:
                ret, _ = _port_perf(r.x, mu, cov)
                v      = obj_var(r.x)
                rows.append({'Return':ret,'Volatility':v,
                             'Sharpe':(ret-rf)/v if v>0 else 0,'Weights':r.x})

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

# ─────────────────────────────────────────────────────────────────────────────
# Download dati
# ─────────────────────────────────────────────────────────────────────────────
def _make_session():
    s = requests.Session()
    s.headers.update({'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    )})
    return s

def _yf_safe(tickers, start, timeout=DOWNLOAD_TIMEOUT):
    sess = _make_session()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        f = ex.submit(yf.download, tickers, start=start,
                      group_by='ticker', auto_adjust=True, progress=False, session=sess)
        try:
            return f.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timeout {timeout}s")

def _process_batch(bt, bd, bv, start, eurusd, eurgbp):
    prices, errors = {}, []
    raw = None
    for attempt in range(2):
        try:
            raw = _yf_safe(bt, start)
            if raw is not None and not raw.empty:
                break
        except Exception as e:
            if attempt == 0:
                import time; time.sleep(1)
            else:
                errors.append(f"{bt[0]}: {e}")
    if raw is not None and not raw.empty:
        for j, t in enumerate(bt):
            try:
                px = raw[(t,'Close')].copy() if isinstance(raw.columns, pd.MultiIndex) \
                     else raw['Close'].copy()
                px = px.ffill()
                if bv[j] == 'USD' and eurusd is not None:
                    px = px / eurusd.reindex(px.index).ffill()
                elif bv[j] == 'GBP' and eurgbp is not None:
                    px = px / eurgbp.reindex(px.index).ffill()
                prices[bd[j]] = px
            except Exception as e2:
                errors.append(f"{t}: {e2}")
    return prices, errors

def _download_worker(tickers, descrizione, valuta, start_date):
    global _DL_STATE, _DL_BUFFER
    total = len(tickers)
    with _DL_LOCK:
        _DL_STATE  = {'status':'running','current':0,'total':total,'errors':[]}
        _DL_BUFFER = {}

    fx = None
    try:
        fx = _yf_safe(['EURUSD=X','EURGBP=X'], start_date, timeout=30)
    except Exception:
        pass

    def _fx(name):
        if fx is None or fx.empty: return None
        try:
            return fx[(name,'Close')] if isinstance(fx.columns, pd.MultiIndex) else fx['Close']
        except: return None

    eurusd = _fx('EURUSD=X')
    eurgbp = _fx('EURGBP=X')

    batches = [(tickers[i:i+DOWNLOAD_BATCH], descrizione[i:i+DOWNLOAD_BATCH],
                valuta[i:i+DOWNLOAD_BATCH]) for i in range(0, total, DOWNLOAD_BATCH)]

    all_prices = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fmap = {ex.submit(_process_batch, bt, bd, bv, start_date, eurusd, eurgbp): len(bt)
                for bt, bd, bv in batches}
        for fut in concurrent.futures.as_completed(fmap):
            bs = fmap[fut]
            try:
                prices, errors = fut.result()
                all_prices.update(prices)
                with _DL_LOCK:
                    _DL_STATE['errors'].extend(errors)
            except Exception as e:
                with _DL_LOCK:
                    _DL_STATE['errors'].append(str(e))
            with _DL_LOCK:
                _DL_STATE['current'] = min(_DL_STATE['current'] + bs, total)

    if all_prices:
        prices_df = pd.DataFrame(all_prices)
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.ffill()
        returns_df = prices_df.pct_change(fill_method=None)
        with _DL_LOCK:
            _DL_BUFFER['prices']  = prices_df
            _DL_BUFFER['returns'] = returns_df
            _DL_STATE['status']   = 'done'
    else:
        with _DL_LOCK:
            _DL_STATE['status'] = 'error'

# ─────────────────────────────────────────────────────────────────────────────
# Carica nomi asset da XLSX
# ─────────────────────────────────────────────────────────────────────────────
def _load_asset_list():
    try:
        df = pd.read_excel(_XLSX)
        cols = df.columns.tolist()
        tickers    = list(df[cols[0]])
        descrizione= list(df[cols[1]])
        valuta     = list(df[cols[2]]) if len(cols) > 2 else ['EUR']*len(tickers)
        return tickers, descrizione, valuta
    except Exception:
        return [], [], []

_TICKERS, _DESCRIZIONI, _VALUTA = _load_asset_list()

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
_MODAL_HIDDEN = {'display':'none','position':'fixed','top':'0','left':'0',
                 'width':'100%','height':'100%','background':'rgba(26,58,92,0.45)',
                 'zIndex':'2000','justifyContent':'center','alignItems':'center'}
_MODAL_SHOWN  = {**_MODAL_HIDDEN, 'display':'flex'}
_FILL_LOADING = {'height':'100%','width':'0%',
                 'background':'linear-gradient(90deg,#0066cc,#3399ff)',
                 'borderRadius':'8px','transition':'width 0.5s ease'}

# Colori frontiere
_FC = {'F1': '#0066cc', 'F2': '#2ca02c', 'F3': '#e6550d'}
_CML_C = {'F1': '#6633cc', 'F2': '#007700', 'F3': '#cc4400'}


def _w_cell(w, color):
    if w is None or w < 0.05:
        return html.Span('—', style={'fontSize':'8px','color':'#bbb'})
    return html.Span(f'{w:.1f}%', style={'fontSize':'8px','fontWeight':'700','color': color})

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
        for asset in (chart_assets or []):
            if asset in prices_df.columns:
                s = prices_df[asset].dropna()
                if len(s) > 1:
                    cum = (s / s.iloc[0] - 1) * 100
                    fig2.add_trace(go.Scatter(
                        x=cum.index, y=cum.values, mode='lines', name=asset,
                        line=dict(width=1.5), opacity=0.75,
                        hovertemplate=f'<b>{asset}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.1f}}%<extra></extra>',
                    ))
        for fname, fcolor in _FC.items():
            if not (show_frontiers or {}).get(fname, False):
                continue
            fw = frontier_weights.get(fname, {})
            if not fw:
                continue
            names = prices_df.columns.tolist()
            w_arr = np.array([fw.get(n, 0) / 100 for n in names], dtype=float)
            s_w = w_arr.sum()
            if s_w <= 0:
                continue
            w_arr /= s_w
            port_prices = (prices_df * w_arr).sum(axis=1).dropna()
            if len(port_prices) < 2:
                continue
            cum_p = (port_prices / port_prices.iloc[0] - 1) * 100
            fig2.add_trace(go.Scatter(
                x=cum_p.index, y=cum_p.values, mode='lines',
                name=f'Portafoglio {fname}',
                line=dict(width=3, color=fcolor),
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
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            margin=dict(l=50, r=30, t=40, b=50),
            hovermode='x unified',
        )
        return fig2
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
                dcc.Loading(type='circle', color='#007755', children=[
                    html.Button('⟳ Aggiorna', id='fe-load-btn', n_clicks=0,
                                title='Scarica dati aggiornati da Yahoo Finance',
                                style={'background':'#007755','color':'white','border':'none',
                                       'padding':'7px 16px','borderRadius':'4px',
                                       'cursor':'pointer','fontWeight':'bold','fontSize':'12px'}),
                ]),
                html.Span(id='fe-last-updated',
                          style={'fontSize':'10px','color':'#6b7a99','fontStyle':'italic',
                                 'alignSelf':'center','whiteSpace':'nowrap'}),
                html.Div([
                    html.Div('da:',style={'fontSize':'10px','marginRight':'4px'}),
                    dcc.DatePickerSingle(id='fe-date-start',
                        date=(pd.Timestamp.today()-pd.DateOffset(years=10)).strftime('%Y-%m-%d'),
                        display_format='DD/MM/YYYY',
                        style={'fontSize':'10px'}),
                    html.Div('a:',style={'fontSize':'10px','margin':'0 4px'}),
                    dcc.DatePickerSingle(id='fe-date-end',
                        date=pd.Timestamp.today().strftime('%Y-%m-%d'),
                        display_format='DD/MM/YYYY',
                        style={'fontSize':'10px'}),
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
                         style={'fontSize':'10px','color':'#555','alignSelf':'center'}),
                html.Div([
                    html.Div(id='fe-progress-text',
                             style={'fontSize':'10px','color':'#555','marginRight':'6px'}),
                    html.Div(html.Div(id='fe-progress-fill', style=_FILL_LOADING),
                             id='fe-progress-bar',
                             style={'display':'none','width':'140px','height':'8px',
                                    'background':'#ddd','borderRadius':'8px','overflow':'hidden'}),
                ], style={'display':'flex','alignItems':'center','marginLeft':'auto'}),
            ], style={'display':'flex','alignItems':'center','gap':'8px','flexWrap':'wrap'}),
        ], style={'padding':'8px 10px','background':'#f0f4fb',
                  'borderBottom':'1px solid #ccd9ee','display':'flex',
                  'alignItems':'center','flexWrap':'wrap','gap':'8px'}),

        # ── Intestazione colonne ─────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div('Asset',
                         style={'width':'25%','fontWeight':'bold','fontSize':'8px',
                                'paddingLeft':'4px','color':'#1a3a5c'}),
                html.Div([
                    html.Span('📊', style={'fontSize':'9px'}),
                    html.Button('☑', id='fe-selall-chart', n_clicks=0, title='Seleziona / Deseleziona tutto grafico',
                                style={'fontSize':'9px','border':'none','background':'none','cursor':'pointer',
                                       'color':'#555','padding':'0 2px','lineHeight':'1'}),
                ], style={'width':'7%','textAlign':'center','display':'flex','alignItems':'center',
                          'justifyContent':'center','gap':'2px','position':'relative'}),
                html.Div([
                    html.Span('P1', style={'fontWeight':'bold','fontSize':'8px','color':'#0066cc'}),
                    html.Button('☑', id='fe-selall-p1', n_clicks=0, title='Seleziona / Deseleziona tutto P1',
                                style={'fontSize':'9px','border':'none','background':'none','cursor':'pointer',
                                       'color':'#0066cc','padding':'0 2px','lineHeight':'1'}),
                ], style={'width':'8%','textAlign':'center','display':'flex','alignItems':'center',
                          'justifyContent':'center','gap':'2px','position':'relative'}),
                html.Div([
                    html.Span('P2', style={'fontWeight':'bold','fontSize':'8px','color':'#2ca02c'}),
                    html.Button('☑', id='fe-selall-p2', n_clicks=0, title='Seleziona / Deseleziona tutto P2',
                                style={'fontSize':'9px','border':'none','background':'none','cursor':'pointer',
                                       'color':'#2ca02c','padding':'0 2px','lineHeight':'1'}),
                ], style={'width':'8%','textAlign':'center','display':'flex','alignItems':'center',
                          'justifyContent':'center','gap':'2px','position':'relative'}),
                html.Div([
                    html.Span('P3', style={'fontWeight':'bold','fontSize':'8px','color':'#e6550d'}),
                    html.Button('☑', id='fe-selall-p3', n_clicks=0, title='Seleziona / Deseleziona tutto P3',
                                style={'fontSize':'9px','border':'none','background':'none','cursor':'pointer',
                                       'color':'#e6550d','padding':'0 2px','lineHeight':'1'}),
                ], style={'width':'8%','textAlign':'center','display':'flex','alignItems':'center',
                          'justifyContent':'center','gap':'2px','position':'relative'}),
                html.Div('F1 %', **{'data-tooltip':'Peso Max-Sharpe Frontiera 1'},
                         style={'width':'15%','textAlign':'center','fontWeight':'bold',
                                'fontSize':'8px','color':'#0066cc','position':'relative','cursor':'default'}),
                html.Div('F2 %', **{'data-tooltip':'Peso Max-Sharpe Frontiera 2'},
                         style={'width':'15%','textAlign':'center','fontWeight':'bold',
                                'fontSize':'8px','color':'#2ca02c','position':'relative','cursor':'default'}),
                html.Div('F3 %', **{'data-tooltip':'Peso Max-Sharpe Frontiera 3'},
                         style={'width':'14%','textAlign':'center','fontWeight':'bold',
                                'fontSize':'8px','color':'#e6550d','position':'relative','cursor':'default'}),
            ], style={'width':'35%','display':'flex','alignItems':'center','minHeight':'28px'}),
            html.Div([
                html.Div([
                    html.Label('N. Port:', style={'fontSize':'10px','marginRight':'4px'}),
                    dcc.Input(id='fe-n-portfolios', type='number', value=15, min=5, max=100,
                              style={'width':'50px','fontSize':'10px'}),
                ], style={'display':'flex','alignItems':'center','marginRight':'8px'}),
                html.Div([
                    html.Label('Min %:', style={'fontSize':'10px','marginRight':'4px'}),
                    dcc.Input(id='fe-min-weight', type='number', value=0, min=0, max=100,
                              style={'width':'45px','fontSize':'10px'}),
                ], style={'display':'flex','alignItems':'center','marginRight':'8px'}),
                html.Div([
                    html.Label('Max %:', style={'fontSize':'10px','marginRight':'4px'}),
                    dcc.Input(id='fe-max-weight', type='number', value=100, min=0, max=100,
                              style={'width':'45px','fontSize':'10px'}),
                ], style={'display':'flex','alignItems':'center','marginRight':'8px'}),
                html.Div([
                    html.Label('Risk Free %:', style={'fontSize':'10px','marginRight':'4px'}),
                    dcc.Input(id='fe-rf', type='number', value=2.0, min=0, max=20, step=0.1,
                              style={'width':'50px','fontSize':'10px'}),
                ], style={'display':'flex','alignItems':'center','marginRight':'8px'}),
                dcc.RadioItems(id='fe-risk-measure',
                    options=[{'label':' Volatilità','value':'vol'},
                             {'label':' CVaR 10%','value':'cvar10'},
                             {'label':' CVaR 5%','value':'cvar5'}],
                    value='vol', inline=True,
                    inputStyle={'marginRight':'3px','cursor':'pointer'},
                    labelStyle={'marginRight':'8px','fontSize':'10px','cursor':'pointer'}),
                dcc.RadioItems(id='fe-return-method',
                    options=[{'label':' Standard','value':'standard'},
                             {'label':' ARIMA','value':'arima'}],
                    value='standard', inline=True,
                    inputStyle={'marginRight':'3px','cursor':'pointer'},
                    labelStyle={'marginRight':'8px','fontSize':'10px','cursor':'pointer'}),
                html.Div([
                    html.Label('Orizz.:', style={'fontSize':'10px','marginRight':'4px'}),
                    dcc.Input(id='fe-arima-horizon', type='number', value=252, min=5, max=504,
                              style={'width':'50px','fontSize':'10px'}),
                ], id='fe-arima-horizon-div',
                   style={'display':'none','alignItems':'center','marginRight':'8px'}),
                html.Div([
                    html.Label('Finestra gg:', style={'fontSize':'10px','marginRight':'4px'}),
                    dcc.Input(id='fe-arima-window', type='number', value=250, min=20, max=1260,
                              style={'width':'55px','fontSize':'10px'}),
                ], id='fe-arima-window-div',
                   style={'display':'flex','alignItems':'center','marginRight':'8px'}),
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
                html.Div(id='fe-asset-count', style={'fontSize':'10px','color':'#555',
                                                      'padding':'3px 5px'}),
                html.Div(id='fe-grid', children=[
                    html.Div('Carica i dati e clicca Calcola Frontiera',
                             style={'color':'#888','fontStyle':'italic',
                                    'fontSize':'11px','padding':'12px 8px'})
                ]),
            ], style={'width':'35%','overflowY':'auto','borderRight':'1px solid #ccd9ee',
                      'background':'white'}),

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
                    style={'flex':'55','minHeight':'0'},
                ),
                dcc.Graph(id='fe-perf-chart',
                          style={'flex':'45','minHeight':'0','marginTop':'6px'},
                          config={'displayModeBar':True}),
                html.Div(id='fe-stats-panel',
                         style={'padding':'4px 10px','fontSize':'11px','color':'#1a3a5c',
                                'flexShrink':'0'}),
            ], style={
                'width':'65%','padding':'6px','background':'white',
                'display':'flex','flexDirection':'column','overflow':'hidden',
            }),
        ], style={'display':'flex','height':'calc(100vh - 178px)','overflow':'hidden'}),

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
    dcc.Interval(id='fe-poll',       interval=800,  n_intervals=0, disabled=True),
    dcc.Interval(id='fe-arima-poll', interval=600,  n_intervals=0, disabled=True),
    dcc.Download(id='fe-dl-template'),
    dcc.Download(id='fe-dl-prices'),

    # Modale progresso
    html.Div([
        html.Div([
            html.Div('Download dati in corso…',
                     style={'fontFamily':"'Playfair Display',serif",
                            'fontSize':'1.1rem','color':'#1a3a6b',
                            'fontWeight':'700','marginBottom':'16px','textAlign':'center'}),
            html.Div([html.Div(id='fe-modal-fill', style=_FILL_LOADING)],
                     style={'width':'100%','height':'10px','background':'#dde6f5',
                            'borderRadius':'8px','overflow':'hidden','marginBottom':'10px'}),
            html.Div(id='fe-modal-pct',
                     style={'textAlign':'center','fontSize':'0.9rem','color':'#2554a0',
                            'fontWeight':'600','marginBottom':'4px'}),
            html.Div(id='fe-modal-status',
                     style={'textAlign':'center','fontSize':'0.78rem','color':'#6b7a99'}),
            html.Button('✕', id='fe-modal-close', n_clicks=0,
                        style={'position':'absolute','top':'12px','right':'16px',
                               'background':'none','border':'none','fontSize':'1.2rem',
                               'cursor':'pointer','color':'#6b7a99'}),
        ], style={'background':'white','borderRadius':'16px','padding':'32px 40px',
                  'minWidth':'340px','maxWidth':'420px','position':'relative',
                  'boxShadow':'0 8px 40px rgba(26,58,107,0.18)'}),
    ], id='fe-modal', style=_MODAL_HIDDEN),
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
    # 1. Dati personalizzati caricati in questa sessione (upload / Aggiorna)
    with _DL_LOCK:
        local_buf = dict(_DL_BUFFER)
    if 'returns' in local_buf and 'prices' in local_buf:
        prices  = local_buf['prices']
        returns = local_buf['returns']
        label   = f'Dati personalizzati ({len(prices.columns)} asset)'
        return (returns.to_json(orient='split', date_format='iso'),
                prices.to_json(orient='split', date_format='iso'),
                True, label)

    # 2. Dati condivisi da analisi di portafoglio (live o pkl)
    prices, returns, saved_at = _read_shared_data()
    if prices is not None:
        n = len(prices.columns)
        label = f'Da analisi di portafoglio ({n} asset)' + (f' — {saved_at}' if saved_at else '')
        return (returns.to_json(orient='split', date_format='iso'),
                prices.to_json(orient='split', date_format='iso'),
                True, label)

    raise PreventUpdate


@app.callback(
    Output('fe-arima-horizon-div', 'style'),
    Output('fe-arima-window-div',  'style'),
    Input('fe-return-method', 'value'),
)
def toggle_arima_horizon(method):
    base = {'display':'flex','alignItems':'center','marginRight':'8px'}
    base_none = {'display':'none','alignItems':'center','marginRight':'8px'}
    if method == 'arima':
        return base, base
    return base_none, base


@app.callback(
    Output('fe-poll',       'disabled'),
    Output('fe-modal',      'style'),
    Output('fe-modal-fill', 'style'),
    Output('fe-modal-pct',  'children'),
    Output('fe-modal-status','children'),
    Input('fe-load-btn',    'n_clicks'),
    State('fe-date-start',  'date'),
    prevent_initial_call=True,
)
def start_download(n, start_date):
    if not n:
        raise PreventUpdate
    sd = start_date or (pd.Timestamp.today()-pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    t = threading.Thread(target=_download_worker,
                         args=(_TICKERS, _DESCRIZIONI, _VALUTA, sd), daemon=True)
    t.start()
    return False, _MODAL_SHOWN, _FILL_LOADING, '0 / … (0%)', 'Avvio download…'


@app.callback(
    Output('fe-stock-data',      'data',            allow_duplicate=True),
    Output('fe-prices-data',     'data',            allow_duplicate=True),
    Output('fe-loaded-flag',     'data',            allow_duplicate=True),
    Output('fe-upload-status',   'children'),
    Output('fe-last-updated',    'children'),
    Output('fe-poll',            'disabled',        allow_duplicate=True),
    Output('fe-modal',           'style',           allow_duplicate=True),
    Input('fe-upload-data',      'contents'),
    State('fe-upload-data',      'filename'),
    State('fe-date-start',       'date'),
    prevent_initial_call=True,
)
def upload_file(contents, filename, start_date):
    if not contents:
        raise PreventUpdate
    try:
        _, cs   = contents.split(',')
        decoded = base64.b64decode(cs)
        df      = pd.read_excel(io.BytesIO(decoded))
        cols    = df.columns.tolist()

        # Detect price file: first col parseable as dates + remaining cols numeric
        is_price_file = False
        try:
            pd.to_datetime(df[cols[0]], errors='raise')
            num_cols = df.drop(columns=[cols[0]]).select_dtypes(include='number').columns.tolist()
            is_price_file = len(num_cols) >= 1
        except Exception:
            pass

        if is_price_file:
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
                True, _MODAL_HIDDEN,
            )
        else:
            # Ticker list → background download
            tickers     = list(df[cols[0]])
            descrizione = list(df[cols[1]]) if len(cols) > 1 else [str(t) for t in tickers]
            valuta      = list(df[cols[2]]) if len(cols) > 2 else ['EUR'] * len(tickers)
            sd = start_date or (pd.Timestamp.today()-pd.DateOffset(years=10)).strftime('%Y-%m-%d')
            threading.Thread(target=_download_worker,
                             args=(tickers, descrizione, valuta, sd), daemon=True).start()
            return (
                no_update, no_update, no_update,
                f'⏳ Download avviato — {len(tickers)} asset da Yahoo Finance…',
                '',
                False, _MODAL_SHOWN,
            )
    except Exception as e:
        return no_update, no_update, no_update, f'Errore: {e}', '', no_update, no_update


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
    Output('fe-progress-text',   'children'),
    Output('fe-progress-fill',   'style'),
    Output('fe-progress-bar',    'style'),
    Output('fe-modal-fill',      'style',    allow_duplicate=True),
    Output('fe-modal-pct',       'children', allow_duplicate=True),
    Output('fe-modal-status',    'children', allow_duplicate=True),
    Output('fe-poll',            'disabled', allow_duplicate=True),
    Output('fe-stock-data',      'data',     allow_duplicate=True),
    Output('fe-prices-data',     'data',     allow_duplicate=True),
    Output('fe-loaded-flag',     'data',     allow_duplicate=True),
    Output('fe-modal',           'style',    allow_duplicate=True),
    Output('fe-last-updated',    'children', allow_duplicate=True),
    Input('fe-poll',             'n_intervals'),
    prevent_initial_call=True,
)
def poll_progress(n):
    with _DL_LOCK:
        st  = dict(_DL_STATE)
        buf = dict(_DL_BUFFER)

    if st['status'] == 'idle':
        raise PreventUpdate

    total   = st['total'] or 1
    current = st['current']
    pct     = int(current / total * 100)
    bar_s   = {**_FILL_LOADING, 'width': f'{pct}%'}
    bar_c   = {'display':'block','width':'140px','height':'8px',
               'background':'#ddd','borderRadius':'8px','overflow':'hidden'}

    if st['status'] == 'running':
        txt = f'{current}/{total} ({pct}%)'
        return (txt, bar_s, bar_c, bar_s, f'{current}/{total} ({pct}%)',
                'Download in corso…', False,
                no_update, no_update, no_update, no_update, no_update)

    if st['status'] in ('done','error'):
        if st['status'] == 'done' and 'returns' in buf and 'prices' in buf:
            ret_json    = buf['returns'].to_json(orient='split', date_format='iso')
            prices_json = buf['prices'].to_json(orient='split', date_format='iso')
            n_assets    = len(buf['prices'].columns)
            errs        = len(st['errors'])
            msg_bar     = f'✓ {n_assets} asset caricati' + (f' ({errs} errori)' if errs else '')
            saved_at    = datetime.now().strftime('%d/%m/%Y %H:%M')
            return (msg_bar, {**_FILL_LOADING,'width':'100%'}, bar_c,
                    {**_FILL_LOADING,'width':'100%'},
                    f'{total}/{total} (100%)', '✓ Completato',
                    True, ret_json, prices_json, True, _MODAL_HIDDEN,
                    f'Aggiornati: {saved_at}')
        else:
            return ('✗ Errore', bar_s, bar_c, bar_s, '–',
                    '✗ Download fallito', True,
                    no_update, no_update, False, _MODAL_HIDDEN, no_update)

    raise PreventUpdate


@app.callback(
    Output('fe-modal', 'style', allow_duplicate=True),
    Input('fe-modal-close', 'n_clicks'),
    prevent_initial_call=True,
)
def close_modal(n):
    if n:
        return _MODAL_HIDDEN
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


# ── Aggiorna performance chart al click di 📊 ─────────────────────────────────
@app.callback(
    Output('fe-perf-chart',  'figure', allow_duplicate=True),
    Input({'type':'fe-chart','index':ALL},      'value'),
    Input({'type':'fe-chart-port','index':ALL}, 'value'),
    State('fe-prices-data',  'data'),
    State('fe-f1-weights',   'data'),
    State('fe-f2-weights',   'data'),
    State('fe-f3-weights',   'data'),
    State('fe-date-start',   'date'),
    State('fe-date-end',     'date'),
    prevent_initial_call=True,
)
def update_perf_chart(chart_vals, port_chart_vals, prices_data, f1j, f2j, f3j, date_start, date_end):
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
    return _build_perf_chart(prices_data, chart_assets, fw, show_frontiers, date_start, date_end)


# ── Calcola le 3 frontiere e ricostruisce la griglia ─────────────────────────
@app.callback(
    Output('fe-grid',            'children'),
    Output('fe-asset-count',     'children'),
    Output('fe-frontier-chart',  'figure'),
    Output('fe-perf-chart',      'figure'),
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
    State('fe-return-method',    'value'),
    State('fe-arima-horizon',    'value'),
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
                    n_port, wmin, wmax, rf, risk, return_method, arima_horizon,
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
        return [_PH], '', _EMPTY_FIG, _EMPTY_FIG, '', None, None, None, None, None, True, None, {'display':'none'}

    returns_df = _get_returns(stock_data)
    if returns_df is None or returns_df.empty:
        return [_PH], '', _EMPTY_FIG, _EMPTY_FIG, '', None, None, None, None, None, True, None, {'display':'none'}

    if date_start: returns_df = returns_df.loc[date_start:]
    if date_end:   returns_df = returns_df.loc[:date_end]
    returns_df = returns_df.dropna(how='all', axis=1).dropna(how='all', axis=0).dropna()
    win = int(arima_window or 250)
    if win < len(returns_df):
        returns_df = returns_df.tail(win)
    all_assets = returns_df.columns.tolist()

    # P1: default tutti gli asset; P2/P3: solo se l'utente ha spuntato almeno 2 asset
    def _p_assets(vals, ids, default_all=False):
        if not ids:
            return all_assets if default_all else []
        sel = [pid['index'] for v, pid in zip(vals, ids) if v]
        if len(sel) == 0:
            return all_assets if default_all else []
        return sel if len(sel) >= 2 else []

    p1_sel = _p_assets(p1_vals, p_ids, default_all=True)
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

    arima_label = ''
    def _mu(df_sub):
        return None

    # ── Calcola le 3 frontiere ───────────────────────────────────────────────
    frontier_res  = {}   # fname → (df_f, max_sharpe, min_vol, names)
    frontier_wgts = {}   # fname → {asset: weight%}

    _existing_sel = json.loads(sel_pt_cur) if sel_pt_cur else {}

    # ── ARIMA: avvia thread in background e ritorna subito ────────────────────
    if return_method == 'arima':
        import uuid as _uuid
        req_id = str(_uuid.uuid4())[:8]
        arima_win = int(arima_window or 250)
        arima_hor = int(arima_horizon or 20)
        with _ARIMA_LOCK:
            _ARIMA_STATE.update({
                'req_id': req_id, 'running': True, 'done': False,
                'pct': 0, 'total': len(all_assets), 'error': None, 'mu': None,
            })
        t = threading.Thread(
            target=_run_arima_thread,
            args=(req_id, returns_df, arima_win, arima_hor),
            daemon=True,
        )
        t.start()
        _prog_style = {'display':'flex','alignItems':'center','justifyContent':'center',
                       'gap':'10px','padding':'8px 16px','background':'#eef4ff',
                       'borderRadius':'8px','margin':'4px 0','flexShrink':'0'}
        no_upd = dash.no_update
        return (no_upd, no_upd, no_upd, no_upd, no_upd,
                no_upd, no_upd, no_upd, no_upd, no_upd,
                False, req_id, _prog_style)

    for fname, assets_sel in [('F1', p1_sel), ('F2', p2_sel), ('F3', p3_sel)]:
        valid = [a for a in assets_sel if a in returns_df.columns]
        if len(valid) < 2:
            continue
        df_sub = returns_df[valid].copy()
        try:
            df_f, ms, mv, names = calc_frontier(
                df_sub, n=n_f, wmin=wmin_f, wmax=wmax_f,
                rf=rf_f, risk=risk, mu_override=_mu(df_sub))
            frontier_res[fname] = (df_f, ms, mv, names)
            # Use selected point if any, else Max-Sharpe
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
    mu_all  = returns_df.mean()
    cov_all = returns_df.cov()

    # Singoli asset (sempre visibili)
    for asset in all_assets:
        w = np.zeros(len(all_assets))
        w[all_assets.index(asset)] = 1.0
        ret_a, vol_a = _port_perf(w, mu_all, cov_all)
        fig.add_trace(go.Scatter(
            x=[vol_a * 100], y=[ret_a * 100],
            mode='markers+text', name=asset,
            marker=dict(size=6, symbol='circle', opacity=0.6),
            text=[asset[:9]], textposition='top center',
            textfont=dict(size=7), showlegend=False,
            hovertemplate=f'<b>{asset}</b><br>Rischio: {vol_a*100:.2f}%<br>Rendimento: {ret_a*100:.2f}%<extra></extra>',
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
            fig.add_trace(go.Scatter(
                x=[ms['Volatility'] * 100], y=[ms['Return'] * 100],
                mode='markers', name=f'Max Sharpe {fname}',
                marker=dict(symbol='circle', size=12, color='red',
                            line=dict(color='#880000', width=1.5)),
                hovertemplate=(f'<b>Max Sharpe {fname}: {ms["Sharpe"]:.2f}</b>'
                               f'<br>Rischio: {ms["Volatility"]*100:.2f}%'
                               f'<br>Rendimento: {ms["Return"]*100:.2f}%<extra></extra>'),
            ))

    risk_label = {'vol':   'Volatilità Ann. (%)',
                  'cvar10':'CVaR 10% Ann. (%)',
                  'cvar5': 'CVaR 5% Ann. (%)'}[risk]
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

    # ── Performance chart ────────────────────────────────────────────────────
    show_frontiers = {fname: (fname in port_chart_checked) for fname in frontier_wgts}
    fig2 = _build_perf_chart(prices_data, chart_assets, frontier_wgts, show_frontiers, date_start, date_end)

    # ── Ricostruisci griglia con F1/F2/F3 weights ────────────────────────────
    p1_set    = set(p1_sel)
    p2_set    = set(p2_sel)
    p3_set    = set(p3_sel)
    chart_set = set(chart_assets)


    rows = []
    for i, asset in enumerate(all_assets):
        f1w = frontier_wgts.get('F1', {}).get(asset)
        f2w = frontier_wgts.get('F2', {}).get(asset)
        f3w = frontier_wgts.get('F3', {}).get(asset)
        row = html.Div([
            html.Div(
                html.Span(asset, style={'overflow':'hidden','whiteSpace':'nowrap',
                                        'textOverflow':'ellipsis','maxWidth':'100%',
                                        'fontSize':'8px','color':'#1a3a5c','fontWeight':'600'}),
                **{'data-tooltip': asset},
                style={'width':'25%','height':'28px','display':'flex','alignItems':'center',
                       'paddingLeft':'4px','overflow':'hidden','position':'relative','cursor':'default'}
            ),
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
        ], style={'display':'flex','alignItems':'center','height':'28px',
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
        if pt_idx >= 0 and pt_idx < len(df_f):
            label = f'P({pt_idx + 1})'
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
            f'Metodo: Auto-ARIMA finestra {arima_window or 250}gg · orizz. {arima_horizon or 20}gg',
            style={'fontSize':'10px','color':'#6b7a99','fontStyle':'italic'}))

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

    return (rows, count, fig, fig2,
            html.Div(stats, style={'display':'flex','flexWrap':'wrap','gap':'4px'}),
            f1j, f2j, f3j, rawdata_json, json.dumps(sel_pt),
            True, None, {'display':'none'})


# ── Click su frontiera → aggiorna solo la riga riepilogo (label) ─────────────
@app.callback(
    Output('fe-selected-pt', 'data', allow_duplicate=True),
    Input('fe-frontier-chart', 'clickData'),
    State('fe-selected-pt',    'data'),
    prevent_initial_call=True,
)
def on_frontier_click(click_data, sel_pt_json):
    if not click_data:
        raise PreventUpdate
    pt = click_data['points'][0]
    cd = pt.get('customdata')
    if not cd or len(cd) < 2:
        raise PreventUpdate
    fname, pt_idx = cd[0], int(cd[1])
    sel_pt = json.loads(sel_pt_json) if sel_pt_json else {}
    sel_pt[fname] = {'label': f'P({pt_idx + 1})', 'pt_idx': pt_idx}
    return json.dumps(sel_pt)


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
        result.append(
            html.Span(label or '—',
                      style={'fontSize':'9px','fontWeight':'700',
                             'color': _FC.get(fname, '#888') if label else '#bbb'})
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
    with _ARIMA_LOCK:
        s = dict(_ARIMA_STATE)
    if s.get('req_id') != cur_req_id:
        raise PreventUpdate
    if s.get('error'):
        return cur_req_id, f"❌ Errore: {s['error'][:50]}", True
    if s.get('done'):
        return cur_req_id, '✓ Completato', True
    total = s['total'] or 1
    pct   = int(s['pct'] / total * 100)
    return (dash.no_update,
            f"ARIMA: {s['pct']}/{total} titoli ({pct}%)",
            False)


# ── ARIMA: quando completato → calcola frontiere e aggiorna ──────────────────
@app.callback(
    Output('fe-grid',               'children',   allow_duplicate=True),
    Output('fe-asset-count',        'children',   allow_duplicate=True),
    Output('fe-frontier-chart',     'figure',     allow_duplicate=True),
    Output('fe-perf-chart',         'figure',     allow_duplicate=True),
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
    State('fe-arima-horizon',       'value'),
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
                  n_port, wmin, wmax, rf, risk, arima_horizon, arima_window,
                  date_start, date_end, port_chart_vals, port_chart_ids, sel_pt_cur):
    if not req_id:
        raise PreventUpdate
    with _ARIMA_LOCK:
        s = dict(_ARIMA_STATE)
    if s.get('req_id') != req_id or not s.get('done') or s.get('error'):
        raise PreventUpdate
    mu_series = s.get('mu')
    if mu_series is None:
        raise PreventUpdate
    if not stock_data:
        raise PreventUpdate

    returns_df = _get_returns(stock_data)
    if returns_df is None or returns_df.empty:
        raise PreventUpdate

    if date_start: returns_df = returns_df.loc[date_start:]
    if date_end:   returns_df = returns_df.loc[:date_end]
    returns_df = returns_df.dropna(how='all', axis=1).dropna(how='all', axis=0).dropna()
    win = int(arima_window or 250)
    if win < len(returns_df):
        returns_df = returns_df.tail(win)
    all_assets = returns_df.columns.tolist()

    def _p_assets(vals, ids, default_all=False):
        if not ids:
            return all_assets if default_all else []
        sel = [pid['index'] for v, pid in zip(vals, ids) if v]
        if len(sel) == 0:
            return all_assets if default_all else []
        return sel if len(sel) >= 2 else []

    p1_sel = _p_assets(p1_vals, p_ids, default_all=True)
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
    arima_label = f' [Auto-ARIMA finestra {arima_window or 250}gg]'
    _existing_sel = json.loads(sel_pt_cur) if sel_pt_cur else {}

    frontier_res  = {}
    frontier_wgts = {}
    for fname, assets_sel in [('F1', p1_sel), ('F2', p2_sel), ('F3', p3_sel)]:
        valid = [a for a in assets_sel if a in returns_df.columns]
        if len(valid) < 2:
            continue
        df_sub = returns_df[valid].copy()
        mu_sub = mu_series.reindex(valid).fillna(mu_series.mean()) if mu_series is not None else None
        try:
            df_f, ms, mv, names = calc_frontier(
                df_sub, n=n_f, wmin=wmin_f, wmax=wmax_f,
                rf=rf_f, risk=risk, mu_override=mu_sub)
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
        except Exception:
            pass

    # Re-use the same chart/grid/stats building as calc_and_render
    # (frontier_res, frontier_wgts, all_assets, etc. are all set)
    # Build frontier chart
    _EMPTY_FIG = go.Figure().update_layout(paper_bgcolor='white', plot_bgcolor='#f8faff')
    fig = go.Figure()
    mu_all  = returns_df.mean()
    cov_all = returns_df.cov()
    for asset in all_assets:
        w = np.zeros(len(all_assets))
        w[all_assets.index(asset)] = 1.0
        ret_a, vol_a = _port_perf(w, mu_all, cov_all)
        fig.add_trace(go.Scatter(
            x=[vol_a*100], y=[ret_a*100], mode='markers+text', name=asset,
            marker=dict(size=6, opacity=0.6), text=[asset[:9]], textposition='top center',
            textfont=dict(size=7), showlegend=False,
            hovertemplate=f'<b>{asset}</b><br>Rischio:{vol_a*100:.2f}%<br>Ren:{ret_a*100:.2f}%<extra></extra>',
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
    risk_label = {'vol':'Volatilità Ann. (%)','cvar10':'CVaR 10% Ann. (%)','cvar5':'CVaR 5% Ann. (%)'}[risk]
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
    fig2 = _build_perf_chart(prices_data, chart_assets, frontier_wgts, show_frontiers, date_start, date_end)

    # Build grid rows
    chart_set = set(chart_assets)
    p1_set    = set(p1_sel)
    p2_set    = set(p2_sel)
    p3_set    = set(p3_sel)
    rows = []
    for i, asset in enumerate(all_assets):
        f1w = frontier_wgts.get('F1', {}).get(asset)
        f2w = frontier_wgts.get('F2', {}).get(asset)
        f3w = frontier_wgts.get('F3', {}).get(asset)
        rows.append(html.Div([
            html.Div(html.Span(asset, style={'overflow':'hidden','whiteSpace':'nowrap',
                                             'textOverflow':'ellipsis','maxWidth':'100%',
                                             'fontSize':'8px','color':'#1a3a5c','fontWeight':'600'}),
                     **{'data-tooltip':asset},
                     style={'width':'25%','height':'28px','display':'flex','alignItems':'center',
                            'paddingLeft':'4px','overflow':'hidden','position':'relative','cursor':'default'}),
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
        ], style={'display':'flex','alignItems':'center','height':'28px',
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
                           style={'fontSize':'10px','color':'#6b7a99','fontStyle':'italic'}))

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

    return (rows, count, fig, fig2,
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
