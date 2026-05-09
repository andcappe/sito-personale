"""
Frontiera Efficiente — App standalone
Ottimizzazione di portafoglio alla Markowitz con visualizzazione interattiva.
"""

import io
import json
import pickle
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
_DL_BUFFER = {}
_DL_LOCK   = threading.Lock()

# pkl salvato da portafoglio/app.py dopo ogni download
_PORT_PKL  = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'portafoglio', 'sessions', 'market_data.pkl',
))

# ─────────────────────────────────────────────────────────────────────────────
# Funzioni matematiche
# ─────────────────────────────────────────────────────────────────────────────
def _port_perf(w, mu, cov):
    ret = float(np.sum(mu * w) * 252)
    vol = float(np.sqrt(np.dot(w.T, np.dot(cov * 252, w))))
    return ret, vol

def _port_var(w, returns_df, pct):
    pr = returns_df.values @ w
    return float(-np.percentile(pr, pct) * np.sqrt(252))

def _arima_mu(returns_df, horizon=252):
    """ARIMA(1,0,0) annualized expected returns; falls back to historical mean."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        mu = {}
        for col in returns_df.columns:
            s = returns_df[col].dropna()
            if len(s) < 50:
                mu[col] = float(s.mean()) * 252
                continue
            try:
                fc = ARIMA(s, order=(1, 0, 0)).fit().forecast(steps=horizon).mean()
                mu[col] = float(fc) * 252
            except Exception:
                mu[col] = float(s.mean()) * 252
        return pd.Series(mu)
    except ImportError:
        return returns_df.mean() * 252

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
        pct = 20 if risk == 'var20' else 10
        def obj_var(w): return _port_var(w, returns_df, pct)
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


def _preload_portafoglio_data():
    """Legge market_data.pkl di portafoglio e pre-popola _DL_BUFFER."""
    try:
        if not os.path.exists(_PORT_PKL):
            return
        with open(_PORT_PKL, 'rb') as f:
            data = pickle.load(f)
        prices_df  = data.get('original_prices')
        returns_df = data.get('close_returns')
        if prices_df is None or returns_df is None:
            return
        with _DL_LOCK:
            _DL_BUFFER['prices']  = prices_df
            _DL_BUFFER['returns'] = returns_df
            _DL_STATE['status']   = 'done'
            _DL_STATE['total']    = len(prices_df.columns)
            _DL_STATE['current']  = len(prices_df.columns)
        print(f"✓ Frontiera: dati caricati da portafoglio — {len(prices_df.columns)} asset")
    except Exception as e:
        print(f"⚠ Frontiera: impossibile caricare market_data.pkl: {e}")

_preload_portafoglio_data()

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
                html.Div('Asset', style={'width':'40%','fontWeight':'bold','fontSize':'8px',
                                         'paddingLeft':'5px','color':'#1a3a5c'}),
                html.Div('P1 %', **{'data-tooltip':'Peso Portafoglio 1'},
                         style={'width':'20%','textAlign':'center','fontWeight':'bold',
                                'fontSize':'8px','color':'#e6194b','position':'relative','cursor':'default'}),
                html.Div('P2 %', **{'data-tooltip':'Peso Portafoglio 2'},
                         style={'width':'20%','textAlign':'center','fontWeight':'bold',
                                'fontSize':'8px','color':'#3cb44b','position':'relative','cursor':'default'}),
                html.Div('P3 %', **{'data-tooltip':'Peso Portafoglio 3'},
                         style={'width':'20%','textAlign':'center','fontWeight':'bold',
                                'fontSize':'8px','color':'#4363d8','position':'relative','cursor':'default'}),
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
                             {'label':' VaR 20°','value':'var20'},
                             {'label':' VaR 10°','value':'var10'}],
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

            # Destra: grafici
            html.Div([
                html.Div(id='fe-hint',
                         style={'display':'none','fontSize':'9px','color':'#0066cc',
                                'fontWeight':'600','padding':'2px 5px 4px',
                                'background':'#e8f4ff','borderLeft':'3px solid #0066cc',
                                'marginBottom':'4px','borderRadius':'0 4px 4px 0'}),
                dcc.Graph(id='fe-frontier-chart',
                          style={'height':'48vh'},
                          config={'displayModeBar':True}),
                dcc.Graph(id='fe-perf-chart',
                          style={'height':'38vh','marginTop':'8px'},
                          config={'displayModeBar':True}),
                html.Div(id='fe-stats-panel',
                         style={'padding':'6px 10px','fontSize':'11px','color':'#1a3a5c'}),
            ], style={'width':'65%','padding':'6px','background':'white'}),
        ], style={'display':'flex','height':'calc(100vh - 178px)','overflow':'hidden'}),

    ], style={'marginTop':'64px'}),

    # ── Stores ───────────────────────────────────────────────────────────────
    dcc.Store(id='_fe-page-load',    data=1),
    dcc.Store(id='fe-stock-data',    data=None),
    dcc.Store(id='fe-prices-data',   data=None),
    dcc.Store(id='fe-loaded-flag',   data=False),
    dcc.Store(id='fe-weights-p1',    data={}),
    dcc.Store(id='fe-weights-p2',    data={}),
    dcc.Store(id='fe-weights-p3',    data={}),
    dcc.Store(id='fe-selected',      data=[]),
    dcc.Interval(id='fe-poll', interval=800, n_intervals=0, disabled=True),
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
    with _DL_LOCK:
        buf     = dict(_DL_BUFFER)
        saved_at = buf.get('saved_at', '')
    if 'returns' in buf and 'prices' in buf:
        ret_json    = buf['returns'].to_json(orient='split', date_format='iso')
        prices_json = buf['prices'].to_json(orient='split', date_format='iso')
        n_asset     = len(buf['prices'].columns)
        label       = f'Da portafoglio ({n_asset} asset)' + (f' — {saved_at}' if saved_at else '')
        return ret_json, prices_json, True, label
    raise PreventUpdate


@app.callback(
    Output('fe-arima-horizon-div', 'style'),
    Input('fe-return-method', 'value'),
)
def toggle_arima_horizon(method):
    base = {'alignItems':'center','marginRight':'8px'}
    if method == 'arima':
        return {'display':'flex', **base}
    return {'display':'none', **base}


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
        return {**shown, 'children': '▶ Dati caricati — clicca CALCOLA FRONTIERA'}
    return hidden


@app.callback(
    Output('fe-grid',       'children'),
    Output('fe-asset-count','children'),
    Input('fe-calc-btn',    'n_clicks'),
    State('fe-loaded-flag', 'data'),
    State('fe-stock-data',  'data'),
    State({'type':'fe-chk','index':ALL}, 'value'),
    State({'type':'fe-w1', 'index':ALL}, 'value'),
    State({'type':'fe-w2', 'index':ALL}, 'value'),
    State({'type':'fe-w3', 'index':ALL}, 'value'),
    prevent_initial_call=True,
)
def build_grid(n, loaded, stock_data, chk_vals, w1_vals, w2_vals, w3_vals):
    placeholder = html.Div('Carica i dati e clicca Calcola Frontiera',
                            style={'color':'#888','fontStyle':'italic',
                                   'fontSize':'11px','padding':'12px 8px'})
    if not n or not stock_data:
        return [placeholder], ''

    returns_df = _get_returns(stock_data)
    if returns_df is None:
        return [placeholder], ''

    saved_chk = []
    for v in chk_vals or []:
        if v: saved_chk.extend(v)
    saved_w1, saved_w2, saved_w3 = {}, {}, {}

    rows = []
    for i, asset in enumerate(returns_df.columns):
        is_checked = asset in saved_chk
        row = html.Div([
            html.Div(
                html.Div(html.B(asset, style={'color':'#1a3a5c'}),
                         style={'overflow':'hidden','whiteSpace':'nowrap',
                                'textOverflow':'ellipsis','width':'100%'}),
                **{'data-tooltip': asset},
                style={'width':'40%','height':'28px','display':'flex',
                       'alignItems':'center','paddingLeft':'5px',
                       'fontSize':'8px','overflow':'visible',
                       'position':'relative','cursor':'default'}
            ),
            html.Div(
                dcc.Input(id={'type':'fe-w1','index':asset},
                          type='number', value=saved_w1.get(asset, 0),
                          min=0, max=100, step=1,
                          style={'width':'100%','fontSize':'9px','textAlign':'center',
                                 'border':'1px solid #ffcdd2','borderRadius':'3px',
                                 'background':'#fff8f8'}),
                style={'width':'20%','padding':'1px 2px'}
            ),
            html.Div(
                dcc.Input(id={'type':'fe-w2','index':asset},
                          type='number', value=saved_w2.get(asset, 0),
                          min=0, max=100, step=1,
                          style={'width':'100%','fontSize':'9px','textAlign':'center',
                                 'border':'1px solid #c8e6c9','borderRadius':'3px',
                                 'background':'#f8fff8'}),
                style={'width':'20%','padding':'1px 2px'}
            ),
            html.Div(
                dcc.Input(id={'type':'fe-w3','index':asset},
                          type='number', value=saved_w3.get(asset, 0),
                          min=0, max=100, step=1,
                          style={'width':'100%','fontSize':'9px','textAlign':'center',
                                 'border':'1px solid #c5cae9','borderRadius':'3px',
                                 'background':'#f8f8ff'}),
                style={'width':'20%','padding':'1px 2px'}
            ),
            dcc.Checklist(id={'type':'fe-chk','index':asset},
                          options=[{'label':'','value':asset}],
                          value=[asset] if is_checked else [],
                          style={'display':'none'}),
        ], style={'display':'flex','alignItems':'center','height':'28px',
                  'borderBottom':'1px solid #f0f4fb',
                  'background':'white' if i%2==0 else '#fafcff'})
        rows.append(row)

    count_txt = f'{len(returns_df.columns)} asset disponibili'
    return rows, count_txt


@app.callback(
    Output('fe-selected', 'data'),
    Input({'type':'fe-chk','index':ALL}, 'value'),
)
def collect_selected(vals):
    result = []
    for v in (vals or []):
        if v: result.extend(v)
    return result


@app.callback(
    Output('fe-weights-p1', 'data'),
    Output('fe-weights-p2', 'data'),
    Output('fe-weights-p3', 'data'),
    Input({'type':'fe-w1','index':ALL}, 'value'),
    Input({'type':'fe-w2','index':ALL}, 'value'),
    Input({'type':'fe-w3','index':ALL}, 'value'),
    State({'type':'fe-w1','index':ALL}, 'id'),
    prevent_initial_call=True,
)
def save_weights(w1_vals, w2_vals, w3_vals, ids):
    p1, p2, p3 = {}, {}, {}
    for val, inp_id in zip(w1_vals, ids):
        p1[inp_id['index']] = val or 0
    for val, inp_id in zip(w2_vals, ids):
        p2[inp_id['index']] = val or 0
    for val, inp_id in zip(w3_vals, ids):
        p3[inp_id['index']] = val or 0
    return p1, p2, p3


@app.callback(
    Output('fe-frontier-chart', 'figure'),
    Output('fe-perf-chart',     'figure'),
    Output('fe-stats-panel',    'children'),
    Input('fe-calc-btn',        'n_clicks'),
    State('fe-stock-data',      'data'),
    State('fe-prices-data',     'data'),
    State('fe-selected',        'data'),
    State('fe-weights-p1',      'data'),
    State('fe-weights-p2',      'data'),
    State('fe-weights-p3',      'data'),
    State('fe-n-portfolios',    'value'),
    State('fe-min-weight',      'value'),
    State('fe-max-weight',      'value'),
    State('fe-rf',              'value'),
    State('fe-risk-measure',    'value'),
    State('fe-return-method',   'value'),
    State('fe-arima-horizon',   'value'),
    State('fe-date-start',      'date'),
    State('fe-date-end',        'date'),
    prevent_initial_call=True,
)
def update_frontier(n, stock_data, prices_data, selected, w1, w2, w3,
                    n_port, wmin, wmax, rf, risk, return_method, arima_horizon,
                    date_start, date_end):
    empty = go.Figure().update_layout(
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        font_color='#1a3a5c',
        annotations=[dict(text='Carica dati e clicca Calcola Frontiera',
                          xref='paper', yref='paper', x=0.5, y=0.5,
                          showarrow=False, font=dict(size=14, color='#6b7a99'))])

    if not n or not stock_data:
        return empty, empty, ''

    returns_df = _get_returns(stock_data)
    if returns_df is None or returns_df.empty:
        return empty, empty, ''

    if date_start:
        returns_df = returns_df.loc[date_start:]
    if date_end:
        returns_df = returns_df.loc[:date_end]
    returns_df = returns_df.dropna(how='all', axis=1).dropna(how='all', axis=0)

    if selected:
        cols = [c for c in selected if c in returns_df.columns]
        if cols:
            returns_df = returns_df[cols]

    if returns_df.shape[1] < 2:
        return empty, empty, 'Seleziona almeno 2 asset'

    returns_df = returns_df.dropna()
    if len(returns_df) < 30:
        return empty, empty, 'Dati insufficienti per il periodo selezionato'

    wmin_f = (wmin or 0) / 100
    wmax_f = (wmax or 100) / 100
    rf_f   = (rf or 2.0) / 100
    n_f    = int(n_port or 15)

    # Stima rendimenti attesi (Standard o ARIMA)
    mu_override = None
    arima_label = ''
    if return_method == 'arima':
        horizon    = int(arima_horizon or 252)
        mu_override = _arima_mu(returns_df, horizon=horizon)
        arima_label = f' [ARIMA orizz.{horizon}gg]'

    try:
        df_f, max_sharpe, min_vol, asset_names = calc_frontier(
            returns_df, n=n_f, wmin=wmin_f, wmax=wmax_f, rf=rf_f,
            risk=risk, mu_override=mu_override)
    except Exception as e:
        return empty, empty, f'Errore calcolo: {e}'

    # ── Grafico frontiera ────────────────────────────────────────────────────
    fig = go.Figure()

    # Curva frontiera
    if not df_f.empty:
        fig.add_trace(go.Scatter(
            x=df_f['Volatility']*100, y=df_f['Return']*100,
            mode='lines+markers',
            name='Frontiera Efficiente',
            line=dict(color='#0066cc', width=2),
            marker=dict(size=5, color=df_f['Sharpe'],
                        colorscale='RdYlGn', showscale=True,
                        colorbar=dict(title='Sharpe', len=0.5)),
            hovertemplate='<b>Frontiera</b><br>Rischio: %{x:.2f}%<br>Rendimento: %{y:.2f}%<extra></extra>',
        ))

    # CML (Capital Market Line)
    if max_sharpe is not None and max_sharpe['Sharpe'] > 0:
        ms_vol   = max_sharpe['Volatility']
        ms_sharpe = max_sharpe['Sharpe']
        vol_range = np.linspace(0, ms_vol * 1.8, 100)
        cml_ret   = rf_f + ms_sharpe * vol_range
        fig.add_trace(go.Scatter(
            x=vol_range * 100, y=cml_ret * 100,
            mode='lines', name='CML',
            line=dict(color='purple', dash='dash', width=1.5),
            hovertemplate='<b>CML</b><br>Rischio: %{x:.2f}%<br>Rendimento: %{y:.2f}%<extra></extra>',
        ))

    # Portafoglio max Sharpe
    if max_sharpe is not None:
        fig.add_trace(go.Scatter(
            x=[max_sharpe['Volatility']*100], y=[max_sharpe['Return']*100],
            mode='markers', name='Max Sharpe',
            marker=dict(symbol='circle', size=14, color='red',
                        line=dict(color='#880000', width=1.5)),
            hovertemplate=(f"<b>Max Sharpe: {max_sharpe['Sharpe']:.2f}</b>"
                           f"<br>Rischio: {max_sharpe['Volatility']*100:.2f}%"
                           f"<br>Rendimento: {max_sharpe['Return']*100:.2f}%<extra></extra>"),
        ))

    # Portafoglio min volatilità
    if min_vol is not None:
        fig.add_trace(go.Scatter(
            x=[min_vol['Volatility']*100], y=[min_vol['Return']*100],
            mode='markers', name='Min Rischio',
            marker=dict(symbol='diamond', size=14, color='#ff6b35',
                        line=dict(color='#cc4400', width=1.5)),
            hovertemplate=(f"<b>Min Rischio</b>"
                           f"<br>Rischio: {min_vol['Volatility']*100:.2f}%"
                           f"<br>Rendimento: {min_vol['Return']*100:.2f}%<extra></extra>"),
        ))

    # Singoli asset
    mu  = returns_df.mean() if mu_override is None else mu_override / 252
    cov = returns_df.cov()
    for asset in asset_names:
        w_single = np.zeros(len(asset_names))
        w_single[asset_names.index(asset)] = 1.0
        ret_a, vol_a = _port_perf(w_single, mu, cov)
        fig.add_trace(go.Scatter(
            x=[vol_a*100], y=[ret_a*100],
            mode='markers+text', name=asset,
            marker=dict(size=7, symbol='circle', opacity=0.7),
            text=[asset[:8]], textposition='top center',
            textfont=dict(size=8),
            showlegend=False,
            hovertemplate=f'<b>{asset}</b><br>Rischio: {vol_a*100:.2f}%<br>Rendimento: {ret_a*100:.2f}%<extra></extra>',
        ))

    # P1, P2, P3
    _port_styles = [
        ('P1', w1, '#e6194b', 'pentagon'),
        ('P2', w2, '#3cb44b', 'hexagram'),
        ('P3', w3, '#4363d8', 'star-triangle-up'),
    ]
    for pname, pweights, pcolor, psymbol in _port_styles:
        if pweights and any(v > 0 for v in pweights.values()):
            try:
                ret_p, vol_p, sh_p, _ = calc_single_portfolio(pweights, returns_df, rf_f)
                fig.add_trace(go.Scatter(
                    x=[vol_p*100], y=[ret_p*100],
                    mode='markers', name=pname,
                    marker=dict(size=14, symbol=psymbol, color=pcolor,
                                line=dict(color='white', width=2)),
                    hovertemplate=(f'<b>{pname}</b><br>Rischio: {vol_p*100:.2f}%'
                                   f'<br>Rendimento: {ret_p*100:.2f}%'
                                   f'<br>Sharpe: {sh_p:.2f}<extra></extra>'),
                ))
            except Exception:
                pass

    risk_label = {'vol': 'Volatilità Annualizzata (%)',
                  'var20': 'VaR 80% Annualizzato (%)',
                  'var10': 'VaR 90% Annualizzato (%)'}[risk]
    title_text = f'Frontiera Efficiente{arima_label}'
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color='#1a3a6b'), x=0.02),
        xaxis=dict(title=risk_label, gridcolor='#e8eef8', zeroline=False),
        yaxis=dict(title='Rendimento Atteso Annualizzato (%)', gridcolor='#e8eef8', zeroline=False),
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=50, r=30, t=60, b=40),
        hovermode='closest',
    )

    # ── Grafico performance cumulativa ───────────────────────────────────────
    if prices_data:
        try:
            prices_df = pd.read_json(io.StringIO(prices_data), orient='split')
            prices_df.index = pd.to_datetime(prices_df.index)
            if date_start:
                prices_df = prices_df.loc[date_start:]
            if date_end:
                prices_df = prices_df.loc[:date_end]

            fig2 = go.Figure()
            cols_avail = [c for c in (selected or []) if c in prices_df.columns]
            if not cols_avail:
                cols_avail = prices_df.columns[:5].tolist()

            for asset in cols_avail[:10]:
                s = prices_df[asset].dropna()
                if len(s) > 1:
                    cum = (s / s.iloc[0] - 1) * 100
                    fig2.add_trace(go.Scatter(
                        x=cum.index, y=cum.values, mode='lines',
                        name=asset, line=dict(width=1.5), opacity=0.8,
                        hovertemplate=f'<b>{asset}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.1f}}%<extra></extra>',
                    ))

            for pname, pweights, pcolor, _ in _port_styles:
                if pweights and any(v > 0 for v in pweights.values()):
                    try:
                        names  = prices_df.columns.tolist()
                        w_arr  = np.array([pweights.get(nm, 0)/100 for nm in names])
                        s_w    = w_arr.sum()
                        if s_w > 0:
                            w_arr = w_arr / s_w
                        port_prices = (prices_df * w_arr).sum(axis=1)
                        cum_p = (port_prices / port_prices.iloc[0] - 1) * 100
                        fig2.add_trace(go.Scatter(
                            x=cum_p.index, y=cum_p.values, mode='lines',
                            name=pname, line=dict(width=2.5, color=pcolor, dash='dot'),
                            hovertemplate=f'<b>{pname}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.1f}}%<extra></extra>',
                        ))
                    except Exception:
                        pass

            fig2.update_layout(
                title=dict(text='Performance Cumulativa (%)',
                           font=dict(size=13, color='#1a3a6b'), x=0.02),
                xaxis=dict(gridcolor='#e8eef8', zeroline=False),
                yaxis=dict(title='Rendimento cumulativo (%)', gridcolor='#e8eef8',
                           zeroline=True, zerolinecolor='#aaa'),
                paper_bgcolor='white', plot_bgcolor='#f8faff',
                font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                margin=dict(l=50, r=30, t=50, b=40),
                hovermode='x unified',
            )
        except Exception:
            fig2 = empty
    else:
        fig2 = empty

    # ── Stats panel ─────────────────────────────────────────────────────────
    stats = []
    if max_sharpe is not None:
        stats.append(html.Span([
            html.B('Max Sharpe: ', style={'color':'#cc0000'}),
            f"{max_sharpe['Return']*100:.1f}% rend · {max_sharpe['Volatility']*100:.1f}% rischio · Sharpe {max_sharpe['Sharpe']:.2f}",
        ], style={'marginRight':'20px'}))
    if min_vol is not None:
        stats.append(html.Span([
            html.B('Min Rischio: ', style={'color':'#ff6b35'}),
            f"{min_vol['Return']*100:.1f}% rend · {min_vol['Volatility']*100:.1f}% rischio",
        ]))
    if arima_label:
        stats.append(html.Span(
            f'Metodo: ARIMA(1,0,0) orizz. {arima_horizon}gg',
            style={'marginLeft':'16px','fontSize':'10px','color':'#6b7a99','fontStyle':'italic'}
        ))
    return fig, fig2, html.Div(stats, style={'display':'flex','flexWrap':'wrap','gap':'8px'})


# ─────────────────────────────────────────────────────────────────────────────
server = app.server

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8052))
    app.run(debug=False, port=port, host='0.0.0.0')
