"""
Frontiera Efficiente — App standalone
Ottimizzazione di portafoglio alla Markowitz con visualizzazione interattiva.
"""

import io
import json
import threading
import concurrent.futures
import os
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
app  = Dash(__name__,
            suppress_callback_exceptions=True,
            external_stylesheets=_EXT,
            requests_pathname_prefix='/frontiera-efficiente/',
            routes_pathname_prefix='/')
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
DOWNLOAD_TIMEOUT= 40
_DL_STATE  = {'status': 'idle', 'current': 0, 'total': 0, 'errors': []}
_DL_BUFFER = {}
_DL_LOCK   = threading.Lock()

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

def calc_frontier(returns_df, n=20, wmin=0.0, wmax=1.0, rf=0.02, risk='vol'):
    mu  = returns_df.mean()
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
        idx_ms   = df_f['Sharpe'].idxmax()
        max_sharpe = df_f.loc[idx_ms]
        idx_mv   = df_f['Volatility'].idxmin()
        min_vol  = df_f.loc[idx_mv]
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
def _download_single(ticker, start):
    result = [None]
    done   = threading.Event()

    def _fetch():
        try:
            result[0] = yf.download(ticker, start=start, auto_adjust=True,
                                    progress=False, threads=False)
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=_fetch, daemon=True).start()

    if done.wait(timeout=DOWNLOAD_TIMEOUT):
        df = result[0]
        if df is not None and not df.empty:
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.dropna().copy()
    return None

def _download_worker(tickers, descrizione, valuta, start_date):
    global _DL_STATE, _DL_BUFFER
    total = len(tickers)
    with _DL_LOCK:
        _DL_STATE  = {'status':'running','current':0,'total':total,'errors':[]}
        _DL_BUFFER = {}

    try:
        all_items = (
            [('EURUSD=X', '__eurusd__', None), ('EURGBP=X', '__eurgbp__', None)]
            + list(zip(tickers, descrizione, valuta))
        )
        raw = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            fmap = {
                ex.submit(_download_single, t, start_date): (desc, curr)
                for t, desc, curr in all_items
            }
            for fut in concurrent.futures.as_completed(fmap):
                desc, curr = fmap[fut]
                try:
                    px = fut.result()
                except Exception as e:
                    px = None
                    with _DL_LOCK:
                        _DL_STATE['errors'].append(str(e))
                if desc not in ('__eurusd__', '__eurgbp__'):
                    raw[desc] = (px, curr)
                    with _DL_LOCK:
                        _DL_STATE['current'] = min(_DL_STATE['current'] + 1, total)
                else:
                    raw[desc] = px

        eurusd = raw.get('__eurusd__')
        eurgbp = raw.get('__eurgbp__')

        all_prices = {}
        for desc, (px, curr) in ((k, v) for k, v in raw.items()
                                 if k not in ('__eurusd__', '__eurgbp__')):
            if px is None:
                with _DL_LOCK:
                    _DL_STATE['errors'].append(f"{desc}: nessun dato")
                continue
            if curr == 'USD' and eurusd is not None:
                px = px / eurusd.reindex(px.index).ffill()
            elif curr == 'GBP' and eurgbp is not None:
                px = px / eurgbp.reindex(px.index).ffill()
            all_prices[desc] = px

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

    except Exception as e:
        print(f"❌ Download worker crash: {e}")
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
_ASSET_OPTIONS = [{'label': d, 'value': d} for d in _DESCRIZIONI]

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
# Stili modali
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
    ls = {'fontSize':'0.82rem','fontWeight':'600','color':'#6b7a99',
          'letterSpacing':'0.04em','textTransform':'uppercase',
          'textDecoration':'none','fontFamily':'Inter, sans-serif'}
    return html.Nav([
        html.A([
            html.Span('A·C', style={'fontFamily':"'Playfair Display',serif",
                'fontSize':'1.1rem','color':'#1a3a6b','fontWeight':'700','marginRight':'10px'}),
            html.Span('FinecoBank', style={'fontFamily':'Inter,sans-serif',
                'fontSize':'0.62rem','fontWeight':'700','letterSpacing':'0.1em',
                'textTransform':'uppercase','color':'#f37021',
                'background':'rgba(243,112,33,0.1)','border':'1px solid rgba(243,112,33,0.3)',
                'padding':'3px 8px','borderRadius':'4px'}),
        ], href='https://andcappe.github.io', target='_blank',
           style={'textDecoration':'none','display':'flex','alignItems':'center'}),
        html.Ul([
            html.Li(html.A('Chi Sono',     href='https://andcappe.github.io#chi-sono',   target='_blank', style=ls)),
            html.Li(html.A('Strumenti',    href='https://andcappe.github.io#dashboard',  target='_blank', style=ls)),
            html.Li(html.A('Prenota Call', href='https://andcappe.github.io#prenota',    target='_blank', style=ls)),
            html.Li(html.A('Contatti',     href='https://andcappe.github.io#contatti',   target='_blank', style=ls)),
        ], style={'display':'flex','gap':'2rem','listStyle':'none',
                  'margin':'0','padding':'0','alignItems':'center'}),
        html.A([html.I(className='fa-regular fa-calendar', style={'marginRight':'7px'}),
                'Prenota call'],
               href='https://andcappe.github.io#prenota', target='_blank',
               style={'padding':'9px 20px','background':'#1a3a6b','color':'white',
                      'borderRadius':'7px','fontSize':'0.8rem','fontWeight':'700',
                      'letterSpacing':'0.04em','textTransform':'uppercase',
                      'textDecoration':'none','display':'inline-flex',
                      'alignItems':'center','fontFamily':'Inter,sans-serif'}),
    ], style={'position':'fixed','top':'0','left':'0','right':'0','zIndex':'100',
              'display':'flex','alignItems':'center','justifyContent':'space-between',
              'padding':'0 3%','height':'64px','background':'rgba(255,255,255,0.97)',
              'backdropFilter':'blur(14px)','borderBottom':'1px solid #ccd9ee',
              'boxShadow':'0 2px 12px rgba(26,58,107,0.08)'})

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    _navbar(),

    html.Div([
        # ── Barra comandi ────────────────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Loading(type='circle', color='#0066cc', children=[
                    html.Button('▶ Carica Dati', id='fe-load-btn', n_clicks=0,
                                style={'background':'#0066cc','color':'white','border':'none',
                                       'padding':'7px 14px','borderRadius':'4px',
                                       'cursor':'pointer','fontWeight':'bold','fontSize':'11px'})
                ]),
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
                ], style={'display':'flex','alignItems':'center','marginLeft':'12px'}),

                # Barra progresso
                html.Div([
                    html.Div(id='fe-progress-text',
                             style={'fontSize':'10px','color':'#555','marginRight':'6px'}),
                    html.Div(html.Div(id='fe-progress-fill', style=_FILL_LOADING),
                             id='fe-progress-bar',
                             style={'display':'none','width':'140px','height':'8px',
                                    'background':'#ddd','borderRadius':'8px','overflow':'hidden'}),
                ], style={'display':'flex','alignItems':'center','marginLeft':'12px'}),
            ], style={'display':'flex','alignItems':'center','gap':'8px','flexWrap':'wrap'}),
        ], style={'padding':'8px 10px','background':'#f0f4fb',
                  'borderBottom':'1px solid #ccd9ee','display':'flex',
                  'alignItems':'center','flexWrap':'wrap','gap':'8px'}),

        # ── Barra controlli calcolo ───────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label('N. Port:', style={'fontSize':'10px','marginRight':'4px'}),
                dcc.Input(id='fe-n-portfolios', type='number', value=15, min=5, max=100,
                          style={'width':'50px','fontSize':'10px'}),
            ], style={'display':'flex','alignItems':'center','marginRight':'12px'}),
            html.Div([
                html.Label('Min %:', style={'fontSize':'10px','marginRight':'4px'}),
                dcc.Input(id='fe-min-weight', type='number', value=0, min=0, max=100,
                          style={'width':'45px','fontSize':'10px'}),
            ], style={'display':'flex','alignItems':'center','marginRight':'12px'}),
            html.Div([
                html.Label('Max %:', style={'fontSize':'10px','marginRight':'4px'}),
                dcc.Input(id='fe-max-weight', type='number', value=100, min=0, max=100,
                          style={'width':'45px','fontSize':'10px'}),
            ], style={'display':'flex','alignItems':'center','marginRight':'12px'}),
            html.Div([
                html.Label('Risk Free %:', style={'fontSize':'10px','marginRight':'4px'}),
                dcc.Input(id='fe-rf', type='number', value=2.0, min=0, max=20, step=0.1,
                          style={'width':'50px','fontSize':'10px'}),
            ], style={'display':'flex','alignItems':'center','marginRight':'12px'}),
            dcc.RadioItems(id='fe-risk-measure',
                options=[{'label':' Volatilità','value':'vol'},
                         {'label':' VaR 20°','value':'var20'},
                         {'label':' VaR 10°','value':'var10'}],
                value='vol', inline=True,
                inputStyle={'marginRight':'3px','cursor':'pointer'},
                labelStyle={'marginRight':'10px','fontSize':'10px','cursor':'pointer'}),
            html.Button('Calcola Frontiera', id='fe-calc-btn', n_clicks=0,
                        style={'background':'#0066cc','color':'white','border':'none',
                               'padding':'6px 14px','borderRadius':'4px','cursor':'pointer',
                               'fontWeight':'bold','fontSize':'11px',
                               'boxShadow':'0 2px 6px rgba(0,102,204,0.35)'}),
        ], style={'display':'flex','alignItems':'center','flexWrap':'wrap','gap':'4px',
                  'padding':'6px 10px','background':'#e8f0fb',
                  'borderBottom':'1px solid #aed6f1'}),

        # ── Griglia + Grafici ────────────────────────────────────────────────
        html.Div([
            # Sinistra: griglia asset
            html.Div([
                html.Div([
                    html.Div(id='fe-asset-count',
                             style={'fontSize':'10px','color':'#555','padding':'3px 5px'}),
                    html.Div(id='fe-hint',
                             style={'display':'none','fontSize':'9px','color':'#0066cc',
                                    'fontWeight':'600','padding':'2px 5px 4px',
                                    'background':'#e8f4ff','borderLeft':'3px solid #0066cc',
                                    'marginBottom':'4px','borderRadius':'0 4px 4px 0'}),
                ]),
                html.Div(id='fe-grid', children=[
                    html.Div('Carica i dati e clicca Calcola Frontiera',
                             style={'color':'#888','fontStyle':'italic',
                                    'fontSize':'11px','padding':'12px 8px'})
                ]),
            ], style={'width':'40%','overflowY':'auto','borderRight':'1px solid #ccd9ee',
                      'background':'white'}),

            # Destra: grafici
            html.Div([
                dcc.Graph(id='fe-frontier-chart',
                          style={'height':'48vh'},
                          config={'displayModeBar':True}),
                dcc.Graph(id='fe-perf-chart',
                          style={'height':'38vh','marginTop':'8px'},
                          config={'displayModeBar':True}),
                html.Div(id='fe-stats-panel',
                         style={'padding':'6px 10px','fontSize':'11px','color':'#1a3a5c'}),
            ], style={'width':'60%','padding':'6px','background':'white'}),
        ], style={'display':'flex','height':'calc(100vh - 180px)','overflow':'hidden'}),

    ], style={'marginTop':'64px'}),

    # ── Stores ───────────────────────────────────────────────────────────────
    dcc.Store(id='fe-stock-data',    data=None),
    dcc.Store(id='fe-prices-data',   data=None),
    dcc.Store(id='fe-loaded-flag',   data=False),
    dcc.Store(id='fe-weights-p1',    data={}),
    dcc.Store(id='fe-weights-p2',    data={}),
    dcc.Store(id='fe-weights-p3',    data={}),
    dcc.Store(id='fe-pesi-result',   data=None),
    dcc.Interval(id='fe-poll', interval=800, n_intervals=0, disabled=True),

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
# Callbacks — Download / polling
# ─────────────────────────────────────────────────────────────────────────────

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
    Output('fe-progress-text',  'children'),
    Output('fe-progress-fill',  'style'),
    Output('fe-progress-bar',   'style'),
    Output('fe-modal-fill',     'style', allow_duplicate=True),
    Output('fe-modal-pct',      'children', allow_duplicate=True),
    Output('fe-modal-status',   'children', allow_duplicate=True),
    Output('fe-poll',           'disabled', allow_duplicate=True),
    Output('fe-stock-data',     'data',     allow_duplicate=True),
    Output('fe-prices-data',    'data',     allow_duplicate=True),
    Output('fe-loaded-flag',    'data',     allow_duplicate=True),
    Output('fe-modal',          'style',    allow_duplicate=True),
    Input('fe-poll',            'n_intervals'),
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
                no_update, no_update, no_update, no_update)

    if st['status'] in ('done','error'):
        if st['status'] == 'done' and 'returns' in buf and 'prices' in buf:
            ret_json    = buf['returns'].to_json(orient='split', date_format='iso')
            prices_json = buf['prices'].to_json(orient='split', date_format='iso')
            msg  = f'✓ {len(buf["prices"].columns)} asset caricati'
            errs = len(st['errors'])
            if errs:
                msg += f' ({errs} errori)'
            return (msg, {**_FILL_LOADING,'width':'100%'}, bar_c,
                    {**_FILL_LOADING,'width':'100%'},
                    f'{total}/{total} (100%)', '✓ Completato',
                    True, ret_json, prices_json, True, _MODAL_HIDDEN)
        else:
            return ('✗ Errore', bar_s, bar_c, bar_s, '–',
                    '✗ Download fallito', True,
                    no_update, no_update, False, _MODAL_HIDDEN)

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
    Output('fe-hint', 'children'),
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
        return hidden, no_update
    if trig == 'fe-loaded-flag' and loaded:
        return shown, '▶ Dati caricati — clicca CALCOLA FRONTIERA'
    return hidden, no_update

# ─────────────────────────────────────────────────────────────────────────────
# Griglia asset (11 colonne)
# ─────────────────────────────────────────────────────────────────────────────

def _lbl(w, txt, color='#1a3a5c'):
    return html.Div(txt, style={
        'width': w, 'textAlign': 'center', 'fontWeight': 'bold',
        'fontSize': '9px', 'color': color,
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
        'whiteSpace': 'pre-line',
    })

def _btn_des(btn_id, w):
    return html.Div(
        html.Button('Des', id=btn_id, n_clicks=0,
                    style={'fontSize': '8px', 'padding': '1px 4px',
                           'width': '88%', 'cursor': 'pointer',
                           'background': '#f8d7da', 'border': '1px solid #f5c6cb',
                           'borderRadius': '3px', 'color': '#721c24'}),
        style={'width': w, 'textAlign': 'center',
               'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})

def _emp(w):
    return html.Div('', style={'width': w})


@app.callback(
    Output('fe-grid',       'children'),
    Output('fe-asset-count','children'),
    Input('fe-calc-btn',    'n_clicks'),
    State('fe-loaded-flag', 'data'),
    State('fe-stock-data',  'data'),
    State({'type':'fe-chart-chk','index':ALL}, 'value'),
    State({'type':'fe-main-chk', 'index':ALL}, 'value'),
    State({'type':'fe-fr1-chk',  'index':ALL}, 'value'),
    State({'type':'fe-fr2-chk',  'index':ALL}, 'value'),
    State({'type':'fe-w1',       'index':ALL}, 'value'),
    State({'type':'fe-w2',       'index':ALL}, 'value'),
    State({'type':'fe-w3',       'index':ALL}, 'value'),
    prevent_initial_call=True,
)
def build_grid(n, loaded, stock_data,
               chart_vals, main_vals, fr1_vals, fr2_vals,
               w1_vals, w2_vals, w3_vals):
    placeholder = html.Div('Carica i dati e clicca Calcola Frontiera',
                            style={'color':'#888','fontStyle':'italic',
                                   'fontSize':'11px','padding':'12px 8px'})
    if not n or not stock_data:
        return [placeholder], ''

    returns_df = _get_returns(stock_data)
    if returns_df is None:
        return [placeholder], ''

    asset_names = returns_df.columns.tolist()

    # Raccoglie stato precedente
    def _collect_chk(vals):
        result = set()
        for v in (vals or []):
            if v:
                result.update(v)
        return result

    saved_chart = _collect_chk(chart_vals)
    saved_main  = _collect_chk(main_vals)
    saved_fr1   = _collect_chk(fr1_vals)
    saved_fr2   = _collect_chk(fr2_vals)

    # Header
    header = html.Div([
        html.Div('Asset', style={'width':'15%','fontWeight':'bold','fontSize':'9px',
                                  'paddingLeft':'5px','color':'#1a3a5c',
                                  'display':'flex','alignItems':'center'}),
        _lbl('5%',  'Chart'),
        _lbl('7%',  'Main'),
        _lbl('7%',  'Fr1'),
        _lbl('7%',  'Fr2'),
        _lbl('8%',  'F1',  '#e6194b'),
        _lbl('8%',  'F2',  '#3cb44b'),
        _lbl('8%',  'F3',  '#4363d8'),
        _lbl('10%', 'Main\n(pesi)'),
        _lbl('12%', 'Fr1\n(pesi)'),
        _lbl('13%', 'Fr2\n(pesi)'),
    ], style={'display':'flex','padding':'4px 0 2px',
              'background':'#eaf4fb','borderTop':'2px solid #2e6da4',
              'borderBottom':'1px solid #aed6f1'})

    # Des-buttons row
    des_row = html.Div([
        _emp('15%'),
        _btn_des('fe-desel-chart', '5%'),
        _btn_des('fe-desel-main',  '7%'),
        _btn_des('fe-desel-fr1',   '7%'),
        _btn_des('fe-desel-fr2',   '7%'),
        _btn_des('fe-reset-f1',    '8%'),
        _btn_des('fe-reset-f2',    '8%'),
        _btn_des('fe-reset-f3',    '8%'),
        _emp('10%'),
        _emp('12%'),
        _emp('13%'),
    ], style={'display':'flex','padding':'3px 0 5px',
              'background':'#f5faff','borderBottom':'2px solid #2e6da4',
              'marginBottom':'4px'})

    rows = [header, des_row]

    # main_vals is [] only on first render (no prior fe-main-chk components);
    # on re-render it's a list of lists (possibly empty per asset).
    first_render = not main_vals

    for i, asset in enumerate(asset_names):
        bg = 'white' if i % 2 == 0 else '#fafcff'

        chart_val = [asset]           if asset in saved_chart else []
        main_val  = [f'{asset}_Main'] if (first_render or asset in saved_main) else []
        fr1_val   = [f'{asset}_Fr1']  if asset in saved_fr1   else []
        fr2_val   = [f'{asset}_Fr2']  if asset in saved_fr2   else []

        def _inp(typ, idx, border_color, bg_inp):
            return dcc.Input(id={'type': typ, 'index': idx},
                             type='number', value=0, min=0, max=100, step=0.1,
                             placeholder='0',
                             style={'width':'90%','textAlign':'right','fontSize':'9px',
                                    'border':f'1px solid {border_color}',
                                    'borderRadius':'3px','background':bg_inp})

        row = html.Div([
            html.Div(
                html.Div(html.B(asset, style={'color':'#1a3a5c'}),
                         style={'overflow':'hidden','whiteSpace':'nowrap',
                                'textOverflow':'ellipsis','width':'100%'}),
                **{'data-tooltip': asset},
                style={'width':'15%','height':'30px','display':'flex',
                       'alignItems':'center','paddingLeft':'5px',
                       'fontSize':'9px','overflow':'visible',
                       'position':'relative','cursor':'default'}
            ),
            # Chart checkbox
            html.Div(
                dcc.Checklist(id={'type':'fe-chart-chk','index':asset},
                              options=[{'label':'','value':asset}],
                              value=chart_val,
                              style={'justifyContent':'center','width':'100%'}),
                style={'width':'5%','height':'30px','display':'flex',
                       'alignItems':'center','justifyContent':'center'}),
            # Main checkbox
            html.Div(
                dcc.Checklist(id={'type':'fe-main-chk','index':asset},
                              options=[{'label':'','value':f'{asset}_Main'}],
                              value=main_val,
                              style={'justifyContent':'center','width':'100%'}),
                style={'width':'7%','height':'30px','display':'flex',
                       'alignItems':'center','justifyContent':'center'}),
            # Fr1 checkbox
            html.Div(
                dcc.Checklist(id={'type':'fe-fr1-chk','index':asset},
                              options=[{'label':'','value':f'{asset}_Fr1'}],
                              value=fr1_val,
                              style={'justifyContent':'center','width':'100%'}),
                style={'width':'7%','height':'30px','display':'flex',
                       'alignItems':'center','justifyContent':'center'}),
            # Fr2 checkbox
            html.Div(
                dcc.Checklist(id={'type':'fe-fr2-chk','index':asset},
                              options=[{'label':'','value':f'{asset}_Fr2'}],
                              value=fr2_val,
                              style={'justifyContent':'center','width':'100%'}),
                style={'width':'7%','height':'30px','display':'flex',
                       'alignItems':'center','justifyContent':'center'}),
            # F1 input
            html.Div(_inp('fe-w1', asset, '#ffcdd2', '#fff8f8'), style={'width':'8%'}),
            # F2 input
            html.Div(_inp('fe-w2', asset, '#c8e6c9', '#f8fff8'), style={'width':'8%'}),
            # F3 input
            html.Div(_inp('fe-w3', asset, '#c5cae9', '#f8f8ff'), style={'width':'8%'}),
            # Pesi display
            html.Div(id={'type':'fe-wm','index':asset},  children='--',
                     style={'width':'10%','textAlign':'center','fontSize':'8px',
                            'display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wf1','index':asset}, children='--',
                     style={'width':'12%','textAlign':'center','fontSize':'8px',
                            'display':'flex','alignItems':'center','justifyContent':'center'}),
            html.Div(id={'type':'fe-wf2','index':asset}, children='--',
                     style={'width':'13%','textAlign':'center','fontSize':'8px',
                            'display':'flex','alignItems':'center','justifyContent':'center'}),
        ], style={'display':'flex','alignItems':'center','height':'30px',
                  'borderBottom':'1px dotted #eee','background':bg})
        rows.append(row)

    # Port1/Port2/Port3 rows
    port_colors = {'Port1':'#e6194b','Port2':'#ff7f0e','Port3':'#9467bd'}
    for p_name, p_color in port_colors.items():
        p_val = [p_name] if p_name in saved_chart else []
        rows.append(html.Div([
            html.Div(html.B(p_name, style={'color':p_color}),
                     style={'width':'15%','height':'30px','display':'flex',
                            'alignItems':'center','paddingLeft':'5px','fontSize':'9px'}),
            html.Div(
                dcc.Checklist(id={'type':'fe-chart-chk','index':p_name},
                              options=[{'label':'','value':p_name}],
                              value=p_val,
                              style={'justifyContent':'center'}),
                style={'width':'5%','height':'30px','display':'flex',
                       'alignItems':'center','justifyContent':'center'}),
            _emp('7%'), _emp('7%'), _emp('7%'),
            _emp('8%'), _emp('8%'), _emp('8%'),
            _emp('10%'), _emp('12%'), _emp('13%'),
        ], style={'display':'flex','borderBottom':'1px dotted #eee','background':'#f0f0f0'}))

    # Frontiera F1/F2/F3 rows
    frontier_labels = [('Frontiera F1','#e6194b'),('Frontiera F2','#3cb44b'),('Frontiera F3','#4363d8')]
    for f_name, f_color in frontier_labels:
        f_val = [f_name] if f_name in saved_chart else []
        rows.append(html.Div([
            html.Div(html.B(f_name, style={'color':f_color}),
                     style={'width':'15%','height':'30px','display':'flex',
                            'alignItems':'center','paddingLeft':'5px','fontSize':'9px'}),
            html.Div(
                dcc.Checklist(id={'type':'fe-chart-chk','index':f_name},
                              options=[{'label':'','value':f_name}],
                              value=f_val,
                              style={'justifyContent':'center'}),
                style={'width':'5%','height':'30px','display':'flex',
                       'alignItems':'center','justifyContent':'center'}),
            _emp('7%'), _emp('7%'), _emp('7%'),
            _emp('8%'), _emp('8%'), _emp('8%'),
            _emp('10%'), _emp('12%'), _emp('13%'),
        ], style={'display':'flex','borderBottom':'1px dotted #eee','background':'#fff0e6'}))

    # TOTALE PESI row (mostra somma pesi ottimali per le 3 frontiere)
    totals_row = html.Div([
        html.Div(html.B('TOTALE PESI', style={'color':'#d62728','fontSize':'9px'}),
                 style={'width':'15%','height':'35px','display':'flex',
                        'alignItems':'center','paddingLeft':'5px'}),
        _emp('5%'), _emp('7%'), _emp('7%'), _emp('7%'),
        _emp('8%'), _emp('8%'), _emp('8%'),
        html.Div(id='fe-total-main',
                 style={'width':'10%','textAlign':'center','fontSize':'10px',
                        'fontWeight':'bold','color':'#d62728',
                        'display':'flex','alignItems':'center','justifyContent':'center'}),
        html.Div(id='fe-total-fr1',
                 style={'width':'12%','textAlign':'center','fontSize':'10px',
                        'fontWeight':'bold','color':'#d62728',
                        'display':'flex','alignItems':'center','justifyContent':'center'}),
        html.Div(id='fe-total-fr2',
                 style={'width':'13%','textAlign':'center','fontSize':'10px',
                        'fontWeight':'bold','color':'#d62728',
                        'display':'flex','alignItems':'center','justifyContent':'center'}),
    ], style={'display':'flex','borderTop':'2px solid #d62728',
              'background':'#fff5f5','marginTop':'5px','paddingTop':'5px'})
    rows.append(totals_row)

    # TOTALE PESI F1-F2-F3 row (somma input pesi manuale)
    totals_p_row = html.Div([
        html.Div(html.B('TOTALE F1-F2-F3', style={'color':'#0066cc','fontSize':'9px'}),
                 style={'width':'15%','height':'35px','display':'flex',
                        'alignItems':'center','paddingLeft':'5px'}),
        _emp('5%'), _emp('7%'), _emp('7%'), _emp('7%'),
        html.Div(id='fe-total-p1',
                 style={'width':'8%','textAlign':'center','fontSize':'10px',
                        'fontWeight':'bold','color':'#d62728',
                        'display':'flex','alignItems':'center','justifyContent':'center'}),
        html.Div(id='fe-total-p2',
                 style={'width':'8%','textAlign':'center','fontSize':'10px',
                        'fontWeight':'bold','color':'#d62728',
                        'display':'flex','alignItems':'center','justifyContent':'center'}),
        html.Div(id='fe-total-p3',
                 style={'width':'8%','textAlign':'center','fontSize':'10px',
                        'fontWeight':'bold','color':'#d62728',
                        'display':'flex','alignItems':'center','justifyContent':'center'}),
        _emp('10%'), _emp('12%'), _emp('13%'),
    ], style={'display':'flex','borderTop':'2px solid #0066cc',
              'background':'#f0f8ff','marginTop':'5px','paddingTop':'5px'})
    rows.append(totals_p_row)

    count_txt = f'{len(asset_names)} asset disponibili'
    return rows, count_txt


# ─────────────────────────────────────────────────────────────────────────────
# Salva pesi portafogli
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Totali pesi input F1/F2/F3
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output('fe-total-p1', 'children'),
    Output('fe-total-p2', 'children'),
    Output('fe-total-p3', 'children'),
    Input({'type':'fe-w1','index':ALL}, 'value'),
    Input({'type':'fe-w2','index':ALL}, 'value'),
    Input({'type':'fe-w3','index':ALL}, 'value'),
    prevent_initial_call=True,
)
def update_weight_totals(w1_vals, w2_vals, w3_vals):
    def _fmt(vals):
        s = sum(v for v in (vals or []) if v is not None)
        color = '#2ca02c' if 99.5 <= s <= 100.5 else '#d62728'
        return html.Span(f'{s:.1f}%', style={'color': color})
    return _fmt(w1_vals), _fmt(w2_vals), _fmt(w3_vals)


# ─────────────────────────────────────────────────────────────────────────────
# Des / Reset buttons
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output({'type':'fe-chart-chk','index':ALL}, 'value', allow_duplicate=True),
    Output('fe-desel-chart', 'children'),
    Input('fe-desel-chart', 'n_clicks'),
    State({'type':'fe-chart-chk','index':ALL}, 'value'),
    State({'type':'fe-chart-chk','index':ALL}, 'options'),
    prevent_initial_call=True,
)
def desel_chart(n, vals, opts):
    if not n:
        raise PreventUpdate
    all_sel = all(v for v in vals)
    if all_sel:
        return [[] for _ in vals], 'Sel'
    return [[o['value'] for o in opt] for opt, _ in zip(opts, vals)], 'Des'


@app.callback(
    Output({'type':'fe-main-chk','index':ALL}, 'value', allow_duplicate=True),
    Output('fe-desel-main', 'children'),
    Input('fe-desel-main', 'n_clicks'),
    State({'type':'fe-main-chk','index':ALL}, 'value'),
    State({'type':'fe-main-chk','index':ALL}, 'options'),
    prevent_initial_call=True,
)
def desel_main(n, vals, opts):
    if not n:
        raise PreventUpdate
    all_sel = all(v for v in vals)
    if all_sel:
        return [[] for _ in vals], 'Sel'
    return [[o['value'] for o in opt] for opt, _ in zip(opts, vals)], 'Des'


@app.callback(
    Output({'type':'fe-fr1-chk','index':ALL}, 'value', allow_duplicate=True),
    Output('fe-desel-fr1', 'children'),
    Input('fe-desel-fr1', 'n_clicks'),
    State({'type':'fe-fr1-chk','index':ALL}, 'value'),
    State({'type':'fe-fr1-chk','index':ALL}, 'options'),
    prevent_initial_call=True,
)
def desel_fr1(n, vals, opts):
    if not n:
        raise PreventUpdate
    all_sel = all(v for v in vals)
    if all_sel:
        return [[] for _ in vals], 'Sel'
    return [[o['value'] for o in opt] for opt, _ in zip(opts, vals)], 'Des'


@app.callback(
    Output({'type':'fe-fr2-chk','index':ALL}, 'value', allow_duplicate=True),
    Output('fe-desel-fr2', 'children'),
    Input('fe-desel-fr2', 'n_clicks'),
    State({'type':'fe-fr2-chk','index':ALL}, 'value'),
    State({'type':'fe-fr2-chk','index':ALL}, 'options'),
    prevent_initial_call=True,
)
def desel_fr2(n, vals, opts):
    if not n:
        raise PreventUpdate
    all_sel = all(v for v in vals)
    if all_sel:
        return [[] for _ in vals], 'Sel'
    return [[o['value'] for o in opt] for opt, _ in zip(opts, vals)], 'Des'


@app.callback(
    Output({'type':'fe-w1','index':ALL}, 'value', allow_duplicate=True),
    Input('fe-reset-f1', 'n_clicks'),
    State({'type':'fe-w1','index':ALL}, 'id'),
    prevent_initial_call=True,
)
def reset_f1(n, ids):
    if not n:
        raise PreventUpdate
    return [0] * len(ids)


@app.callback(
    Output({'type':'fe-w2','index':ALL}, 'value', allow_duplicate=True),
    Input('fe-reset-f2', 'n_clicks'),
    State({'type':'fe-w2','index':ALL}, 'id'),
    prevent_initial_call=True,
)
def reset_f2(n, ids):
    if not n:
        raise PreventUpdate
    return [0] * len(ids)


@app.callback(
    Output({'type':'fe-w3','index':ALL}, 'value', allow_duplicate=True),
    Input('fe-reset-f3', 'n_clicks'),
    State({'type':'fe-w3','index':ALL}, 'id'),
    prevent_initial_call=True,
)
def reset_f3(n, ids):
    if not n:
        raise PreventUpdate
    return [0] * len(ids)


# ─────────────────────────────────────────────────────────────────────────────
# Calcola frontiera — produce grafici + salva pesi ottimali in store
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output('fe-frontier-chart', 'figure'),
    Output('fe-perf-chart',     'figure'),
    Output('fe-stats-panel',    'children'),
    Output('fe-pesi-result',    'data'),
    Output('fe-total-main',     'children'),
    Output('fe-total-fr1',      'children'),
    Output('fe-total-fr2',      'children'),
    Input('fe-calc-btn',        'n_clicks'),
    State('fe-stock-data',      'data'),
    State('fe-prices-data',     'data'),
    State({'type':'fe-chart-chk','index':ALL}, 'value'),
    State({'type':'fe-chart-chk','index':ALL}, 'id'),
    State({'type':'fe-main-chk', 'index':ALL}, 'value'),
    State({'type':'fe-fr1-chk',  'index':ALL}, 'value'),
    State({'type':'fe-fr2-chk',  'index':ALL}, 'value'),
    State('fe-weights-p1',      'data'),
    State('fe-weights-p2',      'data'),
    State('fe-weights-p3',      'data'),
    State('fe-n-portfolios',    'value'),
    State('fe-min-weight',      'value'),
    State('fe-max-weight',      'value'),
    State('fe-rf',              'value'),
    State('fe-risk-measure',    'value'),
    State('fe-date-start',      'date'),
    State('fe-date-end',        'date'),
    prevent_initial_call=True,
)
def update_frontier(n, stock_data, prices_data,
                    chart_vals, chart_ids,
                    main_vals, fr1_vals, fr2_vals,
                    w1, w2, w3,
                    n_port, wmin, wmax, rf, risk,
                    date_start, date_end):

    empty = go.Figure().update_layout(
        paper_bgcolor='white', plot_bgcolor='#f8faff', font_color='#1a3a5c',
        annotations=[dict(text='Carica dati e clicca Calcola Frontiera',
                          xref='paper', yref='paper', x=0.5, y=0.5,
                          showarrow=False, font=dict(size=14, color='#6b7a99'))])

    _no_pesi = None

    if not n or not stock_data:
        return empty, empty, '', _no_pesi, '--', '--', '--'

    returns_df = _get_returns(stock_data)
    if returns_df is None or returns_df.empty:
        return empty, empty, '', _no_pesi, '--', '--', '--'

    # Filtra per data
    if date_start:
        returns_df = returns_df.loc[date_start:]
    if date_end:
        returns_df = returns_df.loc[:date_end]
    returns_df = returns_df.dropna(how='all', axis=1).dropna(how='all', axis=0)

    wmin_f = (wmin or 0) / 100
    wmax_f = (wmax or 100) / 100
    rf_f   = (rf or 2.0) / 100
    n_f    = int(n_port or 15)

    # Raccoglie asset selezionati per Chart
    chart_selected = set()
    for val in (chart_vals or []):
        if val:
            chart_selected.update(val)

    # Raccoglie asset per Main/Fr1/Fr2
    def _collect_assets(vals, suffix):
        result = []
        for val in (vals or []):
            if val:
                for v in val:
                    result.append(v.replace(suffix, ''))
        return result

    main_assets = _collect_assets(main_vals, '_Main')
    fr1_assets  = _collect_assets(fr1_vals,  '_Fr1')
    fr2_assets  = _collect_assets(fr2_vals,  '_Fr2')

    # Calcola le 3 frontiere
    results = {}
    for label, assets in [('Main', main_assets), ('Fr1', fr1_assets), ('Fr2', fr2_assets)]:
        cols = [a for a in assets if a in returns_df.columns]
        if len(cols) < 2:
            results[label] = None
            continue
        sub = returns_df[cols].dropna()
        if len(sub) < 30:
            results[label] = None
            continue
        try:
            df_f, ms, mv, names = calc_frontier(sub, n=n_f, wmin=wmin_f, wmax=wmax_f,
                                                  rf=rf_f, risk=risk)
            results[label] = {'df': df_f, 'ms': ms, 'mv': mv, 'names': names, 'ret': sub}
        except Exception:
            results[label] = None

    # ── Grafico frontiera ────────────────────────────────────────────────────
    fig = go.Figure()
    palette = {'Main': '#0066cc', 'Fr1': '#3cb44b', 'Fr2': '#9467bd'}
    star_colors = {'Main': 'gold', 'Fr1': '#aaff00', 'Fr2': '#ff00ff'}

    for label, res in results.items():
        if res is None:
            continue
        df_f = res['df']
        ms   = res['ms']
        mv   = res['mv']
        col  = palette[label]

        show_f = f'Frontiera {label}' in chart_selected or not chart_selected
        if not show_f and f'Frontiera {label}' not in ['Frontiera F1','Frontiera F2','Frontiera F3']:
            show_f = True

        # Frontiera F1/F2/F3 hanno ID diversi nella griglia
        f_key = {'Main':'Frontiera F1','Fr1':'Frontiera F2','Fr2':'Frontiera F3'}[label]
        show_curve = (f_key in chart_selected) or (not chart_selected)

        if not df_f.empty and show_curve:
            fig.add_trace(go.Scatter(
                x=df_f['Volatility']*100, y=df_f['Return']*100,
                mode='lines+markers', name=f'Frontiera {label}',
                line=dict(color=col, width=2),
                marker=dict(size=5, color=df_f['Sharpe'],
                            colorscale='RdYlGn', showscale=(label=='Main'),
                            colorbar=dict(title='Sharpe', len=0.4)),
                hovertemplate=f'<b>Frontiera {label}</b><br>Rischio: %{{x:.2f}}%<br>Rendimento: %{{y:.2f}}%<extra></extra>',
            ))

        if ms is not None:
            fig.add_trace(go.Scatter(
                x=[ms['Volatility']*100], y=[ms['Return']*100],
                mode='markers', name=f'Max Sharpe {label}',
                marker=dict(symbol='star', size=14, color=star_colors[label],
                            line=dict(color='#555', width=1)),
                hovertemplate=f"<b>Max Sharpe {label}: {ms['Sharpe']:.2f}</b><br>Rischio: {ms['Volatility']*100:.2f}%<br>Rendimento: {ms['Return']*100:.2f}%<extra></extra>",
            ))

    # Singoli asset (solo se in chart_selected o nessuna selezione chart)
    if results.get('Main') and results['Main']:
        mu_all  = results['Main']['ret'].mean()
        cov_all = results['Main']['ret'].cov()
        for asset in results['Main']['names']:
            if asset not in chart_selected and chart_selected:
                continue
            idx = results['Main']['names'].index(asset)
            w_s = np.zeros(len(results['Main']['names']))
            w_s[idx] = 1.0
            ret_a, vol_a = _port_perf(w_s, mu_all, cov_all)
            fig.add_trace(go.Scatter(
                x=[vol_a*100], y=[ret_a*100],
                mode='markers+text', name=asset,
                marker=dict(size=7, symbol='circle', opacity=0.7),
                text=[asset[:8]], textposition='top center',
                textfont=dict(size=8), showlegend=False,
                hovertemplate=f'<b>{asset}</b><br>Rischio: {vol_a*100:.2f}%<br>Rendimento: {ret_a*100:.2f}%<extra></extra>',
            ))

    # Portafogli P1/P2/P3
    port_show = {'Port1': w1, 'Port2': w2, 'Port3': w3}
    port_colors_map = {'Port1':'#e6194b','Port2':'#ff7f0e','Port3':'#9467bd'}
    port_symbols    = {'Port1':'pentagon','Port2':'hexagon','Port3':'octagon'}
    for p_name, pweights in port_show.items():
        if (p_name not in chart_selected and chart_selected):
            continue
        if not pweights or not any(v > 0 for v in pweights.values()):
            continue
        # Usa la frontiera Main per il calcolo del punto
        base_ret = (results.get('Main') or {}).get('ret')
        if base_ret is None:
            base_ret = returns_df
        try:
            ret_p, vol_p, sh_p, _ = calc_single_portfolio(pweights, base_ret, rf_f)
            fig.add_trace(go.Scatter(
                x=[vol_p*100], y=[ret_p*100],
                mode='markers', name=p_name,
                marker=dict(size=14, symbol=port_symbols[p_name],
                            color=port_colors_map[p_name],
                            line=dict(color='white', width=2)),
                hovertemplate=f'<b>{p_name}</b><br>Rischio: {vol_p*100:.2f}%<br>Rendimento: {ret_p*100:.2f}%<br>Sharpe: {sh_p:.2f}<extra></extra>',
            ))
        except Exception:
            pass

    risk_label = {'vol':'Volatilità Annualizzata (%)','var20':'VaR 80% Ann. (%)','var10':'VaR 90% Ann. (%)'}[risk]
    fig.update_layout(
        title=dict(text='Frontiera Efficiente', font=dict(size=14,color='#1a3a6b'), x=0.02),
        xaxis=dict(title=risk_label, gridcolor='#e8eef8', zeroline=False),
        yaxis=dict(title='Rendimento Atteso Annualizzato (%)', gridcolor='#e8eef8', zeroline=False),
        paper_bgcolor='white', plot_bgcolor='#f8faff',
        font=dict(family='Inter, sans-serif', color='#1a3a5c', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=50, r=30, t=60, b=40),
        hovermode='closest',
    )

    # ── Grafico performance cumulativa ───────────────────────────────────────
    fig2 = empty
    if prices_data:
        try:
            prices_df = pd.read_json(io.StringIO(prices_data), orient='split')
            prices_df.index = pd.to_datetime(prices_df.index)
            if date_start:
                prices_df = prices_df.loc[date_start:]
            if date_end:
                prices_df = prices_df.loc[:date_end]

            fig2 = go.Figure()
            shown_assets = main_assets[:10] if main_assets else prices_df.columns[:5].tolist()
            for asset in shown_assets:
                if asset not in chart_selected and chart_selected:
                    continue
                if asset not in prices_df.columns:
                    continue
                s = prices_df[asset].dropna()
                if len(s) > 1:
                    cum = (s / s.iloc[0] - 1) * 100
                    fig2.add_trace(go.Scatter(
                        x=cum.index, y=cum.values, mode='lines',
                        name=asset, line=dict(width=1.5), opacity=0.8,
                        hovertemplate=f'<b>{asset}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.1f}}%<extra></extra>',
                    ))

            for p_name, pweights in port_show.items():
                if (p_name not in chart_selected and chart_selected):
                    continue
                if not pweights or not any(v > 0 for v in pweights.values()):
                    continue
                try:
                    names = prices_df.columns.tolist()
                    w_arr = np.array([pweights.get(nm, 0)/100 for nm in names])
                    s_w   = w_arr.sum()
                    if s_w > 0:
                        w_arr = w_arr / s_w
                    port_prices = (prices_df * w_arr).sum(axis=1)
                    cum_p = (port_prices / port_prices.iloc[0] - 1) * 100
                    fig2.add_trace(go.Scatter(
                        x=cum_p.index, y=cum_p.values, mode='lines',
                        name=p_name, line=dict(width=2.5,color=port_colors_map[p_name],dash='dot'),
                        hovertemplate=f'<b>{p_name}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.1f}}%<extra></extra>',
                    ))
                except Exception:
                    pass

            fig2.update_layout(
                title=dict(text='Performance Cumulativa (%)', font=dict(size=13,color='#1a3a6b'), x=0.02),
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
            pass

    # ── Pesi ottimali Max Sharpe per ogni frontiera ──────────────────────────
    pesi_data = {}
    total_labels = {'Main': '--', 'Fr1': '--', 'Fr2': '--'}

    for label, res in results.items():
        if res is None or res['ms'] is None:
            continue
        ms = res['ms']
        names = res['names']
        pesi_data[label] = {n: float(w)*100 for n, w in zip(names, ms['Weights'])}
        total_labels[label] = f"{sum(ms['Weights'])*100:.1f}%"

    def _fmt_total(val):
        if val == '--':
            return '--'
        return html.Span(val, style={'color':'#2ca02c'})

    # ── Stats panel ──────────────────────────────────────────────────────────
    stats = []
    for label, res in results.items():
        if res is None:
            continue
        ms = res['ms']
        mv = res['mv']
        col = palette[label]
        if ms is not None:
            stats.append(html.Span([
                html.B(f'{label} Max Sharpe: ', style={'color':col}),
                f"{ms['Return']*100:.1f}% rend · {ms['Volatility']*100:.1f}% rischio · Sharpe {ms['Sharpe']:.2f}",
            ], style={'marginRight':'16px','fontSize':'11px'}))

    return (fig, fig2,
            html.Div(stats, style={'display':'flex','flexWrap':'wrap','gap':'6px'}),
            pesi_data,
            _fmt_total(total_labels['Main']),
            _fmt_total(total_labels['Fr1']),
            _fmt_total(total_labels['Fr2']))


# ─────────────────────────────────────────────────────────────────────────────
# Secondo giro: aggiorna celle pesi ottimali dalla store
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output({'type':'fe-wm', 'index':ALL}, 'children'),
    Output({'type':'fe-wf1','index':ALL}, 'children'),
    Output({'type':'fe-wf2','index':ALL}, 'children'),
    Input('fe-pesi-result', 'data'),
    State({'type':'fe-wm', 'index':ALL}, 'id'),
    prevent_initial_call=True,
)
def update_pesi_display(pesi_data, ids):
    if not pesi_data or not ids:
        raise PreventUpdate

    wm_list, wf1_list, wf2_list = [], [], []
    for inp_id in ids:
        asset = inp_id['index']
        wm  = pesi_data.get('Main', {}).get(asset)
        wf1 = pesi_data.get('Fr1',  {}).get(asset)
        wf2 = pesi_data.get('Fr2',  {}).get(asset)

        def _fmt(v):
            if v is None:
                return '--'
            if v < 0.05:
                return html.Span('0%', style={'color':'#aaa'})
            return html.Span(f'{v:.1f}%',
                             style={'color':'#0066cc','fontWeight':'bold'})

        wm_list.append(_fmt(wm))
        wf1_list.append(_fmt(wf1))
        wf2_list.append(_fmt(wf2))

    return wm_list, wf1_list, wf2_list


# server esposto per DispatcherMiddleware in wsgi.py
frontier_server = app.server
