"""
Rendimenti Storici — App standalone
Analisi dei rendimenti per periodo: YTD, annuali, T-N, Information Ratio, Sharpe Ratio.
Legge i dati direttamente dal portafoglio condiviso (market_data.pkl / buffer live).
"""

import io
import json
import pickle
import sys
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update, ALL
from dash.exceptions import PreventUpdate

_sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from settings.browser_css import BROWSER_RESET_CSS
from navbar import make_navbar

# ─── App ─────────────────────────────────────────────────────────────────────
_EXT = [
    'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=_EXT,
           requests_pathname_prefix='/rendimenti/',
           routes_pathname_prefix='/rendimenti/')

# ─── Percorso dati condivisi ──────────────────────────────────────────────────
_PORT_PKL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'portafoglio', 'sessions', 'market_data.pkl',
))

# ─── Helpers ─────────────────────────────────────────────────────────────────
_NU = no_update

def _get_df(js):
    if not js:
        return None
    df = pd.read_json(io.StringIO(js), orient='split')
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def _get_username():
    try:
        from flask import session as _fs
        return _fs.get('username') or 'anon'
    except Exception:
        return 'anon'

def _read_user_json():
    try:
        u = _get_username()
        root = Path(os.path.dirname(os.path.abspath(__file__))).parent
        return json.load(open(root / 'sessions' / u / 'current.json'))
    except Exception:
        return {}

def _reconstruct_from_json(ns):
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

def _read_weights_from_json():
    """Legge i pesi P1/P2/P3 salvati nel portafoglio (current.json)."""
    p1, p2, p3 = {}, {}, {}
    try:
        ns = _read_user_json()
        for desc, v in ns.items():
            w1 = float(v.get('P1') or 0)
            w2 = float(v.get('P2') or 0)
            w3 = float(v.get('P3') or 0)
            if w1: p1[desc] = w1
            if w2: p2[desc] = w2
            if w3: p3[desc] = w3
    except Exception:
        pass
    return p1, p2, p3


def _read_shared_data():
    """Legge prezzi/rendimenti dal portafoglio: JSON utente → buffer live → pkl."""
    try:
        ns = _read_user_json()
        if ns:
            op, cr = _reconstruct_from_json(ns)
            if op is not None and cr is not None:
                return op, cr, ''
    except Exception:
        pass
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

# ─── Calcolo rendimenti ───────────────────────────────────────────────────────
def calculate_return_for_period(prices_series, days_back):
    prices_clean = prices_series.dropna()
    if len(prices_clean) < days_back + 1:
        return None
    price_now  = prices_clean.iloc[-1]
    price_then = prices_clean.iloc[-(days_back + 1)]
    if price_then == 0 or pd.isna(price_then):
        return None
    return (price_now / price_then) - 1

def calculate_year_return(prices_series, year):
    prices_clean = prices_series.dropna()
    if prices_clean.empty:
        return None
    jan_first = pd.Timestamp(f'{year}-01-01')
    dec_31    = pd.Timestamp(f'{year}-12-31')
    before = prices_clean[prices_clean.index < jan_first]
    if before.empty:
        return None
    price_start = before.iloc[-1]
    within = prices_clean[(prices_clean.index >= jan_first) & (prices_clean.index <= dec_31)]
    if within.empty:
        return None
    price_end = within.iloc[-1]
    if price_start == 0 or pd.isna(price_start) or pd.isna(price_end):
        return None
    return (price_end / price_start) - 1

def calculate_ytd_return(prices_series):
    prices_clean = prices_series.dropna()
    if prices_clean.empty:
        return None
    current_year = prices_clean.index[-1].year
    jan_first = pd.Timestamp(f'{current_year}-01-01')
    before_year = prices_clean[prices_clean.index < jan_first]
    price_start = before_year.iloc[-1] if not before_year.empty else prices_clean.iloc[0]
    price_now = prices_clean.iloc[-1]
    if price_start == 0 or pd.isna(price_start):
        return None
    return (price_now / price_start) - 1

def format_return(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/D"
    return f"{'+' if value >= 0 else ''}{value * 100:.2f}%"

def get_cell_style(value, is_portfolio=False):
    base = {
        'padding': '8px 10px', 'textAlign': 'right', 'fontSize': '12px',
        'whiteSpace': 'nowrap', 'border': '1px solid #ddd',
    }
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {**base, 'backgroundColor': '#f9f9f9', 'color': '#aaa'}
    if value >= 0:
        return {**base, 'backgroundColor': '#e8f5e9', 'color': '#1b5e20',
                'fontWeight': 'bold' if is_portfolio else 'normal'}
    return {**base, 'backgroundColor': '#ffebee', 'color': '#b71c1c',
            'fontWeight': 'bold' if is_portfolio else 'normal'}

def compute_akr_last(asset_ret_series, benchmark_ret_series, window, ma_window):
    """Ultimo valore della rolling IR (AKRatio), opzionalmente lisciato con MA."""
    try:
        combined = pd.concat([asset_ret_series, benchmark_ret_series], axis=1).dropna()
        if len(combined) < window + 1:
            return None
        active    = combined.iloc[:, 0] - combined.iloc[:, 1]
        ir_series = (active.rolling(window).mean() / active.rolling(window).std()) * np.sqrt(252)
        if ma_window and ma_window > 1:
            ir_series = ir_series.rolling(int(ma_window), min_periods=1).mean()
        valid = ir_series.dropna()
        return float(valid.iloc[-1]) if not valid.empty else None
    except Exception:
        return None


def calculate_ir_for_period(asset_returns, benchmark_returns, days_back):
    if benchmark_returns is None or len(benchmark_returns) == 0:
        return None
    if isinstance(asset_returns, np.ndarray) and isinstance(benchmark_returns, np.ndarray):
        if len(asset_returns) < days_back + 1:
            return None
        active = asset_returns[-days_back:] - benchmark_returns[-days_back:]
    else:
        combined = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        if len(combined) < days_back + 1:
            return None
        active = combined.iloc[-days_back:, 0].values - combined.iloc[-days_back:, 1].values
    std = active.std()
    if std == 0 or np.isnan(std):
        return None
    return (active.mean() / std) * np.sqrt(252)

def calculate_sharpe_for_period(asset_returns, days_back, annual_rf_pct):
    if asset_returns is None:
        return None
    arr = asset_returns if isinstance(asset_returns, np.ndarray) else np.asarray(asset_returns)
    arr = arr[~np.isnan(arr)]
    if len(arr) < days_back + 1:
        return None
    w = arr[-days_back:]
    ann_ret = w.mean() * 252
    ann_std = w.std() * np.sqrt(252)
    if ann_std == 0 or np.isnan(ann_std):
        return None
    rf = (annual_rf_pct or 0.0) / 100.0
    return (ann_ret - rf) / ann_std

# ─── Index string ─────────────────────────────────────────────────────────────
app.index_string = '''<!DOCTYPE html><html>
<head>{%metas%}<title>Rendimenti Storici — Andrea Cappelletti</title>{%favicon%}{%css%}
<style>
''' + BROWSER_RESET_CSS + '''
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>
'''

# ─── Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div([
    make_navbar("Rendimenti"),

    # ── Stores ───────────────────────────────────────────────────────────────
    dcc.Store(id='rend-prices-data'),
    dcc.Store(id='rend-stock-data'),
    dcc.Store(id='rend-weights-p1', data={}),
    dcc.Store(id='rend-weights-p2', data={}),
    dcc.Store(id='rend-weights-p3', data={}),
    dcc.Store(id='rend-selected', data=[]),
    dcc.Store(id='rend-perf-data'),
    dcc.Store(id='rend-sort-state', data={}),
    dcc.Store(id='rend-ir-bench-store'),
    dcc.Store(id='rend-rf-store', data=0.0),
    dcc.Store(id='rend-akr-w-store',      data=30),
    dcc.Store(id='rend-akr-ma-store',     data=1),
    dcc.Store(id='rend-akr-filter-store', data='all'),

    # Trigger caricamento dati al primo render (stesso pattern di frontiera)
    dcc.Store(id='rend-page-load', data=1),
    dcc.Store(id='rend-sync-sig', data=''),
    dcc.Interval(id='rend-live-sync', interval=2000, n_intervals=0, disabled=False),

    html.Div([

        # ── Barra controlli — riga 1 ───────────────────────────────────────
        html.Div([
            html.H2('Rendimenti Storici per Periodo', style={
                'fontFamily': 'Inter, sans-serif', 'fontSize': '1.1rem',
                'fontWeight': '700', 'color': '#1a3a6b',
                'margin': '0', 'marginRight': '24px', 'whiteSpace': 'nowrap',
            }),
            html.Div([
                html.Label('Benchmark:', style={
                    'fontSize': '10px', 'whiteSpace': 'nowrap',
                    'marginRight': '4px', 'fontWeight': 'bold',
                }),
                dcc.Dropdown(id='rend-ir-bench', options=[], value=None,
                             placeholder='Seleziona…', clearable=True,
                             style={'width': '180px', 'fontSize': '10px',
                                    'minWidth': '180px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '8px'}),
            html.Div([
                html.Label('Risk-Free SR %:', style={
                    'fontSize': '10px', 'whiteSpace': 'nowrap',
                    'marginRight': '4px', 'color': '#7a5c00', 'fontWeight': 'bold',
                }),
                dcc.Input(id='rend-rf-input', type='number', value=0.0,
                          min=0, max=20, step=0.1, placeholder='es. 3.5',
                          style={'width': '55px', 'fontSize': '10px',
                                 'border': '1px solid #aaa', 'borderRadius': '3px',
                                 'padding': '3px 5px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '8px'}),
            html.Div([
                html.Label('AK-W:', style={
                    'fontSize': '10px', 'whiteSpace': 'nowrap',
                    'marginRight': '4px', 'fontWeight': 'bold',
                }),
                dcc.Input(id='rend-akr-w', type='number', value=30, min=5, step=1,
                          placeholder='30',
                          style={'width': '45px', 'fontSize': '10px',
                                 'border': '1px solid #aaa', 'borderRadius': '3px',
                                 'padding': '3px 5px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '8px'}),
            html.Div([
                html.Label('MA:', style={
                    'fontSize': '10px', 'whiteSpace': 'nowrap',
                    'marginRight': '4px', 'fontWeight': 'bold',
                }),
                dcc.Input(id='rend-akr-ma', type='number', value=1, min=1, step=1,
                          placeholder='1',
                          style={'width': '40px', 'fontSize': '10px',
                                 'border': '1px solid #aaa', 'borderRadius': '3px',
                                 'padding': '3px 5px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '8px'}),
            html.Div([
                html.Label('Filtro AKR:', style={
                    'fontSize': '9px', 'fontWeight': 'bold', 'color': '#1a3a5c',
                    'marginRight': '4px', 'whiteSpace': 'nowrap',
                }),
                dcc.RadioItems(
                    id='rend-akr-filter',
                    options=[
                        {'label': 'Tutti',  'value': 'all'},
                        {'label': '>−1',    'value': 'gt_minus1'},
                        {'label': '>0',     'value': 'gt_0'},
                    ],
                    value='all', inline=True,
                    inputStyle={'marginRight': '2px'},
                    labelStyle={'marginRight': '6px', 'fontSize': '9px',
                                'cursor': 'pointer'},
                ),
            ], style={
                'display': 'flex', 'alignItems': 'center',
                'padding': '2px 6px', 'background': '#f0f4fa',
                'border': '1px solid #d0d8e8', 'borderRadius': '5px',
                'marginRight': '8px',
            }),
            html.Button('Aggiorna Tabella', id='rend-update-btn', n_clicks=0, style={
                'background': '#c0392b', 'color': 'white', 'border': 'none',
                'padding': '4px 14px', 'borderRadius': '4px', 'cursor': 'pointer',
                'fontWeight': 'bold', 'fontSize': '10px', 'marginLeft': 'auto',
                'whiteSpace': 'nowrap',
                'boxShadow': '0 2px 6px rgba(192,57,43,0.4)',
            }),
        ], style={
            'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px',
            'flexWrap': 'wrap', 'gap': '4px',
            'padding': '5px 8px', 'background': '#f8fafd',
            'border': '1px solid #e0e6ef', 'borderRadius': '6px',
        }),

        html.Div(id='rend-data-info', style={
            'fontSize': '11px', 'color': '#555', 'marginBottom': '10px',
        }),
        html.Hr(style={'margin': '0 0 12px 0', 'borderColor': '#e0e6ef'}),

        # ── Corpo: griglia sinistra + tabella destra ─────────────────────
        html.Div([

            # Colonna sinistra (30%)
            html.Div([
                html.Div(id='rend-asset-grid'),
                html.Hr(style={'margin': '8px 0'}),
                html.Div([
                    html.Div('Totale Pesi:', style={
                        'width': '35%', 'fontWeight': 'bold',
                        'fontSize': '10px', 'paddingLeft': '5px',
                    }),
                    html.Div('', style={'width': '10%'}),
                    html.Div(id='rend-sum-p1', children='0%', style={
                        'width': '15%', 'textAlign': 'center',
                        'color': '#d62728', 'fontSize': '10px', 'fontWeight': 'bold',
                    }),
                    html.Div(id='rend-sum-p2', children='0%', style={
                        'width': '15%', 'textAlign': 'center',
                        'color': '#d62728', 'fontSize': '10px', 'fontWeight': 'bold',
                    }),
                    html.Div(id='rend-sum-p3', children='0%', style={
                        'width': '15%', 'textAlign': 'center',
                        'color': '#d62728', 'fontSize': '10px', 'fontWeight': 'bold',
                    }),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={
                'width': '30%', 'paddingRight': '15px', 'verticalAlign': 'top',
                'overflowY': 'auto', 'maxHeight': '78vh',
            }),

            # Colonna destra (70%)
            html.Div([
                dcc.Loading(type='circle', children=[
                    html.Div(id='rend-perf-table', style={
                        'overflowX': 'auto', 'overflowY': 'auto',
                        'maxHeight': '78vh',
                        'border': '1px solid #ddd', 'borderRadius': '4px',
                    }),
                ]),
            ], style={'width': '70%', 'verticalAlign': 'top'}),

        ], style={'display': 'flex'}),

    ], style={
        'paddingTop': '112px',
        'padding': '112px 5% 32px',
        'fontFamily': 'Inter, sans-serif',
    }),
])

# ─── Callback 1: Carica dati e pesi al primo render ──────────────────────────
@app.callback(
    Output('rend-prices-data', 'data',     allow_duplicate=True),
    Output('rend-stock-data',  'data',     allow_duplicate=True),
    Output('rend-data-info',   'children', allow_duplicate=True),
    Output('rend-weights-p1',  'data'),
    Output('rend-weights-p2',  'data'),
    Output('rend-weights-p3',  'data'),
    Input('rend-page-load', 'data'),
    prevent_initial_call='initial_duplicate',
)
def load_default_data(_):
    prices, returns, saved_at = _read_shared_data()
    if prices is None or returns is None:
        return None, None, html.Span(
            'Nessun dato disponibile. Vai su Analisi di Portafoglio per caricare i dati.',
            style={'color': '#c0392b'},
        ), {}, {}, {}
    p1, p2, p3 = _read_weights_from_json()
    n_assets = len(prices.columns)
    last_date = prices.index[-1].strftime('%d/%m/%Y') if not prices.empty else 'N/D'
    has_ports = any([p1, p2, p3])
    info_parts = [
        html.I(className='fa-solid fa-circle-info', style={'marginRight': '6px', 'color': '#1a3a6b'}),
        f'{n_assets} asset · dati al {last_date}',
    ]
    if saved_at:
        info_parts.append(f' · aggiornati il {saved_at}')
    if has_ports:
        defined = [f'P{i}' for i, w in enumerate([p1, p2, p3], 1) if w]
        info_parts.append(html.Span(
            f' · pesi caricati da Portafoglio ({", ".join(defined)})',
            style={'color': '#1b5e20', 'fontWeight': '600'},
        ))
    info_parts.append(html.Span(
        ' — seleziona asset/portafogli e clicca Aggiorna Tabella',
        style={'color': '#888'},
    ))
    return (prices.to_json(orient='split', date_format='iso'),
            returns.to_json(orient='split', date_format='iso'),
            html.Span(info_parts), p1, p2, p3)


# ─── Callback 2: Costruisce la griglia asset ──────────────────────────────────
@app.callback(
    Output('rend-asset-grid', 'children'),
    Output('rend-ir-bench', 'options'),
    Input('rend-stock-data', 'data'),
    State('rend-selected', 'data'),
    State('rend-weights-p1', 'data'),
    State('rend-weights-p2', 'data'),
    State('rend-weights-p3', 'data'),
)
def build_asset_grid(stock_json, selected, p1, p2, p3):
    if not stock_json:
        return html.Div('Caricamento dati in corso…',
                        style={'padding': '12px', 'color': '#888', 'fontSize': '12px'}), []
    try:
        returns = _get_df(stock_json)
    except Exception as _e:
        print(f'[rendimenti] build_asset_grid error: {traceback.format_exc()}')
        return html.Div(f'Errore nel caricamento dei dati: {type(_e).__name__}: {_e}',
                        style={'padding': '12px', 'color': '#c0392b', 'fontSize': '12px'}), []

    asset_names = list(returns.columns)
    selected = selected or []
    p1 = p1 or {}
    p2 = p2 or {}
    p3 = p3 or {}

    def has_weights(w):
        return bool(w and any(v and v > 0 for v in w.values()))

    defined = [f'P{i}' for i, w in enumerate([p1, p2, p3], 1) if has_weights(w)]
    if defined:
        badge = html.Div(
            f'Portafogli con pesi: {", ".join(defined)}',
            style={'fontSize': '10px', 'color': '#1b5e20', 'background': '#e8f5e9',
                   'border': '1px solid #81c784', 'borderRadius': '3px',
                   'padding': '4px 8px', 'marginBottom': '6px'},
        )
    else:
        badge = html.Div(
            'Inserisci pesi P1/P2/P3 e clicca Aggiorna',
            style={'fontSize': '10px', 'color': '#e65100', 'background': '#fff3e0',
                   'border': '1px solid #ffb74d', 'borderRadius': '3px',
                   'padding': '4px 8px', 'marginBottom': '6px'},
        )

    def _col_header(label, btn_id, color, tip):
        return html.Div([
            html.Span(label, style={
                'fontSize': '9px', 'fontWeight': 'bold', 'color': color,
                'lineHeight': '1', 'fontFamily': 'Inter, sans-serif',
            }),
            html.Button('☑', id=btn_id, n_clicks=0, title=tip, style={
                'fontSize': '7px', 'border': 'none', 'background': 'none',
                'cursor': 'pointer', 'color': color,
                'padding': '0', 'margin': '0', 'lineHeight': '1',
            }),
        ], style={
            'display': 'flex', 'flexDirection': 'column',
            'alignItems': 'center', 'justifyContent': 'center',
            'overflow': 'hidden', 'gap': '0px',
        })

    header = html.Div([
        html.Div('Asset', style={
            'width': '35%', 'fontWeight': 'bold', 'paddingLeft': '5px',
            'fontSize': '9px', 'fontFamily': 'Inter, sans-serif',
        }),
        html.Div(_col_header('CH', 'rend-deselect-btn', '#1a3a5c', 'Seleziona / Deseleziona'),
                 style={'width': '10%', 'display': 'flex', 'alignItems': 'center',
                        'justifyContent': 'center'}),
        html.Div(_col_header('P1', 'rend-reset-p1-btn', '#e6194b', 'Azzera pesi P1'),
                 style={'width': '15%', 'display': 'flex', 'alignItems': 'center',
                        'justifyContent': 'center'}),
        html.Div(_col_header('P2', 'rend-reset-p2-btn', '#3cb44b', 'Azzera pesi P2'),
                 style={'width': '15%', 'display': 'flex', 'alignItems': 'center',
                        'justifyContent': 'center'}),
        html.Div(_col_header('P3', 'rend-reset-p3-btn', '#4363d8', 'Azzera pesi P3'),
                 style={'width': '15%', 'display': 'flex', 'alignItems': 'center',
                        'justifyContent': 'center'}),
    ], style={
        'display': 'flex', 'alignItems': 'center', 'minHeight': '18px',
        'background': '#eaf4fb', 'borderTop': '2px solid #2e6da4',
        'borderBottom': '1px solid #aed6f1', 'padding': '2px 0',
    })

    rows = [badge, header]

    for asset in asset_names:
        asset_val = [asset] if asset in selected else []

        def make_weight(p_idx, a=asset):
            val = {1: p1, 2: p2, 3: p3}[p_idx].get(a, 0)
            return dcc.Input(
                id={'type': 'rend-weight', 'index': f'P{p_idx}-{a}'},
                type='number', value=val, min=0, max=100, step=0.1, placeholder='0',
                style={'width': '90%', 'textAlign': 'right', 'fontSize': '8px',
                       'height': '18px', 'padding': '1px 2px', 'boxSizing': 'border-box'},
            )

        row = html.Div([
            html.Div(
                html.Span(asset, style={
                    'color': '#1a3a5c', 'fontWeight': 'bold',
                    'overflow': 'hidden', 'whiteSpace': 'nowrap',
                    'textOverflow': 'ellipsis', 'maxWidth': '100%', 'fontSize': '9px',
                    'fontFamily': 'Inter, sans-serif',
                }),
                style={'width': '35%', 'height': '22px', 'display': 'flex',
                       'alignItems': 'center', 'paddingLeft': '4px', 'overflow': 'hidden'},
            ),
            html.Div(
                dcc.Checklist(
                    id={'type': 'rend-check', 'index': asset},
                    options=[{'label': '', 'value': asset}],
                    value=asset_val,
                    inputStyle={'width': '10px', 'height': '10px',
                                'cursor': 'pointer', 'margin': '0'},
                    style={'display': 'flex', 'justifyContent': 'center',
                           'alignItems': 'center', 'width': '100%'},
                ),
                style={'width': '10%', 'height': '22px', 'display': 'flex',
                       'alignItems': 'center', 'justifyContent': 'center'},
            ),
            html.Div(make_weight(1), style={'width': '15%'}),
            html.Div(make_weight(2), style={'width': '15%'}),
            html.Div(make_weight(3), style={'width': '15%'}),
        ], style={'display': 'flex', 'alignItems': 'center',
                  'borderBottom': '1px dotted #eee'})
        rows.append(row)

    # Righe portafogli (read-only, mostra totale pesi)
    for p_num in [1, 2, 3]:
        p_name  = f'Port{p_num}'
        p_color = {1: '#e6194b', 2: '#3cb44b', 3: '#4363d8'}[p_num]
        port_val = [p_name] if p_name in selected else []
        w_dict = {1: p1, 2: p2, 3: p3}[p_num]
        total_w = sum(v for v in w_dict.values() if v and v > 0) if w_dict else 0
        t_color = '#2ca02c' if 99 <= total_w <= 101 else ('#d62728' if total_w > 0 else '#aaa')
        t_label = f'{total_w:.0f}%'

        port_row = html.Div([
            html.Div(
                html.Span(p_name, style={
                    'color': p_color, 'fontWeight': 'bold',
                    'overflow': 'hidden', 'whiteSpace': 'nowrap',
                    'textOverflow': 'ellipsis', 'maxWidth': '100%', 'fontSize': '9px',
                    'fontFamily': 'Inter, sans-serif',
                }),
                style={'width': '35%', 'height': '22px', 'display': 'flex',
                       'alignItems': 'center', 'paddingLeft': '4px', 'overflow': 'hidden'},
            ),
            html.Div(
                dcc.Checklist(
                    id={'type': 'rend-check', 'index': p_name},
                    options=[{'label': '', 'value': p_name}],
                    value=port_val,
                    inputStyle={'width': '10px', 'height': '10px',
                                'cursor': 'pointer', 'margin': '0'},
                    style={'display': 'flex', 'justifyContent': 'center',
                           'alignItems': 'center', 'width': '100%'},
                ),
                style={'width': '10%', 'height': '22px', 'display': 'flex',
                       'alignItems': 'center', 'justifyContent': 'center'},
            ),
            html.Div(html.Span(t_label, style={'color': t_color, 'fontWeight': 'bold',
                                               'fontSize': '9px'}),
                     style={'width': '15%', 'display': 'flex', 'alignItems': 'center',
                            'justifyContent': 'center'}),
            html.Div('', style={'width': '15%'}),
            html.Div('', style={'width': '15%'}),
        ], style={'display': 'flex', 'alignItems': 'center',
                  'borderBottom': '1px dotted #eee', 'backgroundColor': '#eef4ff'})
        rows.append(port_row)

    # Opzioni benchmark: asset + portafogli configurati
    bench_options = [{'label': a, 'value': a} for a in asset_names]
    for p_num, w_dict in [(1, p1), (2, p2), (3, p3)]:
        if w_dict and any(v and v > 0 for v in w_dict.values()):
            bench_options.append({'label': f'Port{p_num}', 'value': f'Port{p_num}'})

    return rows, bench_options


# ─── Callback 3: Raccoglie asset selezionati ──────────────────────────────────
@app.callback(
    Output('rend-selected', 'data'),
    Input({'type': 'rend-check', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def collect_selected(all_values):
    return [v[0] for v in all_values if v]


# ─── Callback 4: Aggiorna pesi ────────────────────────────────────────────────
@app.callback(
    Output('rend-weights-p1', 'data', allow_duplicate=True),
    Output('rend-weights-p2', 'data', allow_duplicate=True),
    Output('rend-weights-p3', 'data', allow_duplicate=True),
    Input({'type': 'rend-weight', 'index': ALL}, 'value'),
    State({'type': 'rend-weight', 'index': ALL}, 'id'),
    State('rend-weights-p1', 'data'),
    State('rend-weights-p2', 'data'),
    State('rend-weights-p3', 'data'),
    prevent_initial_call=True,
)
def update_weights(all_values, all_ids, p1, p2, p3):
    p1 = p1 or {}
    p2 = p2 or {}
    p3 = p3 or {}
    for inp_id, val in zip(all_ids, all_values):
        if isinstance(inp_id, dict) and inp_id.get('type') == 'rend-weight':
            parts = inp_id['index'].split('-', 1)
            if len(parts) == 2:
                port_id, asset = parts
                w = val if val is not None else 0
                if port_id == 'P1':
                    p1[asset] = w
                elif port_id == 'P2':
                    p2[asset] = w
                elif port_id == 'P3':
                    p3[asset] = w
    return p1, p2, p3


# ─── Callback 5: Totali pesi ─────────────────────────────────────────────────
@app.callback(
    Output('rend-sum-p1', 'children'),
    Output('rend-sum-p1', 'style'),
    Output('rend-sum-p2', 'children'),
    Output('rend-sum-p2', 'style'),
    Output('rend-sum-p3', 'children'),
    Output('rend-sum-p3', 'style'),
    Input('rend-weights-p1', 'data'),
    Input('rend-weights-p2', 'data'),
    Input('rend-weights-p3', 'data'),
)
def update_weight_sums(p1, p2, p3):
    base = {'width': '15%', 'textAlign': 'center', 'fontSize': '10px', 'fontWeight': 'bold'}
    def fmt(w_dict):
        total = sum(w_dict.values()) if w_dict else 0
        color = '#2ca02c' if 99 <= total <= 101 else '#d62728'
        return f'{total:.1f}%', {**base, 'color': color}
    t1, s1 = fmt(p1)
    t2, s2 = fmt(p2)
    t3, s3 = fmt(p3)
    return t1, s1, t2, s2, t3, s3


# ─── Callback 6: Deseleziona / seleziona tutti ───────────────────────────────
@app.callback(
    Output({'type': 'rend-check', 'index': ALL}, 'value'),
    Output('rend-deselect-btn', 'children'),
    Input('rend-deselect-btn', 'n_clicks'),
    State({'type': 'rend-check', 'index': ALL}, 'value'),
    State({'type': 'rend-check', 'index': ALL}, 'options'),
    prevent_initial_call=True,
)
def toggle_all_checks(n, current_values, all_options):
    if not all_options:
        return [], 'Deseleziona'
    if any(v for v in current_values):
        return [[] for _ in all_options], 'Seleziona'
    return [[opts[0]['value']] if opts else [] for opts in all_options], 'Deseleziona'


# ─── Callback 6b: Reset pesi P1 / P2 / P3 ───────────────────────────────────
@app.callback(
    Output('rend-weights-p1', 'data', allow_duplicate=True),
    Input('rend-reset-p1-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def reset_p1(_):
    return {}

@app.callback(
    Output('rend-weights-p2', 'data', allow_duplicate=True),
    Input('rend-reset-p2-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def reset_p2(_):
    return {}

@app.callback(
    Output('rend-weights-p3', 'data', allow_duplicate=True),
    Input('rend-reset-p3-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def reset_p3(_):
    return {}


# ─── Callback 7: Relay stores IR e RF ────────────────────────────────────────
@app.callback(
    Output('rend-ir-bench-store', 'data'),
    Input('rend-ir-bench', 'value'),
    prevent_initial_call=True,
)
def relay_ir_bench(val):
    return val

@app.callback(
    Output('rend-rf-store', 'data'),
    Input('rend-rf-input', 'value'),
    prevent_initial_call=True,
)
def relay_rf(val):
    return val if val is not None else 0.0

@app.callback(
    Output('rend-akr-w-store', 'data'),
    Input('rend-akr-w', 'value'),
    prevent_initial_call=True,
)
def relay_akr_w(val):
    return max(5, int(val)) if val else 30

@app.callback(
    Output('rend-akr-ma-store', 'data'),
    Input('rend-akr-ma', 'value'),
    prevent_initial_call=True,
)
def relay_akr_ma(val):
    return max(1, int(val)) if val else 1

@app.callback(
    Output('rend-akr-filter-store', 'data'),
    Input('rend-akr-filter', 'value'),
    prevent_initial_call=True,
)
def relay_akr_filter(val):
    return val or 'all'


# ─── Callback 8: Calcola dati performance ────────────────────────────────────
@app.callback(
    Output('rend-perf-data', 'data'),
    Input('rend-update-btn', 'n_clicks'),
    State('rend-selected', 'data'),
    State('rend-stock-data', 'data'),
    State('rend-prices-data', 'data'),
    State('rend-weights-p1', 'data'),
    State('rend-weights-p2', 'data'),
    State('rend-weights-p3', 'data'),
    State('rend-ir-bench-store', 'data'),
    State('rend-rf-store', 'data'),
    State('rend-akr-w-store', 'data'),
    State('rend-akr-ma-store', 'data'),
    prevent_initial_call=True,
)
def compute_performance(n_clicks, selected_items, stock_json, prices_json,
                        w_p1, w_p2, w_p3, ir_benchmark_name, annual_rf_pct,
                        akr_window, akr_ma):
    if not stock_json or not prices_json or not selected_items:
        raise PreventUpdate

    try:
        close_returns   = _get_df(stock_json)
        original_prices = _get_df(prices_json)
    except Exception as e:
        print(f'[rendimenti] compute_performance error: {e}')
        raise PreventUpdate

    akr_window = max(5, int(akr_window or 30))
    akr_ma     = max(1, int(akr_ma     or 1))

    # Costruisce prezzi portafogli
    portfolio_prices  = {}
    portfolio_returns = {}

    for p_num, w_dict in [(1, w_p1), (2, w_p2), (3, w_p3)]:
        p_name = f'Port{p_num}'
        if not w_dict or not any(v and v > 0 for v in w_dict.values()):
            continue
        normalized = {
            asset: w / 100.0
            for asset, w in w_dict.items()
            if w and w > 0 and asset in close_returns.columns
        }
        if not normalized:
            continue
        port_ret = pd.Series(0.0, index=close_returns.index)
        for asset, w in normalized.items():
            port_ret += close_returns[asset].fillna(0) * w
        valid_idx = close_returns[list(normalized.keys())].replace(0, np.nan).dropna(how='any').index
        if valid_idx.empty:
            continue
        port_ret    = port_ret.loc[valid_idx.min():]
        port_prices = (1 + port_ret).cumprod() * 100
        portfolio_prices[p_name]  = port_prices
        portfolio_returns[p_name] = port_prices.pct_change().dropna()

    # Benchmark (condiviso tra IR-periodi e AKRatio)
    bmark_ret_raw = None
    if ir_benchmark_name:
        if ir_benchmark_name in portfolio_returns:
            bmark_ret_raw = portfolio_returns[ir_benchmark_name]
        elif ir_benchmark_name in close_returns.columns:
            bmark_ret_raw = close_returns[ir_benchmark_name]

    annual_rf_pct = annual_rf_pct or 0.0

    bmark_arr = np.asarray(bmark_ret_raw.dropna(), dtype=float) if bmark_ret_raw is not None else None

    ret_periods = [
        ('YTD', None), ('2025', 2025), ('2024', 2024), ('2023', 2023),
        ('T-30', 30), ('T-60', 60), ('T-90', 90), ('T-180', 180),
        ('T-250', 250), ('T-500', 500), ('T-750', 750),
    ]
    ir_periods = [('IR-30', 30), ('IR-60', 60), ('IR-100', 100), ('IR-250', 250)]
    sr_periods = [('SR-30', 30), ('SR-60', 60), ('SR-100', 100), ('SR-250', 250)]
    all_periods = ret_periods + ir_periods + sr_periods

    rows_data = []
    for item in (selected_items or []):
        is_portfolio = item.startswith('Port')

        if is_portfolio:
            prices_series = portfolio_prices.get(item)
            asset_ret_s   = portfolio_returns.get(item)
            if prices_series is None:
                row = {'name': item, 'is_portfolio': True, 'AKR': None}
                for p, _ in all_periods:
                    row[p] = None
                rows_data.append(row)
                continue
        else:
            if item not in original_prices.columns:
                continue
            prices_series = original_prices[item].dropna()
            if prices_series.empty:
                continue
            asset_ret_s = close_returns[item] if item in close_returns.columns else None

        asset_arr = np.asarray(asset_ret_s.dropna(), dtype=float) if asset_ret_s is not None else None

        if asset_arr is not None and bmark_arr is not None:
            min_len = min(len(asset_arr), len(bmark_arr))
            asset_aligned = asset_arr[-min_len:]
            bmark_aligned = bmark_arr[-min_len:]
        else:
            asset_aligned = asset_arr
            bmark_aligned = bmark_arr

        row = {'name': item, 'is_portfolio': is_portfolio}

        # AKRatio (rolling IR lisciato con MA, ultimo valore)
        if bmark_ret_raw is not None and asset_ret_s is not None:
            row['AKR'] = compute_akr_last(asset_ret_s.dropna(), bmark_ret_raw.dropna(),
                                          akr_window, akr_ma)
        else:
            row['AKR'] = None

        for period_name, val in ret_periods:
            if period_name == 'YTD':
                row[period_name] = calculate_ytd_return(prices_series)
            elif isinstance(val, int) and val > 1000:
                row[period_name] = calculate_year_return(prices_series, val)
            else:
                row[period_name] = calculate_return_for_period(prices_series, val)

        for period_name, days in ir_periods:
            row[period_name] = calculate_ir_for_period(asset_aligned, bmark_aligned, days) \
                               if asset_aligned is not None else None

        for period_name, days in sr_periods:
            row[period_name] = calculate_sharpe_for_period(asset_arr, days, annual_rf_pct) \
                               if asset_arr is not None else None

        rows_data.append(row)

    last_date = (original_prices.index[-1].strftime('%d/%m/%Y')
                 if not original_prices.empty else 'N/D')

    return {
        'rows': rows_data,
        'last_date': last_date,
        'akr_window': akr_window,
        'akr_ma': akr_ma,
        'benchmark': ir_benchmark_name or '',
    }


# ─── Callback 9: Aggiorna stato ordinamento ──────────────────────────────────
@app.callback(
    Output('rend-sort-state', 'data'),
    Input({'type': 'rend-col-header', 'index': ALL}, 'n_clicks'),
    State('rend-sort-state', 'data'),
    prevent_initial_call=True,
)
def update_sort_state(n_clicks_list, current_sort):
    ctx = callback_context
    if not ctx.triggered or not ctx.triggered[0]['value']:
        raise PreventUpdate
    try:
        id_dict = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        clicked_col = id_dict['index']
    except Exception:
        raise PreventUpdate
    if current_sort and current_sort.get('col') == clicked_col:
        new_dir = 'asc' if current_sort.get('direction') == 'desc' else 'desc'
    else:
        new_dir = 'desc'
    return {'col': clicked_col, 'direction': new_dir}


# ─── Callback 10: Renderizza la tabella ──────────────────────────────────────
@app.callback(
    Output('rend-perf-table', 'children'),
    Input('rend-perf-data', 'data'),
    Input('rend-sort-state', 'data'),
    State('rend-akr-filter-store', 'data'),
    prevent_initial_call=True,
)
def render_table(perf_data, sort_state, akr_filter):
    if not perf_data or not perf_data.get('rows'):
        return html.Div(
            'Seleziona gli asset, configura i pesi e clicca "Aggiorna Tabella".',
            style={'padding': '20px', 'color': '#666', 'fontSize': '13px'},
        )

    rows_data = perf_data['rows']
    last_date  = perf_data.get('last_date', 'N/D')
    akr_window = perf_data.get('akr_window', 30)
    akr_ma     = perf_data.get('akr_ma', 1)
    benchmark  = perf_data.get('benchmark', '')
    akr_filter = akr_filter or 'all'

    # Applica filtro AKR (solo asset, mai portafogli)
    threshold = None
    if akr_filter == 'gt_minus1':
        threshold = -1.0
    elif akr_filter == 'gt_0':
        threshold = 0.0

    if threshold is not None:
        filtered = []
        for row in rows_data:
            if row.get('is_portfolio'):
                filtered.append(row)
                continue
            akr_val = row.get('AKR')
            if akr_val is not None and not (isinstance(akr_val, float) and np.isnan(akr_val)):
                if akr_val <= threshold:
                    continue
            filtered.append(row)
        rows_data = filtered

    ret_cols = ['YTD', '2025', '2024', '2023', 'T-30', 'T-60', 'T-90', 'T-180', 'T-250', 'T-500', 'T-750']
    ir_cols  = ['IR-30', 'IR-60', 'IR-100', 'IR-250']
    sr_cols  = ['SR-30', 'SR-60', 'SR-100', 'SR-250']
    periods  = ['AKR'] + ret_cols + ir_cols + sr_cols

    sort_col = sort_state.get('col') if sort_state else None
    sort_dir = sort_state.get('direction', 'desc') if sort_state else 'desc'

    if sort_col and sort_col in periods:
        def sort_key(row):
            val = row.get(sort_col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return -float('inf') if sort_dir == 'desc' else float('inf')
            return val
        rows_data = sorted(rows_data, key=sort_key, reverse=(sort_dir == 'desc'))

    def make_th(label, col_id=None, bg_override=None):
        is_active = sort_col == col_id if col_id else False
        arrow = (' ▼' if sort_dir == 'desc' else ' ▲') if is_active else (' ↕' if col_id else '')
        base_bg = bg_override if (bg_override and not is_active) else ('#0d2a4a' if is_active else '#1a3a5c')
        th_style = {
            'padding': '0px',
            'backgroundColor': base_bg,
            'color': 'white', 'position': 'sticky', 'top': '0',
            'zIndex': '2', 'minWidth': '90px',
            'border': '1px solid #0d2540', 'whiteSpace': 'nowrap',
        }
        if col_id is None:
            th_style['textAlign'] = 'left'
            th_style['minWidth']  = '160px'

        if col_id is not None:
            btn_style = {
                'background': 'none', 'border': 'none',
                'color': '#ffd700' if is_active else 'white',
                'cursor': 'pointer', 'fontWeight': 'bold' if is_active else 'normal',
                'fontSize': '12px', 'padding': '10px 14px', 'width': '100%',
                'textAlign': 'right', 'userSelect': 'none', 'whiteSpace': 'nowrap',
            }
            return html.Th(
                html.Button(f'{label}{arrow}',
                            id={'type': 'rend-col-header', 'index': col_id},
                            n_clicks=0, style=btn_style),
                style=th_style,
            )
        return html.Th(label, style={**th_style, 'padding': '10px 14px'})

    header_cells = [make_th('Asset / Portafoglio')]
    # AKR — prima colonna numerica, sfondo viola come IR
    header_cells.append(make_th('AKR', col_id='AKR', bg_override='#4a1a7c'))
    for p in ret_cols:
        header_cells.append(make_th(p, col_id=p))
    for p in ir_cols:
        header_cells.append(make_th(p, col_id=p, bg_override='#4a1a7c'))
    for p in sr_cols:
        header_cells.append(make_th(p, col_id=p, bg_override='#7a5c00'))

    table_rows = [html.Tr(header_cells)]

    for row_idx, row in enumerate(rows_data):
        item = row['name']
        is_portfolio = row['is_portfolio']
        row_bg = '#dce8ff' if is_portfolio else ('#ffffff' if row_idx % 2 == 0 else '#f8f8f8')

        name_style = {
            'padding': '8px 10px', 'fontSize': '11px',
            'fontWeight': 'bold' if is_portfolio else 'normal',
            'color': '#0066cc' if is_portfolio else '#222',
            'backgroundColor': row_bg,
            'position': 'sticky', 'left': '0', 'zIndex': '1',
            'border': '1px solid #ddd', 'whiteSpace': 'nowrap', 'minWidth': '160px',
        }
        cells = [html.Td(item, style=name_style)]

        # Cella AKR
        akr_val = row.get('AKR')
        if akr_val is None or (isinstance(akr_val, float) and np.isnan(akr_val)):
            akr_txt = 'N/D'
            akr_cs  = {'padding': '8px 10px', 'textAlign': 'right', 'fontSize': '12px',
                       'whiteSpace': 'nowrap', 'border': '1px solid #ddd',
                       'backgroundColor': '#f3eeff', 'color': '#aaa'}
        else:
            akr_txt = f'{akr_val:+.2f}'
            akr_cs  = {'padding': '8px 10px', 'textAlign': 'right', 'fontSize': '12px',
                       'whiteSpace': 'nowrap', 'border': '1px solid #ddd',
                       'fontWeight': 'bold' if is_portfolio else 'normal',
                       'backgroundColor': '#ede7f6' if akr_val >= 0 else '#fce4ec',
                       'color': '#4a148c' if akr_val >= 0 else '#880e4f'}
        if sort_col == 'AKR' and akr_val is not None and not (isinstance(akr_val, float) and np.isnan(akr_val)):
            akr_cs['backgroundColor'] = '#d1c4e9' if akr_val >= 0 else '#f8bbd0'
        cells.append(html.Td(akr_txt, style=akr_cs))

        for period in ret_cols:
            val = row.get(period)
            cs = get_cell_style(val, is_portfolio=is_portfolio)
            if sort_col == period and val is not None and not (isinstance(val, float) and np.isnan(val)):
                cs['backgroundColor'] = '#d0edce' if val >= 0 else '#ffdde0'
            cells.append(html.Td(format_return(val), style=cs))

        for period in ir_cols:
            val = row.get(period)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                txt = 'N/D'
                cs  = {'padding': '8px 10px', 'textAlign': 'right', 'fontSize': '12px',
                       'whiteSpace': 'nowrap', 'border': '1px solid #ddd',
                       'backgroundColor': '#f3eeff', 'color': '#aaa'}
            else:
                txt = f'{val:+.2f}'
                cs  = {'padding': '8px 10px', 'textAlign': 'right', 'fontSize': '12px',
                       'whiteSpace': 'nowrap', 'border': '1px solid #ddd',
                       'fontWeight': 'bold' if is_portfolio else 'normal',
                       'backgroundColor': '#ede7f6' if val >= 0 else '#fce4ec',
                       'color': '#4a148c' if val >= 0 else '#880e4f'}
            if sort_col == period and val is not None and not (isinstance(val, float) and np.isnan(val)):
                cs['backgroundColor'] = '#d1c4e9' if val >= 0 else '#f8bbd0'
            cells.append(html.Td(txt, style=cs))

        for period in sr_cols:
            val = row.get(period)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                txt = 'N/D'
                cs  = {'padding': '8px 10px', 'textAlign': 'right', 'fontSize': '12px',
                       'whiteSpace': 'nowrap', 'border': '1px solid #ddd',
                       'backgroundColor': '#fdf8e6', 'color': '#aaa'}
            else:
                txt = f'{val:+.2f}'
                cs  = {'padding': '8px 10px', 'textAlign': 'right', 'fontSize': '12px',
                       'whiteSpace': 'nowrap', 'border': '1px solid #ddd',
                       'fontWeight': 'bold' if is_portfolio else 'normal',
                       'backgroundColor': '#fff9e6' if val >= 0 else '#fff0e6',
                       'color': '#7a5c00' if val >= 0 else '#a03000'}
            if sort_col == period and val is not None and not (isinstance(val, float) and np.isnan(val)):
                cs['backgroundColor'] = '#ffe082' if val >= 0 else '#ffccbc'
            cells.append(html.Td(txt, style=cs))

        table_rows.append(html.Tr(cells, style={'backgroundColor': row_bg}))

    table = html.Table(table_rows, style={
        'borderCollapse': 'collapse', 'width': '100%',
        'fontFamily': 'Arial, sans-serif',
    })

    info_parts = [
        html.Span(f'Dati al: {last_date}',
                  style={'fontSize': '11px', 'color': '#555', 'marginRight': '16px'}),
    ]
    if benchmark:
        info_parts.append(html.Span(
            f'Benchmark: {benchmark} · AK-W: {akr_window}' + (f' · MA: {akr_ma}' if akr_ma > 1 else ''),
            style={'fontSize': '11px', 'color': '#4a1a7c', 'fontWeight': '600',
                   'marginRight': '16px'},
        ))
    info_parts.append(html.Span(
        'Verde = positivo · Rosso = negativo · Clicca intestazione per ordinare',
        style={'fontSize': '11px', 'color': '#555'},
    ))

    info_bar = html.Div(info_parts, style={
        'padding': '8px 12px', 'backgroundColor': '#f0f4fa',
        'borderBottom': '1px solid #ddd',
    })

    return html.Div([info_bar, table])


# ─── Sync live con portafoglio: aggiorna dati quando la lista asset cambia ───
@app.callback(
    Output('rend-prices-data', 'data',     allow_duplicate=True),
    Output('rend-stock-data',  'data',     allow_duplicate=True),
    Output('rend-data-info',   'children', allow_duplicate=True),
    Output('rend-sync-sig',    'data'),
    Input('rend-live-sync',    'n_intervals'),
    State('rend-sync-sig',     'data'),
    prevent_initial_call=True,
)
def rend_live_sync(_, sig):
    ns = _read_user_json()
    new_sig = ','.join(sorted(ns.keys())) if ns else ''
    if new_sig == (sig or '') or not ns:
        raise PreventUpdate
    op, cr = _reconstruct_from_json(ns)
    if op is None or cr is None:
        raise PreventUpdate
    n = len(op.columns)
    label = html.Span([
        html.I(className='fa-solid fa-circle-info',
               style={'marginRight': '6px', 'color': '#1a3a6b'}),
        html.Span('👤 Personale', style={'fontWeight': '700', 'color': '#1a5c1a',
                                          'marginRight': '8px'}),
        f'{n} asset',
    ])
    return (op.to_json(orient='split', date_format='iso'),
            cr.to_json(orient='split', date_format='iso'),
            label, new_sig)
