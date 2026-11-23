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
from settings.caratteri import NOME_ASSET
from navbar import make_navbar, ETICHETTA_PERSONALE

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

# Tetto ai rendimenti giornalieri: oltre ±50%/giorno è un tick corrotto che fa
# esplodere le metriche. _get_df qui legge sia rendimenti sia prezzi, quindi il
# clip si applica SOLO alle variabili dei rendimenti (non ai prezzi).
_MAX_DAILY_RET = 0.5

# AKRatio: finestra e media mobile predefinite (come AK-SH-TV W e MA in Analisi Portafoglio)
_AKR_W, _AKR_MA = 100, 10
_RF = 2.0   # come il risk free predefinito della Frontiera

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
    """Solo voci-asset di current.json: scarta le chiavi meta (es. "_tipo": "personale"),
    altrimenti _reconstruct_from_json andrebbe in eccezione sui valori non-dict e
    Rendimenti ripiegherebbe sul pkl di default (asset diversi dal portafoglio)."""
    try:
        u = _get_username()
        root = Path(os.path.dirname(os.path.abspath(__file__))).parent
        raw = json.load(open(root / 'sessions' / u / 'current.json'))
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
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
/* Dentro il Portafoglio (?embed=1, classe nb-embed da NAVBAR_JS): niente spazio per
   la navbar e stessi margini della Frontiera incorporata */
html.nb-embed .rend-corpo { padding: 8px 8px 16px !important; }
/* Tabella: la colonna dei nomi resta ferma (sticky left), le altre scorrono di lato. */
.rend-scorri { overflow: auto; max-height: 78vh; border: 1px solid #ddd; border-radius: 4px;
  -webkit-overflow-scrolling: touch; overscroll-behavior-x: contain; }
.rend-nome { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
/* Colonna AKR ferma accanto ai nomi: il suo «left» è la larghezza della colonna dei
   nomi, misurata dallo script in fondo (--rend-nome-w sul riquadro che scorre). */
.rend-akr { left: var(--rend-nome-w, 0px) !important; }
/* Menu ⚙ «Descrizione dati tabella» accanto a Crea tabella (come in Analisi Portafoglio). */
.rend-menu { position: relative; display: inline-block; margin-left: 4px; }
.rend-menu > summary { list-style: none; }
.rend-menu > summary::-webkit-details-marker { display: none; }
.rend-menu-corpo { position: absolute; top: calc(100% + 4px); right: 0; z-index: 1500;
  background: #fff; border: 1px solid #c0d0e8; border-radius: 6px; padding: 10px 12px;
  box-shadow: 0 6px 20px rgba(26,58,107,0.18); min-width: 260px; max-width: 360px;
  display: flex; flex-direction: column; gap: 6px; font-size: 11px; color: #333; }
@media (max-width: 900px) {
  .rend-menu-corpo { position: fixed; top: auto; left: 16px; right: 16px; min-width: 0; max-width: none; }
  /* Parametri, Asset e Tabella: tre schede uguali, come in Analisi Portafoglio. */
  .rend-sez { background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important; padding: 5px 14px 5px 8px !important;
    margin: 0 0 8px 0 !important; width: 100% !important; box-sizing: border-box; }
  .rend-sez > .nb-espandi-btn { font-size: 12px !important; font-weight: 700 !important;
    color: #1a3a5c !important; padding: 4px 0 !important; border-bottom: none !important; }
  .rend-titolo { display: none !important; }
  .rend-scorri { max-height: 75vh; }
  .rend-nome { max-width: 104px; }
  .rend-th button { padding: 8px 8px !important; font-size: 11px !important; }
  .rend-th { min-width: 0 !important; }
  .rend-td { padding: 6px 7px !important; font-size: 11px !important; }
  .rend-td-nome { padding: 6px 7px !important; }
  /* Dentro il Portafoglio il pulsante sta nella riga in alto, accanto al file. */
  html.nb-embed #rend-update-btn { display: none !important; }
}
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>
// Larghezza della colonna dei nomi → left della colonna AKR (ferma accanto).
(function(){
  function misura(){
    document.querySelectorAll('.rend-scorri').forEach(function(sc){
      var th = sc.querySelector('th.rend-td-nome');
      if (th) sc.style.setProperty('--rend-nome-w', th.getBoundingClientRect().width + 'px');
    });
  }
  new MutationObserver(misura).observe(document.documentElement, {childList: true, subtree: true});
  window.addEventListener('resize', misura);
})();
// Menu ⚙: si chiude con un clic fuori (anche la copia aperta dal ⚙ del Portafoglio).
// Clic su un nome asset tagliato («…»): fumetto col nome intero, sparisce al clic dopo.
document.addEventListener('click', function(e){
  document.querySelectorAll('details.rend-menu[open]').forEach(function(d){
    if (!d.contains(e.target)) d.removeAttribute('open');
  });
  try { var pp = window.parent.document.getElementById('pf-info-pop'); if (pp) pp.remove(); } catch (err) {}
  var vecchio = document.getElementById('rend-nome-pop');
  if (vecchio) vecchio.remove();
  var n = e.target.closest && e.target.closest('.rend-nome');
  if (!n || n.scrollWidth <= n.clientWidth || vecchio && vecchio._nome === n) return;
  var pop = document.createElement('div');
  pop.id = 'rend-nome-pop'; pop._nome = n; pop.textContent = n.textContent;
  var r = n.getBoundingClientRect();
  pop.style.cssText = 'position:fixed;z-index:3000;left:' + Math.round(Math.max(4, r.left)) + 'px;' +
    'top:' + Math.round(r.bottom + 4) + 'px;max-width:' + Math.round(window.innerWidth - 16) + 'px;' +
    'background:#1a3a5c;color:#fff;font-size:12px;font-weight:600;padding:5px 9px;' +
    'border-radius:5px;box-shadow:0 4px 12px rgba(0,0,0,0.25);white-space:normal;';
  document.body.appendChild(pop);
});
// Lo scorrimento della tabella lo stacca dal nome: si toglie.
document.addEventListener('scroll', function(){
  var p = document.getElementById('rend-nome-pop'); if (p) p.remove();
}, true);
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

# ─── Layout ──────────────────────────────────────────────────────────────────
# Sezioni come in Analisi Portafoglio: «Parametri», «Asset» e «Tabella». Sul cellulare
# sono schede che si chiudono (nb-espandi, CSS e JS in navbar.py); Asset e Tabella
# sono aperte all'avvio, Parametri chiusa. Le larghezze delle colonne della griglia
# asset sono quelle di Analisi Portafoglio (_LARG_* in portafoglio/app.py).
_LARG_NOME, _LARG_CH, _LARG_PESO = '34%', '6%', '17%'
_COLONNE_ASSET = [   # larghezza, etichetta, colore, tooltip, id del pulsante ☑
    (_LARG_CH,   'CH', '#1a3a5c', 'Deseleziona tutti (o seleziona tutti se nessuno è spuntato)', 'rend-deselect-btn'),
    (_LARG_PESO, 'P1', '#e6194b', 'Azzera pesi P1 (se già a zero: pesi uguali sugli asset spuntati in CH)', 'rend-reset-p1-btn'),
    (_LARG_PESO, 'P2', '#3cb44b', 'Azzera pesi P2 (se già a zero: pesi uguali sugli asset spuntati in CH)', 'rend-reset-p2-btn'),
    (_LARG_PESO, 'P3', '#4363d8', 'Azzera pesi P3 (se già a zero: pesi uguali sugli asset spuntati in CH)', 'rend-reset-p3-btn'),
]
_BTN_PRINCIPALE = {'backgroundColor': '#1a3a6b', 'color': 'white', 'border': 'none',
                   'padding': '6px 14px', 'borderRadius': '4px', 'cursor': 'pointer',
                   'fontWeight': '600', 'fontSize': '12px', 'lineHeight': '18px',
                   'whiteSpace': 'nowrap'}
_ETIC = {'fontSize': '10px', 'whiteSpace': 'nowrap', 'marginRight': '4px'}
_CAMPO = {'display': 'flex', 'alignItems': 'center', 'marginRight': '10px'}
_INPUT = {'fontSize': '10px'}


def _etichette_colonne():
    """Asset | CH | P1 | P2 | P3 sopra la griglia, come in Analisi Portafoglio;
    i ☑ alternano (vedi toggle_all_checks e reset_pesi)."""
    return html.Div([
        html.Div('Asset', style={
            'width': _LARG_NOME, 'fontWeight': '700', 'fontSize': '11px',
            'paddingLeft': '4px', 'color': '#1a3a5c',
            'display': 'flex', 'alignItems': 'center', 'overflow': 'hidden',
        }),
        *[html.Div([
            html.Span(lbl, style={'fontWeight': '700', 'fontSize': '11px', 'color': col,
                                  'lineHeight': '1', 'whiteSpace': 'nowrap'}),
            html.Button('☑', id=btn_id, n_clicks=0, title=tip, style={
                'fontSize': '10px', 'border': 'none', 'background': 'none',
                'cursor': 'pointer', 'color': col,
                'padding': '0', 'margin': '0', 'lineHeight': '1'}),
        ], style={
            'width': w, 'display': 'flex', 'flexDirection': 'column',
            'alignItems': 'center', 'justifyContent': 'center', 'gap': '1px',
            'overflow': 'hidden',
        }) for w, lbl, col, tip, btn_id in _COLONNE_ASSET],
    ], style={
        'display': 'flex', 'alignItems': 'center', 'minHeight': '30px',
        'background': '#eaf4fb', 'borderBottom': '1px solid #aed6f1',
        'position': 'sticky', 'top': '0', 'zIndex': '2',
    })


def _testa_cell(titolo):
    """Intestazione di una sezione che si chiude solo sul cellulare (nascosta sul computer)."""
    return html.Div(className='nb-espandi-btn nb-testa-cell', children=[
        titolo,
        html.I(className='fa-solid fa-chevron-down nb-espandi-ico'),
    ], style={'fontSize': '12px', 'fontWeight': '700', 'color': '#1a3a5c',
              'padding': '6px 0 4px 4px', 'borderBottom': '1px solid #aed6f1'})


def _somma_pesi(id_):
    return html.Div(id=id_, children='0%', style={
        'width': _LARG_PESO, 'textAlign': 'right', 'paddingRight': '5px',
        'fontWeight': 'bold', 'color': '#d62728', 'fontSize': '10px',
        'boxSizing': 'border-box'})


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
    dcc.Store(id='rend-rf-store', data=_RF),
    dcc.Store(id='rend-akr-w-store',      data=_AKR_W),
    dcc.Store(id='rend-akr-ma-store',     data=_AKR_MA),
    dcc.Store(id='rend-akr-filter-store', data='all'),

    # Trigger caricamento dati al primo render (stesso pattern di frontiera)
    dcc.Store(id='rend-page-load', data=1),
    dcc.Store(id='rend-sync-sig', data=''),
    dcc.Interval(id='rend-live-sync', interval=2000, n_intervals=0, disabled=False),

    html.Div([
        html.H2('Rendimenti Storici per Periodo', className='rend-titolo', style={
            'fontSize': '1.1rem', 'fontWeight': '700', 'color': '#1a3a6b',
            'margin': '0 0 8px 0', 'whiteSpace': 'nowrap',
        }),

        # ── Parametri ──────────────────────────────────────────────────────
        html.Div(className='nb-espandi rend-sez', children=[
            html.Span(className='nb-espandi-btn', children=[
                'Parametri', html.Span(':', className='nb-nascondi-cell'),
                html.I(className='fa-solid fa-chevron-down nb-espandi-ico'),
            ], style={'marginRight': '10px', 'whiteSpace': 'nowrap', 'fontSize': '12px',
                      'fontWeight': '700', 'color': '#1a3a5c'}),
            html.Div(className='nb-espandi-corpo', children=[
                html.Div([
                    html.Label('Benchmark:', style=_ETIC),
                    dcc.Dropdown(id='rend-ir-bench', options=[], value=None,
                                 placeholder='Seleziona…', clearable=True,
                                 style={'width': '180px', 'fontSize': '10px',
                                        'minWidth': '180px'}),
                ], style=_CAMPO),
                html.Div([
                    html.Label('Risk-Free SR %:', style=_ETIC),
                    dcc.Input(id='rend-rf-input', type='number', value=_RF,
                              min=0, max=20, step=0.1, placeholder='es. 3.5',
                              style={**_INPUT, 'width': '55px'}),
                ], style=_CAMPO),
                html.Div([
                    html.Label('AKR W:', style=_ETIC),
                    dcc.Input(id='rend-akr-w', type='number', value=_AKR_W, min=5, step=1,
                              placeholder=str(_AKR_W), style={**_INPUT, 'width': '45px'}),
                ], style=_CAMPO),
                html.Div([
                    html.Label('MA:', style=_ETIC),
                    dcc.Input(id='rend-akr-ma', type='number', value=_AKR_MA, min=1, step=1,
                              placeholder=str(_AKR_MA), style={**_INPUT, 'width': '40px'}),
                ], style=_CAMPO),
                html.Div([
                    html.Label('Filtro AKR:', style={**_ETIC, 'fontWeight': '600'}),
                    dcc.RadioItems(
                        id='rend-akr-filter',
                        options=[
                            {'label': 'Tutti',  'value': 'all'},
                            {'label': '>−1',    'value': 'gt_minus1'},
                            {'label': '>0',     'value': 'gt_0'},
                        ],
                        value='all', inline=True,
                        inputStyle={'marginRight': '3px'},
                        labelStyle={'marginRight': '8px', 'fontSize': '10px',
                                    'cursor': 'pointer'},
                    ),
                ], style=_CAMPO),
                # Dentro il Portafoglio sul cellulare il pulsante sta nella riga in alto
                # (pf-crea-cell, che preme questo).
                html.Button('Crea tabella', id='rend-update-btn', n_clicks=0,
                            style=_BTN_PRINCIPALE),
                # ⚙ Descrizione dati tabella: numero asset, data dei dati, parametri, legenda.
                html.Details(className='rend-menu', children=[
                    html.Summary(html.I(className='fa-solid fa-gear'),
                                 title='Descrizione dati tabella',
                                 style={**_BTN_PRINCIPALE, 'padding': '6px 9px'}),
                    html.Div(className='rend-menu-corpo', children=[
                        html.Div('Descrizione dati tabella',
                                 style={'fontSize': '12px', 'fontWeight': '700', 'color': '#1a3a5c'}),
                        html.Div(id='rend-info-asset'),
                        html.Div('Premi «Crea tabella» per i dati della tabella.',
                                 id='rend-info-tabella', style={'color': '#888'}),
                    ]),
                ]),
            ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap',
                      'gap': '4px 0'}),
        ], style={'marginBottom': '8px', 'display': 'flex', 'alignItems': 'center',
                  'flexWrap': 'wrap', 'padding': '5px 8px', 'background': '#f8fafc',
                  'border': '1px solid #e2e8f0', 'borderRadius': '6px'}),

        # Solo per l'errore «Nessun dato disponibile»; vuota non occupa spazio.
        html.Div(id='rend-data-info', className='rend-info', style={
            'fontSize': '11px', 'color': '#c0392b',
        }),

        # ── Corpo: Asset (sx) + Tabella (dx); impilati sul cellulare ──────
        html.Div([
            html.Div([
                _testa_cell('Asset'),
                html.Div(className='nb-espandi-corpo', children=[
                    _etichette_colonne(),
                    html.Div(id='rend-asset-grid'),
                    html.Div([
                        html.Hr(style={'margin': '10px 0'}),
                        html.Div([
                            html.Div('Totale Pesi:', style={
                                'width': '40%', 'fontWeight': 'bold',
                                'paddingLeft': '4px', 'fontSize': '9px'}),
                            _somma_pesi('rend-sum-p1'),
                            _somma_pesi('rend-sum-p2'),
                            _somma_pesi('rend-sum-p3'),
                        ], style={'display': 'flex', 'alignItems': 'center'}),
                    ], style={'marginTop': '10px'}),
                ]),
            ], className='nb-piena nb-espandi nb-aperto rend-sez',
               style={'width': '35%', 'paddingRight': '12px', 'boxSizing': 'border-box'}),

            html.Div([
                _testa_cell('Tabella'),
                html.Div(className='nb-espandi-corpo', children=[
                    dcc.Loading(type='circle', children=[
                        html.Div(id='rend-perf-table'),
                    ]),
                ]),
            ], className='nb-piena nb-espandi nb-aperto rend-sez',
               style={'width': '65%', 'minWidth': '0'}),
        ], className='nb-impila', style={'display': 'flex', 'alignItems': 'flex-start'}),

    ], className='rend-corpo', style={
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
    # Niente riga «N asset · dati al…» (inutile per l'utente): la riga resta solo per l'errore.
    return (prices.to_json(orient='split', date_format='iso'),
            returns.to_json(orient='split', date_format='iso'),
            None, p1, p2, p3)


# Stile del nome asset nella griglia (come in Analisi Portafoglio): padding fisso,
# il filo rosso del filtro AKR (box-shadow inset) lo mette color_asset_names.
_RASSET_BASE = {
    **NOME_ASSET, 'overflow': 'hidden', 'whiteSpace': 'nowrap',
    'textOverflow': 'ellipsis', 'maxWidth': '100%', 'color': '#1a3a5c',
    'padding': '3px 6px', 'borderRadius': '5px', 'boxSizing': 'border-box',
    'boxShadow': 'none',
}
_ROSSO_AKR = '#c0392b'
_PORT_RIGHE = ('Port1', 'Port2', 'Port3')


def _chk(chk_id, valore, spuntato, w):
    return html.Div(
        dcc.Checklist(id=chk_id, options=[{'label': '', 'value': valore}],
                      value=[valore] if spuntato else [],
                      inputStyle={'width': '10px', 'height': '10px',
                                  'cursor': 'pointer', 'margin': '0'},
                      style={'display': 'flex', 'justifyContent': 'center',
                             'alignItems': 'center', 'width': '100%'}),
        style={'width': w, 'height': '22px', 'display': 'flex',
               'alignItems': 'center', 'justifyContent': 'center'})


# ─── Parametri iniziali presi dalle altre dashboard ───────────────────────────
# Benchmark, AKR W, MA e Filtro AKR = gli ultimi usati in Analisi Portafoglio, risk free =
# l'ultimo della Frontiera (li scrivono nel localStorage, chiavi «par-*»). Mancano: stessi
# predefiniti (benchmark per file come _PF_BENCH_PREDEFINITI, rf 2%, 100/10, Tutti).
# Una volta per pagina: le opzioni si riscrivono a ogni aggiornamento dei dati e non devono
# cancellare quello che l'utente cambia qui.
app.clientside_callback(
    """function(opzioni) {
        var nu = window.dash_clientside.no_update;
        if (!opzioni || !opzioni.length || window._rendParametri) return [nu, nu, nu, nu, nu];
        window._rendParametri = true;
        function leggi(k) {
            try { var v = localStorage.getItem('par-' + k); return v === null ? null : JSON.parse(v); }
            catch (e) { return null; }
        }
        var valori = opzioni.map(function(o) { return o.value; });
        var bench = leggi('bench');
        if (valori.indexOf(bench) < 0) {
            bench = null;
            ['Az. ACWI', 'Bitcoin USD', 'EUR/USD', 'Crude Oil'].some(function(n) {
                var t = valori.find(function(v) { return String(v).indexOf(n) === 0; });
                if (t) { bench = t; return true; }
                return false;
            });
            if (!bench) bench = valori[0];
        }
        var rf = leggi('rf'), w = leggi('akr_w'), ma = leggi('akr_ma'), f = leggi('filtro');
        return [bench,
                (typeof rf === 'number') ? rf : %s,
                (typeof w === 'number' && w >= 5) ? w : %d,
                (typeof ma === 'number' && ma >= 1) ? ma : %d,
                (['all', 'gt_minus1', 'gt_0'].indexOf(f) >= 0) ? f : 'all'];
    }""" % (_RF, _AKR_W, _AKR_MA),
    Output('rend-ir-bench', 'value'),
    Output('rend-rf-input', 'value'),
    Output('rend-akr-w', 'value'),
    Output('rend-akr-ma', 'value'),
    Output('rend-akr-filter', 'value'),
    Input('rend-ir-bench', 'options'),
)


# ─── Menu ⚙: numero asset del file e quanti spuntati ─────────────────────────
@app.callback(
    Output('rend-info-asset', 'children'),
    Input('rend-stock-data', 'data'),
    Input('rend-selected', 'data'),
)
def info_asset(stock_json, selected):
    if not stock_json:
        return ''
    try:
        n = len(_get_df(stock_json).columns)
    except Exception:
        return ''
    sel = [x for x in (selected or []) if x not in _PORT_RIGHE]
    return html.Div([html.B('Numero asset: '), f'{n}', html.Span(f' · selezionati {len(sel)}',
                                                               style={'color': '#666'})])


# ─── Callback 2: Costruisce la griglia asset ──────────────────────────────────
@app.callback(
    Output('rend-asset-grid', 'children'),
    Output('rend-ir-bench', 'options'),
    Output('rend-selected', 'data', allow_duplicate=True),
    Input('rend-stock-data', 'data'),
    State('rend-selected', 'data'),
    State('rend-weights-p1', 'data'),
    State('rend-weights-p2', 'data'),
    State('rend-weights-p3', 'data'),
    prevent_initial_call='initial_duplicate',
)
def build_asset_grid(stock_json, selected, p1, p2, p3):
    if not stock_json:
        return html.Div('Caricamento dati in corso…',
                        style={'padding': '12px', 'color': '#888', 'fontSize': '12px'}), [], no_update
    try:
        returns = _get_df(stock_json)
    except Exception as _e:
        print(f'[rendimenti] build_asset_grid error: {traceback.format_exc()}')
        return html.Div(f'Errore nel caricamento dei dati: {type(_e).__name__}: {_e}',
                        style={'padding': '12px', 'color': '#c0392b', 'fontSize': '12px'}), [], no_update

    asset_names = list(returns.columns)
    # Come in Analisi Portafoglio: all'apertura le spunte salvate in current.json.
    if not selected:
        try:
            selected = [d for d, v in _read_user_json().items()
                        if not d.startswith('_') and isinstance(v, dict) and v.get('checked')]
        except Exception:
            selected = []
    selected = set(selected or [])
    pesi = {1: p1 or {}, 2: p2 or {}, 3: p3 or {}}

    rows = []
    for asset in asset_names:
        def peso(p_idx, a=asset):
            return html.Div(dcc.Input(
                id={'type': 'rend-weight', 'index': f'P{p_idx}-{a}'},
                type='number', value=pesi[p_idx].get(a, 0), min=0, max=100, step=0.1,
                placeholder='0',
                style={'width': '96%', 'textAlign': 'right', 'fontSize': '11px',
                       'height': '20px', 'padding': '1px 3px', 'boxSizing': 'border-box'},
            ), style={'width': _LARG_PESO})

        rows.append(html.Div([
            html.Div(
                html.Span(asset, id={'type': 'rend-asset-name', 'index': asset},
                          style=_RASSET_BASE),
                title=asset,
                style={'width': _LARG_NOME, 'height': '28px', 'display': 'flex',
                       'alignItems': 'center', 'overflow': 'hidden'},
            ),
            _chk({'type': 'rend-check', 'index': asset}, asset, asset in selected, _LARG_CH),
            peso(1), peso(2), peso(3),
        ], style={'display': 'flex', 'alignItems': 'center',
                  'borderBottom': '1px dotted #eee'}))

    # Righe dei portafogli: si spuntano per metterli in tabella, i pesi stanno sopra.
    for p_name in _PORT_RIGHE:
        rows.append(html.Div([
            html.Div(html.Span(p_name, style={
                'color': '#0066cc', **NOME_ASSET, 'overflow': 'hidden',
                'whiteSpace': 'nowrap', 'textOverflow': 'ellipsis', 'maxWidth': '100%'}),
                style={'width': _LARG_NOME, 'height': '28px', 'display': 'flex',
                       'alignItems': 'center', 'paddingLeft': '4px', 'overflow': 'hidden'}),
            _chk({'type': 'rend-check', 'index': p_name}, p_name, p_name in selected, _LARG_CH),
            html.Div('', style={'width': _LARG_PESO}),
            html.Div('', style={'width': _LARG_PESO}),
            html.Div('', style={'width': _LARG_PESO}),
        ], style={'display': 'flex', 'alignItems': 'center',
                  'borderBottom': '1px dotted #eee', 'backgroundColor': '#f0f0f0'}))

    # Opzioni benchmark: asset + portafogli configurati
    bench_options = [{'label': a, 'value': a} for a in asset_names]
    for p_num in (1, 2, 3):
        if any(v and v > 0 for v in pesi[p_num].values()):
            bench_options.append({'label': f'Port{p_num}', 'value': f'Port{p_num}'})

    # le spunte disegnate diventano la selezione (collect_selected parte solo ai clic)
    return rows, bench_options, [a for a in [*asset_names, *_PORT_RIGHE] if a in selected]


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
    base = {'width': _LARG_PESO, 'textAlign': 'right', 'paddingRight': '5px',
            'fontSize': '10px', 'fontWeight': 'bold', 'boxSizing': 'border-box'}
    def fmt(w_dict):
        total = sum(v for v in (w_dict or {}).values() if v) if w_dict else 0
        color = '#2ca02c' if 99 <= total <= 101 else '#d62728'
        return f'{total:.1f}%', {**base, 'color': color}
    t1, s1 = fmt(p1)
    t2, s2 = fmt(p2)
    t3, s3 = fmt(p3)
    return t1, s1, t2, s2, t3, s3


# ─── Callback 6: ☑ CH — se qualcosa è spuntato toglie tutto, altrimenti spunta
#     tutti gli asset (non le righe Port1–3), come in Analisi Portafoglio ────────
@app.callback(
    Output({'type': 'rend-check', 'index': ALL}, 'value'),
    Input('rend-deselect-btn', 'n_clicks'),
    State({'type': 'rend-check', 'index': ALL}, 'value'),
    State({'type': 'rend-check', 'index': ALL}, 'id'),
    prevent_initial_call=True,
)
def toggle_all_checks(n, current_values, ids):
    if not ids:
        raise PreventUpdate
    if any(v for v in current_values):
        return [[] for _ in ids]
    return [[] if d['index'] in _PORT_RIGHE else [d['index']] for d in ids]


# ─── Callback 6b: ☑ P1/P2/P3 — se la colonna ha pesi li azzera, se è tutta a zero
#     dà pesi uguali agli asset spuntati in CH (somma esatta 100) ──────────────
def _equipesa(n):
    """n pesi uguali che sommano esattamente 100 (passo 0,1): lo scarto si distribuisce
    un decimo per asset sui primi. Stessa regola di _pf_equipesa nel Portafoglio."""
    q, r = divmod(1000, n)
    return [(q + 1) / 10 if k < r else q / 10 for k in range(n)]


@app.callback(
    Output({'type': 'rend-weight', 'index': ALL}, 'value'),
    Input('rend-reset-p1-btn', 'n_clicks'),
    Input('rend-reset-p2-btn', 'n_clicks'),
    Input('rend-reset-p3-btn', 'n_clicks'),
    State({'type': 'rend-weight', 'index': ALL}, 'id'),
    State({'type': 'rend-weight', 'index': ALL}, 'value'),
    State('rend-selected', 'data'),
    prevent_initial_call=True,
)
def reset_pesi(n1, n2, n3, ids, valori, selected):
    porta = {'rend-reset-p1-btn': 'P1', 'rend-reset-p2-btn': 'P2',
             'rend-reset-p3-btn': 'P3'}.get(callback_context.triggered_id)
    if not porta or not ids:
        raise PreventUpdate
    mie = [k for k, d in enumerate(ids) if d['index'].split('-', 1)[0] == porta]
    out = [no_update] * len(ids)
    if any(valori[k] for k in mie):
        for k in mie:
            out[k] = 0
        return out
    spuntati = set(selected or []) - set(_PORT_RIGHE)
    scelti = [k for k in mie if ids[k]['index'].split('-', 1)[1] in spuntati]
    if not scelti:
        raise PreventUpdate
    for k, w in zip(scelti, _equipesa(len(scelti))):
        out[k] = w
    return out


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
    return max(5, int(val)) if val else _AKR_W

@app.callback(
    Output('rend-akr-ma-store', 'data'),
    Input('rend-akr-ma', 'value'),
    prevent_initial_call=True,
)
def relay_akr_ma(val):
    return max(1, int(val)) if val else _AKR_MA

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
        if close_returns is not None:
            close_returns = close_returns.clip(-_MAX_DAILY_RET, _MAX_DAILY_RET)   # tick corrotti
        original_prices = _get_df(prices_json)
    except Exception as e:
        print(f'[rendimenti] compute_performance error: {e}')
        raise PreventUpdate

    akr_window = max(5, int(akr_window or _AKR_W))
    akr_ma     = max(1, int(akr_ma     or _AKR_MA))

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
    Output('rend-info-tabella', 'children'),
    Input('rend-perf-data', 'data'),
    Input('rend-sort-state', 'data'),
    Input('rend-akr-filter-store', 'data'),
    prevent_initial_call=True,
)
def render_table(perf_data, sort_state, akr_filter):
    if not perf_data or not perf_data.get('rows'):
        return html.Div(
            'Seleziona gli asset, configura i pesi e clicca "Crea tabella".',
            style={'padding': '20px', 'color': '#666', 'fontSize': '13px'},
        ), no_update

    rows_data = perf_data['rows']
    last_date  = perf_data.get('last_date', 'N/D')
    akr_window = perf_data.get('akr_window', 30)
    akr_ma     = perf_data.get('akr_ma', 1)
    benchmark  = perf_data.get('benchmark', '')
    akr_filter = akr_filter or 'all'

    # Filtro AKR (solo asset, mai portafogli): NON nasconde le righe, ma
    # evidenzia in rosso gli asset che soddisfano la soglia (AKR > threshold).
    threshold = None
    if akr_filter == 'gt_minus1':
        threshold = -1.0
    elif akr_filter == 'gt_0':
        threshold = 0.0

    akr_highlight = set()
    if threshold is not None:
        for row in rows_data:
            if row.get('is_portfolio'):
                continue
            akr_val = row.get('AKR')
            if akr_val is not None and not (isinstance(akr_val, float) and np.isnan(akr_val)):
                if akr_val > threshold:
                    akr_highlight.add(row['name'])

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
            # angolo in alto a sinistra: fermo sia scorrendo in giù sia di lato
            th_style.update({'textAlign': 'left', 'minWidth': '0', 'left': '0', 'zIndex': '3'})
        elif col_id == 'AKR':
            # ferma accanto ai nomi (left dalla classe rend-akr), ombra sul suo bordo destro
            th_style.update({'left': '0', 'zIndex': '3', 'minWidth': '0',
                             'boxShadow': '2px 0 3px rgba(0,0,0,0.12)'})

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
                className='rend-th rend-akr' if col_id == 'AKR' else 'rend-th', style=th_style,
            )
        return html.Th(label, className='rend-th rend-td-nome',
                       style={**th_style, 'padding': '10px 14px', 'fontSize': '12px'})

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

        is_akr_hit = (not is_portfolio) and (item in akr_highlight)
        name_style = {
            'padding': '8px 10px', 'fontSize': '11px',
            'fontWeight': 'bold' if (is_portfolio or is_akr_hit) else 'normal',
            'color': '#0066cc' if is_portfolio else ('#ffffff' if is_akr_hit else '#222'),
            'backgroundColor': '#e53935' if is_akr_hit else row_bg,
            'position': 'sticky', 'left': '0', 'zIndex': '1',
            'border': '1px solid #ddd', 'whiteSpace': 'nowrap',
        }
        # colonna ferma: il nome lungo si accorcia («…») e il nome intero è nel title
        cells = [html.Td(html.Div(item, className='rend-nome', title=item),
                         className='rend-td-nome', style=name_style)]

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
        # ferma accanto ai nomi: sfondo pieno (già dato sopra), left dalla classe rend-akr
        akr_cs.update({'position': 'sticky', 'left': '0', 'zIndex': '1',
                       'boxShadow': '2px 0 3px rgba(0,0,0,0.12)'})
        cells.append(html.Td(akr_txt, className='rend-td rend-akr', style=akr_cs))

        for period in ret_cols:
            val = row.get(period)
            cs = get_cell_style(val, is_portfolio=is_portfolio)
            if sort_col == period and val is not None and not (isinstance(val, float) and np.isnan(val)):
                cs['backgroundColor'] = '#d0edce' if val >= 0 else '#ffdde0'
            cells.append(html.Td(format_return(val), className='rend-td', style=cs))

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
            cells.append(html.Td(txt, className='rend-td', style=cs))

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
            cells.append(html.Td(txt, className='rend-td', style=cs))

        table_rows.append(html.Tr(cells, style={'backgroundColor': row_bg}))

    # border-collapse separate: con collapse i bordi della colonna ferma scorrerebbero via
    table = html.Table(table_rows, style={
        'borderCollapse': 'separate', 'borderSpacing': '0', 'width': '100%',
    })

    # Descrizione della tabella nel menu ⚙ accanto a Crea tabella.
    info = [html.Div([html.B('Dati al: '), last_date])]
    if benchmark:
        info.append(html.Div([html.B('Benchmark: '), f'{benchmark} · AK-W: {akr_window}'
                              + (f' · MA: {akr_ma}' if akr_ma > 1 else '')],
                             style={'color': '#4a1a7c'}))
    info.append(html.Div('Verde = positivo · Rosso = negativo · Clicca intestazione per ordinare',
                         style={'color': '#555'}))
    return html.Div(table, className='rend-scorri'), info


# ─── Colora di rosso i nomi asset nella griglia di sinistra se passano AKR ───
@app.callback(
    Output({'type': 'rend-asset-name', 'index': ALL}, 'style'),
    Input('rend-perf-data',        'data'),
    Input('rend-akr-filter-store', 'data'),
    State({'type': 'rend-asset-name', 'index': ALL}, 'id'),
    prevent_initial_call=True,
)
def color_asset_names(perf_data, akr_filter, ids):
    ids = ids or []
    threshold = (-1.0 if akr_filter == 'gt_minus1'
                 else 0.0 if akr_filter == 'gt_0' else None)
    akr_map = {}
    if perf_data and perf_data.get('rows'):
        for r in perf_data['rows']:
            if not r.get('is_portfolio'):
                akr_map[r['name']] = r.get('AKR')
    out = []
    for d in ids:
        hit = False
        if threshold is not None:
            v = akr_map.get(d['index'])
            if v is not None and not (isinstance(v, float) and np.isnan(v)) and v > threshold:
                hit = True
        out.append({**_RASSET_BASE,
                    'boxShadow': f'inset 0 0 0 1.5px {_ROSSO_AKR}' if hit else 'none'})
    return out


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
    return (op.to_json(orient='split', date_format='iso'),
            cr.to_json(orient='split', date_format='iso'),
            None, new_sig)
