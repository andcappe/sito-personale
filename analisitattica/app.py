"""
Analisi Tattica — App standalone (sezione di menu /analisitattica/).

Contiene l'Analisi ARIMA per singolo asset (estratta da ir_fe_14.py):
  log(P) → detrend lineare → grid-search ARIMA(p,1,q) → residui ε_t
  → test ADF → ACF/PACF → GARCH(1,1) → forecast con cono di confidenza 95%.

I dati (prezzi/rendimenti) sono letti da current.json — la fonte di verità
UNICA per utente, condivisa con tutte le altre sezioni del sito.
"""
import os
import io
import json
import time
import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update, ALL
from dash.exceptions import PreventUpdate

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.browser_css import SITE_CSS            # noqa: E402  (CSS unico del sito)
from navbar import make_navbar                       # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
_EXT = [
    'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=_EXT,
           requests_pathname_prefix='/analisitattica/',
           routes_pathname_prefix='/analisitattica/')
app.title = 'Analisi Tattica — Andrea Cappelletti'
server = app.server

app.index_string = '''<!DOCTYPE html><html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>''' + SITE_CSS + '''</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

_ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent


# ─────────────────────────────────────────────────────────────────────────────
# Dati — TUTTA la logica vive nel modulo CONDIVISO data_core (un solo posto,
# richiamato anche da Portafoglio). Qui solo alias brevi per le callback.
# ─────────────────────────────────────────────────────────────────────────────
import data_core as dc                       # noqa: E402

_get_username         = dc.get_username
_read_current         = dc.read_current
_write_current        = dc.write_current
_asset_options        = dc.asset_options
_build_prices         = dc.build_prices
_cloud_push           = dc.cloud_push
_fx_series            = dc.fx_series
_download_series      = dc.download_series
_add_asset_to_current = dc.add_asset_to_current
_template_bytes       = dc.template_bytes
_export_bytes         = dc.export_bytes
_read_analyses        = dc.read_analyses
_profili_dir          = dc.profili_dir
_list_profili         = dc.list_profili
_save_profilo         = dc.save_profilo
_load_profilo         = dc.load_profilo
_delete_profilo       = dc.delete_profilo


def _quick_forecast(prices):
    """
    Previsione ARIMA leggera a 1 giorno per la tabella riassuntiva.
    Usa ARIMA(1,1,1) sulla ciclicità (log-prezzo detrendizzato) + intervallo di
    confidenza 95% come banda di volatilità. Ritorna prezzo previsto, min/max e
    variazioni % (media/min/max), oppure None se non calcolabile.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
        prices = prices.dropna()
        if len(prices) < 60:
            return None
        if len(prices) > 500:
            prices = prices.iloc[-500:]
        log_p = np.log(prices.values.astype(float))
        x     = np.arange(len(log_p), dtype=float)
        coef  = np.polyfit(x, log_p, 1)
        detr  = log_p - np.polyval(coef, x)
        m     = ARIMAModel(detr, order=(1, 1, 1)).fit()
        fc    = m.get_forecast(steps=1)
        mean_det = float(np.array(fc.predicted_mean)[0])
        ci       = np.array(fc.conf_int(alpha=0.05))[0]
        lo_det, hi_det = float(ci[0]), float(ci[1])
        trend_next = float(np.polyval(coef, len(log_p)))
        prev = float(np.exp(mean_det + trend_next))
        pmin = float(np.exp(lo_det  + trend_next))
        pmax = float(np.exp(hi_det  + trend_next))
        last = float(prices.iloc[-1])
        return {
            'prev': prev, 'min': pmin, 'max': pmax,
            'dpm':  (prev / last - 1) * 100,
            'dmin': (pmin / last - 1) * 100,
            'dmax': (pmax / last - 1) * 100,
        }
    except Exception:
        return None


def _render_file_list(username=None):
    items = _list_profili(username)
    if not items:
        return html.Div('Nessun lavoro salvato.',
                        style={'font-size': '10px', 'color': '#888', 'padding': '6px'})
    _ib = {'border': 'none', 'border-radius': '4px', 'cursor': 'pointer',
           'font-size': '11px', 'padding': '3px 7px', 'margin-left': '4px'}
    rows = []
    for it in items:
        rows.append(html.Div([
            html.Span(it['label'], style={'flex': '1', 'font-size': '10px',
                                          'overflow': 'hidden', 'white-space': 'nowrap'}),
            html.Span(f"{it['kb']} KB", style={'font-size': '9px', 'color': '#999', 'margin': '0 6px'}),
            html.Button('📂', id={'type': 'at-fp-load', 'index': it['name']}, n_clicks=0,
                        title='Carica questo lavoro',
                        style={**_ib, 'background': '#2e6da4', 'color': 'white'}),
            html.Button('🗑', id={'type': 'at-fp-del', 'index': it['name']}, n_clicks=0,
                        title='Elimina', style={**_ib, 'background': '#c0392b', 'color': 'white'}),
        ], style={'display': 'flex', 'align-items': 'center',
                  'border-bottom': '1px dotted #eee', 'padding': '4px 0'}))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
# Larghezze colonne (devono sommare ~100%)
_COLW = {'asset': '24%', 'chk': '8%', 'prev': '13%', 'min': '12%',
         'max': '12%', 'dpm': '11%', 'dmin': '10%', 'dmax': '10%'}


def _cell(text, w, *, bold=False, color='#333', align='right', title=None):
    st = {'width': w, 'height': '28px', 'display': 'flex', 'align-items': 'center',
          'justify-content': ('center' if align == 'center' else
                              ('flex-start' if align == 'left' else 'flex-end')),
          'font-size': '8px', 'overflow': 'hidden', 'white-space': 'nowrap',
          'padding': '0 3px', 'color': color}
    if bold:
        st['font-weight'] = 'bold'
    return html.Div(text, style=st, title=title)


def _build_asset_grid(asset_names, selected=None, forecasts=None):
    """
    Griglia asset di sinistra con checkbox + (se forecasts presente) colonne di
    previsione a 1 giorno: prezzo previsto, min/max (volatilità) e Δ% med/min/max.
    """
    selected  = selected or []
    forecasts = forecasts or {}
    if not asset_names:
        return html.Div("Nessun asset: carica i dati in Analisi di Portafoglio.",
                        style={'color': '#888', 'padding': '12px', 'font-size': '11px'})

    def _hcell(text, w, title=None):
        return html.Div(text, title=title, style={
            'width': w, 'font-weight': 'bold', 'font-size': '8px',
            'text-align': 'center', 'padding': '0 2px', 'white-space': 'nowrap',
            'overflow': 'hidden'})

    header = html.Div([
        html.Div('Asset', style={'width': _COLW['asset'], 'font-weight': 'bold',
                                  'font-size': '9px', 'padding-left': '4px'}),
        html.Div(html.Button('Des', id='deselect-all-arima-tab', n_clicks=0,
                             style={'font-size': '7px', 'padding': '1px 3px', 'width': '95%'}),
                 style={'width': _COLW['chk'], 'text-align': 'center'}),
        _hcell('Prev.',  _COLW['prev'], 'Prezzo previsto giorno successivo'),
        _hcell('Min',    _COLW['min'],  'Minimo (volatilità, IC 95%)'),
        _hcell('Max',    _COLW['max'],  'Massimo (volatilità, IC 95%)'),
        _hcell('Δ%',     _COLW['dpm'],  'Variazione % media attesa'),
        _hcell('Δ%min',  _COLW['dmin'], 'Variazione % minima'),
        _hcell('Δ%max',  _COLW['dmax'], 'Variazione % massima'),
    ], style={'display': 'flex', 'margin-bottom': '5px',
              'border-bottom': '2px solid #ccc', 'padding-bottom': '4px', 'align-items': 'center'})

    rows = [header]
    for asset in asset_names:
        asset_val = [asset] if asset in selected else []
        fc = forecasts.get(asset)
        if fc:
            pcol = '#1b7a34' if fc['dpm'] >= 0 else '#c0392b'
            cells = [
                _cell(f"{fc['prev']:.2f}", _COLW['prev'], bold=True),
                _cell(f"{fc['min']:.2f}",  _COLW['min'],  color='#666'),
                _cell(f"{fc['max']:.2f}",  _COLW['max'],  color='#666'),
                _cell(f"{fc['dpm']:+.1f}", _COLW['dpm'],  bold=True, color=pcol),
                _cell(f"{fc['dmin']:+.1f}", _COLW['dmin'], color='#c0392b'),
                _cell(f"{fc['dmax']:+.1f}", _COLW['dmax'], color='#1b7a34'),
            ]
        else:
            cells = [_cell('—', _COLW[k], color='#bbb', align='center')
                     for k in ('prev', 'min', 'max', 'dpm', 'dmin', 'dmax')]
        rows.append(html.Div([
            html.Div(html.B(asset), style={
                'width': _COLW['asset'], 'height': '28px', 'display': 'flex',
                'align-items': 'center', 'padding-left': '4px',
                'font-size': '8px', 'overflow': 'hidden', 'white-space': 'nowrap'}),
            html.Div(dcc.Checklist(
                id={'type': 'graph-select-checkbox-arima', 'index': asset},
                options=[{'label': '', 'value': asset}], value=asset_val,
                style={'justify-content': 'center'}),
                style={'width': _COLW['chk'], 'height': '28px', 'display': 'flex',
                       'align-items': 'center', 'justify-content': 'center'}),
            *cells,
        ], style={'display': 'flex', 'border-bottom': '1px dotted #eee', 'align-items': 'center'}))

    # ── Riga "Aggiungi asset" IN CODA (come in Analisi di Portafoglio) ────────
    _ai = {'font-size': '9px', 'padding': '4px 6px', 'border': '1px solid #aaa',
           'border-radius': '4px'}
    rows.append(html.Div([
        html.Div('➕ Aggiungi asset (in coda)',
                 style={'font-size': '9px', 'font-weight': '700', 'color': '#1a3a5c',
                        'margin-bottom': '4px'}),
        html.Div([
            dcc.Input(id='at-add-desc', placeholder='descrizione',
                      style={**_ai, 'width': '40%'}),
            dcc.Input(id='at-add-ticker', placeholder='ticker',
                      style={**_ai, 'width': '28%'}),
            dcc.Input(id='at-add-cur', value='EUR',
                      style={**_ai, 'width': '20%'}),
        ], style={'display': 'flex', 'gap': '3px', 'margin-bottom': '4px'}),
        html.Button('➕ Aggiungi', id='at-add-btn', n_clicks=0,
                    style={'width': '100%', 'font-size': '10px', 'padding': '5px',
                           'border': 'none', 'border-radius': '4px', 'cursor': 'pointer',
                           'color': 'white', 'background': '#1b7a34', 'font-weight': 'bold'}),
    ], style={'margin-top': '10px', 'border-top': '2px solid #ccc', 'padding-top': '8px'}))
    return rows


def get_arima_analysis_tab(options_tickers):
    """Layout dell'Analisi ARIMA per singolo asset."""
    asset_names = [o['value'] for o in (options_tickers or [])]
    first = asset_names[0] if asset_names else None
    return html.Div([
        # ── Header con controlli ──────────────────────────────────────────
        html.Div([
            html.H3('Analisi ARIMA per Asset',
                    style={'margin-right': '20px', 'white-space': 'nowrap', 'font-size': '16px'}),
            html.Div([
                html.Label("Asset:", style={'margin-right': '6px', 'font-size': '11px',
                                             'white-space': 'nowrap'}),
                dcc.Dropdown(id='arima-asset-dropdown', options=options_tickers or [],
                             value=first, placeholder="Seleziona asset…",
                             style={'width': '230px', 'font-size': '11px'}),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '12px'}),
            html.Div([
                html.Label("Orizzonte (gg):", style={'margin-right': '5px', 'font-size': '11px',
                                                      'white-space': 'nowrap'}),
                dcc.Input(id='arima-tab-horizon', type='number', value=30,
                          min=5, max=252, step=5, style={'width': '55px'}),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '12px'}),
            html.Div([
                html.Label("Max p:", style={'margin-right': '5px', 'font-size': '11px'}),
                dcc.Input(id='arima-tab-max-p', type='number', value=4,
                          min=1, max=8, style={'width': '42px'}),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '8px'}),
            html.Div([
                html.Label("Max q:", style={'margin-right': '5px', 'font-size': '11px'}),
                dcc.Input(id='arima-tab-max-q', type='number', value=4,
                          min=1, max=8, style={'width': '42px'}),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '12px'}),
            html.Div([
                html.Label("Criterio:", style={'font-size': '11px', 'margin-right': '6px',
                                               'white-space': 'nowrap'}),
                dcc.RadioItems(id='arima-tab-criterion',
                               options=[{'label': ' AIC', 'value': 'aic'},
                                        {'label': ' BIC', 'value': 'bic'}],
                               value='aic', inline=True, style={'font-size': '11px'},
                               inputStyle={'margin-right': '4px'},
                               labelStyle={'margin-right': '10px'}),
            ], style={'display': 'flex', 'align-items': 'center', 'border': '1px solid #ccc',
                      'border-radius': '4px', 'padding': '3px 8px', 'margin-right': '12px',
                      'background': '#f5f5f5'}),
            html.Button('▶  Esegui Analisi ARIMA', id='run-arima-tab-button', n_clicks=0,
                        style={'background-color': '#0066cc', 'color': 'white', 'border': 'none',
                               'padding': '8px 18px', 'border-radius': '4px', 'cursor': 'pointer',
                               'font-weight': 'bold', 'font-size': '12px'}),
            html.Button('📊 Previsioni 1gg (tutti)', id='calc-arima-forecasts-btn', n_clicks=0,
                        title='Calcola la previsione a 1 giorno per TUTTI gli asset '
                              '(ARIMA(1,1,1) + IC 95%); può richiedere qualche secondo',
                        style={'background-color': '#1b7a34', 'color': 'white', 'border': 'none',
                               'padding': '8px 14px', 'border-radius': '4px', 'cursor': 'pointer',
                               'font-weight': 'bold', 'font-size': '12px', 'margin-left': '8px'}),
            html.Span(id='arima-forecast-status',
                      style={'font-size': '10px', 'color': '#555', 'margin-left': '8px'}),
        ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap',
                  'gap': '4px', 'margin-bottom': '10px'}),

        html.Hr(style={'margin': '8px 0'}),

        # ── Body ─────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Loading(type='circle', color='#1b7a34', children=[
                    html.Div(id='weights-grid-container-arima',
                             children=_build_asset_grid(asset_names, [first] if first else [])),
                ]),
            ], style={'width': '46%', 'vertical-align': 'top', 'padding-right': '10px',
                      'border-right': '1px solid #eee', 'overflow-y': 'auto', 'max-height': '90vh'}),
            html.Div([
                html.Div(id='arima-tab-status',
                         style={'font-size': '11px', 'color': '#333', 'padding': '5px 10px',
                                'background': '#f0f8ff', 'border': '1px solid #bee3f8',
                                'border-radius': '4px', 'margin-bottom': '8px',
                                'white-space': 'pre-wrap', 'font-family': 'monospace'}),
                dcc.Loading(id='loading-arima-tab', type='circle', children=[
                    dcc.Graph(id='arima-analysis-chart',
                              style={'width': '100%', 'height': '92vh'},
                              config={'responsive': True, 'scrollZoom': True}),
                ]),
            ], style={'width': '54%', 'vertical-align': 'top', 'padding-left': '10px'}),
        ], style={'display': 'flex'}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Posizionamento COT (Commitments of Traders, CFTC)
#
# Fonte: CFTC Public Reporting (Socrata Open Data API), aggiornato ogni venerdì.
#   • TFF  (Traders in Financial Futures) — indici azionari, valute, tassi:
#       Asset Manager / Institutional  = fondi pensione, comuni, assicurazioni
#                                        (posizioni medio/lungo o di copertura)
#       Leveraged Money                = hedge fund, CTA, gestori speculativi
#                                        (posizioni attive e direzionali)
#   • Disaggregated                       — materie prime (oro, petrolio, agricoli):
#       Managed Money                  = speculazione gestita da hedge fund
# Posizione NETTA = contratti long − short di ciascuna categoria.
# ─────────────────────────────────────────────────────────────────────────────
_COT_TFF_URL = 'https://publicreporting.cftc.gov/resource/gpe5-46if.json'
_COT_DIS_URL = 'https://publicreporting.cftc.gov/resource/72hh-3qpy.json'

# chiave interna → (categoria, etichetta IT, dataset, contract_market_name esatto)
_COT_INSTRUMENTS = {
    # ── Indici azionari (TFF) ─────────────────────────────────────────────
    'sp500':   ('Indici azionari', 'S&P 500',      'tff', 'E-MINI S&P 500'),
    'nasdaq':  ('Indici azionari', 'Nasdaq 100',   'tff', 'NASDAQ MINI'),
    'russell': ('Indici azionari', 'Russell 2000', 'tff', 'RUSSELL E-MINI'),
    # ── Valute (TFF) ──────────────────────────────────────────────────────
    'eur': ('Valute', 'Euro FX',             'tff', 'EURO FX'),
    'jpy': ('Valute', 'Yen giapponese',      'tff', 'JAPANESE YEN'),
    'gbp': ('Valute', 'Sterlina britannica', 'tff', 'BRITISH POUND'),
    'chf': ('Valute', 'Franco svizzero',     'tff', 'SWISS FRANC'),
    'cad': ('Valute', 'Dollaro canadese',    'tff', 'CANADIAN DOLLAR'),
    'aud': ('Valute', 'Dollaro australiano', 'tff', 'AUSTRALIAN DOLLAR'),
    # ── Tassi d'interesse (TFF) ───────────────────────────────────────────
    'ust2':  ('Tassi', 'Treasury 2 anni',   'tff', 'UST 2Y NOTE'),
    'ust5':  ('Tassi', 'Treasury 5 anni',   'tff', 'UST 5Y NOTE'),
    'ust10': ('Tassi', 'Treasury 10 anni',  'tff', 'UST 10Y NOTE'),
    'ustb':  ('Tassi', 'Treasury Bond 30a', 'tff', 'UST BOND'),
    'ff':    ('Tassi', 'Fed Funds 30gg',    'tff', 'FED FUNDS'),
    # ── Materie prime (Disaggregated → Managed Money) ─────────────────────
    'gold':     ('Materie prime', 'Oro',          'dis', 'GOLD'),
    'silver':   ('Materie prime', 'Argento',      'dis', 'SILVER'),
    'copper':   ('Materie prime', 'Rame',         'dis', 'COPPER- #1'),
    'wti':      ('Materie prime', 'Petrolio WTI', 'dis', 'CRUDE OIL, LIGHT SWEET-WTI'),
    'natgas':   ('Materie prime', 'Gas naturale', 'dis', 'NAT GAS NYME'),
    'platinum': ('Materie prime', 'Platino',      'dis', 'PLATINUM'),
    'corn':     ('Materie prime', 'Mais',         'dis', 'CORN'),
    'wheat':    ('Materie prime', 'Grano',        'dis', 'WHEAT-SRW'),
    'soy':      ('Materie prime', 'Soia',         'dis', 'SOYBEANS'),
    'sugar':    ('Materie prime', 'Zucchero',     'dis', 'SUGAR NO. 11'),
    'coffee':   ('Materie prime', 'Caffè',        'dis', 'COFFEE C'),
    'cotton':   ('Materie prime', 'Cotone',       'dis', 'COTTON NO. 2'),
}

_COT_CAT_ICON = {'Indici azionari': '📈', 'Valute': '💱',
                 'Tassi': '🏦', 'Materie prime': '🛢️'}
_COT_CAT_ORDER = ['Indici azionari', 'Valute', 'Tassi', 'Materie prime']

# cache in-processo: chiave → (timestamp_epoch, DataFrame). TTL 6h — i dati CFTC
# escono una volta a settimana (venerdì), quindi il refetch frequente è inutile.
_COT_CACHE = {}
_COT_TTL   = 6 * 3600

# Ticker Yahoo Finance del sottostante da disegnare sotto al posizionamento.
# Per gli INDICI azionari si usa l'indice cash (^GSPC, ^NDX, ^RUT) invece del
# future mini: serie continua e pulita, coerente con il riferimento del COT.
# Per valute/tassi/materie prime il future CME è già il contratto del COT.
_COT_PRICE_TICKER = {
    'sp500': '^GSPC', 'nasdaq': '^NDX', 'russell': '^RUT',
    'eur': '6E=F', 'jpy': '6J=F', 'gbp': '6B=F',
    'chf': '6S=F', 'cad': '6C=F', 'aud': '6A=F',
    'ust2': 'ZT=F', 'ust5': 'ZF=F', 'ust10': 'ZN=F', 'ustb': 'ZB=F', 'ff': 'ZQ=F',
    'gold': 'GC=F', 'silver': 'SI=F', 'copper': 'HG=F', 'wti': 'CL=F', 'natgas': 'NG=F',
    'platinum': 'PL=F', 'corn': 'ZC=F', 'wheat': 'ZW=F', 'soy': 'ZS=F',
    'sugar': 'SB=F', 'coffee': 'KC=F', 'cotton': 'CT=F',
}
# etichetta del pannello prezzo quando differisce dal nome del contratto COT
_COT_PRICE_NAME = {
    'sp500': 'S&P 500 (indice)', 'nasdaq': 'Nasdaq 100 (indice)',
    'russell': 'Russell 2000 (indice)',
}
_COT_PRICE_CACHE = {}
_COT_PRICE_TTL   = 12 * 3600
# Scarichiamo sempre lo storico completo (i report COT TFF/Disaggregated partono dal 2006):
# così il prezzo copre tutto l'orizzonte e lo ritagliamo a valle sulla finestra mostrata.
_COT_PRICE_START = '2005-01-01'


def _cot_options():
    """Opzioni dropdown ordinate per categoria, con icona e nome IT."""
    opts = []
    for cat in _COT_CAT_ORDER:
        for key, (c, label, kind, contract) in _COT_INSTRUMENTS.items():
            if c == cat:
                opts.append({'label': f"{_COT_CAT_ICON.get(cat, '')}  {label}", 'value': key})
    return opts


def _cot_fetch(key, force=False):
    """Scarica (e mette in cache) lo storico posizioni per uno strumento."""
    now = time.time()
    hit = _COT_CACHE.get(key)
    if hit and not force and (now - hit[0]) < _COT_TTL:
        return hit[1]
    cat, label, kind, contract = _COT_INSTRUMENTS[key]
    if kind == 'tff':
        url = _COT_TFF_URL
        sel = ('report_date_as_yyyy_mm_dd,asset_mgr_positions_long,'
               'asset_mgr_positions_short,lev_money_positions_long,'
               'lev_money_positions_short')
    else:
        url = _COT_DIS_URL
        # Managed Money (speculatori) + lato Commercial (hedger) = Producer/Merchant
        # + Swap Dealers. NB: il campo short degli swap ha il doppio underscore.
        sel = ('report_date_as_yyyy_mm_dd,m_money_positions_long_all,'
               'm_money_positions_short_all,prod_merc_positions_long,'
               'prod_merc_positions_short,swap_positions_long_all,'
               'swap__positions_short_all')
    params = {
        '$select': sel,
        '$where': f"contract_market_name='{contract}'",
        '$order': 'report_date_as_yyyy_mm_dd ASC',
        '$limit': 3000,
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        _COT_CACHE[key] = (now, df)
        return df
    df['date'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
    for c in df.columns:
        if c not in ('report_date_as_yyyy_mm_dd', 'date'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if kind == 'tff':
        df['am_net'] = df['asset_mgr_positions_long'] - df['asset_mgr_positions_short']
        df['lm_net'] = df['lev_money_positions_long'] - df['lev_money_positions_short']
    else:
        df['mm_net'] = df['m_money_positions_long_all'] - df['m_money_positions_short_all']
        # Commercial (hedger) = Producer/Merchant + Swap Dealers, netto long − short
        comm_cols = ('prod_merc_positions_long', 'prod_merc_positions_short',
                     'swap_positions_long_all', 'swap__positions_short_all')
        if all(c in df.columns for c in comm_cols):
            comm_long = (df['prod_merc_positions_long'].fillna(0)
                         + df['swap_positions_long_all'].fillna(0))
            comm_short = (df['prod_merc_positions_short'].fillna(0)
                          + df['swap__positions_short_all'].fillna(0))
            df['comm_net'] = comm_long - comm_short
    df = df.sort_values('date').set_index('date')
    _COT_CACHE[key] = (now, df)
    return df


def _cot_price(key):
    """Prezzo di chiusura del sottostante da Yahoo Finance (storico completo), cache 12h.

    Si scarica sempre tutto lo storico (da _COT_PRICE_START) e si affetta a valle
    sulla finestra mostrata: così l'S&P 500 copre l'intero orizzonte temporale
    qualunque sia il periodo selezionato, senza dipendere dall'ordine delle chiamate.
    """
    ticker = _COT_PRICE_TICKER.get(key)
    if not ticker:
        return None
    now = time.time()
    hit = _COT_PRICE_CACHE.get(ticker)
    if hit and (now - hit[0]) < _COT_PRICE_TTL:
        return hit[1]
    s = None
    try:
        import yfinance as yf
        data = yf.download(ticker, start=_COT_PRICE_START, progress=False,
                           auto_adjust=True, threads=False)
        if data is not None and not data.empty and 'Close' in data:
            close = data['Close']
            if isinstance(close, pd.DataFrame):          # colonne MultiIndex → 1ª col
                close = close.iloc[:, 0]
            close = close.dropna()
            s = close if not close.empty else None
    except Exception:
        s = None
    _COT_PRICE_CACHE[ticker] = (now, s)
    return s


def _cot_empty_fig(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color='#888'),
                       xref='paper', yref='paper', x=0.5, y=0.5)
    fig.update_layout(height=540, plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _cot_summary(kind, df, label):
    """Riepilogo (card) dell'ultimo dato settimanale con variazione."""
    if df is None or df.empty:
        return ''
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    d = df.index[-1].strftime('%d/%m/%Y')

    def _num(v):
        return f"{v:,.0f}".replace(',', '.')

    def card(title, net, net_prev, color):
        delta = net - net_prev
        arrow = '▲' if delta > 0 else ('▼' if delta < 0 else '■')
        dcol = '#1b7a34' if delta > 0 else ('#c0392b' if delta < 0 else '#666')
        pos = 'NET LONG' if net >= 0 else 'NET SHORT'
        return html.Div([
            html.Div(title, style={'font-size': '11px', 'font-weight': '700',
                                   'color': color, 'margin-bottom': '4px'}),
            html.Div(_num(net), style={'font-size': '20px', 'font-weight': '700',
                                       'color': '#1a3a5c'}),
            html.Div(pos, style={'font-size': '10px', 'font-weight': '700',
                                 'color': '#1b7a34' if net >= 0 else '#c0392b'}),
            html.Div(f"{arrow} {'+' if delta >= 0 else ''}{_num(delta)} vs sett. prec.",
                     style={'font-size': '10px', 'color': dcol, 'margin-top': '3px'}),
        ], style={'flex': '1', 'padding': '10px 14px', 'background': '#f8fafd',
                  'border': '1px solid #e3e9f2', 'border-radius': '6px'})

    if kind == 'tff':
        cards = [
            card('Asset Manager (istituzionali)', last['am_net'], prev['am_net'], '#1a3a5c'),
            card('Leveraged Money (hedge fund)',  last['lm_net'], prev['lm_net'], '#c0392b'),
        ]
    else:
        cards = [card('Managed Money (hedge fund)', last['mm_net'], prev['mm_net'], '#1b7a34')]
        if 'comm_net' in df.columns and pd.notna(last.get('comm_net')):
            cards.append(card('Commercial (Producer/Merchant + Swap)',
                              last['comm_net'], prev['comm_net'], '#1a3a5c'))
    return html.Div([
        html.Div(f"Ultimo dato CFTC: {d}",
                 style={'font-size': '11px', 'color': '#555', 'margin-bottom': '6px',
                        'font-weight': '600'}),
        html.Div(cards, style={'display': 'flex', 'gap': '10px'}),
    ])


_COT_WIN = 52   # settimane (1 anno) usate da COT Index e Z-score


def _cot_index_series(s, win=_COT_WIN, minp=26):
    """COT Index (Williams): 0 = minimo, 100 = massimo della finestra mobile."""
    lo = s.rolling(win, min_periods=minp).min()
    hi = s.rolling(win, min_periods=minp).max()
    rng = hi - lo
    return (100.0 * (s - lo) / rng).where(rng > 0)


def _cot_z_series(s, win=_COT_WIN, minp=26):
    """Z-score: scostamento in deviazioni standard dalla media della finestra mobile."""
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return ((s - m) / sd).where(sd > 0)


def _drawdown_series(price):
    """Drawdown %: distanza dal massimo progressivo (0 sui picchi, negativo sotto)."""
    return (price / price.cummax() - 1.0) * 100.0


def _stoch_series(close, n, smooth=3):
    """Stocastico %K su n giorni (prezzi di chiusura), filtrato con media mobile a `smooth` giorni."""
    lo = close.rolling(n, min_periods=max(2, n // 3)).min()
    hi = close.rolling(n, min_periods=max(2, n // 3)).max()
    rng = hi - lo
    k = (100.0 * (close - lo) / rng).where(rng > 0)
    if smooth and smooth > 1:
        k = k.rolling(smooth, min_periods=1).mean()
    return k


def _cot_fig_and_summary(key, year_range):
    cat, label, kind, contract = _COT_INSTRUMENTS[key]
    try:
        df = _cot_fetch(key)
    except Exception as e:
        return _cot_empty_fig(f'Errore nel recupero dei dati CFTC: {e}'), ''
    if df is None or df.empty:
        return _cot_empty_fig('Nessun dato COT disponibile per questo strumento'), ''
    full = df
    # year_range = [anno_inizio, anno_fine] scelti dallo slider; ritaglio la finestra
    # mostrata mantenendo full per il calcolo di COT Index / Z-score / stocastico.
    if year_range and len(year_range) == 2:
        y0, y1 = sorted(int(v) for v in year_range)
        start = pd.Timestamp(year=y0, month=1, day=1)
        end = pd.Timestamp(year=y1, month=12, day=31, hour=23, minute=59, second=59)
        df = df.loc[(df.index >= start) & (df.index <= end)]
    if df.empty:
        df = full.tail(2)

    # prezzo del sottostante (asse secondario): storico completo ritagliato sulla
    # finestra COT mostrata, così copre tutto l'orizzonte temporale del grafico.
    # Teniamo anche lo storico intero (price_full) per lo stocastico a 120 giorni.
    price_full = _cot_price(key)
    price = None
    if price_full is not None and not price_full.empty:
        price = price_full[(price_full.index >= df.index.min()) &
                           (price_full.index <= df.index.max())]
    has_price = price is not None and not price.empty
    price_label = _COT_PRICE_NAME.get(key, label)

    def _yr(series):
        """Range Y ritagliato su [min − 10, max + 10] per riempire il pannello."""
        s = series.dropna()
        if s.empty:
            return None
        return [float(s.min()) - 10, float(s.max()) + 10]

    def _add_price(row):
        """Aggiunge l'S&P 500 (o il sottostante) sull'asse secondario del pannello."""
        if has_price:
            fig.add_trace(go.Scatter(
                x=price.index, y=price.values, name=f'Prezzo {price_label}',
                mode='lines', line=dict(color='#444', width=1.5),
                showlegend=(row == 1),
                hovertemplate='%{x|%d %b %Y}<br>Prezzo: %{y:,.2f}<extra></extra>'),
                row=row, col=1, secondary_y=True)

    def _fmt_price_axis(row):
        if has_price:
            fig.update_yaxes(title_text=f'{price_label}', showgrid=False, zeroline=False,
                             row=row, col=1, secondary_y=True)
        else:
            fig.update_yaxes(visible=False, row=row, col=1, secondary_y=True)

    disp = df.index

    def _add_cot_index(series_full, color, row, name):
        """COT Index (0–100) sullo storico completo, lisciato a 8 settimane e ritagliato."""
        ci = _cot_index_series(series_full).rolling(8, min_periods=1).mean().reindex(disp)
        fig.add_trace(go.Scatter(
            x=disp, y=ci, name=name, showlegend=False, mode='lines',
            line=dict(color=color, width=1.8),
            hovertemplate='%{x|%d %b %Y}<br>' + name + ': %{y:.0f}<extra></extra>'),
            row=row, col=1)

    def _add_zscore(series_full, color, row, name):
        """Z-score sullo storico completo, lisciato a 8 settimane e ritagliato."""
        z = _cot_z_series(series_full).rolling(8, min_periods=1).mean().reindex(disp)
        fig.add_trace(go.Scatter(
            x=disp, y=z, name=name, showlegend=False, mode='lines',
            line=dict(color=color, width=1.8),
            hovertemplate='%{x|%d %b %Y}<br>' + name + ': %{y:.2f}<extra></extra>'),
            row=row, col=1)

    def _fmt_index_axis(row):
        fig.add_hline(y=80, line=dict(color='#bbb', width=1, dash='dot'), row=row, col=1)
        fig.add_hline(y=20, line=dict(color='#bbb', width=1, dash='dot'), row=row, col=1)
        fig.update_yaxes(title_text='COT Index', range=[0, 100], showgrid=True,
                         gridcolor='#eef1f5', row=row, col=1)

    def _fmt_z_axis(row):
        fig.add_hline(y=0, line=dict(color='#999', width=1), row=row, col=1)
        fig.add_hline(y=2, line=dict(color='#bbb', width=1, dash='dot'), row=row, col=1)
        fig.add_hline(y=-2, line=dict(color='#bbb', width=1, dash='dot'), row=row, col=1)
        fig.update_yaxes(title_text='Z-score', showgrid=True, gridcolor='#eef1f5', row=row, col=1)

    def _add_wow(row, pairs):
        """Variazione settimanale dei contratti netti (diff sett-su-sett), lisciata
        con media a 4 settimane; la diff grezza è tracciata tenue sotto.

        pairs = [(series_full, colore, nome), ...]. Calcolata sullo storico completo
        e ritagliata alla finestra, così la media a 4 settimane è continua anche al
        bordo sinistro del periodo mostrato. Asse Y su [min − 10, max + 10] delle sole
        medie a 4 settimane: le diff grezze possono essere tagliate, così le medie
        riempiono il pannello e restano leggibili.
        """
        shown = []
        for series_full, color, name in pairs:
            raw = series_full.diff()
            ma = raw.rolling(4, min_periods=1).mean()
            raw_d = raw.reindex(disp)
            ma_d = ma.reindex(disp)
            shown.append(ma_d)
            fig.add_trace(go.Scatter(
                x=disp, y=raw_d, name=f'Δsett {name}', showlegend=False,
                mode='lines', line=dict(color=color, width=0.8), opacity=0.30,
                hoverinfo='skip'),
                row=row, col=1)
            fig.add_trace(go.Scatter(
                x=disp, y=ma_d, name=f'Δsett {name} (media 4 sett.)',
                showlegend=False, mode='lines', line=dict(color=color, width=2.0),
                hovertemplate='%{x|%d %b %Y}<br>Δ4s ' + name + ': %{y:,.0f}<extra></extra>'),
                row=row, col=1)
        fig.add_hline(y=0, line=dict(color='#999', width=1), row=row, col=1)
        fig.update_yaxes(title_text='Δ contratti / sett.', showgrid=True,
                         gridcolor='#eef1f5', zeroline=False, row=row, col=1)
        if shown:
            allv = pd.concat(shown).dropna()
            if not allv.empty:
                fig.update_yaxes(range=[float(allv.min()) - 10, float(allv.max()) + 10],
                                 row=row, col=1)

    def _add_sum(row):
        """Somma AM + LM (contratti netti totali) con media a 4 settimane.

        Essendo già posizioni nette, la somma dà il posizionamento netto complessivo
        di istituzionali e hedge fund. Asse Y su [min − 10, max + 10] come i pannelli
        di posizionamento.
        """
        am_ma = full['am_net'].rolling(4, min_periods=1).mean()
        lm_ma = full['lm_net'].rolling(4, min_periods=1).mean()
        s = (am_ma + lm_ma).reindex(disp)
        fig.add_trace(go.Scatter(
            x=disp, y=s, name='Somma AM + LM', showlegend=False,
            mode='lines', line=dict(color='#6a3d9a', width=2.2),
            fill='tozeroy', fillcolor='rgba(106,61,154,0.10)',
            hovertemplate='%{x|%d %b %Y}<br>AM + LM: %{y:,.0f}<extra></extra>'),
            row=row, col=1)
        fig.add_hline(y=0, line=dict(color='#999', width=1), row=row, col=1)
        fig.update_yaxes(title_text='AM + LM (netti)', showgrid=True, gridcolor='#eef1f5',
                         zeroline=False, row=row, col=1)
        yr = _yr(s)
        if yr:
            fig.update_yaxes(range=yr, row=row, col=1)

    def _add_drawdown(row):
        """Drawdown % del sottostante dal massimo progressivo della finestra mostrata."""
        dd = _drawdown_series(price)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=f'Drawdown {price_label}', showlegend=False,
            mode='lines', line=dict(color='#8a1a1a', width=1.3),
            fill='tozeroy', fillcolor='rgba(192,57,43,0.15)',
            hovertemplate='%{x|%d %b %Y}<br>Drawdown: %{y:.1f}%<extra></extra>'),
            row=row, col=1)
        fig.update_yaxes(title_text='Drawdown', ticksuffix='%', showgrid=True,
                         gridcolor='#eef1f5', row=row, col=1)

    def _add_stoch(row):
        """Stocastico %K del sottostante con filtro dedicato per finestra.

        30gg → media 5; 60gg → media 10; 120gg → media 20; 240gg → media 30.
        Le due linee lunghe (120gg rossa, 240gg blu) sono più marcate e disegnate
        per ultime così restano in primo piano.
        """
        # (n_giorni, media_filtro, colore, spessore)
        for n, smooth, color, width in ((30, 5, '#0f8a8a', 1.4),
                                        (60, 10, '#e08a1e', 1.4),
                                        (120, 20, '#c0392b', 2.8),
                                        (240, 30, '#1a4fd6', 2.8)):
            st = _stoch_series(price_full, n, smooth=smooth).reindex(price.index)
            fig.add_trace(go.Scatter(
                x=st.index, y=st.values, name=f'Stocastico {n}gg', showlegend=True,
                mode='lines', line=dict(color=color, width=width),
                hovertemplate=f'%{{x|%d %b %Y}}<br>Stocastico {n}gg: %{{y:.0f}}<extra></extra>'),
                row=row, col=1)
        fig.add_hline(y=80, line=dict(color='#bbb', width=1, dash='dot'), row=row, col=1)
        fig.add_hline(y=20, line=dict(color='#bbb', width=1, dash='dot'), row=row, col=1)
        fig.update_yaxes(title_text='Stocastico', range=[0, 100], showgrid=True,
                         gridcolor='#eef1f5', row=row, col=1)

    if kind == 'tff':
        # Pannelli in ordine: Asset Manager, Leveraged Money (posizionamento + prezzo),
        # Somma AM + LM, Variazione settimanale contratti (media 4 sett.),
        # poi Drawdown e Stocastico sul sottostante, infine COT Index e Z-score.
        titles  = ['Asset Manager — istituzionali (media 4 sett.)',
                   'Leveraged Money — hedge fund / CTA (media 4 sett.)',
                   'Somma AM + LM (contratti netti totali, media 4 sett.)',
                   'Variazione settimanale contratti netti (media 4 sett.) — AM e LM']
        specs   = [[{'secondary_y': True}], [{'secondary_y': True}],
                   [{'secondary_y': False}], [{'secondary_y': False}]]
        heights = [1.0, 1.0, 1.0, 1.40]   # AM, LM, somma, variazione settimanale (doppia)
        r_sum, r_wow = 3, 4
        if has_price:
            titles  += [f'Drawdown {price_label} (%)',
                        f'Stocastico {price_label} — 240 gg (media 30) · 120 gg (media 20) · 60 gg (media 10) · 30 gg (media 5)']
            specs   += [[{'secondary_y': False}], [{'secondary_y': False}]]
            heights += [0.70, 0.70]
            r_dd, r_st, r_ci, r_z = 5, 6, 7, 8
        else:
            r_dd = r_st = None
            r_ci, r_z = 5, 6
        titles  += ['COT Index (0–100, finestra 1 anno, media 8 sett.)',
                    'Z-score (deviazioni standard, finestra 1 anno, media 8 sett.)']
        specs   += [[{'secondary_y': False}], [{'secondary_y': False}]]
        heights += [0.70, 0.70]
        _tot = sum(heights)
        fig = make_subplots(
            rows=len(specs), cols=1, shared_xaxes=True, vertical_spacing=0.045,
            row_heights=[h / _tot for h in heights], specs=specs,
            subplot_titles=tuple(titles))
        # posizionamento AM/LM lisciato con media a 4 settimane (calcolata su full,
        # ritagliata alla finestra per continuità al bordo sinistro)
        am_disp = full['am_net'].rolling(4, min_periods=1).mean().reindex(disp)
        lm_disp = full['lm_net'].rolling(4, min_periods=1).mean().reindex(disp)
        fig.add_trace(go.Scatter(
            x=disp, y=am_disp, name='Asset Manager (istituzionali)',
            mode='lines', line=dict(color='#1a3a5c', width=2.2),
            hovertemplate='%{x|%d %b %Y}<br>Asset Manager: %{y:,.0f}<extra></extra>'),
            row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=disp, y=lm_disp, name='Leveraged Money (hedge fund / CTA)',
            mode='lines', line=dict(color='#c0392b', width=2.2),
            hovertemplate='%{x|%d %b %Y}<br>Leveraged Money: %{y:,.0f}<extra></extra>'),
            row=2, col=1, secondary_y=False)
        _add_price(1)
        _add_price(2)
        for r, series in ((1, am_disp), (2, lm_disp)):
            fig.add_hline(y=0, line=dict(color='#999', width=1), row=r, col=1, secondary_y=False)
            fig.update_yaxes(title_text='Contratti netti', showgrid=True, gridcolor='#eef1f5',
                             zeroline=False, row=r, col=1, secondary_y=False)
            yr = _yr(series)
            if yr:
                fig.update_yaxes(range=yr, row=r, col=1, secondary_y=False)
            _fmt_price_axis(r)
        _add_sum(r_sum)
        _add_wow(r_wow, [(full['am_net'], '#1a3a5c', 'Asset Manager'),
                         (full['lm_net'], '#c0392b', 'Leveraged Money')])
        if has_price:
            _add_drawdown(r_dd)
            _add_stoch(r_st)
        _add_cot_index(full['am_net'], '#1a3a5c', r_ci, 'COT Index AM')
        _add_cot_index(full['lm_net'], '#c0392b', r_ci, 'COT Index LM')
        _fmt_index_axis(r_ci)
        _add_zscore(full['am_net'], '#1a3a5c', r_z, 'Z-score AM')
        _add_zscore(full['lm_net'], '#c0392b', r_z, 'Z-score LM')
        _fmt_z_axis(r_z)
        fig.update_annotations(font_size=12)
        height = 2100 if has_price else 1700
    else:
        # Materie prime: Managed Money (speculatori) e Commercial (hedger = Producer/
        # Merchant + Swap Dealers), poi Variazione settimanale, Drawdown e Stocastico
        # sul sottostante, infine COT Index e Z-score. Se il Commercial non è
        # disponibile si mostra il solo Managed Money.
        has_comm = 'comm_net' in df.columns and df['comm_net'].notna().any()
        mm_disp = full['mm_net'].rolling(4, min_periods=1).mean().reindex(disp)
        if has_comm:
            titles  = ['Managed Money — hedge fund (media 4 sett.)',
                       'Commercial — Producer/Merchant + Swap Dealers (media 4 sett.)']
            specs   = [[{'secondary_y': True}], [{'secondary_y': True}]]
            heights = [1.0, 1.0]
            pos_rows = 2
        else:
            titles  = ['Managed Money — hedge fund (media 4 sett.)']
            specs   = [[{'secondary_y': True}]]
            heights = [1.3]
            pos_rows = 1
        titles  += ['Variazione settimanale contratti netti (media 4 sett.)']
        specs   += [[{'secondary_y': False}]]
        heights += [1.40]
        r_wow = pos_rows + 1
        if has_price:
            titles  += [f'Drawdown {price_label} (%)',
                        f'Stocastico {price_label} — 240 gg (media 30) · 120 gg (media 20) · 60 gg (media 10) · 30 gg (media 5)']
            specs   += [[{'secondary_y': False}], [{'secondary_y': False}]]
            heights += [0.70, 0.70]
            r_dd, r_st = r_wow + 1, r_wow + 2
            r_ci, r_z = r_wow + 3, r_wow + 4
        else:
            r_dd = r_st = None
            r_ci, r_z = r_wow + 1, r_wow + 2
        titles  += ['COT Index (0–100, finestra 1 anno, media 8 sett.)',
                    'Z-score (deviazioni standard, finestra 1 anno, media 8 sett.)']
        specs   += [[{'secondary_y': False}], [{'secondary_y': False}]]
        heights += [0.80, 0.80]
        _tot = sum(heights)
        fig = make_subplots(
            rows=len(specs), cols=1, shared_xaxes=True, vertical_spacing=0.045,
            row_heights=[h / _tot for h in heights], specs=specs,
            subplot_titles=tuple(titles))
        # posizionamento Managed Money, lisciato a 4 settimane
        fig.add_trace(go.Scatter(
            x=disp, y=mm_disp, name='Managed Money (hedge fund)',
            mode='lines', line=dict(color='#1b7a34', width=2.2),
            hovertemplate='%{x|%d %b %Y}<br>Managed Money: %{y:,.0f}<extra></extra>'),
            row=1, col=1, secondary_y=False)
        _add_price(1)
        fig.add_hline(y=0, line=dict(color='#999', width=1), row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text='Contratti netti', showgrid=True, gridcolor='#eef1f5',
                         zeroline=False, row=1, col=1, secondary_y=False)
        yr = _yr(mm_disp)
        if yr:
            fig.update_yaxes(range=yr, row=1, col=1, secondary_y=False)
        _fmt_price_axis(1)
        if has_comm:
            comm_disp = full['comm_net'].rolling(4, min_periods=1).mean().reindex(disp)
            fig.add_trace(go.Scatter(
                x=disp, y=comm_disp, name='Commercial (Producer/Merchant + Swap)',
                mode='lines', line=dict(color='#1a3a5c', width=2.2),
                hovertemplate='%{x|%d %b %Y}<br>Commercial: %{y:,.0f}<extra></extra>'),
                row=2, col=1, secondary_y=False)
            _add_price(2)
            fig.add_hline(y=0, line=dict(color='#999', width=1), row=2, col=1, secondary_y=False)
            fig.update_yaxes(title_text='Contratti netti', showgrid=True, gridcolor='#eef1f5',
                             zeroline=False, row=2, col=1, secondary_y=False)
            yr = _yr(comm_disp)
            if yr:
                fig.update_yaxes(range=yr, row=2, col=1, secondary_y=False)
            _fmt_price_axis(2)
        wow_pairs = [(full['mm_net'], '#1b7a34', 'Managed Money')]
        if has_comm:
            wow_pairs.append((full['comm_net'], '#1a3a5c', 'Commercial'))
        _add_wow(r_wow, wow_pairs)
        if has_price:
            _add_drawdown(r_dd)
            _add_stoch(r_st)
        _add_cot_index(full['mm_net'], '#1b7a34', r_ci, 'COT Index MM')
        if has_comm:
            _add_cot_index(full['comm_net'], '#1a3a5c', r_ci, 'COT Index Commercial')
        _fmt_index_axis(r_ci)
        _add_zscore(full['mm_net'], '#1b7a34', r_z, 'Z-score MM')
        if has_comm:
            _add_zscore(full['comm_net'], '#1a3a5c', r_z, 'Z-score Commercial')
        _fmt_z_axis(r_z)
        fig.update_annotations(font_size=12)
        height = int(290 * _tot)

    fig.update_layout(
        title=dict(text=f"Posizionamento COT — {label}",
                   font=dict(size=15, color='#1a3a5c')),
        height=height, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=70, r=70, t=80, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
        hovermode='closest')
    fig.update_xaxes(showgrid=True, gridcolor='#eef1f5')
    return fig, _cot_summary(kind, full, label)


# Slider periodo: dallo storico COT (report TFF/Disaggregated dal 2006) all'anno corrente.
_COT_YEAR_MIN = 2006
_COT_YEAR_MAX = pd.Timestamp.today().year


def _cot_year_marks():
    """Tacche dello slider: un'etichetta ogni 2 anni ('06, '08, …) + estremo destro."""
    st = {'font-size': '9px', 'color': '#777'}
    marks = {y: {'label': f"'{str(y)[2:]}", 'style': st}
             for y in range(_COT_YEAR_MIN, _COT_YEAR_MAX + 1, 2)}
    marks[_COT_YEAR_MAX] = {'label': f"'{str(_COT_YEAR_MAX)[2:]}", 'style': st}
    return marks


def get_cot_tab():
    """Layout della tab Posizionamento COT."""
    opts = _cot_options()
    return html.Div([
        html.Div([
            html.H3('Posizionamento COT',
                    style={'margin-right': '20px', 'white-space': 'nowrap', 'font-size': '16px'}),
            html.Div([
                html.Label("Strumento:", style={'margin-right': '6px', 'font-size': '11px',
                                                 'white-space': 'nowrap'}),
                dcc.Dropdown(id='cot-instrument', options=opts, value='sp500',
                             clearable=False, style={'width': '250px', 'font-size': '11px'}),
            ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '14px'}),
            html.Div([
                html.Label("Periodo:", style={'margin-right': '12px', 'font-size': '11px',
                                              'white-space': 'nowrap'}),
                html.Div([
                    dcc.RangeSlider(
                        id='cot-range', min=_COT_YEAR_MIN, max=_COT_YEAR_MAX, step=1,
                        value=[_COT_YEAR_MAX - 2, _COT_YEAR_MAX], allowCross=False,
                        marks=_cot_year_marks(),
                        tooltip={'placement': 'bottom', 'always_visible': False}),
                ], style={'width': '360px'}),
            ], style={'display': 'flex', 'align-items': 'center', 'border': '1px solid #ccc',
                      'border-radius': '4px', 'padding': '3px 14px 3px 10px', 'margin-right': '12px',
                      'background': '#f5f5f5'}),
            html.Button('🔄 Aggiorna', id='cot-refresh', n_clicks=0,
                        title='Forza il ri-scaricamento dei dati dalla CFTC',
                        style={'background-color': '#1a3a5c', 'color': 'white', 'border': 'none',
                               'padding': '8px 16px', 'border-radius': '4px', 'cursor': 'pointer',
                               'font-weight': 'bold', 'font-size': '12px'}),
        ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap',
                  'gap': '4px', 'margin-bottom': '8px'}),

        html.Div('Fonte: CFTC — Commitments of Traders, pubblicato ogni venerdì. '
                 'Per indici azionari, valute e tassi (report TFF) sono mostrate le posizioni '
                 'nette di Asset Manager (fondi pensione/comuni/assicurazioni, medio-lungo termine) '
                 'e Leveraged Money (hedge fund/CTA, speculativi). Per le materie prime '
                 '(report Disaggregated) sono mostrati il Managed Money (speculazione hedge fund) '
                 'e il Commercial (hedger = Producer/Merchant + Swap Dealers).',
                 style={'font-size': '10px', 'color': '#777', 'margin-bottom': '10px',
                        'line-height': '1.4', 'max-width': '920px'}),

        html.Hr(style={'margin': '4px 0 10px'}),

        html.Div(id='cot-summary', style={'margin-bottom': '10px'}),
        dcc.Loading(id='loading-cot', type='circle', color='#1a3a5c', children=[
            dcc.Graph(id='cot-chart', style={'width': '100%', 'height': '1400px'},
                      config={'responsive': True, 'displaylogo': False}),
        ]),
    ])


# Stile tab (uguale ad Analisi di Portafoglio)
_TAB_STYLE = {'font-size': '12px', 'padding': '8px 18px'}
_TAB_SEL   = {'font-size': '12px', 'padding': '8px 18px',
              'font-weight': 'bold', 'border-top': '3px solid #1a3a5c'}

_INP = {'font-size': '11px', 'padding': '5px 8px', 'border': '1px solid #aaa',
        'border-radius': '4px'}
_SEP = {'width': '1px', 'height': '26px', 'background': '#d0d8e4', 'margin': '0 8px'}


def _btn(bg):
    return {'font-size': '11px', 'padding': '6px 12px', 'border': 'none',
            'border-radius': '4px', 'cursor': 'pointer', 'color': 'white',
            'background': bg, 'font-weight': 'bold'}


def _file_panel():
    """Solo il pulsante 📁 File (il pannello vero è _file_modal(), alla radice)."""
    return html.Button('📁 File', id='at-file-btn', n_clicks=0,
                       style={'border': 'none', 'border-radius': '4px', 'cursor': 'pointer',
                              'font-weight': 'bold', 'background-color': '#5a1a6a',
                              'color': 'white', 'padding': '6px 14px', 'font-size': '12px'})


def _file_modal():
    """Pannello 📁 File montato alla RADICE della pagina (fixed, sopra a tutto)."""
    _bb = {'border': 'none', 'border-radius': '4px', 'cursor': 'pointer',
           'font-size': '11px', 'padding': '4px 10px', 'font-weight': 'bold'}
    return html.Div(id='at-file-panel', style={'display': 'none'}, children=[
        html.Div([
            html.Button('✕', id='at-file-close', n_clicks=0, title='Chiudi',
                        style={'position': 'absolute', 'top': '6px', 'right': '8px', 'border': 'none',
                               'background': 'transparent', 'cursor': 'pointer', 'font-size': '15px',
                               'color': '#888', 'font-weight': 'bold'}),
            html.Div([
                # Sinistra: salva tutto come…
                html.Div([
                    html.B('💾 Salva tutto il lavoro come…',
                           style={'font-size': '11px', 'color': '#1a3a5c',
                                  'display': 'block', 'margin-bottom': '8px'}),
                    dcc.Input(id='at-save-name', type='text', placeholder='Es. Tattica_Maggio…',
                              style={'width': '100%', 'padding': '5px 8px', 'border': '1px solid #aaa',
                                     'border-radius': '4px', 'font-size': '11px', 'margin-bottom': '6px'}),
                    html.Button('💾 Salva', id='at-save-btn', n_clicks=0,
                                style={**_bb, 'background': '#1b7a34', 'color': 'white', 'width': '100%'}),
                    html.Div(id='at-save-status',
                             style={'font-size': '10px', 'margin-top': '5px', 'color': '#555',
                                    'min-height': '16px'}),
                    html.Div('Salva dataset + pesi + analisi (tutti i dati).',
                             style={'font-size': '9px', 'color': '#888', 'margin-top': '6px'}),
                ], style={'width': '220px', 'padding-right': '20px', 'border-right': '1px solid #ddd'}),
                # Destra: i miei lavori salvati
                html.Div([
                    html.Div([
                        html.B('📁 I miei lavori salvati',
                               style={'font-size': '11px', 'color': '#1a3a5c'}),
                        html.Button('🔄', id='at-refresh-btn', n_clicks=0,
                                    style={**_bb, 'background': '#e8e8e8', 'color': '#333',
                                           'margin-left': '8px', 'padding': '3px 8px'}),
                    ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '8px'}),
                    html.Div(id='at-file-list', children=_render_file_list(),
                             style={'max-height': '280px', 'overflow-y': 'auto'}),
                ], style={'flex': '1', 'padding-left': '20px'}),
            ], style={'display': 'flex'}),
            # Footer: pulsante Chiudi in basso a destra
            html.Div([
                html.Button('Chiudi', id='at-file-close-btn', n_clicks=0,
                            style={'border': 'none', 'border-radius': '4px', 'cursor': 'pointer',
                                   'font-weight': 'bold', 'font-size': '12px', 'padding': '7px 20px',
                                   'background': '#5a1a6a', 'color': 'white'}),
            ], style={'display': 'flex', 'justify-content': 'flex-end', 'margin-top': '14px',
                      'border-top': '1px solid #eee', 'padding-top': '10px'}),
        ], style={'position': 'relative'}),
    ])


def _data_toolbar():
    return html.Div([
        # ── Toolbar ───────────────────────────────────────────────────────
        html.Div([
            _file_panel(),
            html.Div(style=_SEP),
            # ⬆ Carica File · 📋 Template · 📤 Esporta
            dcc.Upload(id='at-upload', children=html.Div(['⬆ Carica File']), multiple=False,
                       style={'height': '30px', 'lineHeight': '30px', 'padding': '0 12px',
                              'borderWidth': '1px', 'borderStyle': 'dashed', 'borderColor': '#9bb0cc',
                              'borderRadius': '4px', 'textAlign': 'center', 'fontSize': '11px',
                              'color': '#1a3a5c', 'background': '#f5f8fc', 'cursor': 'pointer'}),
            html.Button('📋 Template', id='at-template-btn', n_clicks=0,
                        style={'font-size': '11px', 'padding': '6px 12px', 'border-radius': '4px',
                               'cursor': 'pointer', 'background': '#e8f5e9',
                               'border': '1px solid #a5d6a7', 'color': '#1b5e20'}),
            html.Button('📤 Esporta Dati', id='at-export-btn', n_clicks=0,
                        style={'font-size': '11px', 'padding': '6px 12px', 'border-radius': '4px',
                               'cursor': 'pointer', 'background': '#eafaf1',
                               'border': '1px solid #1a7a4a', 'color': '#1a7a4a', 'font-weight': 'bold'}),
        ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap',
                  'gap': '6px', 'margin-bottom': '6px'}),
        dcc.Loading(type='default', color='#1b7a34', children=[
            html.Div(id='at-data-status',
                     style={'font-size': '11px', 'color': '#555', 'min-height': '16px',
                            'margin-bottom': '6px'})]),
        dcc.Download(id='at-dl-template'),
        dcc.Download(id='at-dl-export'),
    ])


def serve_layout():
    opts = _asset_options()
    return html.Div([
        make_navbar('Analisi Tattica'),
        html.Div([
            # ── Intestazione pagina (stile Analisi di Portafoglio) ─────────
            html.Div([
                html.H1([
                    'Analisi Tattica',
                    html.Span(' - ', style={'color': '#9baabf'}),
                    html.Span('Analisi per una gestione tattica del portafoglio',
                              className='sub'),
                ]),
            ], className='page-head'),
            _data_toolbar(),
            dcc.Tabs(id='at-tabs', value='tab-arima',
                     colors={'border': '#dee2e6', 'primary': '#1a3a5c',
                             'background': '#f0f4fa'},
                     style={'margin-bottom': '10px'},
                     children=[
                         dcc.Tab(label='📉 Analisi ARIMA', value='tab-arima',
                                 style=_TAB_STYLE, selected_style=_TAB_SEL),
                         dcc.Tab(label='📊 Posizionamento COT', value='tab-cot',
                                 style=_TAB_STYLE, selected_style=_TAB_SEL),
                     ]),
            # Contenuto delle tab: tutte restano nel DOM e si mostrano/nascondono
            # via callback (Input at-tabs), così le callback ARIMA che dipendono
            # dai suoi componenti restano sempre agganciate.
            html.Div(id='tab-arima-content', children=get_arima_analysis_tab(opts),
                     style={'display': 'block'}),
            html.Div(id='tab-cot-content', children=get_cot_tab(),
                     style={'display': 'none'}),
        ], className='page-wrap'),
        # Pannello File montato ALLA RADICE → galleggia sopra a tutto
        _file_modal(),
    ])


app.layout = serve_layout


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('arima-asset-dropdown', 'value', allow_duplicate=True),
    Input({'type': 'graph-select-checkbox-arima', 'index': ALL}, 'value'),
    State('arima-asset-dropdown', 'value'),
    prevent_initial_call=True,
)
def sync_arima_dropdown_from_grid(all_checkbox_values, current_dropdown_value):
    """Spuntare un asset nella griglia aggiorna il dropdown (ultimo spuntato)."""
    checked = [v[0] for v in all_checkbox_values if v]
    if not checked:
        raise PreventUpdate
    new_asset = checked[-1]
    if new_asset == current_dropdown_value:
        raise PreventUpdate
    return new_asset


@app.callback(
    Output({'type': 'graph-select-checkbox-arima', 'index': ALL}, 'value'),
    Output('deselect-all-arima-tab', 'children'),
    Input('deselect-all-arima-tab', 'n_clicks'),
    State({'type': 'graph-select-checkbox-arima', 'index': ALL}, 'value'),
    State({'type': 'graph-select-checkbox-arima', 'index': ALL}, 'options'),
    prevent_initial_call=True,
)
def deselect_all_arima_tab(n, current_values, all_options):
    if not all_options:
        return [], 'Des'
    if any(v for v in current_values):
        return [[] for _ in all_options], 'Sel'
    return [[opts[0]['value']] if opts else [] for opts in all_options], 'Des'


@app.callback(
    Output('weights-grid-container-arima', 'children'),
    Output('arima-forecast-status', 'children'),
    Input('calc-arima-forecasts-btn', 'n_clicks'),
    State({'type': 'graph-select-checkbox-arima', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def compute_all_forecasts(n, checked_vals):
    """Calcola la previsione ARIMA a 1 giorno per TUTTI gli asset e riempie la griglia."""
    if not n:
        raise PreventUpdate
    prices_df = _build_prices()
    if prices_df is None or prices_df.empty:
        return _build_asset_grid([], []), "⚠ Nessun dato disponibile"
    selected = [v[0] for v in (checked_vals or []) if v]
    asset_names = list(prices_df.columns)
    forecasts = {}
    for asset in asset_names:
        fc = _quick_forecast(prices_df[asset])
        if fc:
            forecasts[asset] = fc
    grid = _build_asset_grid(asset_names, selected, forecasts)
    return grid, f"✓ Previsione 1gg calcolata per {len(forecasts)}/{len(asset_names)} asset"


@app.callback(
    Output('arima-analysis-chart', 'figure'),
    Output('arima-tab-status', 'children'),
    Input('run-arima-tab-button', 'n_clicks'),
    State('arima-asset-dropdown', 'value'),
    State('arima-tab-horizon', 'value'),
    State('arima-tab-max-p', 'value'),
    State('arima-tab-max-q', 'value'),
    State('arima-tab-criterion', 'value'),
    prevent_initial_call=True,
)
def run_arima_tab_analysis(n_clicks, selected_asset, horizon, max_p, max_q, criterion):
    """Analisi ARIMA completa per un singolo asset (vedi docstring del modulo)."""
    if not n_clicks:
        raise PreventUpdate
    if not selected_asset:
        return go.Figure(), "⚠ Seleziona un asset."

    horizon   = int(horizon or 30)
    max_p     = int(max_p   or 4)
    max_q     = int(max_q   or 4)
    criterion = criterion or 'aic'

    ARCH_AVAILABLE = False
    try:
        from arch import arch_model as arch_garch_model
        ARCH_AVAILABLE = True
    except ImportError:
        pass

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
    from statsmodels.tsa.arima.model import ARIMA as ARIMAModel

    try:
        original_prices = _build_prices()
        if original_prices is None or selected_asset not in original_prices.columns:
            return go.Figure(), f"⚠ Asset '{selected_asset}' non trovato nei dati (current.json)."

        prices = original_prices[selected_asset].dropna()
        if len(prices) < 100:
            return go.Figure(), "⚠ Dati insufficienti (< 100 osservazioni)."
        if len(prices) > 750:
            prices = prices.iloc[-750:]

        # Step 1: log-prezzi
        log_prices = np.log(prices.values.astype(float))
        idx        = prices.index
        x_idx      = np.arange(len(log_prices), dtype=float)

        # Step 2: trend lineare → ciclicità
        trend_coef = np.polyfit(x_idx, log_prices, 1)
        trend_line = np.polyval(trend_coef, x_idx)
        detrended  = log_prices - trend_line

        diff_detrended = np.diff(detrended)
        idx_diff       = idx[1:]

        # Step 3: grid search ARIMA(p,1,q)
        best_score = np.inf
        best_order = (1, 1, 1)
        best_model = None
        aic_grid   = {}
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    m = ARIMAModel(detrended, order=(p, 1, q)).fit()
                    score = m.aic if criterion == 'aic' else m.bic
                    aic_grid[(p, q)] = round(score, 2)
                    if score < best_score:
                        best_score = score
                        best_order = (p, 1, q)
                        best_model = m
                except Exception:
                    continue
        if best_model is None:
            return go.Figure(), "❌ Nessun modello ARIMA convergente trovato."

        # Step 4: residui ε_t + ADF + ACF/PACF
        residuals = best_model.resid
        adf_res       = adfuller(residuals, autolag='AIC')
        adf_pval      = float(adf_res[1])
        is_stationary = adf_pval < 0.05

        n_lags   = min(40, len(diff_detrended) // 4)
        ci_bound = 1.96 / np.sqrt(len(diff_detrended))
        acf_vals, _  = sm_acf (diff_detrended, nlags=n_lags, alpha=0.05, fft=True)
        pacf_vals, _ = sm_pacf(diff_detrended, nlags=n_lags, alpha=0.05, method='ywm')
        lags_arr = np.arange(len(acf_vals))

        # Step 5: GARCH(1,1)
        garch_ok     = False
        cond_vol     = None
        garch_fc_vol = None
        garch_info   = "arch non installato"
        if ARCH_AVAILABLE:
            try:
                garch_spec   = arch_garch_model(residuals * 100, vol='Garch', p=1, q=1, dist='Normal')
                garch_fit    = garch_spec.fit(disp='off', show_warning=False)
                cond_vol     = garch_fit.conditional_volatility / 100
                garch_fc     = garch_fit.forecast(horizon=horizon, reindex=False)
                garch_fc_vol = np.sqrt(garch_fc.variance.values[-1]) / 100
                garch_ok     = True
                garch_info   = f"GARCH(1,1) ✓ | vol. ultima: {float(cond_vol[-1]) * 100:.3f}%"
            except Exception as eg:
                garch_info = f"GARCH errore: {str(eg)[:60]}"

        # Step 6: forecast → ricostruzione prezzi
        fc_result   = best_model.get_forecast(steps=horizon)
        fc_mean_det = np.array(fc_result.predicted_mean)
        fc_ci       = np.array(fc_result.conf_int(alpha=0.05))
        fc_lo_det   = fc_ci[:, 0]
        fc_hi_det   = fc_ci[:, 1]

        future_dates = pd.bdate_range(start=idx[-1] + pd.Timedelta(days=1), periods=horizon)
        x_future     = np.arange(len(log_prices), len(log_prices) + horizon, dtype=float)
        trend_future = np.polyval(trend_coef, x_future)

        log_fc_mean = fc_mean_det + trend_future
        log_fc_lo   = fc_lo_det   + trend_future
        log_fc_hi   = fc_hi_det   + trend_future

        if garch_ok and garch_fc_vol is not None:
            garch_std_cum = np.sqrt(np.cumsum(garch_fc_vol ** 2))
            log_fc_lo = log_fc_mean - 1.96 * garch_std_cum
            log_fc_hi = log_fc_mean + 1.96 * garch_std_cum

        price_fc_mean = np.exp(log_fc_mean)
        price_fc_lo   = np.exp(log_fc_lo)
        price_fc_hi   = np.exp(log_fc_hi)

        last_price = float(prices.iloc[-1])
        last_date  = idx[-1]
        anchor_dates = [last_date]  + list(future_dates)
        anchor_mean  = [last_price] + list(price_fc_mean)
        anchor_lo    = [last_price] + list(price_fc_lo)
        anchor_hi    = [last_price] + list(price_fc_hi)

        expected_ret = (price_fc_mean[-1] / last_price - 1) * 100
        ret_sign     = '+' if expected_ret >= 0 else ''

        # ── Figura 5×2 ────────────────────────────────────────────────────
        specs_layout = [
            [{"colspan": 2, "type": "scatter"}, None],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"},     {"type": "bar"}],
            [{"colspan": 2, "type": "scatter"}, None],
            [{"colspan": 2, "type": "scatter"}, None],
        ]
        garch_title = ('GARCH(1,1) — Volatilità Condizionale degli Errori'
                       if garch_ok else 'Errori Stazionari ε_t (ARIMA residui)')
        fig = make_subplots(
            rows=5, cols=2, specs=specs_layout,
            subplot_titles=[
                f'{selected_asset} — Log Prezzi + Trend (slope={trend_coef[0]:+.5f}/gg)',
                'Ciclicità — Serie Detrended I(1)',
                f'Δ Ciclicità = Errori Grezzi I(0)  |  ADF p={adf_pval:.4f}'
                f'  {"✓ Staz." if is_stationary else "⚠ Non staz."}',
                f'ACF  Δ-Ciclicità  (nlags={n_lags})',
                f'PACF Δ-Ciclicità  (nlags={n_lags})',
                garch_title,
                f'Proiezione {horizon}gg — Prezzi + Cono 95%'
                f'  |  Rendimento atteso: {ret_sign}{expected_ret:.2f}%',
            ],
            row_heights=[0.17, 0.13, 0.16, 0.12, 0.42],
            vertical_spacing=0.065, horizontal_spacing=0.07,
        )

        # R1
        fig.add_trace(go.Scatter(x=idx, y=log_prices, name='Log Prezzi',
                                 line=dict(color='#1f77b4', width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=idx, y=trend_line, name='Trend Lineare',
                                 line=dict(color='#d62728', width=2, dash='dash')), row=1, col=1)
        # R2c1
        fig.add_trace(go.Scatter(x=idx, y=detrended, name='Ciclicità',
                                 line=dict(color='#2ca02c', width=1.1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=[idx[0], idx[-1]], y=[0, 0], showlegend=False,
                                 line=dict(color='red', width=0.8, dash='dot')), row=2, col=1)
        # R2c2
        col_diff = ['#2ca02c' if v >= 0 else '#d62728' for v in diff_detrended]
        fig.add_trace(go.Bar(x=idx_diff, y=diff_detrended, name='Δ Ciclicità',
                             marker_color=col_diff, marker_line_width=0), row=2, col=2)
        # R3c1 ACF
        acf_col = ['#d62728' if abs(v) > ci_bound else '#1f77b4' for v in acf_vals]
        fig.add_trace(go.Bar(x=lags_arr, y=acf_vals, name='ACF',
                             marker_color=acf_col, marker_line_width=0), row=3, col=1)
        for b in [ci_bound, -ci_bound]:
            fig.add_trace(go.Scatter(x=[0, n_lags], y=[b, b], showlegend=False,
                                     line=dict(color='orange', dash='dash', width=1.2)), row=3, col=1)
        # R3c2 PACF
        pacf_col = ['#d62728' if abs(v) > ci_bound else '#ff7f0e' for v in pacf_vals]
        fig.add_trace(go.Bar(x=lags_arr, y=pacf_vals, name='PACF',
                             marker_color=pacf_col, marker_line_width=0), row=3, col=2)
        for b in [ci_bound, -ci_bound]:
            fig.add_trace(go.Scatter(x=[0, n_lags], y=[b, b], showlegend=False,
                                     line=dict(color='orange', dash='dash', width=1.2)), row=3, col=2)
        # R4 GARCH / residui
        if garch_ok and cond_vol is not None:
            vol_idx = idx_diff[-len(cond_vol):]
            fig.add_trace(go.Scatter(x=vol_idx, y=cond_vol * 100, name='σ_t GARCH (%)',
                                     line=dict(color='#d62728', width=1.3),
                                     fill='tozeroy', fillcolor='rgba(214,39,40,0.1)'), row=4, col=1)
            mean_vol = float(np.mean(cond_vol)) * 100
            fig.add_trace(go.Scatter(x=[vol_idx[0], vol_idx[-1]], y=[mean_vol, mean_vol],
                                     name=f'σ media ({mean_vol:.3f}%)',
                                     line=dict(color='navy', dash='dash', width=1.2)), row=4, col=1)
        else:
            res_idx = idx_diff[-len(residuals):]
            fig.add_trace(go.Scatter(x=res_idx, y=residuals, name='ε_t (residui)',
                                     line=dict(color='#8c564b', width=0.7), opacity=0.8), row=4, col=1)
            fig.add_trace(go.Scatter(x=[res_idx[0], res_idx[-1]], y=[0, 0], showlegend=False,
                                     line=dict(color='red', width=1, dash='dot')), row=4, col=1)
        # R5 prezzi + forecast + cono
        lookback    = min(252 * 2, len(prices))
        hist_prices = prices.iloc[-lookback:]
        y_min = min(float(np.min(price_fc_lo)), float(hist_prices.min())) * 0.98
        y_max = max(float(np.max(price_fc_hi)), float(hist_prices.max())) * 1.02
        fig.add_trace(go.Scatter(x=hist_prices.index, y=hist_prices.values, name='Prezzo Storico',
                                 line=dict(color='#1f77b4', width=1.8)), row=5, col=1)
        fig.add_trace(go.Scatter(x=[last_date, last_date], y=[y_min, y_max], name='Oggi',
                                 line=dict(color='gray', width=1.5, dash='dash')), row=5, col=1)
        fig.add_trace(go.Scatter(x=anchor_dates + anchor_dates[::-1], y=anchor_hi + anchor_lo[::-1],
                                 fill='toself', fillcolor='rgba(31,119,180,0.13)',
                                 line=dict(color='rgba(0,0,0,0)'), name='Cono 95%'), row=5, col=1)
        fig.add_trace(go.Scatter(x=anchor_dates, y=anchor_hi, name='CI +95%',
                                 line=dict(color='rgba(31,119,180,0.55)', dash='dot', width=1.2)), row=5, col=1)
        fig.add_trace(go.Scatter(x=anchor_dates, y=anchor_lo, name='CI -95%',
                                 line=dict(color='rgba(31,119,180,0.55)', dash='dot', width=1.2)), row=5, col=1)
        fig.add_trace(go.Scatter(x=anchor_dates, y=anchor_mean, name='Previsione Media',
                                 line=dict(color='#d62728', width=2.5), marker=dict(size=4)), row=5, col=1)

        arima_lbl = f'ARIMA{best_order}'
        title_str = (f'🔬  {selected_asset}  —  {arima_lbl}  |  {criterion.upper()}={best_score:.1f}  |  '
                     f'ADF p={adf_pval:.4f}  {"✓ Staz." if is_stationary else "⚠ Non staz."}  |  '
                     f'Forecast {horizon}gg: {ret_sign}{expected_ret:.2f}%')
        fig.update_layout(
            height=1150, title=dict(text=title_str, font=dict(size=12), x=0.01),
            showlegend=True,
            legend=dict(x=1.01, y=1, xanchor='left', yanchor='top', font=dict(size=9),
                        bgcolor='rgba(255,255,255,0.8)', bordercolor='#ccc', borderwidth=1),
            margin=dict(t=70, b=30, l=55, r=175), hovermode='closest',
        )
        for row_n, col_n, lbl in [(1, 1, 'Log Prezzo'), (2, 1, 'Ciclicità'), (2, 2, 'Δ Ciclicità'),
                                  (3, 1, 'ACF'), (3, 2, 'PACF'),
                                  (4, 1, 'σ_t (%)' if garch_ok else 'ε_t'), (5, 1, 'Prezzo')]:
            fig.update_yaxes(title_text=lbl, row=row_n, col=col_n, title_font=dict(size=9))

        best_pq = (best_order[0], best_order[2])
        aic_lines = []
        for (p, q), sc in sorted(aic_grid.items(), key=lambda x: x[1])[:8]:
            mk = '★' if (p, q) == best_pq else ' '
            aic_lines.append(f"  {mk}ARIMA({p},1,{q})  {criterion.upper()}={sc}")
        status_text = (
            f"✓ Modello: {arima_lbl}  ({criterion.upper()}={best_score:.2f})\n"
            f"✓ ADF sui residui ε_t: p={adf_pval:.4f}  "
            f"→  {'Stazionari ✓' if is_stationary else 'Non stazionari ⚠'}\n"
            f"✓ {garch_info}\n"
            f"✓ Rendimento atteso ({horizon}gg): {ret_sign}{expected_ret:.2f}%  "
            f"(CI: {(price_fc_lo[-1]/last_price-1)*100:+.2f}% / "
            f"{(price_fc_hi[-1]/last_price-1)*100:+.2f}%)\n\n"
            f"Top modelli ({criterion.upper()}):\n" + "\n".join(aic_lines)
        )
        return fig, status_text

    except Exception as exc:
        import traceback as _tb
        print(f"❌ ARIMA Tab error:\n{_tb.format_exc()}", flush=True)
        empty = go.Figure()
        empty.add_annotation(text=f"Errore: {str(exc)}", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False, font=dict(size=14, color='red'))
        return empty, f"❌ Errore: {str(exc)}"


# ── Barra dati: aggiungi asset / carica file / template / esporta / file lavoro ─
@app.callback(
    Output('tab-arima-content', 'children', allow_duplicate=True),
    Output('at-data-status', 'children', allow_duplicate=True),
    Input('at-add-btn', 'n_clicks'),
    State('at-add-ticker', 'value'),
    State('at-add-desc', 'value'),
    State('at-add-cur', 'value'),
    prevent_initial_call=True,
)
def at_add_asset(n, ticker, desc, cur):
    if not n:
        raise PreventUpdate
    ok, msg = _add_asset_to_current(ticker, desc, cur)
    if ok:
        return get_arima_analysis_tab(_asset_options()), msg
    return no_update, msg


@app.callback(
    Output('tab-arima-content', 'children', allow_duplicate=True),
    Output('at-data-status', 'children', allow_duplicate=True),
    Input('at-upload', 'contents'),
    State('at-upload', 'filename'),
    prevent_initial_call=True,
)
def at_upload(contents, filename):
    if not contents:
        raise PreventUpdate
    import base64
    try:
        _, b64 = contents.split(',', 1)
        df = pd.read_excel(io.BytesIO(base64.b64decode(b64)))
    except Exception as e:
        return no_update, f"⚠ File non leggibile: {str(e)[:80]}"
    cols = {str(c).upper().strip(): c for c in df.columns}
    tcol, dcol, ccol = cols.get('TICKER'), cols.get('DESCRIZIONE'), cols.get('VALUTA')
    if not tcol:
        return no_update, "⚠ Manca la colonna TICKER (scarica il Template)"
    added, errs = 0, 0
    for _, r in df.iterrows():
        tk = str(r[tcol]).strip()
        if not tk or tk.lower() == 'nan':
            continue
        ds = str(r[dcol]).strip() if dcol and pd.notna(r[dcol]) else tk
        cu = str(r[ccol]).strip() if ccol and pd.notna(r[ccol]) else 'EUR'
        ok, _m = _add_asset_to_current(tk, ds, cu)
        added += int(ok)
        errs  += int(not ok)
    return (get_arima_analysis_tab(_asset_options()),
            f"✓ Caricati {added} asset dal file" + (f" ({errs} non trovati)" if errs else ""))


@app.callback(
    Output('at-dl-template', 'data'),
    Input('at-template-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def at_template(n):
    if not n:
        raise PreventUpdate
    return dcc.send_bytes(lambda b: b.write(_template_bytes()), 'template_asset.xlsx')


@app.callback(
    Output('at-dl-export', 'data'),
    Input('at-export-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def at_export(n):
    if not n:
        raise PreventUpdate
    return dcc.send_bytes(lambda b: b.write(_export_bytes()), 'dati_analisi_tattica.xlsx')


# ── Pannello File: toggle, lista, salva, carica, elimina ──────────────────────
@app.callback(
    Output('at-file-panel', 'style'),
    Input('at-file-btn', 'n_clicks'),
    Input('at-file-close', 'n_clicks'),
    Input('at-file-close-btn', 'n_clicks'),
    State('at-file-panel', 'style'),
    prevent_initial_call=True,
)
def at_toggle_file_panel(n_open, n_close, n_close2, st):
    # position:fixed + z-index altissimo + montato alla radice → SOPRA tutto
    base = {'position': 'fixed', 'top': '150px', 'left': '1.5%', 'background': 'white',
            'border': '1px solid #ccc', 'border-radius': '8px',
            'box-shadow': '0 8px 30px rgba(0,0,0,0.28)', 'padding': '16px',
            'z-index': 5000, 'min-width': '480px', 'max-width': '95vw'}
    if callback_context.triggered_id in ('at-file-close', 'at-file-close-btn'):
        return {**base, 'display': 'none'}
    if st and st.get('display') != 'none':
        return {**base, 'display': 'none'}
    return {**base, 'display': 'block'}


@app.callback(
    Output('at-file-list', 'children'),
    Input('at-refresh-btn', 'n_clicks'),
    Input('at-save-btn', 'n_clicks'),
    Input('at-file-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def at_refresh_files(*_):
    return _render_file_list()


@app.callback(
    Output('at-save-status', 'children'),
    Output('at-save-name', 'value'),
    Input('at-save-btn', 'n_clicks'),
    State('at-save-name', 'value'),
    prevent_initial_call=True,
)
def at_save(n, name):
    if not n:
        raise PreventUpdate
    ok, msg = _save_profilo(name)
    return msg, ('' if ok else no_update)


@app.callback(
    Output('tab-arima-content', 'children', allow_duplicate=True),
    Output('at-data-status', 'children', allow_duplicate=True),
    Output('at-file-panel', 'style', allow_duplicate=True),
    Input({'type': 'at-fp-load', 'index': ALL}, 'n_clicks'),
    State('at-file-panel', 'style'),
    prevent_initial_call=True,
)
def at_fp_load(all_n, st):
    if not any(all_n or []):
        raise PreventUpdate
    trg = callback_context.triggered_id
    fn = trg.get('index') if isinstance(trg, dict) else None
    ok, msg = _load_profilo(fn)
    if ok:
        return get_arima_analysis_tab(_asset_options()), msg, {**(st or {}), 'display': 'none'}
    return no_update, msg, no_update


@app.callback(
    Output('at-file-list', 'children', allow_duplicate=True),
    Output('at-data-status', 'children', allow_duplicate=True),
    Input({'type': 'at-fp-del', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def at_fp_del(all_n):
    if not any(all_n or []):
        raise PreventUpdate
    trg = callback_context.triggered_id
    fn = trg.get('index') if isinstance(trg, dict) else None
    ok, msg = _delete_profilo(fn)
    return _render_file_list(), msg


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Posizionamento COT: switch tab + aggiornamento grafico
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('tab-arima-content', 'style'),
    Output('tab-cot-content', 'style'),
    Input('at-tabs', 'value'),
)
def _at_switch_tab(tab):
    show = {'display': 'block'}
    hide = {'display': 'none'}
    if tab == 'tab-cot':
        return hide, show
    return show, hide


@app.callback(
    Output('cot-chart', 'figure'),
    Output('cot-summary', 'children'),
    Output('cot-chart', 'style'),
    Input('cot-instrument', 'value'),
    Input('cot-range', 'value'),
    Input('cot-refresh', 'n_clicks'),
)
def _update_cot(key, year_range, n):
    if not key:
        raise PreventUpdate
    # il pulsante 🔄 forza il refetch invalidando la cache dello strumento
    if callback_context.triggered_id == 'cot-refresh':
        _COT_CACHE.pop(key, None)
    fig, summary = _cot_fig_and_summary(key, year_range)
    h = int(getattr(fig.layout, 'height', None) or 1400)
    style = {'width': '100%', 'height': f'{h}px'}
    return fig, summary, style


if __name__ == '__main__':
    app.run(debug=True, port=8060)
