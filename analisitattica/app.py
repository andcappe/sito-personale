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
import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd
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
                     ]),
            # Contenuto della tab ARIMA (altre tab si aggiungeranno qui in futuro)
            html.Div(id='tab-arima-content', children=get_arima_analysis_tab(opts)),
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
            margin=dict(t=70, b=30, l=55, r=175), hovermode='x unified',
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
    State('at-file-panel', 'style'),
    prevent_initial_call=True,
)
def at_toggle_file_panel(n_open, n_close, st):
    # position:fixed + z-index altissimo + montato alla radice → SOPRA tutto
    base = {'position': 'fixed', 'top': '150px', 'left': '1.5%', 'background': 'white',
            'border': '1px solid #ccc', 'border-radius': '8px',
            'box-shadow': '0 8px 30px rgba(0,0,0,0.28)', 'padding': '16px',
            'z-index': 5000, 'min-width': '480px', 'max-width': '95vw'}
    if callback_context.triggered_id == 'at-file-close':
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


if __name__ == '__main__':
    app.run(debug=True, port=8060)
