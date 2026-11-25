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
from navbar import make_navbar, ETICHETTA_PERSONALE, tendina_cell  # noqa: E402
import valutazione                                   # noqa: E402  (sotto-tab Valutazione)
from analisitattica import indicatori                # noqa: E402  (sotto-tab Indicatori)

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
<style>''' + SITE_CSS + indicatori.CSS + valutazione.CSS + '''
  /* Menu «Strumenti» (come in Analisi di Portafoglio): <details> nativo; si chiude
     con un clic fuori o su una voce (script sotto). */
  .pf-strumenti { position: relative; display: inline-block; }
  .pf-strumenti > summary { list-style: none; }
  .pf-strumenti > summary::-webkit-details-marker { display: none; }
  .pf-strumenti-menu { position: absolute; top: calc(100% + 4px); left: 0; z-index: 1500;
    background: #fff; border: 1px solid #c0d0e8; border-radius: 6px; padding: 4px;
    box-shadow: 0 6px 20px rgba(26,58,107,0.18); min-width: 220px;
    display: flex; flex-direction: column; gap: 2px; }
  .pf-strumenti-menu > * { width: 100%; box-sizing: border-box; }
  .pf-voce:hover { background: #f0f4fa !important; }
  @media (max-width: 600px) { .pf-strumenti-menu { left: auto; right: 0; } }
  .at-via { display: none !important; }
  @media (min-width: 901px) { .at-ar-sez, .at-ar-corpo { display: contents; } }
  @media (max-width: 900px) {
    .at-ar-sez { flex: 1 1 100%; border: 1px solid #e2e8f0; border-radius: 6px;
                 background: #f8fafc; padding: 8px 10px; margin: 6px 0; }
    .at-ar-sez .nb-espandi-btn { font-size: 12px; font-weight: 700; color: #1a3a5c; }
    .at-ar-corpo { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  }
  @media (max-width: 900px) {
    .at-info-btn { background-color: #0066cc !important; padding: 7px 10px !important; }
  }
  .at-info-testo { font-size: 12px; color: #33475b; line-height: 1.45; margin: 0 0 8px; }
  .at-info-testo b { color: #1a3a5c; }
  @media (max-width: 900px) {
    .pf-info > .pf-strumenti-menu { position: fixed; top: auto; left: 16px; right: 16px;
      min-width: 0 !important; max-width: none !important; }
  }
  .at-file-aperto:hover { border-color: #1a3a6b !important; }
  /* Analisi ARIMA: sezioni Asset e Grafici, una sotto l'altra a tutta larghezza */
  .at-ar-box { border: 1px solid #e2e8f0; border-radius: 6px; background: #f8fafc;
               padding: 8px 10px; margin: 0 0 10px; }
  .at-ar-box > .nb-espandi-btn { font-size: 13px; font-weight: 700; color: #1a3a5c; margin-bottom: 6px; }
  .at-ar-box #weights-grid-container-arima { background: #fff; padding: 4px; border-radius: 4px; }
  .at-ar-stato:empty { display: none; }
  .at-ar-stato { font-size: 11px; color: #333; padding: 5px 10px; background: #f0f8ff;
                 border: 1px solid #bee3f8; border-radius: 4px; margin-bottom: 8px;
                 white-space: pre-wrap; font-family: monospace; overflow-x: auto; }
  .at-ar-titolo { font-size: 13px; font-weight: 700; color: #1a3a5c; margin: 4px 0 8px; }
  .at-ar-graf { background: #fff; border: 1px solid #e2e8f0; border-radius: 4px;
                margin-bottom: 10px; box-sizing: border-box; padding-top: 8px; }
  .at-ar-gtit { font-size: 13px; font-weight: 600; color: #1a3a5c; margin: 0 10px; }
  .at-ar-gsub { font-size: 11px; color: #555; margin-right: 10px; }
  .cot-nota { font-size: 11px; color: #555; margin-bottom: 6px; }
  .at-ar-gsub::before { content: '· '; }
  @media (max-width: 900px) {
    .at-ar-gtit, .at-ar-gsub { display: block; margin: 0 10px; }
    .at-ar-gsub::before { content: ''; }
  }
  @media (max-width: 900px) {
    .at-ar-box { padding: 8px; }
    .at-ar-graf .modebar-container { display: none; }
    .at-ar-titolo { font-size: 12px; }
  }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}
<script>
document.addEventListener('click', function(e){
  document.querySelectorAll('details.pf-strumenti[open]').forEach(function(d){
    var dentro = d.contains(e.target);
    if (!dentro || e.target.closest('.pf-voce')) setTimeout(function(){ d.removeAttribute('open'); }, 0);
  });
  // Valutazione sul cellulare: un grafico disegnato a sezione chiusa nasce largo 0.
  var testa = e.target.closest && e.target.closest('.val-sezioni .nb-espandi-btn, .at-ar-box > .nb-espandi-btn');
  if (testa && window.Plotly) {
    setTimeout(function(){
      testa.parentNode.querySelectorAll('.js-plotly-plot').forEach(function(g){
        try { Plotly.Plots.resize(g); } catch (err) {}
      });
    }, 60);
  }
});
</script></footer></body></html>'''

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
def _build_asset_grid(asset_names, selected=None):
    """Griglia degli asset del file aperto: una spunta sceglie l'asset per l'analisi
    (una sola alla volta) e lo mette nella tendina in alto (ar_griglia_tendina)."""
    selected = selected or []
    if not asset_names:
        return html.Div("Nessun asset: carica i dati in Analisi di Portafoglio.",
                        style={'color': '#888', 'padding': '12px', 'font-size': '11px'})
    header = html.Div([
        html.Div('Asset', style={'flex': '1 1 auto', 'font-weight': 'bold',
                                  'font-size': '11px', 'padding-left': '4px'}),
        html.Div('Analizza', style={'flex': '0 0 64px', 'font-weight': 'bold',
                                     'font-size': '11px', 'text-align': 'center'}),
    ], style={'display': 'flex', 'margin-bottom': '5px',
              'border-bottom': '2px solid #ccc', 'padding-bottom': '4px', 'align-items': 'center'})

    rows = [header]
    for asset in asset_names:
        rows.append(html.Div([
            html.Div(html.B(asset, style={'overflow': 'hidden', 'text-overflow': 'ellipsis'}), style={
                'flex': '1 1 auto', 'min-width': '0', 'height': '28px', 'display': 'flex',
                'align-items': 'center', 'padding-left': '4px',
                'font-size': '12px', 'overflow': 'hidden', 'white-space': 'nowrap',
                'text-overflow': 'ellipsis'}, title=asset),
            html.Div(dcc.Checklist(
                id={'type': 'graph-select-checkbox-arima', 'index': asset},
                options=[{'label': '', 'value': asset}],
                value=[asset] if asset in selected else [],
                style={'justify-content': 'center'}),
                style={'flex': '0 0 64px', 'height': '28px', 'display': 'flex',
                       'align-items': 'center', 'justify-content': 'center'}),
        ], style={'display': 'flex', 'border-bottom': '1px dotted #eee', 'align-items': 'center'}))

    # ── Riga "Aggiungi asset" IN CODA (come in Analisi di Portafoglio) ────────
    _ai = {'font-size': '12px', 'padding': '4px 6px', 'border': '1px solid #aaa',
           'border-radius': '4px', 'box-sizing': 'border-box', 'min-width': '0'}
    rows.append(html.Div([
        html.Div('➕ Aggiungi asset al file personale',
                 style={'font-size': '12px', 'font-weight': '700', 'color': '#1a3a5c',
                        'margin-bottom': '4px'}),
        html.Div([
            dcc.Input(id='at-add-desc', placeholder='Descrizione',
                      style={**_ai, 'flex': '3 1 0'}),
            dcc.Input(id='at-add-ticker', placeholder='Ticker Yahoo',
                      style={**_ai, 'flex': '2 1 0'}),
            dcc.Input(id='at-add-cur', value='EUR',
                      style={**_ai, 'flex': '0 0 60px'}),
            html.Button('➕ Aggiungi', id='at-add-btn', n_clicks=0,
                    style={'flex': '0 0 auto', 'font-size': '12px', 'padding': '5px 12px',
                           'border': 'none', 'border-radius': '4px', 'cursor': 'pointer',
                           'color': 'white', 'background': '#1b7a34', 'font-weight': 'bold'}),
        ], style={'display': 'flex', 'gap': '4px', 'flex-wrap': 'wrap'}),
    ], style={'margin-top': '10px', 'border-top': '2px solid #ccc', 'padding-top': '8px'}))
    return rows


def get_arima_analysis_tab(options_tickers):
    """Layout dell'Analisi ARIMA per singolo asset."""
    asset_names = [o['value'] for o in (options_tickers or [])]
    first = asset_names[0] if asset_names else None
    return html.Div([
        # ── Header con controlli ──────────────────────────────────────────
        html.Div([
            html.H3('Analisi ARIMA per Asset', className='nb-nascondi-cell',
                    style={'margin-right': '20px', 'white-space': 'nowrap', 'font-size': '16px'}),
            # L'asset si sceglie nella riga in alto (arima-asset-dropdown, _data_toolbar).
            # «Parametri»: sul computer i campi restano nella riga (display: contents,
            # intestazione nascosta); sul cellulare scheda espandibile chiusa.
            html.Div(className='nb-espandi at-ar-sez', children=[
            html.Div(['Parametri', html.I(className='fa-solid fa-chevron-down nb-espandi-ico')],
                     className='nb-espandi-btn nb-testa-cell'),
            html.Div(className='nb-espandi-corpo at-ar-corpo', children=[
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
            ]),
            ]),
            # Sul cellulare il pulsante sta nella riga in alto (at-arima-cell).
            html.Button('▶  Esegui Analisi ARIMA', id='run-arima-tab-button', n_clicks=0,
                        className='nb-nascondi-cell',
                        style={'background-color': '#0066cc', 'color': 'white', 'border': 'none',
                               'padding': '8px 18px', 'border-radius': '4px', 'cursor': 'pointer',
                               'font-weight': 'bold', 'font-size': '12px'}),
        ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap',
                  'gap': '4px', 'margin-bottom': '10px'}),

        # ── Sezioni: Asset, poi Grafici (sul cellulare schede espandibili aperte) ──
        _ar_sezione('Asset', [
            dcc.Loading(type='circle', color='#1b7a34', children=[
                html.Div(id='weights-grid-container-arima',
                         children=_build_asset_grid(asset_names, [])),
            ]),
            html.Div(id='at-add-stato', style={'font-size': '12px', 'color': '#1a3a5c',
                                               'margin-top': '6px'}),
        ]),
        _ar_sezione('Grafici modello ARIMA', [
            html.Div(id='arima-tab-status', className='at-ar-stato'),
            dcc.Loading(id='loading-arima-tab', type='circle', children=[
                html.Div(id='arima-grafici', children=html.Div(
                    "Scegli un asset nella riga in alto e premi «▶ Esegui Analisi ARIMA».",
                    style={'color': '#888', 'font-size': '12px', 'padding': '10px 2px'})),
            ]),
        ]),
    ])


def _ar_sezione(titolo, corpo, aperto=True):
    """Riquadro a tutta larghezza; sul cellulare sezione espandibile, aperta all'avvio
    (aperto=False: chiusa sul cellulare; sul computer è sempre aperta)."""
    return html.Div(className='nb-espandi at-ar-box' + (' nb-aperto' if aperto else ''), children=[
        html.Div([titolo, html.I(className='fa-solid fa-chevron-down nb-espandi-ico')],
                 className='nb-espandi-btn at-ar-box-tit'),
        html.Div(corpo, className='nb-espandi-corpo'),
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
        ], style={'flex': '1 1 140px', 'min-width': '0', 'padding': '10px 14px', 'background': '#f8fafd',
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
        html.Div(cards, style={'display': 'flex', 'gap': '10px', 'flex-wrap': 'wrap'}),
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


# Stocastici del grafico prezzo: finestra base × moltiplicatori (vuoto = non usato).
# 30 × 1, 2, 4, 8 = 30 / 60 / 120 / 240 giorni, le finestre di prima.
_COT_ST_BASE = 30
_COT_ST_MOLT = (2, 4, 8)
_COT_ST_COLORI = (('#0f8a8a', 1.4), ('#e08a1e', 1.4), ('#c0392b', 2.6), ('#1a4fd6', 2.6))


def _cot_finestre_stoc(base, molt):
    """[n1, n2, n3, n4] dei quattro stocastici: base, poi base × ogni moltiplicatore
    (es. 10 · 2 3 4 → 10, 20, 30, 40). Valori vuoti o non validi si saltano."""
    try:
        base = int(float(base))
    except (TypeError, ValueError):
        base = _COT_ST_BASE
    base = max(2, min(base, 1000))
    out = [base]
    for m in molt:
        try:
            n = int(round(base * float(m)))
        except (TypeError, ValueError):
            continue
        if 2 <= n <= 2000 and n not in out:
            out.append(n)
    return out


def _cot_media_stoc(n):
    """Media mobile del %K: come prima 30→5, 60→10, 120→20, 240→30."""
    return int(min(30, max(3, round(n / 6))))


def _cot_figure(key, year_range, finestre=None, cell=False):
    """Grafici separati del tab COT (uno sotto l'altro, titoli fuori da Plotly in
    layout.meta.titolo) + riepilogo. Il primo è il prezzo del sottostante (asse y1)
    con i quattro stocastici (asse y2); poi posizionamento, variazione settimanale,
    drawdown, COT Index e Z-score."""
    finestre = finestre or _cot_finestre_stoc(_COT_ST_BASE, _COT_ST_MOLT)
    cat, label, kind, contract = _COT_INSTRUMENTS[key]
    try:
        df = _cot_fetch(key)
    except Exception as e:
        return [_cot_empty_fig(f'Errore nel recupero dei dati CFTC: {e}')], '', label
    if df is None or df.empty:
        return [_cot_empty_fig('Nessun dato COT disponibile per questo strumento')], '', label
    full = df
    # year_range = [anno_inizio, anno_fine] dallo slider; COT Index / Z-score / stocastico
    # si calcolano su tutta la storia (full) e si ritagliano alla finestra.
    if year_range and len(year_range) == 2:
        y0, y1 = sorted(int(v) for v in year_range)
        start = pd.Timestamp(year=y0, month=1, day=1)
        end = pd.Timestamp(year=y1, month=12, day=31, hour=23, minute=59, second=59)
        df = df.loc[(df.index >= start) & (df.index <= end)]
    if df.empty:
        df = full.tail(2)

    price_full = _cot_price(key)
    price = None
    if price_full is not None and not price_full.empty:
        price = price_full[(price_full.index >= df.index.min()) &
                           (price_full.index <= df.index.max())]
    has_price = price is not None and not price.empty
    price_label = _COT_PRICE_NAME.get(key, label)
    disp = df.index
    x0 = min(disp.min(), price.index.min()) if has_price else disp.min()
    x1 = max(disp.max(), price.index.max()) if has_price else disp.max()

    figure = []
    fig = None

    def nuovo(titolo, alto_pc, alto_cell, sec=False):
        nonlocal fig
        fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': sec}]])
        fig.update_layout(meta={'titolo': titolo}, height=alto_cell if cell else alto_pc)
        figure.append(fig)

    def _yr(series):
        s = series.dropna()
        if s.empty:
            return None
        return [float(s.min()) - 10, float(s.max()) + 10]

    def _add_price():
        if has_price:
            fig.add_trace(go.Scatter(
                x=price.index, y=price.values, name=f'Prezzo {price_label}',
                mode='lines', line=dict(color='#444', width=1.5),
                hovertemplate='%{x|%d %b %Y}<br>Prezzo: %{y:,.2f}<extra></extra>'),
                secondary_y=True)
            fig.update_yaxes(title_text=price_label, showgrid=False, zeroline=False,
                             secondary_y=True)

    def _add_pos(serie, colore, nome, ax_bg):
        """Posizionamento netto (media 4 sett.) + eco della variazione settimanale + prezzo."""
        _add_wow_bg(full_col[nome], colore, nome)
        s = serie.rolling(4, min_periods=1).mean().reindex(disp)
        fig.add_trace(go.Scatter(
            x=disp, y=s, name=nome, mode='lines', line=dict(color=colore, width=2.2),
            hovertemplate='%{x|%d %b %Y}<br>' + nome + ': %{y:,.0f}<extra></extra>'),
            secondary_y=False)
        _add_price()
        fig.add_hline(y=0, line=dict(color='#999', width=1), secondary_y=False)
        fig.update_yaxes(title_text='Contratti netti', showgrid=True, gridcolor='#eef1f5',
                         zeroline=False, secondary_y=False)
        yr = _yr(s)
        if yr:
            fig.update_yaxes(range=yr, secondary_y=False)

    def _add_wow_bg(series_full, hexcol, name):
        """Eco tenue della variazione settimanale (media 4 sett.) dietro il posizionamento,
        su un asse nascosto con scala propria (va aggiunta prima della linea)."""
        h = hexcol.lstrip('#')
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        var = series_full.diff().rolling(4, min_periods=1).mean().reindex(disp)
        fig.add_trace(go.Scatter(
            x=disp, y=var, xaxis='x', yaxis='y9', showlegend=False,
            name=f'Variazione sett. {name} (media 4 sett.)',
            mode='lines', line=dict(color=f'rgba({r},{g},{b},0.35)', width=1.1),
            fill='tozeroy', fillcolor=f'rgba({r},{g},{b},0.08)',
            hovertemplate='%{x|%d %b %Y}<br>Variazione sett. ' + name
                          + ' (media 4 sett.): %{y:,.0f}<extra></extra>'))
        fig.update_layout(yaxis9=dict(overlaying='y', anchor='x', showgrid=False,
                                      zeroline=False, visible=False))

    def _add_wow(pairs):
        shown = []
        for series_full, color, name in pairs:
            raw = series_full.diff()
            ma = raw.rolling(4, min_periods=1).mean()
            shown.append(ma.reindex(disp))
            fig.add_trace(go.Scatter(
                x=disp, y=raw.reindex(disp), name=f'Δsett {name}', showlegend=False,
                mode='lines', line=dict(color=color, width=0.8), opacity=0.30,
                hoverinfo='skip'))
            fig.add_trace(go.Scatter(
                x=disp, y=ma.reindex(disp), name=f'{name} (media 4 sett.)',
                mode='lines', line=dict(color=color, width=2.0),
                hovertemplate='%{x|%d %b %Y}<br>Δ4s ' + name + ': %{y:,.0f}<extra></extra>'))
        fig.add_hline(y=0, line=dict(color='#999', width=1))
        fig.update_yaxes(title_text='Δ contratti / sett.', showgrid=True,
                         gridcolor='#eef1f5', zeroline=False)
        allv = pd.concat(shown).dropna() if shown else pd.Series(dtype=float)
        if not allv.empty:
            fig.update_yaxes(range=[float(allv.min()) - 10, float(allv.max()) + 10])

    def _add_cot_index(series_full, color, name):
        ci = _cot_index_series(series_full).rolling(8, min_periods=1).mean().reindex(disp)
        fig.add_trace(go.Scatter(
            x=disp, y=ci, name=name, mode='lines', line=dict(color=color, width=1.8),
            hovertemplate='%{x|%d %b %Y}<br>' + name + ': %{y:.0f}<extra></extra>'))

    def _add_zscore(series_full, color, name):
        z = _cot_z_series(series_full).rolling(8, min_periods=1).mean().reindex(disp)
        fig.add_trace(go.Scatter(
            x=disp, y=z, name=name, mode='lines', line=dict(color=color, width=1.8),
            hovertemplate='%{x|%d %b %Y}<br>' + name + ': %{y:.2f}<extra></extra>'))

    def _soglie(valori, secondary_y=None):
        """Linee di riferimento come tracce (non shape: il tab può essere nascosto)."""
        for y, c, d in valori:
            kw = {} if secondary_y is None else {'secondary_y': secondary_y}
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode='lines', showlegend=False,
                                     hoverinfo='skip', line=dict(color=c, width=1, dash=d)), **kw)

    # ── 1) Prezzo (y1) e stocastici (y2) ──────────────────────────────────────
    if has_price:
        fin_txt = ' · '.join(f'{n} gg' for n in finestre)
        nuovo(f'{price_label}: prezzo e stocastico  ·  {fin_txt} (%K con media mobile)',
              560, 440, sec=True)
        fig.add_trace(go.Scatter(
            x=price.index, y=price.values, name=f'Prezzo {price_label}',
            mode='lines', line=dict(color='#222', width=1.8),
            hovertemplate='%{x|%d %b %Y}<br>Prezzo: %{y:,.2f}<extra></extra>'),
            secondary_y=False)
        _soglie([(80, '#bbb', 'dot'), (20, '#bbb', 'dot')], secondary_y=True)
        for i, n in enumerate(finestre):
            colore, larg = _COT_ST_COLORI[i % len(_COT_ST_COLORI)]
            sm = _cot_media_stoc(n)
            st = _stoch_series(price_full, n, smooth=sm).reindex(price.index)
            fig.add_trace(go.Scatter(
                x=st.index, y=st.values, name=f'Stocastico {n} gg', opacity=0.85,
                mode='lines', line=dict(color=colore, width=larg),
                hovertemplate=f'%{{x|%d %b %Y}}<br>Stocastico {n} gg (media {sm}): '
                              f'%{{y:.0f}}<extra></extra>'),
                secondary_y=True)
        fig.update_yaxes(title_text=f'Prezzo {price_label}', showgrid=True, gridcolor='#eef1f5',
                         secondary_y=False)
        fig.update_yaxes(title_text='Stocastico', range=[0, 100], showgrid=False,
                         secondary_y=True)

    # ── Drawdown del prezzo, subito sotto prezzo e stocastici ─────────────────
    if has_price:
        nuovo(f'Drawdown {price_label}  ·  % dal massimo progressivo', 300, 260)
        dd = _drawdown_series(price)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=f'Drawdown {price_label}', mode='lines',
            line=dict(color='#8a1a1a', width=1.3),
            fill='tozeroy', fillcolor='rgba(192,57,43,0.15)',
            hovertemplate='%{x|%d %b %Y}<br>Drawdown: %{y:.1f}%<extra></extra>'))
        fig.update_yaxes(title_text='Drawdown', ticksuffix='%', showgrid=True,
                         gridcolor='#eef1f5')

    # ── 2) Posizionamento ─────────────────────────────────────────────────────
    if kind == 'tff':
        full_col = {'Asset Manager': full['am_net'], 'Leveraged Money': full['lm_net']}
        nuovo('Asset Manager — istituzionali  ·  contratti netti, media 4 sett.', 400, 320,
              sec=True)
        _add_pos(full['am_net'], '#1a3a5c', 'Asset Manager', None)
        nuovo('Leveraged Money — hedge fund / CTA  ·  contratti netti, media 4 sett.', 400, 320,
              sec=True)
        _add_pos(full['lm_net'], '#c0392b', 'Leveraged Money', None)
        nuovo('Somma AM + LM  ·  contratti netti totali, media 4 sett.', 340, 280)
        s = (full['am_net'].rolling(4, min_periods=1).mean()
             + full['lm_net'].rolling(4, min_periods=1).mean()).reindex(disp)
        fig.add_trace(go.Scatter(
            x=disp, y=s, name='Somma AM + LM', mode='lines',
            line=dict(color='#6a3d9a', width=2.2),
            fill='tozeroy', fillcolor='rgba(106,61,154,0.10)',
            hovertemplate='%{x|%d %b %Y}<br>AM + LM: %{y:,.0f}<extra></extra>'))
        fig.add_hline(y=0, line=dict(color='#999', width=1))
        fig.update_yaxes(title_text='AM + LM (netti)', showgrid=True, gridcolor='#eef1f5',
                         zeroline=False)
        yr = _yr(s)
        if yr:
            fig.update_yaxes(range=yr)
        gruppi = [(full['am_net'], '#1a3a5c', 'Asset Manager'),
                  (full['lm_net'], '#c0392b', 'Leveraged Money')]
        nomi_idx = [(full['am_net'], '#1a3a5c', 'AM'), (full['lm_net'], '#c0392b', 'LM')]
    else:
        has_comm = 'comm_net' in df.columns and df['comm_net'].notna().any()
        full_col = {'Managed Money': full['mm_net']}
        nuovo('Managed Money — hedge fund  ·  contratti netti, media 4 sett.', 400, 320, sec=True)
        _add_pos(full['mm_net'], '#1b7a34', 'Managed Money', None)
        gruppi = [(full['mm_net'], '#1b7a34', 'Managed Money')]
        nomi_idx = [(full['mm_net'], '#1b7a34', 'MM')]
        if has_comm:
            full_col['Commercial'] = full['comm_net']
            nuovo('Commercial — Producer/Merchant + Swap Dealers  ·  contratti netti, media 4 sett.',
                  400, 320, sec=True)
            _add_pos(full['comm_net'], '#1a3a5c', 'Commercial', None)
            gruppi.append((full['comm_net'], '#1a3a5c', 'Commercial'))
            nomi_idx.append((full['comm_net'], '#1a3a5c', 'Commercial'))

    # ── 3) Variazione settimanale, drawdown, COT Index, Z-score ───────────────
    nuovo('Variazione settimanale dei contratti netti  ·  media 4 sett. (linea chiara = dato '
          'della settimana)', 380, 300)
    _add_wow(gruppi)
    nuovo('COT Index  ·  0–100, finestra 1 anno, media 8 sett.', 300, 260)
    _soglie([(80, '#bbb', 'dot'), (20, '#bbb', 'dot')])
    for s, c, n in nomi_idx:
        _add_cot_index(s, c, f'COT Index {n}')
    fig.update_yaxes(title_text='COT Index', range=[0, 100], showgrid=True, gridcolor='#eef1f5')
    nuovo('Z-score  ·  deviazioni standard, finestra 1 anno, media 8 sett.', 300, 260)
    _soglie([(0, '#999', 'solid'), (2, '#bbb', 'dot'), (-2, '#bbb', 'dot')])
    for s, c, n in nomi_idx:
        _add_zscore(s, c, f'Z-score {n}')
    fig.update_yaxes(title_text='Z-score', showgrid=True, gridcolor='#eef1f5')

    # ── Aspetto comune: stessa finestra sull'asse x, legenda in alto ─────────
    for f in figure:
        f.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', hovermode='x unified' if not cell else 'closest',
            margin=(dict(l=44, r=44, t=36, b=28) if cell else dict(l=70, r=70, t=36, b=30)),
            legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0,
                        font=dict(size=10 if cell else 11)),
            font=dict(size=10 if cell else 12))
        f.update_xaxes(showgrid=True, gridcolor='#eef1f5', range=[x0, x1])
        if cell:   # sul cellulare i titoli degli assi rubano metà della larghezza
            f.update_yaxes(title_text=None, tickfont=dict(size=9))
    return figure, _cot_summary(kind, full, label), label


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
    """Layout della tab Posizionamento COT. Strumento e 🔄 Aggiorna stanno nella riga in
    alto (cot-cerca-riga, _data_toolbar); qui le sezioni Finestra Temporale, Parametri
    e Grafici (sul cellulare schede espandibili)."""
    _num = {'width': '64px', 'font-size': '12px', 'padding': '4px 6px',
            'border': '1px solid #aaa', 'border-radius': '4px'}
    _et = {'font-size': '11px', 'color': '#1a3a5c', 'font-weight': '600', 'display': 'block',
           'margin-bottom': '2px'}

    def campo(etichetta, cid, valore):
        return html.Div([html.Label(etichetta, style=_et),
                         dcc.Input(id=cid, type='number', value=valore, min=0, step='any',
                                   debounce=True, persistence=True, persistence_type='local',
                                   style=_num)])

    return html.Div([
        _ar_sezione('Finestra Temporale', [
            html.Div(dcc.RangeSlider(
                id='cot-range', min=_COT_YEAR_MIN, max=_COT_YEAR_MAX, step=1,
                value=[_COT_YEAR_MAX - 2, _COT_YEAR_MAX], allowCross=False,
                marks=_cot_year_marks(),
                tooltip={'placement': 'bottom', 'always_visible': False}),
                style={'padding': '6px 4px 0', 'max-width': '900px'}),
        ], aperto=False),
        _ar_sezione('Parametri', [
            html.Div('Stocastici del grafico prezzo: finestra base in giorni e tre '
                     'moltiplicatori (10 · 2 · 3 · 4 = 10, 20, 30, 40 giorni; vuoto = non usato).',
                     className='cot-nota'),
            html.Div([
                campo('Stocastico (gg)', 'cot-st-base', _COT_ST_BASE),
                campo('× 2º', 'cot-st-m1', _COT_ST_MOLT[0]),
                campo('× 3º', 'cot-st-m2', _COT_ST_MOLT[1]),
                campo('× 4º', 'cot-st-m3', _COT_ST_MOLT[2]),
                html.Div(id='cot-st-finestre', className='cot-nota',
                         style={'align-self': 'flex-end', 'padding-bottom': '6px'}),
            ], style={'display': 'flex', 'gap': '10px', 'flex-wrap': 'wrap'}),
        ], aperto=False),
        _ar_sezione('Grafici', [
            html.Div(id='cot-summary', style={'margin-bottom': '10px'}),
            dcc.Loading(id='loading-cot', type='circle', color='#1a3a5c', children=[
                html.Div(id='cot-grafici'),
            ]),
        ]),
    ])


# Stile tab (uguale ad Analisi di Portafoglio)
_TAB_STYLE = {'font-size': '12px', 'padding': '8px 18px'}
_TAB_SEL   = {'font-size': '12px', 'padding': '8px 18px',
              'font-weight': 'bold', 'border-top': '3px solid #1a3a5c'}

_INP = {'font-size': '11px', 'padding': '5px 8px', 'border': '1px solid #aaa',
        'border-radius': '4px'}


def _btn(bg):
    return {'font-size': '11px', 'padding': '6px 12px', 'border': 'none',
            'border-radius': '4px', 'cursor': 'pointer', 'color': 'white',
            'background': bg, 'font-weight': 'bold'}


# Pulsanti principali e voci del menu «Strumenti»: stessi stili di Analisi di
# Portafoglio (_PF_BTN_PRINCIPALE, _PF_VOCE in portafoglio/app.py).
_BTN_PRINCIPALE = {'background-color': '#1a3a6b', 'color': 'white', 'border': 'none',
                   'padding': '6px 14px', 'border-radius': '4px', 'cursor': 'pointer',
                   'font-weight': '600', 'font-size': '12px', 'line-height': '18px',
                   'display': 'inline-block', 'user-select': 'none'}
_VOCE = {'font-size': '12px', 'padding': '7px 12px', 'border-radius': '4px',
         'cursor': 'pointer', 'background': '#ffffff', 'border': '1px solid transparent',
         'color': '#1a3a5c', 'text-align': 'left', 'width': '100%',
         'font-weight': '500', 'display': 'block'}


def _file_aperto():
    """Indicatore del file di lavoro (current.json): personale o uno dei predefiniti,
    con lo stesso nome del selettore di Analisi di Portafoglio. Si cambia lì."""
    tipo = str(dc.read_meta().get('_tipo') or '')
    n = len(dc.read_current())
    if tipo.startswith('default:'):
        nome, nota = Path(tipo.split(':', 1)[1].strip()).stem, 'file predefinito'
    else:                       # 'personale' o tipo assente (lavoro caricato / file vecchio)
        nome, nota = ETICHETTA_PERSONALE, 'file personale'
    return [html.Span(nome, style={'font-weight': '600', 'color': '#1a3a5c'})], \
        f'File aperto: {nome} ({nota}, {n} asset). Per cambiarlo vai in Analisi Strategica.'


# ── Menu ⚙ «Descrizione»: i testi esplicativi tolti dalle pagine, per sotto-tab ──
_INFO_TITOLO = {'font-size': '11px', 'font-weight': '600', 'color': '#5a6b85',
                'text-transform': 'uppercase', 'letter-spacing': '0.04em', 'margin-bottom': '6px'}

_INFO = {
    'tab-valutazione': ('💹 Valutazione', [
        ('Modelli', 'DCF · DDM · Graham · P/E relativo · EV/EBITDA · sensitività WACC × g (dentro DCF) · '
                    'SaaS. Dati fondamentali da Yahoo Finance, prospetti di bilancio da Alpha Vantage.'),
        ('Cerca titolo', 'Scrivi il nome della società o il ticker Yahoo nella casella in alto '
                         '(es. «apple», «eni», ENI.MI), scegli dalla lista e premi ▶ Carica ticker.'),
        ('⟳ Aggiorna bilanci', 'Solo dopo una trimestrale appena uscita: costa 3 delle 25 '
                               'richieste giornaliere di Alpha Vantage.'),
        ('🎯 Valori consigliati', 'Li applica già ▶ al caricamento. Il tab 🎯 Parametri mostra '
                                 'da dove viene ogni numero.'),
        ('Tasso finale', 'Crescita perpetua: il massimo segue il WACC (resta un quarto di punto '
                         'sotto), perché una crescita perpetua pari o superiore al tasso di sconto fa '
                         'divergere Gordon. Alza il WACC se serve più corsa. In pratica 2-3% '
                         '(inflazione + crescita reale): per una crescita alta ma temporanea usa le '
                         'due fasi sopra.'),
        ('Bilanci — base di calcolo', 'Vale per i grafici e le tabelle del tab Bilanci. Per la '
                                      'variazione anno su anno scegli «Annuali».'),
    ]),
    'tab-indicatori': ('📐 Indicatori', [
        ('Finestre', 'Nell\'unità della periodicità (giorni di borsa, settimane o mesi). RSI e '
                     'stocastico: la finestra più altre tre = finestra × moltiplicatore (vuoto = non usata).'),
        ('Medie mobili', 'Le tre medie si confrontano a coppie (veloce − lenta, in % della lenta).'),
        ('Dati', 'Prezzi in EUR del file aperto; lo stocastico usa le sole chiusure.'),
        ('Colori', 'Rosso = sopra la soglia alta (ipercomprato), verde = sotto la bassa '
                   '(ipervenduto); per le medie verde = veloce sopra la lenta.'),
        ('Finestra Temporale', 'È la stessa delle altre pagine: la tabella dà i valori alla data A. '
                               'Il periodo del grafico parte da lì e si cambia sopra il grafico '
                               '(vale solo per il grafico). Clic su una riga per il grafico.'),
        ('Aggiungi asset', 'Scarica 10 anni da Yahoo, converte in EUR e accoda l\'asset al file '
                           'personale: lo trovi anche in Analisi di Portafoglio (senza spunte né pesi). '
                           'Una valuta sbagliata si corregge da sola con quella dichiarata da Yahoo.'),
    ]),
    'tab-arima': ('📉 Analisi ARIMA', [
        ('Analisi', 'Per l\'asset scelto cerca l\'ARIMA migliore fino a Max p / Max q secondo il '
                    'criterio (AIC o BIC) e prevede per l\'orizzonte indicato, in giorni di borsa.'),
    ]),
    'tab-cot': ('📊 Posizionamento COT', [
        ('Mercato', 'Si sceglie nella riga in alto, accanto al file aperto; 🔄 Aggiorna '
                    'riscarica subito i dati CFTC dello strumento.'),
        ('Fonte', 'CFTC — Commitments of Traders, pubblicato ogni venerdì.'),
        ('Finestra Temporale', 'Anni mostrati nei grafici. COT Index, Z-score e stocastici '
                               'si calcolano su tutta la storia e poi si ritagliano.'),
        ('Parametri', 'Quattro stocastici sul grafico del prezzo: finestra base in giorni e tre '
                      'moltiplicatori del primo valore (10 · 2 · 3 · 4 = 10, 20, 30, 40 giorni). '
                      'Il %K è smussato con una media di circa 1/6 della finestra (min 3, max 30). '
                      'Un moltiplicatore vuoto toglie quello stocastico.'),
        ('Grafici', 'Primo: prezzo (asse sinistro) e stocastici (asse destro, 0–100, linee 80/20); '
                    'sotto il drawdown del prezzo. Poi posizionamento netto, variazione settimanale, '
                    'COT Index, Z-score.'),
        ('Indici, valute e tassi', 'Report TFF: posizioni nette di Asset Manager (fondi '
                                   'pensione/comuni/assicurazioni, medio-lungo termine) e Leveraged '
                                   'Money (hedge fund/CTA, speculativi).'),
        ('Materie prime', 'Report Disaggregated: Managed Money (speculazione hedge fund) e '
                          'Commercial (hedger = Producer/Merchant + Swap Dealers).'),
    ]),
}


def _info_tab(tab):
    """Contenuto del menu ⚙ per il sotto-tab aperto."""
    titolo, voci = _INFO.get(tab) or _INFO['tab-valutazione']
    return [html.Div(f'Descrizione · {titolo}', style=_INFO_TITOLO)] + [
        html.P([html.B(f'{t}: '), testo], className='at-info-testo') for t, testo in voci]


def _file_panel():
    """Solo il pulsante 📁 File (il pannello vero è _file_modal(), alla radice)."""
    return html.Button('📁 File', id='at-file-btn', n_clicks=0, className='pf-voce', style=_VOCE)


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
                                title='Salva dataset + pesi + analisi (tutti i dati)',
                                style={**_bb, 'background': '#1b7a34', 'color': 'white', 'width': '100%'}),
                    html.Div(id='at-save-status',
                             style={'font-size': '10px', 'margin-top': '5px', 'color': '#555',
                                    'min-height': '16px'}),
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
            # File aperto (personale o predefinito): solo indicatore, si cambia nel Portafoglio.
            html.A(id='at-file-aperto', className='at-file-aperto', href='/portafoglio/',
                   children=_file_aperto()[0], title=_file_aperto()[1],
                   style={'display': 'inline-flex', 'align-items': 'center', 'gap': '2px',
                          'font-size': '12px', 'padding': '5px 12px', 'border': '1px solid #b8c4d4',
                          'border-radius': '4px', 'background': '#fff', 'text-decoration': 'none',
                          'white-space': 'nowrap', 'line-height': '18px'}),
            # Valutazione: ricerca del titolo + ▶ Carica ticker (sul cellulare a capo,
            # sotto il file aperto). Visibili solo sul sotto-tab Valutazione (at_info).
            html.Div(id='val-a-capo', className='val-a-capo'),
            valutazione.barra_ricerca(),
            # ARIMA: asset da analizzare = ticker/nome cercato su Yahoo («yf:TICKER») oppure,
            # dalla freccia, uno degli asset del file aperto (personale o predefinito).
            # Visibile solo sul sotto-tab ARIMA (at_info).
            html.Div(id='ar-cerca-riga', className='at-via',
                     title='Usa nome o ticker di Yahoo, oppure scegli un asset del file', children=[
                dcc.Dropdown(id='arima-asset-dropdown',
                             options=[{'label': 'AAPL — Apple Inc. · Yahoo', 'value': 'yf:AAPL'}],
                             value='yf:AAPL', searchable=True, clearable=False,
                             placeholder='Usa nome o ticker di Yahoo…',
                             className='val-cerca-campo', optionHeight=44,
                             style={'font-size': '12px'}),
            ]),
            # COT: mercato + 🔄 Aggiorna, visibili solo sul sotto-tab COT (at_info).
            html.Div(id='cot-cerca-riga', className='at-via', children=[
                dcc.Dropdown(id='cot-instrument', options=_cot_options(), value='sp500',
                             clearable=False, searchable=True, className='val-cerca-campo',
                             style={'font-size': '12px'}),
                html.Button('🔄 Aggiorna', id='cot-refresh', n_clicks=0,
                            title='Riscarica subito i dati CFTC dello strumento',
                            style={k: v for k, v in _BTN_PRINCIPALE.items() if k != 'display'}),
            ]),
            # Menu «Strumenti»: File · Carica file · Template · Esporta dati, voci uguali.
            # Solo da PC: sul cellulare il menu Strumenti non c'è (scelta dell'utente).
            html.Details(className='pf-strumenti nb-nascondi-cell', children=[
                html.Summary('🧰 Strumenti ▾', style=_BTN_PRINCIPALE),
                html.Div(className='pf-strumenti-menu', children=[
                    _file_panel(),
                    dcc.Upload(id='at-upload', className='pf-voce', multiple=False,
                               children=html.Div(['⬆ Carica file']), style=_VOCE),
                    html.Button('📋 Template', id='at-template-btn', n_clicks=0, className='pf-voce',
                                title='Scarica il file Excel da compilare con TICKER, DESCRIZIONE, VALUTA',
                                style=_VOCE),
                    html.Button('📤 Esporta dati', id='at-export-btn', n_clicks=0, className='pf-voce',
                                title='Scarica i dati di Analisi Tattica come file Excel', style=_VOCE),
                ]),
            ]),
            # Sul cellulare, tab Indicatori: preme ▶ Calcola indicatori (nb-clicca, delega in NAVBAR_JS).
            html.Button('▶ Calcola indicatori', id='at-calcola-cell', className='at-via',
                        **{'data-bersaglio': 'ind-calcola'},
                        style={**{k: v for k, v in _BTN_PRINCIPALE.items() if k != 'display'},
                               'white-space': 'nowrap', 'padding': '7px 12px'}),
            # Sul cellulare, tab ARIMA: preme ▶ Esegui Analisi ARIMA.
            html.Button('▶ Esegui ARIMA', id='at-arima-cell', className='at-via',
                        **{'data-bersaglio': 'run-arima-tab-button'},
                        style={**{k: v for k, v in _BTN_PRINCIPALE.items() if k != 'display'},
                               'white-space': 'nowrap', 'padding': '7px 12px'}),
            # ⚙ Descrizione: i testi esplicativi del sotto-tab aperto (come il ⚙ del Portafoglio).
            html.Details(className='pf-strumenti pf-info', children=[
                html.Summary([html.I(className='fa-solid fa-gear nb-nascondi-cell'),
                              html.Span('⚙', className='nb-solo-cell')],
                             className='at-info-btn', title='Descrizione',
                             style={**_BTN_PRINCIPALE, 'padding': '6px 9px'}),
                html.Div(id='at-info-corpo', className='pf-strumenti-menu',
                         children=_info_tab('tab-valutazione'),
                         style={'padding': '10px 12px', 'gap': '0', 'min-width': '320px',
                                'max-width': '420px'}),
            ]),
        ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap',
                  'gap': '6px', 'margin-bottom': '6px'}),
        dcc.Loading(type='default', color='#1b7a34', children=[
            html.Div(id='at-data-status',
                     style={'font-size': '11px', 'color': '#555', 'min-height': '16px',
                            'margin-bottom': '6px'})]),
        dcc.Download(id='at-dl-template'),
        dcc.Download(id='at-dl-export'),
    ])


# Sotto-tab: una sola lista per i dcc.Tabs e la tendina del cellulare.
_AT_SOTTOTAB = [
    ('💹 Valutazione',          'tab-valutazione'),
    ('📐 Indicatori',           'tab-indicatori'),
    ('📉 Analisi ARIMA',        'tab-arima'),
    ('📊 Posizionamento COT',   'tab-cot'),
]


def serve_layout():
    opts = _asset_options()
    return html.Div([
        make_navbar('Analisi Tattica', margine='calc(8px + 1%)'),  # = .page-wrap
        html.Div([
            # Niente intestazione: il nome della dashboard sta già nella navbar (scelta dell'utente).
            _data_toolbar(),
            # `serve_layout` è una funzione: a ogni ricarica della pagina il
            # valore tornerebbe al primo tab. Con persistence il tab aperto sopravvive.
            # Ordine (24/11/2026, scelta dell'utente): Valutazione, Indicatori, ARIMA, COT.
            # Sul cellulare i dcc.Tabs si nascondono e al loro posto c'è la tendina
            # «Tabs Strumenti», identica a quella dell'Analisi Strategica.
            tendina_cell('at-tendina', _AT_SOTTOTAB),
            dcc.Tabs(id='at-tabs', value='tab-valutazione',
                     persistence=True, persistence_type='session',
                     parent_className='nb-nascondi-cell',
                     colors={'border': '#dee2e6', 'primary': '#1a3a5c',
                             'background': '#f0f4fa'},
                     style={'margin-bottom': '10px'},
                     children=[
                         dcc.Tab(label=lab, value=val, style=_TAB_STYLE, selected_style=_TAB_SEL)
                         for lab, val in _AT_SOTTOTAB
                     ]),
            # Contenuto delle tab: tutte restano nel DOM e si mostrano/nascondono
            # via callback (Input at-tabs), così le callback ARIMA che dipendono
            # dai suoi componenti restano sempre agganciate.
            html.Div(id='tab-valutazione-content', children=valutazione.layout(),
                     style={'display': 'block'}),
            html.Div(id='tab-indicatori-content', children=indicatori.layout(),
                     style={'display': 'none'}),
            html.Div(id='tab-arima-content', children=get_arima_analysis_tab(opts),
                     style={'display': 'none'}),
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
    Output({'type': 'graph-select-checkbox-arima', 'index': ALL}, 'value'),
    Output('arima-asset-dropdown', 'value', allow_duplicate=True),
    Output('arima-asset-dropdown', 'options', allow_duplicate=True),
    Input({'type': 'graph-select-checkbox-arima', 'index': ALL}, 'value'),
    Input('arima-asset-dropdown', 'value'),
    State({'type': 'graph-select-checkbox-arima', 'index': ALL}, 'id'),
    prevent_initial_call=True,
)
def ar_griglia_tendina(valori, scelto, ids):
    """Griglia e tendina in alto restano allineate, una sola spunta alla volta.
    Spunta nella griglia → l'asset va subito nella tendina (valore e opzioni);
    asset scelto dalla tendina → la griglia spunta lui, o niente se è un titolo Yahoo."""
    tid = callback_context.triggered_id
    nomi = [i['index'] for i in ids]
    if isinstance(tid, dict):                       # clic su una spunta della griglia
        if tid['index'] not in nomi:
            raise PreventUpdate
        acceso = bool(valori[nomi.index(tid['index'])])
        if not acceso:                               # tolta: la tendina resta com'è
            return [[] for _ in nomi], no_update, no_update
        a = tid['index']
        return ([[n] if n == a else [] for n in nomi], a, _ar_opzioni('', a))
    attese = [[n] if n == scelto else [] for n in nomi]
    if attese == [list(v or []) for v in valori]:
        raise PreventUpdate
    return attese, no_update, no_update


def _ar_grafici(titolo, figure):
    """Titolo del modello + un dcc.Graph per figura, uno sotto l'altro.
    responsive=True con l'altezza nello style: un grafico disegnato a tab o sezione
    nascosta nascerebbe largo quanto misurato da nascosto."""
    out = [html.Div(titolo, className='at-ar-titolo')]
    for f in figure:
        f.update_traces(mode='lines', selector=dict(type='scatter'))   # niente pallini sulle linee
        testa, _, coda = (f.layout.meta or {}).get('titolo', '').partition('  ·  ')
        out.append(html.Div(className='at-ar-graf', children=[
            html.Div([html.Span(testa, className='at-ar-gtit'),
                      html.Span(coda, className='at-ar-gsub') if coda else None]),
            dcc.Graph(figure=f, responsive=True,
                      style={'height': f'{f.layout.height}px', 'width': '100%'},
                      config={'scrollZoom': True, 'displaylogo': False}),
        ]))
    return out


@app.callback(
    Output('arima-grafici', 'children'),
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
        return [], "⚠ Seleziona un asset."

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
        if str(selected_asset).startswith('yf:'):
            # Ticker cercato su Yahoo, non nel file: 10 anni scaricati al volo, in EUR.
            tk = selected_asset[3:]
            px = dc.download_series(tk, 'EUR')
            if px is None or px.empty:
                return [], f"⚠ Nessun dato da Yahoo per '{tk}' (ticker corretto?)."
            prices = px.dropna()
            selected_asset = _AR_NOMI.get(selected_asset, tk).split(' · ')[0]
        else:
            original_prices = _build_prices()
            if original_prices is None or selected_asset not in original_prices.columns:
                return [], f"⚠ Asset '{selected_asset}' non trovato nei dati (current.json)."
            prices = original_prices[selected_asset].dropna()
        if len(prices) < 100:
            return [], "⚠ Dati insufficienti (< 100 osservazioni)."
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
            return [], "❌ Nessun modello ARIMA convergente trovato."

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

        # ── Un grafico per riquadro, uno sotto l'altro: prima la previsione ──
        def _fig(titolo, altezza):
            # Il titolo non va in Plotly ma in un'intestazione HTML sopra il grafico
            # (_ar_grafici), che sul cellulare va a capo da sola.
            f = go.Figure()
            f.update_layout(
                height=altezza, meta={'titolo': titolo},
                margin=dict(t=40, b=30, l=55, r=15), hovermode='x unified',
                plot_bgcolor='white', paper_bgcolor='white',
                legend=dict(orientation='h', x=0, y=1.0, xanchor='left', yanchor='bottom',
                            font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
            )
            f.update_xaxes(showgrid=True, gridcolor='#eef1f5')
            f.update_yaxes(showgrid=True, gridcolor='#eef1f5', title_font=dict(size=10))
            return f

        stato_staz = "✓ Staz." if is_stationary else "⚠ Non staz."
        figure = []

        # 1. Proiezione prezzi + cono 95%
        lookback    = min(252 * 2, len(prices))
        hist_prices = prices.iloc[-lookback:]
        f = _fig(f'Proiezione {horizon}gg — Prezzi + Cono 95%  ·  '
                 f'rendimento atteso {ret_sign}{expected_ret:.2f}%', 520)
        f.add_trace(go.Scatter(x=hist_prices.index, y=hist_prices.values, name='Prezzo storico',
                               line=dict(color='#1f77b4', width=1.8)))
        f.add_trace(go.Scatter(x=anchor_dates + anchor_dates[::-1], y=anchor_hi + anchor_lo[::-1],
                               fill='toself', fillcolor='rgba(31,119,180,0.13)',
                               line=dict(color='rgba(0,0,0,0)'), name='Cono 95%', hoverinfo='skip'))
        f.add_trace(go.Scatter(x=anchor_dates, y=anchor_hi, name='IC +95%',
                               line=dict(color='rgba(31,119,180,0.55)', dash='dot', width=1.2)))
        f.add_trace(go.Scatter(x=anchor_dates, y=anchor_lo, name='IC −95%',
                               line=dict(color='rgba(31,119,180,0.55)', dash='dot', width=1.2)))
        f.add_trace(go.Scatter(x=anchor_dates, y=anchor_mean, name='Previsione media',
                               line=dict(color='#d62728', width=2.5), marker=dict(size=4)))
        y_min = min(float(np.min(price_fc_lo)), float(hist_prices.min())) * 0.98
        y_max = max(float(np.max(price_fc_hi)), float(hist_prices.max())) * 1.02
        # «Oggi» come traccia, non come shape: il tab può essere ridisegnato da nascosto.
        f.add_trace(go.Scatter(x=[last_date, last_date], y=[y_min, y_max], name='Oggi',
                               mode='lines', line=dict(color='gray', width=1.2, dash='dash'),
                               hoverinfo='skip'))
        f.update_yaxes(title_text='Prezzo (EUR)', range=[y_min, y_max])
        figure.append(f)

        # 2. Log prezzi + trend
        f = _fig(f'Log prezzi + trend lineare (pendenza {trend_coef[0]:+.5f}/gg)', 380)
        f.add_trace(go.Scatter(x=idx, y=log_prices, name='Log prezzi',
                               line=dict(color='#1f77b4', width=1.2)))
        f.add_trace(go.Scatter(x=idx, y=trend_line, name='Trend lineare',
                               line=dict(color='#d62728', width=2, dash='dash')))
        f.update_yaxes(title_text='Log prezzo')
        figure.append(f)

        # 3. Ciclicità (serie senza trend)
        f = _fig('Ciclicità — serie senza trend, I(1)', 340)
        f.add_trace(go.Scatter(x=idx, y=detrended, name='Ciclicità',
                               line=dict(color='#2ca02c', width=1.1)))
        f.add_trace(go.Scatter(x=[idx[0], idx[-1]], y=[0, 0], showlegend=False, hoverinfo='skip',
                               line=dict(color='red', width=0.8, dash='dot')))
        f.update_yaxes(title_text='Ciclicità')
        figure.append(f)

        # 4. Δ ciclicità = errori grezzi
        f = _fig(f'Δ Ciclicità = errori grezzi I(0)  ·  ADF p={adf_pval:.4f} {stato_staz}', 340)
        f.add_trace(go.Bar(x=idx_diff, y=diff_detrended, name='Δ Ciclicità', marker_line_width=0,
                           marker_color=['#2ca02c' if v >= 0 else '#d62728' for v in diff_detrended]))
        f.update_yaxes(title_text='Δ Ciclicità')
        figure.append(f)

        # 5–6. ACF e PACF
        for nome, valori, base in (('ACF', acf_vals, '#1f77b4'), ('PACF', pacf_vals, '#ff7f0e')):
            f = _fig(f'{nome} della Δ ciclicità ({n_lags} ritardi)  ·  in rosso i ritardi '
                     f'fuori dalla banda 95%', 320)
            f.add_trace(go.Bar(x=lags_arr, y=valori, name=nome, marker_line_width=0,
                               marker_color=['#d62728' if abs(v) > ci_bound else base for v in valori]))
            f.add_trace(go.Scatter(x=[0, n_lags, None, 0, n_lags],
                                   y=[ci_bound, ci_bound, None, -ci_bound, -ci_bound],
                                   name='Banda 95%', hoverinfo='skip',
                                   line=dict(color='orange', dash='dash', width=1.2)))
            f.update_layout(hovermode='closest')
            f.update_xaxes(title_text='Ritardo')
            f.update_yaxes(title_text=nome)
            figure.append(f)

        # 7. GARCH (o residui se arch manca)
        if garch_ok and cond_vol is not None:
            f = _fig('GARCH(1,1) — volatilità condizionale degli errori', 340)
            vol_idx = idx_diff[-len(cond_vol):]
            f.add_trace(go.Scatter(x=vol_idx, y=cond_vol * 100, name='σ_t GARCH (%)',
                                   line=dict(color='#d62728', width=1.3),
                                   fill='tozeroy', fillcolor='rgba(214,39,40,0.1)'))
            mean_vol = float(np.mean(cond_vol)) * 100
            f.add_trace(go.Scatter(x=[vol_idx[0], vol_idx[-1]], y=[mean_vol, mean_vol],
                                   name=f'σ media ({mean_vol:.3f}%)',
                                   line=dict(color='navy', dash='dash', width=1.2)))
            f.update_yaxes(title_text='σ_t (%)')
        else:
            f = _fig('Errori stazionari ε_t (residui ARIMA)', 340)
            res_idx = idx_diff[-len(residuals):]
            f.add_trace(go.Scatter(x=res_idx, y=residuals, name='ε_t (residui)',
                                   line=dict(color='#8c564b', width=0.7), opacity=0.8))
            f.add_trace(go.Scatter(x=[res_idx[0], res_idx[-1]], y=[0, 0], showlegend=False,
                                   hoverinfo='skip', line=dict(color='red', width=1, dash='dot')))
            f.update_yaxes(title_text='ε_t')
        figure.append(f)

        arima_lbl = f'ARIMA{best_order}'
        title_str = (f'🔬  {selected_asset}  —  {arima_lbl}  |  {criterion.upper()}={best_score:.1f}  |  '
                     f'ADF p={adf_pval:.4f}  {stato_staz}  |  '
                     f'Forecast {horizon}gg: {ret_sign}{expected_ret:.2f}%')

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
        return _ar_grafici(title_str, figure), status_text

    except Exception as exc:
        import traceback as _tb
        print(f"❌ ARIMA Tab error:\n{_tb.format_exc()}", flush=True)
        return [], f"❌ Errore: {str(exc)}"


# ── Barra dati: aggiungi asset / carica file / template / esporta / file lavoro ─
@app.callback(
    Output('weights-grid-container-arima', 'children', allow_duplicate=True),
    Output('at-add-stato', 'children'),
    Output('arima-asset-dropdown', 'value', allow_duplicate=True),
    Output('arima-asset-dropdown', 'options', allow_duplicate=True),
    Input('at-add-btn', 'n_clicks'),
    Input('at-add-ticker', 'n_submit'),
    State('at-add-ticker', 'value'),
    State('at-add-desc', 'value'),
    State('at-add-cur', 'value'),
    running=[(Output('at-add-btn', 'disabled'), True, False)],
    prevent_initial_call=True,
)
def at_add_asset(n, invio, ticker, desc, cur):
    """Aggiunge un titolo al file personale (stessi controlli degli Indicatori), ridisegna
    solo la griglia (i parametri restano) e lo sceglie per l'analisi ARIMA."""
    if not (n or invio):
        raise PreventUpdate
    ticker = (ticker or '').strip().upper()
    if not ticker:
        return no_update, '⚠ Scrivi il ticker Yahoo (es. AAPL, SWDA.MI, BTC-USD)', no_update, no_update
    desc = (desc or '').strip() or ticker
    attuali = dc.read_current()
    if desc in attuali:
        return no_update, f'⚠ «{desc}» è già nel file', no_update, no_update
    doppio = next((k for k, v in attuali.items() if (v.get('ticker') or '').upper() == ticker), None)
    if doppio:
        return no_update, f'⚠ {ticker} è già nel file come «{doppio}»', no_update, no_update
    ok, msg = _add_asset_to_current(ticker, desc, cur or 'EUR')
    if not ok:
        return no_update, msg, no_update, no_update
    nomi = [o['value'] for o in _asset_options()]
    return _build_asset_grid(nomi, [desc]), msg, desc, _ar_opzioni('', desc)


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


# ── Indicatore del file aperto: si rilegge quando i dati possono essere cambiati ─
@app.callback(
    Output('at-file-aperto', 'children'), Output('at-file-aperto', 'title'),
    Input('at-tabs', 'value'), Input('at-data-status', 'children'), Input('ind-nuovo', 'data'),
    prevent_initial_call=True,
)
def at_file_aperto(*_):
    return _file_aperto()


# ── Sotto-tab sul cellulare: tendina «Tabs Strumenti» (come il Portafoglio) ──
@app.callback(
    Output('at-tabs', 'value'),
    Input({'type': 'at-tendina-voce', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def _at_tendina_scegli(clicks):
    tid = callback_context.triggered_id
    if not tid or not any(clicks):
        raise PreventUpdate
    return tid['index']


app.clientside_callback(
    """function (val) {
        var voci = %s, lab = voci[0][0];
        voci.forEach(function (v) { if (v[1] === val) lab = v[0]; });
        return [lab, voci.map(function (v) {
            return v[1] === val ? 'nb-tendina-voce nb-attiva' : 'nb-tendina-voce'; })];
    }""" % json.dumps(_AT_SOTTOTAB),
    Output('at-tendina-att', 'children'),
    Output({'type': 'at-tendina-voce', 'index': ALL}, 'className'),
    Input('at-tabs', 'value'),
)


@app.callback(Output('at-info-corpo', 'children'), Output('at-calcola-cell', 'className'),
              Output('val-cerca-riga', 'className'), Output('val-a-capo', 'className'),
              Output('at-arima-cell', 'className'), Output('ar-cerca-riga', 'className'),
              Output('cot-cerca-riga', 'className'),
              Input('at-tabs', 'value'))
def at_info(tab):
    val = tab in (None, 'tab-valutazione')
    return (_info_tab(tab), ('nb-solo-cell nb-clicca' if tab == 'tab-indicatori' else 'at-via'),
            'val-cerca' if val else 'at-via', 'val-a-capo' if val else 'at-via',
            'nb-solo-cell nb-clicca' if tab == 'tab-arima' else 'at-via',
            'val-cerca' if tab == 'tab-arima' else 'at-via',
            'val-cerca' if tab == 'tab-cot' else 'at-via')


_AR_NOMI = {'yf:AAPL': 'AAPL — Apple Inc. · Yahoo'}   # valore → etichetta (ticker Yahoo)


def _ar_opzioni(testo, valore):
    """Opzioni della tendina ARIMA: asset del file aperto + (se si scrive) risultati di
    Yahoo. I titoli Yahoo non si salvano da nessuna parte: restano fra le opzioni solo
    finché sono quelli scelti, poi spariscono."""
    import re
    testo = (testo or '').strip()
    file_opts = [{'label': f"📁 {o['label']}", 'value': o['value'],
                  'search': f"{o['label']} {o['value']}"} for o in (_asset_options() or [])]
    yahoo = []
    if len(testo) >= 2:
        for q in valutazione._yahoo_cerca(testo):
            v = f"yf:{q['symbol']}"
            et = f"{q['symbol']} — {q['nome']}" + (f" · {q['borsa']}" if q['borsa'] else '')
            _AR_NOMI[v] = et
            yahoo.append({'label': et, 'value': v, 'search': f"{testo} {et}"})
        su = testo.upper()
        if re.fullmatch(r"[A-Z0-9.\-=^]{1,15}", su) and f"yf:{su}" not in {o['value'] for o in yahoo}:
            _AR_NOMI.setdefault(f"yf:{su}", su)
            yahoo.append({'label': f"🔎 {su} — usa come ticker", 'value': f"yf:{su}",
                          'search': f"{testo} {su}"})
    # Prima riga, non selezionabile: come si usa la casella (la casella mostra sempre
    # l'asset scelto, quindi un placeholder non si vedrebbe mai). Sparisce quando si scrive.
    guida = [{'label': '✏️ Usa nome o ticker di Yahoo, oppure scegli un asset del file',
              'value': '_guida', 'disabled': True}] if not testo else []
    opts = guida + yahoo + file_opts
    if valore and valore not in {o['value'] for o in opts}:
        et = _AR_NOMI.get(valore, valore[3:] if str(valore).startswith('yf:') else valore)
        # Resta fra le opzioni (altrimenti Dash toglie il valore), in fondo e filtrata
        # dal testo come le altre: in cima, Invio riprenderebbe il titolo di prima.
        opts.append({'label': et, 'value': valore, 'search': et})
    return opts


@app.callback(Output('arima-asset-dropdown', 'options'),
              Input('arima-asset-dropdown', 'search_value'),
              Input('at-tabs', 'value'),
              Input('tab-arima-content', 'children'),     # file cambiato
              State('arima-asset-dropdown', 'value'))
def ar_cerca(testo, _tab, _contenuto, valore):
    return _ar_opzioni(testo, valore)


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
    Output('tab-valutazione-content', 'style'),
    Output('tab-indicatori-content', 'style'),
    Input('at-tabs', 'value'),
)
def _at_switch_tab(tab):
    ordine = ('tab-arima', 'tab-cot', 'tab-valutazione', 'tab-indicatori')
    tab = tab if tab in ordine else 'tab-valutazione'
    return tuple({'display': 'block' if t == tab else 'none'} for t in ordine)


@app.callback(
    Output('cot-grafici', 'children'),
    Output('cot-summary', 'children'),
    Output('cot-st-finestre', 'children'),
    Input('cot-instrument', 'value'),
    Input('cot-range', 'value'),
    Input('cot-refresh', 'n_clicks'),
    Input('cot-st-base', 'value'),
    Input('cot-st-m1', 'value'),
    Input('cot-st-m2', 'value'),
    Input('cot-st-m3', 'value'),
    Input('val-cell', 'data'),
)
def _update_cot(key, year_range, n, base, m1, m2, m3, cell):
    if not key:
        raise PreventUpdate
    # il pulsante 🔄 forza il refetch invalidando la cache dello strumento
    if callback_context.triggered_id == 'cot-refresh':
        _COT_CACHE.pop(key, None)
    finestre = _cot_finestre_stoc(base, (m1, m2, m3))
    figure, summary, label = _cot_figure(key, year_range, finestre, bool(cell))
    fin = '= ' + ' · '.join(f'{x} gg' for x in finestre)
    return _ar_grafici(f'Posizionamento COT — {label}', figure), summary, fin


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Valutazione titolo azionario (modulo condiviso valutazione.py)
# ─────────────────────────────────────────────────────────────────────────────
valutazione.register_callbacks(app)
indicatori.register_callbacks(app)


if __name__ == '__main__':
    app.run(debug=True, port=8060)
