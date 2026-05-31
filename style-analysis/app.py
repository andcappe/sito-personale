"""Style Analysis — App standalone.
Legge i dati da portafoglio/sessions o da file caricato dall'utente.
"""
import io
import os
import sys
import pickle
import base64
from pathlib import Path

import numpy as np
import pandas as pd

from dash import Dash, html, dcc, Input, Output, State, no_update, callback_context
from dash.exceptions import PreventUpdate

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SELF = os.path.dirname(os.path.abspath(__file__))
_PORT = os.path.join(_ROOT, 'portafoglio')
if _PORT not in sys.path:
    sys.path.insert(0, _PORT)

from navbar import make_navbar
from style_analysis import (get_style_analysis_tab,
                             register_style_analysis_callbacks)

# ─── PKL paths ────────────────────────────────────────────────────────────────
_MARKET_PKL  = Path(_PORT) / 'sessions' / 'market_data.pkl'
_SESSIONS_DIR = Path(_PORT) / 'sessions'


def _get_username():
    try:
        from flask import session as _fs
        return _fs.get('username') or 'anon'
    except Exception:
        return 'anon'


def _load_pkl_data(username='anon'):
    """Carica dati utente o default dal pkl."""
    user_pkl = _SESSIONS_DIR / username / 'market_data_ETF_user_{}.pkl'.format(username)
    for path in [user_pkl, _MARKET_PKL]:
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    d = pickle.load(f)
                cr = d.get('close_returns')
                op = d.get('original_prices')
                tm = d.get('ticker_map', {})
                saved_at = d.get('saved_at', '')
                if cr is not None and not cr.empty:
                    return cr, op, tm, saved_at
            except Exception:
                pass
    return None, None, {}, ''


# ─── App ─────────────────────────────────────────────────────────────────────
_EXT = [
    'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=_EXT,
           requests_pathname_prefix='/style-analysis/',
           routes_pathname_prefix='/style-analysis/')

# ─── Layout ───────────────────────────────────────────────────────────────────
app.layout = html.Div([
    make_navbar('Style Analysis'),

    html.Div([

        # Toolbar caricamento dati
        html.Div([
            html.Div([
                html.I(className='fa-solid fa-chart-line',
                       style={'margin-right': '8px', 'color': '#1a3a5c'}),
                html.B('Style Analysis', style={'font-size': '14px', 'color': '#1a3a5c'}),
                html.Span(' — Sharpe (1992) + OLS rolling',
                          style={'font-size': '11px', 'color': '#666', 'margin-left': '8px'}),
            ], style={'display': 'flex', 'align-items': 'center'}),

            html.Div([
                html.Span(id='sa-data-status',
                          style={'font-size': '11px', 'color': '#555',
                                 'margin-right': '12px', 'font-style': 'italic'}),
                dcc.Upload(
                    id='sa-upload-file',
                    children=html.Div(['📂 Carica file prezzi',],
                                      style={'font-size': '11px'}),
                    style={'border': '1px dashed #aaa', 'border-radius': '4px',
                           'padding': '5px 12px', 'cursor': 'pointer',
                           'background': '#f8f9fa', 'color': '#555'},
                    multiple=False,
                ),
            ], style={'display': 'flex', 'align-items': 'center'}),

        ], style={'display': 'flex', 'justify-content': 'space-between',
                  'align-items': 'center', 'padding': '8px 16px',
                  'background': '#eef4fb', 'border-bottom': '2px solid #c0d4ee'}),

        # Style Analysis tab content (caricato dinamicamente)
        html.Div(id='sa-main-content'),

    ], style={'margin-top': '106px'}),

    # Stores
    dcc.Store(id='stock-data',            data=None),
    dcc.Store(id='asset-checklist',       data=[]),
    dcc.Store(id='style-analysis-store',  data=None),
    dcc.Store(id='sa-nav-reload',         data=0, storage_type='memory'),
    dcc.Interval(id='sa-init-interval',   interval=200, n_intervals=0, max_intervals=1),
])


# ─── Callback: init — carica dati dal pkl al primo avvio ─────────────────────
@app.callback(
    Output('stock-data',      'data'),
    Output('asset-checklist', 'data'),
    Output('sa-data-status',  'children'),
    Output('sa-main-content', 'children'),
    Input('sa-init-interval', 'n_intervals'),
    prevent_initial_call=False,
)
def sa_init(n):
    u = _get_username()
    cr, op, tm, saved_at = _load_pkl_data(u)
    if cr is None:
        return (None, [],
                '⚠ Nessun dato — carica un file Excel',
                get_style_analysis_tab([]))
    opts = [{'label': c, 'value': c} for c in cr.columns]
    cr_json = cr.to_json(date_format='iso', orient='split')
    lbl = f'✓ {len(opts)} asset' + (f' — {saved_at}' if saved_at else '')
    return cr_json, opts, lbl, get_style_analysis_tab(opts)


# ─── Callback: upload file prezzi (Excel con colonne = asset, righe = date) ──
@app.callback(
    Output('stock-data',      'data',     allow_duplicate=True),
    Output('asset-checklist', 'data',     allow_duplicate=True),
    Output('sa-data-status',  'children', allow_duplicate=True),
    Output('sa-main-content', 'children', allow_duplicate=True),
    Input('sa-upload-file',   'contents'),
    State('sa-upload-file',   'filename'),
    prevent_initial_call=True,
)
def sa_upload(contents, filename):
    if not contents:
        raise PreventUpdate
    try:
        _, b64 = contents.split(',', 1)
        raw = base64.b64decode(b64)
        df = pd.read_excel(io.BytesIO(raw))
        # Prova formato prezzi (prima colonna = date)
        col0 = df.columns[0]
        try:
            df[col0] = pd.to_datetime(df[col0])
            df = df.set_index(col0)
            df = df.select_dtypes(include='number').ffill().dropna(how='all')
        except Exception:
            return no_update, no_update, f'⚠ Formato non riconosciuto: {filename}', no_update

        cr = df.pct_change(fill_method=None).dropna(how='all')
        opts = [{'label': c, 'value': c} for c in cr.columns]
        return (cr.to_json(date_format='iso', orient='split'),
                opts,
                f'✓ {len(opts)} asset da {filename}',
                get_style_analysis_tab(opts))
    except Exception as e:
        return no_update, no_update, f'⚠ Errore: {e}', no_update


register_style_analysis_callbacks(app)
server = app.server
