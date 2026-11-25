"""Style Analysis — estratto da ir_fe_14.py e adattato per portafoglio."""
import os
import json as _json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

from dash import html, dcc, no_update, ALL, ctx

import data_core as _dc   # Finestra Temporale condivisa
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import sessions_manager as _sm


# `statsmodels.api` costa da solo ~73 MB di memoria, piu' di tutti e sette i
# cruscotti messi insieme, e qui serve solo dentro le funzioni di analisi.
# Importarlo in cima significava pagarlo a ogni avvio anche per chi apre un
# grafico e basta - su un'istanza da 512 MB e' la differenza fra un download
# che finisce e un container ammazzato a meta'. Questo segnaposto si comporta
# come il modulo ma lo carica alla prima `sm.qualcosa` davvero eseguita.
class _StatsmodelsPigro:
    _mod = None

    def __getattr__(self, nome):
        if _StatsmodelsPigro._mod is None:
            import statsmodels.api as _m
            _StatsmodelsPigro._mod = _m
        return getattr(_StatsmodelsPigro._mod, nome)


sm = _StatsmodelsPigro()


_SA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Sito personale


def _sa_username():
    try:
        from flask import session as _fs
        return _fs.get('username') or 'anon'
    except Exception:
        return 'anon'


def _sa_current_json(username):
    """Legge current.json dell'utente (per ticker/valuta degli asset)."""
    try:
        path = os.path.join(_SA_ROOT, 'sessions', username, 'current.json')
        with open(path) as f:
            return _json.load(f)
    except Exception:
        return {}

# ─── palette colori ───────────────────────────────────────────────────────────
_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#9edae5', '#dbdb8d', '#c7c7c7',
]


# ─── helper: JSON → DataFrame ─────────────────────────────────────────────────
def _get_df(json_str):
    if not json_str:
        return None
    try:
        import io
        df = pd.read_json(io.StringIO(json_str), orient='split')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def _sa_build_cr_from_current(username=None):
    """Ricostruisce il DataFrame dei rendimenti da current.json (la fonte di
    verità UNICA, persistita lato server). È il fallback quando lo store
    'stock-data' del browser è vuoto — tipico nella sessione online."""
    data = _sa_current_json(username or _sa_username())
    if not data:
        return None
    cols = {}
    for asset, v in data.items():
        if not isinstance(v, dict):
            continue
        dates, rets = v.get('dates'), v.get('returns')
        if not dates or not rets or len(dates) != len(rets):
            continue
        try:
            cols[asset] = pd.Series(rets, index=pd.to_datetime(dates))
        except Exception:
            continue
    if not cols:
        return None
    try:
        return pd.DataFrame(cols).sort_index()
    except Exception:
        return None


# ─── helper functions ─────────────────────────────────────────────────────────
def _returns_monthly(close_returns: pd.DataFrame) -> pd.DataFrame:
    cr = close_returns.copy()
    cr.index = pd.to_datetime(cr.index)
    if hasattr(cr.index, 'tz') and cr.index.tz is not None:
        cr.index = cr.index.tz_localize(None)
    # 'ME' = pandas ≥2.2, 'M' = pandas <2.2
    try:
        return (1 + cr).resample('ME').prod() - 1
    except ValueError:
        return (1 + cr).resample('M').prod() - 1


def _make_stat_table_sa(rows, header_bg='#1a3a5c'):
    """Tabella dei risultati: dentro .sa-scorri si scorre di lato e la prima colonna
    (.sa-prima, sticky) resta ferma; ogni cella ha uno sfondo pieno, altrimenti i
    valori scorrerebbero sotto i nomi."""
    prima = lambda j: 'sa-prima' if j == 0 else None
    return html.Div(className='sa-scorri', style={'border': '1px solid #dee2e6'}, children=html.Table([
        html.Thead(html.Tr([
            html.Th(c, className=prima(j), style={
                'background': header_bg, 'color': 'white',
                'padding': '5px 10px', 'font-size': '11px',
                'text-align': 'left',
            }) for j, c in enumerate(rows[0])
        ])),
        html.Tbody([
            html.Tr([
                html.Td(cell, className=prima(j), style={
                    'padding': '4px 10px', 'font-size': '11px',
                    'border-bottom': '1px solid #eee',
                    'font-family': 'monospace',
                    'font-weight': '600' if j == 0 else '400',
                    'background': '#fff' if ri % 2 == 0 else '#f8f9fa',
                }) for j, cell in enumerate(row)
            ]) for ri, row in enumerate(rows[1:])
        ])
    ], className='sa-tab'))


def _fit_ols(y: pd.Series, X: pd.DataFrame, cov_type: str, nonneg: bool = True):
    from scipy.optimize import minimize as _min
    X_c = sm.add_constant(X)

    if nonneg:
        y_v = y.values
        X_v = X_c.values
        n_params = X_v.shape[1]
        bounds = [(None, None)] + [(0.0, 1.0)] * (n_params - 1)

        def obj(b):
            return float(((y_v - X_v @ b) ** 2).sum())

        x0 = np.zeros(n_params)
        res = _min(obj, x0, method='SLSQP', bounds=bounds,
                   options={'ftol': 1e-12, 'maxiter': 1000})
        b = res.x

        fitted = pd.Series(X_v @ b, index=y.index)
        resid = y - fitted
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n, k = len(y), n_params
        r2_adj = 1.0 - (1 - r2) * (n - 1) / (n - k)
        sigma2 = ss_res / max(n - k, 1)
        try:
            cov_b = sigma2 * np.linalg.inv(X_v.T @ X_v)
            se = np.sqrt(np.diag(cov_b))
        except Exception:
            se = np.full(n_params, np.nan)

        params = pd.Series(b, index=X_c.columns)
        bse = pd.Series(se, index=X_c.columns)
        tvalues = params / bse
        pvalues = pd.Series(
            2 * (1 - __import__('scipy').stats.t.cdf(np.abs(tvalues.values), df=n - k)),
            index=X_c.columns
        )
        conf_int = pd.DataFrame({0: params - 1.96 * bse, 1: params + 1.96 * bse})

        class _FitResult:
            pass

        m = _FitResult()
        m.params = params
        m.bse = bse
        m.tvalues = tvalues
        m.pvalues = pvalues
        m.rsquared = r2
        m.rsquared_adj = r2_adj
        m.resid = resid
        m.fittedvalues = fitted
        m.aic = n * np.log(ss_res / n) + 2 * k
        m.bic = n * np.log(ss_res / n) + k * np.log(n)
        m.fvalue = ((ss_tot - ss_res) / (k - 1)) / (ss_res / (n - k))
        m.f_pvalue = 1 - __import__('scipy').stats.f.cdf(m.fvalue, k - 1, n - k)
        m.df_model = k - 1
        m.model = type('M', (), {'exog': X_v, 'exog_names': list(X_c.columns)})()
        m.conf_int = lambda alpha=0.05: conf_int
        return m

    ols = sm.OLS(y, X_c)
    if cov_type == 'HAC':
        nlag = max(1, int(4 * (len(y) / 100) ** (2 / 9)))
        return ols.fit(cov_type='HAC', cov_kwds={'maxlags': nlag})
    elif cov_type == 'HC3':
        return ols.fit(cov_type='HC3')
    return ols.fit()


def _fit_sharpe(y: pd.Series, X: pd.DataFrame):
    cols = X.columns.tolist()
    n = len(cols)
    y_v = y.values
    X_v = X.values

    def obj(w):
        resid = y_v - X_v @ w
        return float(resid @ resid)

    def grad(w):
        resid = y_v - X_v @ w
        return -2 * X_v.T @ resid

    bounds = [(0.0, 1.0)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    x0 = np.full(n, 1.0 / n)
    res = minimize(obj, x0, jac=grad, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-12, 'maxiter': 500})
    w = res.x if res.success else x0
    fitted = X_v @ w
    resid = y_v - fitted
    ss_tot = ((y_v - y_v.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (dict(zip(cols, w)),
            pd.Series(fitted, index=y.index),
            pd.Series(resid, index=y.index),
            r2)


def _empty_fig(msg=''):
    fig = go.Figure()
    if msg:
        fig.add_annotation(text=msg, xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=12, color='#bbb'))
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='#f8f8f8',
                      margin=dict(t=30, b=20, l=40, r=20))
    return fig


def _pstar(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '·'
    return ''


def _hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ─── Layout ───────────────────────────────────────────────────────────────────
# Sezioni a scheda, una sotto l'altra (23/11/2026, richiesta dell'utente): Parametri,
# Finestra Temporale (condivisa con Analisi Portafoglio e Frontiera), Fattori di stile,
# Risultati, Style Weights Rolling, R² rolling, Asset Y, Alpha rolling, Contributo al
# rendimento. Sul cellulare sono sezioni espandibili (nb-espandi) e «Esegui» + ⚙ stanno
# nella riga in alto del Portafoglio (pf-sa-cell, pf-info-cell).
_SA_BTN = {'background-color': '#1a3a6b', 'color': 'white', 'border': 'none',
           'padding': '6px 14px', 'border-radius': '4px', 'cursor': 'pointer',
           'font-weight': '600', 'font-size': '12px', 'line-height': '18px',
           'display': 'inline-block', 'user-select': 'none', 'white-space': 'nowrap'}
_SA_CAMPO = {'display': 'flex', 'flex-direction': 'column', 'gap': '3px'}
_SA_ETIC = {'font-size': '11px', 'color': '#6b7a99', 'font-weight': '600', 'white-space': 'nowrap'}
_SA_NOTA = {'font-size': '11px', 'color': '#6b7a99', 'font-weight': '400', 'margin-left': '8px'}

# CSS delle sezioni: lo inserisce portafoglio/app.py nel suo index_string.
SA_CSS = """
  .sa-sez { background: #fff; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 8px 12px; margin: 0 0 10px 0; box-sizing: border-box; min-width: 0; }
  .sa-sez > .nb-espandi-btn { font-size: 13px; font-weight: 700; color: #1a3a5c;
    padding: 2px 0 6px; display: block; }
  .sa-grafico { height: var(--sa-h, 320px); }
  /* Tabelle dei risultati: si scorrono di lato, la prima colonna resta ferma. */
  .sa-scorri { overflow-x: auto; max-width: 100%; -webkit-overflow-scrolling: touch; }
  .sa-tab { border-collapse: separate; border-spacing: 0; min-width: 100%; }
  .sa-tab th, .sa-tab td { white-space: nowrap; }
  .sa-tab .sa-prima { position: sticky; left: 0; z-index: 1; border-right: 1px solid #dee2e6; }
  .sa-fattori { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 2px 14px; max-height: 320px; overflow-y: auto; }
  .sa-date .SingleDatePickerInput { border: 1px solid #ccd9ee; border-radius: 4px; }
  .sa-date .DateInput { width: 104px; }
  .sa-date .DateInput_input { font-size: 13px; padding: 4px 8px; height: 28px; line-height: 20px; }
  @media (max-width: 900px) {
    .sa-grafico .modebar-container { display: none !important; }  /* copriva i titoli */
    .sa-sez { background: #f8fafc; padding: 5px 14px 5px 8px; margin: 0 0 8px 0; }
    .sa-sez > .nb-espandi-btn { font-size: 12px; padding: 4px 0; }
    .sa-fattori { grid-template-columns: 1fr; max-height: 50vh; }
    .sa-grafico { height: var(--sa-hc, 300px); }
  }
"""


def _sa_sez(titolo, corpo, aperta=True, nota=None, cls=''):
    """Sezione a scheda; sul cellulare espandibile (aperta o chiusa all'avvio)."""
    testa = [titolo]
    if nota:
        testa.append(html.Span(nota, className='nb-nascondi-cell', style=_SA_NOTA))
    testa.append(html.I(className='fa-solid fa-chevron-down nb-espandi-ico'))
    return html.Div(className=f"nb-espandi sa-sez{' nb-aperto' if aperta else ''} {cls}".strip(),
                    children=[html.Div(testa, className='nb-espandi-btn'),
                              html.Div(corpo, className='nb-espandi-corpo')])


def _sa_grafico(id_, alto, alto_cell=None):
    # responsive=True (prop di dcc.Graph, non di config): il grafico prende altezza e
    # larghezza del riquadro; disegnato a tab nascosto nascerebbe 700×450.
    return dcc.Graph(id=id_, className='sa-grafico', responsive=True,
                     figure=_empty_fig('Premi «Esegui Style Analysis»'),
                     style={'--sa-h': f'{alto}px', '--sa-hc': f'{alto_cell or alto}px'},
                     config={'displaylogo': False})


def _sa_riga(etichetta, valore, colore='#1a3a5c'):
    return html.Div([
        html.Span(etichetta, style={'color': '#5a6b85', 'font-size': '12px'}),
        html.Span(valore, style={'color': colore, 'font-size': '12px', 'font-weight': '600',
                                 'white-space': 'nowrap'}),
    ], style={'display': 'flex', 'justify-content': 'space-between', 'gap': '16px',
              'padding': '3px 0', 'border-bottom': '1px dotted #e2e8f0'})


_SA_RIEPILOGO_VUOTO = [html.Div('Riepilogo Style Analysis', style={'font-size': '12px', 'font-weight': '700',
                                                                   'color': '#1a3a5c', 'margin-bottom': '4px'}),
                       html.Div('Premi «Esegui Style Analysis»: qui compaiono asset, periodo, '
                                'fattori e statistiche principali.',
                                style={'font-size': '12px', 'color': '#5a6b85', 'white-space': 'normal'})]


def get_style_analysis_tab(options_tickers):
    opts = options_tickers or []
    campo = lambda etic, comp: html.Div([html.Div(etic, style=_SA_ETIC), comp], style=_SA_CAMPO)
    radio = dict(inline=True, style={'font-size': '12px'}, inputStyle={'margin-right': '4px'},
                 labelStyle={'margin-right': '12px', 'cursor': 'pointer', 'color': '#1a3a5c'})

    comandi = html.Div(className='nb-nascondi-cell', style={'display': 'flex', 'align-items': 'center',
                                                            'gap': '6px', 'margin': '0 0 10px 0'}, children=[
        html.Button('▶ Esegui Style Analysis', id='btn-run-sa', n_clicks=0, style=_SA_BTN),
        # Riepilogo dei dati: sul cellulare lo apre il ⚙ della riga in alto (pf-info-cell).
        html.Details(className='pf-strumenti pf-info sa-info', children=[
            html.Summary(html.I(className='fa-solid fa-gear'), title='Riepilogo Style Analysis',
                         style={**_SA_BTN, 'padding': '6px 9px'}),
            html.Div(id='sa-riepilogo', className='pf-strumenti-menu', children=_SA_RIEPILOGO_VUOTO,
                     style={'padding': '10px 12px', 'gap': '0', 'min-width': '280px'}),
        ]),
    ])

    parametri = _sa_sez('Parametri', aperta=False, corpo=html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '12px 22px', 'align-items': 'flex-end'},
        children=[
            campo('Asset Y (da spiegare)',
                  dcc.Dropdown(id='sa-asset-y', options=opts,
                               value=opts[0]['value'] if opts else None,
                               placeholder='Asset da analizzare…', clearable=False,
                               style={'width': '240px', 'font-size': '12px'})),
            campo('Finestra rolling (mesi)',
                  dcc.Input(id='sa-window', type='number', value=36, min=12, max=120, step=6,
                            style={'width': '80px', 'font-size': '12px', 'padding': '4px 6px'})),
            campo('Lag X (mesi)',
                  dcc.Input(id='sa-lag', type='number', value=0, min=0, max=24, step=1,
                            style={'width': '80px', 'font-size': '12px', 'padding': '4px 6px'})),
            campo('Tipo', dcc.RadioItems(id='sa-reg-type', value='ols', **radio,
                                         options=[{'label': 'OLS libero', 'value': 'ols'},
                                                  {'label': 'Sharpe (≥0, Σ=1)', 'value': 'sharpe'}])),
            campo('Std Error', dcc.RadioItems(id='sa-cov-type', value='HAC', **radio,
                                              options=[{'label': 'OLS', 'value': 'nonrobust'},
                                                       {'label': 'HC3', 'value': 'HC3'},
                                                       {'label': 'HAC', 'value': 'HAC'}])),
        ]))

    # Stessa Finestra Temporale di Analisi Portafoglio e Frontiera (sessions/<u>/finestra.json).
    finestra = _sa_sez('Finestra Temporale', aperta=False, nota='condivisa con Analisi Portafoglio e Frontiera',
                       cls='sa-date', corpo=html.Div(style={'display': 'flex', 'gap': '10px'}, children=[
        campo('Da', dcc.DatePickerSingle(id='sa-date-start', display_format='DD/MM/YYYY',
                                         first_day_of_week=1, clearable=False)),
        campo('A', dcc.DatePickerSingle(id='sa-date-end', display_format='DD/MM/YYYY',
                                        first_day_of_week=1, clearable=False)),
    ]))

    fattori = _sa_sez('Fattori di stile', nota='variabili X della regressione', corpo=[
        html.Div(style={'display': 'flex', 'gap': '6px', 'margin-bottom': '6px'}, children=[
            html.Button('✔ Tutti', id='sa-sel-all', n_clicks=0,
                        style={**_SA_BTN, 'background-color': '#eef3fb', 'color': '#1a3a6b',
                               'border': '1px solid #c0d0e8', 'padding': '3px 10px'}),
            html.Button('✘ Nessuno', id='sa-sel-none', n_clicks=0,
                        style={**_SA_BTN, 'background-color': '#eef3fb', 'color': '#1a3a6b',
                               'border': '1px solid #c0d0e8', 'padding': '3px 10px'}),
        ]),
        html.Div(id='sa-x-checklist', className='sa-fattori'),
    ])

    risultati = _sa_sez('Risultati Style Analysis', nota='equazione, statistiche e parametri', corpo=dcc.Loading(
        type='circle', children=[
            html.Div(id='sa-equation', className='sa-scorri',
                     children='Premi «Esegui Style Analysis».',
                     style={'font-family': 'monospace', 'font-size': '12px', 'background': '#f8f9fa',
                            'border': '1px solid #dee2e6', 'border-radius': '4px', 'padding': '8px 14px',
                            'white-space': 'pre', 'color': '#1a3a5c'}),
            html.Div(id='sa-stats-table', style={'margin-top': '10px'}),
            html.Div(id='sa-coeff-table', style={'margin-top': '4px'}),
        ]))

    pesi = _sa_sez('Style Weights Rolling', nota='peso di ciascun fattore nella finestra mobile', corpo=[
        _sa_grafico('sa-chart-weights', 440, 380),
        html.Div(id='sa-portfolio-suggest', style={'padding': '10px 4px 4px'}),
        # Esporta il portafoglio target come Analisi: si carica in P1/P2/P3 da
        # Analisi Portafoglio → 🧰 Strumenti → Importa/Esporta.
        html.Div(style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap', 'gap': '6px',
                        'padding': '10px 4px 4px', 'border-top': '1px solid #e2e8f0', 'margin-top': '8px'},
                 children=[
            html.Span('📤 Esporta il portafoglio per Analisi Portafoglio',
                      style={'font-size': '12px', 'font-weight': '600', 'color': '#1a3a5c'}),
            dcc.Input(id='sa-export-name', placeholder='Nome analisi…',
                      style={'font-size': '12px', 'padding': '4px 8px', 'border': '1px solid #ccd9ee',
                             'border-radius': '4px', 'width': '180px'}),
            html.Button('Esporta come Analisi', id='sa-export-btn', n_clicks=0,
                        style={**_SA_BTN, 'background-color': '#1b7a34'}),
            html.Span(id='sa-export-status', style={'font-size': '12px', 'color': '#1b5e20',
                                                    'font-weight': '600'}),
        ]),
        html.Div('Poi in Analisi Portafoglio: 🧰 Strumenti → Importa/Esporta → carica l\'analisi in P1, P2 o P3.',
                 style={'font-size': '11px', 'color': '#6b7a99', 'padding': '2px 4px'}),
    ])

    return html.Div(className='sa-corpo', style={'padding': '8px calc(8px + 1%) 24px'}, children=[
        comandi,
        html.Div(id='sa-status', style={'font-size': '12px', 'color': '#b71c1c', 'font-weight': '600',
                                        'margin': '0 0 8px 0'}),
        parametri,
        finestra,
        fattori,
        risultati,
        pesi,
        _sa_sez('R² rolling', nota='potere esplicativo del modello nella finestra mobile',
                corpo=_sa_grafico('sa-chart-r2', 380, 320)),
        _sa_sez('Asset Y — osservato vs stimato', nota='rendimento cumulato',
                corpo=_sa_grafico('sa-chart-fit', 340, 300)),
        _sa_sez('Alpha rolling', nota='residuo non spiegato dai fattori: positivo = sovraperformance',
                corpo=_sa_grafico('sa-chart-alpha', 300, 280)),
        _sa_sez('Contributo al rendimento di ogni asset', nota='contributo medio per fattore, %/anno',
                corpo=_sa_grafico('sa-chart-decomp', 360, 340)),
    ])


# ─── Callbacks ────────────────────────────────────────────────────────────────
def register_style_analysis_callbacks(app):

    @app.callback(
        Output('sa-asset-y', 'options'),
        Output('sa-asset-y', 'value'),
        Input('asset-checklist', 'data'),
        State('sa-asset-y', 'value'),
        prevent_initial_call=False,
    )
    def sa_populate_y(options_tickers, current):
        if not options_tickers:
            cr = _sa_build_cr_from_current()   # fonte unica: current.json
            if cr is not None and not cr.empty:
                options_tickers = [{'label': c, 'value': c} for c in cr.columns]
        if not options_tickers:
            return [], None
        if current and any(o['value'] == current for o in options_tickers):
            return options_tickers, current
        return options_tickers, (options_tickers[0]['value'] if options_tickers else None)

    def _sa_dati(stock_data):
        cr = _get_df(stock_data)
        if cr is None or cr.empty:
            cr = _sa_build_cr_from_current()   # fonte unica: current.json
        return cr if cr is not None and not cr.empty else None

    # Finestra Temporale condivisa con Analisi Portafoglio e Frontiera: si rilegge dal
    # file a ogni apertura del tab (una data cambiata là arriva qui) e a ogni dato nuovo.
    @app.callback(
        Output('sa-date-start', 'date'),
        Output('sa-date-end', 'date'),
        Input('stock-data', 'data'),
        Input('main-tabs', 'value'),
        prevent_initial_call=False,
    )
    def sa_date_condivise(stock_data, tab):
        if ctx.triggered_id == 'main-tabs' and tab != 'tab-sa':
            raise PreventUpdate
        cr = _sa_dati(stock_data)
        if cr is None:
            return None, None
        return _dc.finestra_iniziale(cr.index.min(), cr.index.max())

    # Una data cambiata qui vale anche per Analisi Portafoglio e Frontiera.
    @app.callback(
        Output('sa-status', 'children', allow_duplicate=True),
        Input('sa-date-start', 'date'),
        Input('sa-date-end', 'date'),
        State('stock-data', 'data'),
        prevent_initial_call=True,
    )
    def sa_salva_finestra(da, a, stock_data):
        if da and a:
            cr = _sa_dati(stock_data)
            _dc.scrivi_finestra(da, a, dmax=cr.index.max() if cr is not None else None)
        return no_update

    @app.callback(
        Output('sa-x-checklist', 'children'),
        Input('stock-data', 'data'),
        prevent_initial_call=False,
    )
    def sa_populate_x(stock_data):
        cr = _sa_dati(stock_data)
        if cr is None:
            return []
        checks = [
            html.Div(
                dcc.Checklist(
                    id={'type': 'sa-x-check', 'index': c},
                    options=[{'label': f' {c}', 'value': c}],
                    value=[],
                    style={'font-size': '12px', 'color': '#1a3a5c'},
                    inputStyle={'margin-right': '5px'},
                ),
                style={'padding': '2px 0', 'white-space': 'nowrap', 'overflow': 'hidden',
                       'text-overflow': 'ellipsis'}, title=c,
            )
            for c in cr.columns
        ]
        return checks

    @app.callback(
        Output({'type': 'sa-x-check', 'index': ALL}, 'value'),
        Input('sa-sel-all',  'n_clicks'),
        Input('sa-sel-none', 'n_clicks'),
        State({'type': 'sa-x-check', 'index': ALL}, 'id'),
        prevent_initial_call=True,
    )
    def sa_sel_x(a, b, ids):
        if ctx.triggered_id == 'sa-sel-none':
            return [[] for _ in ids]
        if ctx.triggered_id != 'sa-sel-all':
            raise PreventUpdate
        return [[i['index']] for i in ids]

    @app.callback(
        Output('sa-equation',          'children'),
        Output('sa-stats-table',       'children'),
        Output('sa-coeff-table',       'children'),
        Output('sa-chart-weights',     'figure'),
        Output('sa-portfolio-suggest', 'children'),
        Output('sa-chart-r2',          'figure'),
        Output('sa-chart-fit',         'figure'),
        Output('sa-chart-alpha',       'figure'),
        Output('sa-chart-decomp',      'figure'),
        Output('sa-status',            'children'),
        Output('style-analysis-store', 'data'),
        Output('sa-riepilogo',         'children'),
        Input('btn-run-sa', 'n_clicks'),
        State('stock-data',   'data'),
        State('sa-asset-y',   'value'),
        State({'type': 'sa-x-check', 'index': ALL}, 'value'),
        State({'type': 'sa-x-check', 'index': ALL}, 'id'),
        State('sa-window',    'value'),
        State('sa-lag',       'value'),
        State('sa-reg-type',  'value'),
        State('sa-cov-type',  'value'),
        State('sa-date-start', 'date'),
        State('sa-date-end',   'date'),
        prevent_initial_call=True,
    )
    def run_style_analysis(n_clicks, stock_data, y_col, x_vals, x_ids,
                           window_months, lag_months, reg_type, cov_type,
                           date_start, date_end):

        _e = lambda msg: (msg, None, None,
                          _empty_fig('Nessun risultato'), '',
                          _empty_fig(''), _empty_fig(''),
                          _empty_fig(''), _empty_fig(''),
                          f'❌ {msg}', None, no_update)

        if not y_col:            return _e('Seleziona la variabile Y')

        x_selected = [ids['index'] for vals, ids in zip(x_vals, x_ids) if vals]
        if not x_selected:       return _e('Seleziona almeno un fattore X')
        if y_col in x_selected:  return _e('Y non può essere tra gli X')

        cr = _get_df(stock_data)
        if cr is None or cr.empty:
            cr = _sa_build_cr_from_current()   # fonte unica: current.json
        if cr is None or cr.empty:  return _e('Nessun dato caricato')

        if date_start: cr = cr.loc[pd.Timestamp(date_start):]
        if date_end:   cr = cr.loc[:pd.Timestamp(date_end)]

        monthly = _returns_monthly(cr)
        cols_needed = [y_col] + x_selected
        missing = [c for c in cols_needed if c not in monthly.columns]
        if missing: return _e(f'Colonne mancanti: {missing}')

        data = monthly[cols_needed].dropna()
        if len(data) < 24: return _e(f'Troppo pochi dati mensili ({len(data)}), min 24')

        y_full = data[y_col]
        X_full = data[x_selected]
        lag = int(lag_months or 0)
        if lag > 0:
            X_full = X_full.shift(lag)

        combined = pd.concat([y_full, X_full], axis=1).dropna()
        y_fit = combined[y_col]
        X_fit = combined[x_selected]
        n_obs = len(combined)
        W = int(window_months or 36)
        if n_obs < max(24, W + 1): return _e(f'Dati insufficienti ({n_obs} mesi)')

        # ── Modello globale ───────────────────────────────────────────────────
        if reg_type == 'sharpe':
            g_weights, g_fitted, g_resid, g_r2 = _fit_sharpe(y_fit, X_fit)
            g_r2_adj = 1 - (1 - g_r2) * (n_obs - 1) / (n_obs - len(x_selected) - 1)
            terms = [f'  {v:.4f} · {k}' for k, v in g_weights.items()]
            alpha_glob = float(g_resid.mean())
            eq_text = (f'{y_col} ≈\n'
                       + '\n'.join(f'  + {t}' for t in terms)
                       + f'\n  + α = {alpha_glob:+.6f}')
            stats_rows = [
                ['Statistica', 'Valore', 'Note'],
                ['N osservazioni',   f'{n_obs}',                      ''],
                ['N fattori X',      f'{len(x_selected)}',            ''],
                ['Tipo regressione', 'Sharpe constrained',            'w≥0, Σw=1'],
                ['R²',               f'{g_r2:.6f}',                   ''],
                ['R² adj.',          f'{g_r2_adj:.6f}',               ''],
                ['Alpha mensile',    f'{alpha_glob:+.6f}',            ''],
                ['Alpha annualiz.',  f'{alpha_glob * 12:+.4f}%',      ''],
            ]
            stat_table = _make_stat_table_sa(stats_rows)
            coef_rows = [['Fattore', 'Peso', 'Peso %']]
            for k, v in g_weights.items():
                coef_rows.append([k, f'{v:.6f}', f'{v * 100:.2f}%'])
            coef_table = html.Div([
                html.Div('Pesi stimati (Sharpe style)',
                         style={'font-size': '11px', 'font-weight': 'bold',
                                'color': '#1a3a5c', 'background': '#eaf4fb',
                                'padding': '4px 10px', 'border-radius': '4px 4px 0 0',
                                'border': '1px solid #aed6f1', 'border-bottom': 'none',
                                'margin-top': '10px'}),
                _make_stat_table_sa(coef_rows, '#2e6da4'),
            ])
        else:
            model = _fit_ols(y_fit, X_fit, cov_type, nonneg=True)
            g_fitted  = model.fittedvalues
            g_resid   = model.resid
            g_r2      = model.rsquared
            g_r2_adj  = model.rsquared_adj
            terms = []
            for cn, cv in model.params.items():
                if cn == 'const':
                    terms.append(f'  α = {cv:+.6f}')
                else:
                    terms.append(f'  {"+" if cv >= 0 else "−"} {abs(cv):.4f} · {cn}')
            eq_text = f'{y_col} =\n' + '\n'.join(terms) + '  + ε'

            from statsmodels.stats.stattools import durbin_watson, jarque_bera
            from statsmodels.stats.diagnostic import het_breuschpagan
            dw = float(durbin_watson(model.resid))
            jb_s, jb_p, jb_sk, jb_ku = jarque_bera(model.resid)
            bp_lm, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
            cov_lbl = {'nonrobust': 'OLS classico', 'HC3': 'HC3 robusto', 'HAC': 'HAC Newey-West'}.get(cov_type, cov_type)
            stats_rows = [
                ['Statistica', 'Valore', 'Note'],
                ['N osservazioni', f'{n_obs}', ''],
                ['N parametri', f'{int(model.df_model + 1)}', 'incl. costante'],
                ['Std Error type', cov_lbl, ''],
                ['R²', f'{g_r2:.6f}', ''],
                ['R² adj.', f'{g_r2_adj:.6f}', ''],
                ['F-stat', f'{model.fvalue:.4f}', f'p = {model.f_pvalue:.4e} {_pstar(model.f_pvalue)}'],
                ['AIC', f'{model.aic:.2f}', ''],
                ['BIC', f'{model.bic:.2f}', ''],
                ['Durbin-Watson', f'{dw:.4f}', '~2 = no autocorr.'],
                ['Jarque-Bera', f'{jb_s:.4f}', f'p={jb_p:.4e} {_pstar(jb_p)} | sk={jb_sk:.3f} ku={jb_ku:.3f}'],
                ['Breusch-Pagan', f'{bp_lm:.4f}', f'p={bp_p:.4e} {_pstar(bp_p)}'],
            ]
            stat_table = _make_stat_table_sa(stats_rows)
            conf = model.conf_int(alpha=0.05)
            x_cols_v = [c for c in model.model.exog_names if c != 'const']
            Xc = sm.add_constant(X_fit)
            vif_d = {}
            if len(x_cols_v) > 1:
                for xc in x_cols_v:
                    oth = [o for o in x_cols_v if o != xc]
                    try:
                        r2x = sm.OLS(Xc[xc], sm.add_constant(Xc[oth])).fit().rsquared
                        vif_d[xc] = 1 / (1 - r2x) if r2x < 1 else np.inf
                    except Exception:
                        vif_d[xc] = np.nan
            else:
                vif_d = {c: np.nan for c in x_cols_v}
            coef_rows = [['Variabile', 'Coeff.', 'Std Err', 't-stat', 'p-val', 'Sig.', 'IC95 inf', 'IC95 sup', 'VIF']]
            for var in model.params.index:
                p = model.pvalues[var]
                vf = vif_d.get(var, np.nan)
                coef_rows.append([var, f'{model.params[var]:.6f}', f'{model.bse[var]:.6f}',
                                   f'{model.tvalues[var]:.4f}', f'{p:.4e}', _pstar(p),
                                   f'{conf.loc[var, 0]:.6f}', f'{conf.loc[var, 1]:.6f}',
                                   f'{vf:.2f}' if not np.isnan(vf) else '—'])
            coef_table = html.Div([
                html.Div('Coefficienti (full sample)',
                         style={'font-size': '11px', 'font-weight': 'bold',
                                'color': '#1a3a5c', 'background': '#eaf4fb',
                                'padding': '4px 10px', 'border-radius': '4px 4px 0 0',
                                'border': '1px solid #aed6f1', 'border-bottom': 'none',
                                'margin-top': '10px'}),
                _make_stat_table_sa(coef_rows, '#2e6da4'),
                html.Div('*** p<0.001  ** p<0.01  * p<0.05  · p<0.10',
                         style={'font-size': '10px', 'color': '#777', 'margin-top': '3px',
                                'font-style': 'italic'}),
            ])
            g_weights = {var: float(model.params[var])
                         for var in model.params.index if var != 'const'}

        # ── Rolling ───────────────────────────────────────────────────────────
        roll_dates = []
        roll_coefs = {c: [] for c in x_selected}
        roll_r2    = []
        roll_alpha = []

        for i in range(W, n_obs + 1):
            idx_w = combined.index[i - W:i]
            y_w = combined.loc[idx_w, y_col]
            X_w = combined.loc[idx_w, x_selected]
            roll_dates.append(idx_w[-1])
            try:
                if reg_type == 'sharpe':
                    w_d, _, _, r2_w = _fit_sharpe(y_w, X_w)
                    for c in x_selected:
                        roll_coefs[c].append(w_d.get(c, 0.0))
                    roll_r2.append(r2_w)
                    fitted_w = X_w.values @ np.array([w_d[c] for c in x_selected])
                    roll_alpha.append(float((y_w.values - fitted_w).mean()))
                else:
                    m_w = _fit_ols(y_w, X_w, cov_type, nonneg=True)
                    for c in x_selected:
                        roll_coefs[c].append(float(m_w.params.get(c, 0.0)))
                    roll_r2.append(float(m_w.rsquared))
                    roll_alpha.append(float(m_w.params.get('const', m_w.resid.mean())))
            except Exception:
                for c in x_selected: roll_coefs[c].append(np.nan)
                roll_r2.append(np.nan)
                roll_alpha.append(np.nan)

        roll_dates = pd.DatetimeIndex(roll_dates)

        # ── Grafico 1: Rolling weights ────────────────────────────────────────
        fig_w = go.Figure()
        for ci, col in enumerate(x_selected):
            arr = np.array(roll_coefs[col])
            clr = _COLORS[ci % len(_COLORS)]
            fig_w.add_trace(go.Bar(x=roll_dates, y=arr, name=col,
                                   marker_color=_hex_to_rgba(clr, 0.85),
                                   marker_line_width=0,
                                   hovertemplate=f'<b>{col}</b><br>%{{x|%b %Y}}: %{{y:.2%}}<extra></extra>'))
        fig_w.add_hline(y=1.0, line_color='#555', line_dash='dot', line_width=1,
                        annotation_text='Σw=1', annotation_font_size=9,
                        annotation_position='bottom left')
        fig_w.update_layout(
            barmode='stack',
            xaxis=dict(tickformat='%b %Y', tickangle=-45, showgrid=False),
            yaxis=dict(title='Peso', tickformat='.0%', range=[0, 1.05], gridcolor='#e8e8e8'),
            legend=dict(orientation='h', y=-0.22, font=dict(size=10)),
            plot_bgcolor='#f8f8f8', paper_bgcolor='white',
            font=dict(color='#1a3a5c', size=10),
            margin=dict(l=45, r=20, t=12, b=90),
        )

        # Portafoglio suggerito ultima finestra
        last_weights = {c: float(roll_coefs[c][-1]) for c in x_selected
                        if roll_coefs[c] and not np.isnan(roll_coefs[c][-1])}
        sig_w = sorted([(c, w) for c, w in last_weights.items() if w > 0.005],
                       key=lambda t: t[1], reverse=True)
        try:
            te_last = float(np.std(y_fit.values[-W:] - sum(
                X_fit[c].values[-W:] * last_weights.get(c, 0.0) for c in x_selected
            ))) if sig_w and len(y_fit) >= W else np.nan
        except Exception:
            te_last = np.nan

        suggest_rows = []
        for c, w in sig_w:
            clr = _COLORS[x_selected.index(c) % len(_COLORS)]
            suggest_rows.append(html.Div([
                html.Span(f'{c}', title=c, style={'flex': '1', 'font-size': '12px', 'color': '#1a3a5c',
                                                  'min-width': '0', 'overflow': 'hidden',
                                                  'text-overflow': 'ellipsis', 'white-space': 'nowrap'}),
                html.Div(style={'width': f'{w*100:.1f}%', 'min-width': '2px', 'height': '10px',
                                'background': clr, 'border-radius': '2px', 'margin': '0 8px',
                                'flex': '0 0 auto', 'max-width': '40%'}),
                html.Span(f'{w:.1%}', style={'width': '56px', 'text-align': 'right',
                                              'font-size': '12px', 'font-weight': '700',
                                              'color': '#1b7a34'}),
            ], style={'display': 'flex', 'align-items': 'center', 'margin': '3px 0'}))

        suggest_block = html.Div([
            html.B('🎯 Percentuali del portafoglio — ultima finestra',
                   style={'font-size': '12px', 'color': '#1a3a5c', 'display': 'block',
                          'margin-bottom': '4px'}),
            html.Span(
                (f'Finestra: {roll_dates[-1].strftime("%b %Y") if len(roll_dates) else "—"}'
                 f' | Assets: {len(sig_w)}'
                 + (f' | TE: {te_last*100:.2f}%/mese' if not np.isnan(te_last) else '')),
                style={'font-size': '11px', 'color': '#6b7a99', 'display': 'block', 'margin-bottom': '8px'}),
            *suggest_rows,
        ], style={'max-width': '720px'}) if sig_w else html.Span('Nessun peso significativo',
                                    style={'color': '#888', 'font-size': '12px'})

        # ── Grafico 2: R² rolling ────────────────────────────────────────────
        r2_arr = np.array(roll_r2)
        mean_r2 = float(np.nanmean(r2_arr))
        fig_r2 = go.Figure()
        # Fasce di lettura: quanto del rendimento di Y spiegano i fattori.
        for y0, y1, col, txt in ((0, 50, 'rgba(214,39,40,0.06)', 'debole'),
                                 (50, 80, 'rgba(255,165,0,0.07)', 'discreto'),
                                 (80, 100, 'rgba(44,160,44,0.07)', 'buono')):
            fig_r2.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0, layer='below')
            fig_r2.add_annotation(xref='paper', x=0.005, y=(y0 + y1) / 2, yref='y', text=txt,
                                  showarrow=False, xanchor='left',
                                  font=dict(size=10, color='#8a94a6'))
        fig_r2.add_trace(go.Scatter(x=roll_dates, y=r2_arr * 100, name='R² rolling',
                                    line=dict(color='#1a3a6b', width=2.6),
                                    hovertemplate='%{y:.1f}%<extra>R² rolling</extra>'))
        n = len(roll_dates)
        fig_r2.add_trace(go.Scatter(x=roll_dates, y=[mean_r2 * 100] * n,
                                    name=f'Media rolling {mean_r2*100:.1f}%',
                                    line=dict(color='#d62728', dash='dash', width=1.6),
                                    hoverinfo='skip'))
        fig_r2.add_trace(go.Scatter(x=roll_dates, y=[g_r2 * 100] * n,
                                    name=f'Intero periodo {g_r2*100:.1f}%',
                                    line=dict(color='#2ca02c', dash='dot', width=2),
                                    hoverinfo='skip'))
        fig_r2.update_layout(hovermode='x unified', margin=dict(t=36, b=30, l=50, r=12),
                             paper_bgcolor='white', plot_bgcolor='white', font=dict(size=12),
                             legend=dict(orientation='h', x=0, y=1.02, yanchor='bottom',
                                         font=dict(size=12)))
        fig_r2.update_yaxes(range=[0, 101], dtick=20, ticksuffix='%', showgrid=True,
                            gridcolor='#e3e8ef', title_text='R²', zeroline=False)
        fig_r2.update_xaxes(showgrid=True, gridcolor='#eef1f5', hoverformat='%b %Y')

        # ── Grafico 3: Fit storico ───────────────────────────────────────────
        fig_fit = go.Figure()
        cum_y = (1 + y_fit).cumprod() - 1
        cum_fitted = (1 + g_fitted).cumprod() - 1
        fig_fit.add_trace(go.Scatter(x=y_fit.index, y=cum_y.values * 100,
                                     name=f'{y_col} (osservato)',
                                     line=dict(color='#1f77b4', width=2)))
        fig_fit.add_trace(go.Scatter(x=g_fitted.index, y=cum_fitted.values * 100,
                                     name='Stimato', line=dict(color='#d62728', width=1.8, dash='dot')))
        fig_fit.update_layout(
            hovermode='x unified',
            legend=dict(orientation='h', y=1.02, x=0, font=dict(size=9)),
            margin=dict(t=30, b=30, l=55, r=20),
            paper_bgcolor='white', plot_bgcolor='#f8f8f8')
        fig_fit.update_xaxes(showgrid=True, gridcolor='#e8e8e8')
        fig_fit.update_yaxes(showgrid=True, gridcolor='#e8e8e8', title_text='Rend. cumulato (%)')
        fig_fit.add_annotation(text=f'R²={g_r2:.4f}  R²adj={g_r2_adj:.4f}',
                               xref='paper', yref='paper', x=0.99, y=0.02,
                               showarrow=False, font=dict(size=10),
                               bgcolor='rgba(255,255,255,0.8)', xanchor='right', yanchor='bottom')

        # ── Grafico 4: Alpha rolling ─────────────────────────────────────────
        alpha_arr = np.array(roll_alpha) * 12 * 100
        mean_alpha = float(np.nanmean(alpha_arr))
        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Bar(x=roll_dates, y=alpha_arr,
                                   marker_color=['#2ca02c' if v >= 0 else '#d62728' for v in alpha_arr],
                                   marker_line_width=0,
                                   hovertemplate='%{x|%b %Y}: %{y:.2f}%<extra>Alpha</extra>'))
        fig_alpha.add_hline(y=0, line_color='#555', line_dash='dot', line_width=1)
        fig_alpha.add_hline(y=mean_alpha, line_color='navy', line_dash='dash', line_width=1.5,
                            annotation_text=f'μ α={mean_alpha:+.2f}%/anno',
                            annotation_position='right')
        fig_alpha.update_layout(hovermode='x unified', margin=dict(t=14, b=25, l=55, r=120),
                                paper_bgcolor='white', plot_bgcolor='#f8f8f8', showlegend=False)
        fig_alpha.update_xaxes(showgrid=True, gridcolor='#e8e8e8')
        fig_alpha.update_yaxes(showgrid=True, gridcolor='#e8e8e8', title_text='Alpha %/anno')

        # ── Grafico 5: Decomposizione ────────────────────────────────────────
        mean_X_monthly = X_fit.mean()
        contrib = {c: float(np.nanmean(roll_coefs[c])) * float(mean_X_monthly[c]) * 12 * 100
                   for c in x_selected}
        sorted_c = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
        names_d = [k for k, _ in sorted_c] + ['Alpha (residuo)']
        vals_d  = [v for _, v in sorted_c] + [mean_alpha]
        colors_d = [(_COLORS[i % len(_COLORS)] if v >= 0 else '#d62728')
                    for i, (_, v) in enumerate(sorted_c)] + ['#9467bd']
        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Bar(x=names_d, y=vals_d, marker_color=colors_d,
                                    marker_line_width=0,
                                    text=[f'{v:+.2f}%' for v in vals_d],
                                    textposition='outside', textfont=dict(size=9),
                                    hovertemplate='<b>%{x}</b><br>%{y:+.2f}%/anno<extra></extra>'))
        fig_decomp.add_hline(y=0, line_color='#555', line_dash='dot', line_width=1)
        total_c = sum(vals_d)
        fig_decomp.add_hline(y=total_c, line_color='navy', line_dash='dash', line_width=1.5,
                             annotation_text=f'Tot={total_c:+.2f}%', annotation_position='right')
        fig_decomp.update_layout(
            margin=dict(t=24, b=60, l=55, r=90),
            paper_bgcolor='white', plot_bgcolor='#f8f8f8', showlegend=False)
        fig_decomp.update_xaxes(tickangle=-25, tickfont=dict(size=10))
        fig_decomp.update_yaxes(showgrid=True, gridcolor='#e8e8e8', title_text='Contributo %/anno')

        status = (f'✅ Style Analysis | {n_obs} mesi | {len(x_selected)} fattori | '
                  f'R²={g_r2:.4f} | α={mean_alpha:+.2f}%/anno | rolling={W}m')

        # Portafoglio target = pesi ultima finestra rolling, normalizzati a 100 e in %
        target = {c: float(last_weights[c]) for c in x_selected if last_weights.get(c, 0) > 0.0001}
        tot = sum(target.values())
        if tot > 0:
            target = {c: round(w / tot * 100, 2) for c, w in target.items()}
        store_data = {'eq_text': eq_text, 'g_r2': g_r2, 'g_r2_adj': g_r2_adj,
                      'status': status, 'target': target, 'y': y_col}

        # Riepilogo nel menu ⚙ (sul cellulare lo copia il ⚙ della riga in alto).
        _d = lambda t: pd.Timestamp(t).strftime('%m/%Y')
        tipo = 'Sharpe (≥0, Σ=1)' if reg_type == 'sharpe' else 'OLS libero'
        std = ('—' if reg_type == 'sharpe' else
               {'nonrobust': 'OLS classico', 'HC3': 'HC3 robusto', 'HAC': 'HAC Newey-West'}.get(cov_type, cov_type))
        riepilogo = [
            _SA_RIEPILOGO_VUOTO[0],
            _sa_riga('Asset Y:', y_col),
            _sa_riga('Periodo:', f'{_d(combined.index.min())} — {_d(combined.index.max())}'),
            _sa_riga('Mesi usati:', f'{n_obs}'),
            _sa_riga('Fattori di stile:', f'{len(x_selected)}'),
            _sa_riga('Tipo:', tipo),
            _sa_riga('Std Error:', std),
            _sa_riga('R²:', f'{g_r2:.4f}'),
            _sa_riga('R² adj.:', f'{g_r2_adj:.4f}'),
            _sa_riga('Alpha medio:', f'{mean_alpha:+.2f}%/anno',
                     '#1b7a34' if mean_alpha >= 0 else '#b71c1c'),
            _sa_riga('Finestra rolling:', f'{W} mesi'),
            _sa_riga('Lag X:', f'{lag} mesi'),
            _sa_riga('Dati della finestra:', f'{_d(date_start) if date_start else "—"} — '
                                             f'{_d(date_end) if date_end else "—"}', '#5a6b85'),
        ]

        return (eq_text, stat_table, coef_table,
                fig_w, suggest_block,
                fig_r2, fig_fit, fig_alpha, fig_decomp,
                '', store_data, riepilogo)

    # ── Esporta il portafoglio target come Analisi (stesso archivio analyses.json) ──
    @app.callback(
        Output('sa-export-status', 'children'),
        Input('sa-export-btn', 'n_clicks'),
        State('style-analysis-store', 'data'),
        State('sa-export-name', 'value'),
        prevent_initial_call=True,
    )
    def sa_export(n, store, name):
        if not n:
            raise PreventUpdate
        target = (store or {}).get('target') or {}
        if not target:
            return '⚠ Esegui prima la Style Analysis'
        name = (name or '').strip()
        if not name:
            return '⚠ Scrivi un nome per l\'analisi'
        u = _sa_username()
        ns = _sa_current_json(u)
        meta = {a: {'ticker': (ns.get(a, {}) or {}).get('ticker', ''),
                    'valuta': (ns.get(a, {}) or {}).get('currency', 'EUR')}
                for a in target}
        ok = _sm.save_analysis(u, name, target, meta=meta)
        if ok:
            return f'✅ Salvata analisi "{name}" ({len(target)} asset) — importabile dal pulsante generale'
        return '⚠ Errore durante il salvataggio'
