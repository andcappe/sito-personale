"""
Analisi Tattica — sotto-tab «📐 Indicatori».

Tabella con gli asset di current.json (prezzi in EUR) e, per ognuno, l'ultimo
valore degli indicatori tecnici sulle finestre scelte dall'utente:
  - RSI (Wilder) e Stocastico %K (solo chiusure: current.json non ha massimi e
    minimi) con %D, ciascuno su una finestra base e altre tre = base × moltiplicatore
    (predefiniti 2, 3, 4: 14 → 28, 42, 56);
  - tre medie mobili sui prezzi (SMA o EMA) e le loro differenze a coppie, in %
    della più lenta.
Periodicità giornaliera, settimanale o mensile (i prezzi si ricampionano).
Un clic su una riga della tabella apre il grafico dell'asset: prezzo con le medie,
differenze fra medie, RSI e stocastico.

Si calcola solo col pulsante (e all'apertura); il grafico usa i parametri
dell'ultimo calcolo, salvati in `ind-par`.

Finestra Temporale Da/A condivisa con Analisi, Frontiera, Style Analysis e
Correlazioni (`sessions/<u>/finestra.json`, data_core.leggi/scrivi_finestra):
il grafico mostra Da → A, la tabella dà i valori alla data A. Gli indicatori si
calcolano su tutta la storia fino ad A, così le finestre lunghe hanno dati già
al primo giorno mostrato.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, dash_table, Input, Output, State, no_update, callback_context
from dash.exceptions import PreventUpdate

import data_core as dc

# ── Predefiniti ──────────────────────────────────────────────────────────────
_BASE_DEF = 14                   # finestra base di RSI e stocastico
_MOLT_DEF = (2, 3, 4)            # altre tre finestre = base × moltiplicatore
_MM_DEF = (20, 50, 200)          # tre medie mobili sui prezzi

_PERIODICITA = [{'label': 'Giornaliera', 'value': 'D'},
                {'label': 'Settimanale', 'value': 'W'},
                {'label': 'Mensile', 'value': 'M'}]
_UNITA = {'D': 'gg', 'W': 'sett.', 'M': 'mesi'}
_NOME_PER = {o['value']: o['label'] for o in _PERIODICITA}

# Calendari compatti (stesse regole di .fe-box in Frontiera); va nell'index_string.
CSS = '''
  .ind-date .SingleDatePickerInput { border: 1px solid #b8c4d4; border-radius: 4px; }
  .ind-date .DateInput { width: 100px; }
  .ind-date .DateInput_input { font-size: 12px; padding: 3px 7px; height: 26px; line-height: 20px; }
  @media (max-width: 900px) { .ind-sez { flex: 1 1 100%; } #ind-grafico .modebar-container { display: none; } }
  /* Tabella degli asset: intestazione ferma in alto, Asset e Prezzo fermi a sinistra */
  #ind-tabella table { border-collapse: separate !important; border-spacing: 0; }
  #ind-tabella th.dash-header { position: sticky; top: 0; z-index: 2; }
  #ind-tabella .column-0, #ind-tabella .column-1 { position: sticky; z-index: 1; background-color: #fff;
                                                   box-sizing: border-box; }
  #ind-tabella .column-0 { left: 0; width: 125px !important; min-width: 125px !important;
                           max-width: 125px !important; }
  #ind-tabella .column-1 { left: 125px; width: 68px !important; min-width: 68px !important;
                           max-width: 68px !important; border-right: 2px solid #b8c4d4 !important; }
  #ind-tabella th.column-0, #ind-tabella th.column-1 { z-index: 3; background-color: #f0f4fa; }
  .ind-add input { font-size: 12px; padding: 4px 7px; border: 1px solid #b8c4d4; border-radius: 4px; }
'''

_VALUTE = ['EUR', 'USD', 'GBP', 'CHF', 'JPY']

_COLORI = ['#0066cc', '#e8730c', '#2ca02c', '#9467bd']
_ROSSO, _ROSSO_BG = '#b91c1c', '#fde2e1'
_VERDE, _VERDE_BG = '#15803d', '#dcfce7'

# ── Stili ────────────────────────────────────────────────────────────────────
_BOX = {'border': '1px solid #e2e8f0', 'border-radius': '6px', 'background': '#f8fafc',
        'padding': '8px 12px', 'display': 'flex', 'flex-direction': 'column', 'gap': '5px'}
_TIT = {'font-size': '12px', 'font-weight': '700', 'color': '#1a3a5c'}
_LAB = {'font-size': '11px', 'color': '#4a5d7a'}
_INP = {'font-size': '12px', 'padding': '4px 7px', 'border': '1px solid #b8c4d4',
        'border-radius': '4px'}
_NUM = {**_INP, 'width': '58px'}
_SOTTO = {'font-size': '11.5px', 'font-weight': '700', 'color': '#33475b'}
# Gruppo dentro la sezione Parametri (RSI, Stocastico, Medie): senza riquadro proprio.
_GRUPPO = {'display': 'flex', 'flex-direction': 'column', 'gap': '5px'}


# ─────────────────────────────────────────────────────────────────────────────
# Calcolo
# ─────────────────────────────────────────────────────────────────────────────
def _num(v, predef, lo=0, hi=100):
    try:
        v = float(v)
        return v if lo <= v <= hi else predef
    except (TypeError, ValueError):
        return predef


def _fmt_m(m):
    return f'{m:g}'.replace('.', ',')


def _finestre(base, molt):
    """Finestra base + base × ciascun moltiplicatore → ([14, 28, 42, 56], ['', '×2', '×3', '×4']).
    Moltiplicatori vuoti o non validi si saltano; finestre 2…500, senza doppioni."""
    base = int(_num(base, _BASE_DEF, 2, 500))
    n_out, x_out = [base], ['']
    for m in molt:
        try:
            m = float(m)
        except (TypeError, ValueError):
            continue
        if not 0 < m <= 50:
            continue
        n = int(round(base * m))
        if 2 <= n <= 500 and n not in n_out:
            n_out.append(n)
            x_out.append(f'×{_fmt_m(m)}')
    return n_out, x_out


def _medie(valori):
    """Tre lunghezze di media mobile (2…500, senza doppioni, crescenti)."""
    out = sorted({int(v) for v in (_num(x, None, 2, 500) for x in valori) if v is not None})
    return out or list(_MM_DEF)


def parametri(per, rsi_base, rsi_m, rsi_lo, rsi_hi, stoc_base, stoc_m, stoc_d, stoc_lo, stoc_hi,
              mm, mm_tipo):
    rsi_n, rsi_x = _finestre(rsi_base, rsi_m)
    stoc_n, stoc_x = _finestre(stoc_base, stoc_m)
    medie = _medie(mm)
    return {
        'per': per if per in _UNITA else 'D',
        'rsi': rsi_n, 'rsi_x': rsi_x, 'rsi_lo': _num(rsi_lo, 30), 'rsi_hi': _num(rsi_hi, 70),
        'stoc': stoc_n, 'stoc_x': stoc_x, 'stoc_d': int(_num(stoc_d, 3, 1, 50)),
        'stoc_lo': _num(stoc_lo, 20), 'stoc_hi': _num(stoc_hi, 80),
        'medie': medie,
        'mm': [(a, b) for i, a in enumerate(medie) for b in medie[i + 1:]],
        'mm_tipo': 'EMA' if mm_tipo == 'EMA' else 'SMA',
    }


def ricampiona(s, per):
    s = s.dropna()
    if per == 'W':
        return s.resample('W-FRI').last().dropna()
    if per == 'M':
        try:
            return s.resample('ME').last().dropna()
        except ValueError:
            return s.resample('M').last().dropna()
    return s


def rsi(s, n):
    """RSI di Wilder (media esponenziale con alpha = 1/n)."""
    d = s.diff()
    su = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    giu = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    r = 100 - 100 / (1 + su / giu)
    return r.where(giu > 0, 100.0).where(su.notna())


def stocastico(s, n, d=3):
    """(%K, %D) sulle sole chiusure: posizione del prezzo nel range di n periodi."""
    lo = s.rolling(n, min_periods=n).min()
    hi = s.rolling(n, min_periods=n).max()
    k = (100 * (s - lo) / (hi - lo)).where(hi > lo)
    return k, k.rolling(d, min_periods=1).mean()


def media(s, n, tipo):
    if tipo == 'EMA':
        return s.ewm(span=n, adjust=False, min_periods=n).mean()
    return s.rolling(n, min_periods=n).mean()


def diff_mm(s, a, b, tipo):
    """Media veloce − lenta, in % della lenta."""
    return (media(s, a, tipo) / media(s, b, tipo) - 1) * 100


def _col(tipo, *n):
    return f"{tipo}_{'_'.join(map(str, n))}"


def _ultimo(x):
    x = x.dropna()
    return round(float(x.iloc[-1]), 1) if len(x) else None


def taglia(prezzi, fine):
    """Prezzi fino alla data `fine` (inclusa); senza data, tutti."""
    if not fine:
        return prezzi
    return prezzi[prezzi.index <= pd.Timestamp(fine) + pd.Timedelta(hours=23, minutes=59)]


def tabella(prezzi, p):
    """Righe della tabella (una per asset) e colonne di dash_table."""
    u = _UNITA[p['per']]
    # Asset e Prezzo sono le due colonne ferme (fixed_columns=2), poi nell'ordine
    # voluto: stocastici, RSI, medie, differenze fra medie; la data in fondo.
    colonne = [{'name': 'Asset', 'id': 'asset'},
               {'name': 'Prezzo €', 'id': 'prezzo', 'type': 'numeric'}]
    colonne += [{'name': f'Stoc. {n} {x}'.strip(), 'id': _col('stoc', n), 'type': 'numeric'}
                for n, x in zip(p['stoc'], p['stoc_x'])]
    colonne += [{'name': f'RSI {n} {x}'.strip(), 'id': _col('rsi', n), 'type': 'numeric'}
                for n, x in zip(p['rsi'], p['rsi_x'])]
    colonne += [{'name': f"{p['mm_tipo']} {n} €", 'id': _col('med', n), 'type': 'numeric'}
                for n in p['medie']]
    colonne += [{'name': f"{p['mm_tipo']} {a}/{b} %", 'id': _col('mm', a, b), 'type': 'numeric'}
                for a, b in p['mm']]
    colonne += [{'name': 'Dati al', 'id': 'data'}]
    righe = []
    for asset in prezzi.columns:
        s = ricampiona(prezzi[asset], p['per'])
        if len(s) < 3 or prezzi[asset].dropna().empty:
            continue
        r = {'asset': asset, 'prezzo': round(float(s.iloc[-1]), 2),
             'data': prezzi[asset].dropna().index[-1].strftime('%d/%m/%y')}
        for n in p['stoc']:
            r[_col('stoc', n)] = _ultimo(stocastico(s, n, p['stoc_d'])[0])
        for n in p['rsi']:
            r[_col('rsi', n)] = _ultimo(rsi(s, n))
        for n in p['medie']:
            m = media(s, n, p['mm_tipo']).dropna()
            r[_col('med', n)] = round(float(m.iloc[-1]), 2) if len(m) else None
        for a, b in p['mm']:
            r[_col('mm', a, b)] = _ultimo(diff_mm(s, a, b, p['mm_tipo']))
        righe.append(r)
    return righe, colonne, u


def _stili(p):
    """Colori delle celle: ipercomprato rosso, ipervenduto verde, MM sopra/sotto zero."""
    st = []
    for tipo, lo, hi, finestre in (('rsi', p['rsi_lo'], p['rsi_hi'], p['rsi']),
                                    ('stoc', p['stoc_lo'], p['stoc_hi'], p['stoc'])):
        for n in finestre:
            c = _col(tipo, n)
            st.append({'if': {'filter_query': f'{{{c}}} >= {hi}', 'column_id': c},
                       'backgroundColor': _ROSSO_BG, 'color': _ROSSO, 'fontWeight': '700'})
            st.append({'if': {'filter_query': f'{{{c}}} <= {lo}', 'column_id': c},
                       'backgroundColor': _VERDE_BG, 'color': _VERDE, 'fontWeight': '700'})
    for a, b in p['mm']:
        c = _col('mm', a, b)
        st.append({'if': {'filter_query': f'{{{c}}} > 0', 'column_id': c}, 'color': _VERDE})
        st.append({'if': {'filter_query': f'{{{c}}} < 0', 'column_id': c}, 'color': _ROSSO})
    return st


# ─────────────────────────────────────────────────────────────────────────────
# Grafico
# ─────────────────────────────────────────────────────────────────────────────
def _soglia(fig, x0, x1, y, riga, colore):
    """Linea orizzontale come traccia (le shape su un tab nascosto danno MNaN)."""
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode='lines', hoverinfo='skip',
                             showlegend=False, line=dict(color=colore, width=1, dash='dot')),
                  row=riga, col=1)


def figura(prezzi_asset, asset, p, da=None, a=None):
    s = ricampiona(taglia(prezzi_asset, a), p['per'])
    if len(s) < 3:
        return _vuota('Dati insufficienti per questo asset'), 300
    inizio = max(pd.Timestamp(da), s.index[0]) if da else s.index[0]
    if inizio >= s.index[-1]:
        return _vuota('Nessun dato nella finestra temporale scelta'), 300

    # Una legenda per pannello, col nome del pannello come titolo: i colori
    # ripartono da capo in ogni pannello e una legenda unica li confonderebbe.
    # Ordine (scelta dell'utente): prezzo, stocastico, RSI, differenze fra medie.
    pannelli = ['Prezzo € e medie', f"Stocastico (%D a {p['stoc_d']})", 'RSI',
                f"{p['mm_tipo']} veloce − lenta (%)"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.085,
                        row_heights=[0.37, 0.21, 0.21, 0.21])
    # La legenda dello stocastico (8 voci) va spesso su due righe: pannello più in basso.
    d2 = fig.layout.yaxis2.domain
    fig.layout.yaxis2.domain = [d2[0], d2[1] - 0.05]
    v = lambda x: x[x.index >= inizio]           # noqa: E731
    x0, x1 = max(inizio, s.index[0]), s.index[-1]
    leg = lambda r: 'legend' if r == 1 else f'legend{r}'   # noqa: E731

    def linea(x, nome, riga, colore, larg=1.3, tratto=None, fmt='%{y:.1f}'):
        fig.add_trace(go.Scatter(x=x.index, y=x, name=nome, legend=leg(riga), hovertemplate=fmt,
                                 line=dict(color=colore, width=larg, dash=tratto)), row=riga, col=1)

    linea(v(s), asset, 1, '#1a3a5c', 1.7, fmt='%{y:.2f} €')
    for i, n in enumerate(p['medie']):
        linea(v(media(s, n, p['mm_tipo'])), f"{p['mm_tipo']} {n}", 1, _COLORI[(i + 1) % 4], 1.2,
              fmt='%{y:.2f} €')
    for i, (a, b) in enumerate(p['mm']):
        linea(v(diff_mm(s, a, b, p['mm_tipo'])), f'{a}/{b}', 4, _COLORI[i % 4], fmt='%{y:.2f}%')
    _soglia(fig, x0, x1, 0, 4, '#64748b')
    for i, (n, x) in enumerate(zip(p['rsi'], p['rsi_x'])):
        linea(v(rsi(s, n)), f'RSI {n} {x}'.strip(), 3, _COLORI[i % 4])
    _soglia(fig, x0, x1, p['rsi_hi'], 3, _ROSSO)
    _soglia(fig, x0, x1, p['rsi_lo'], 3, _VERDE)
    # Prima tutte le %D (linee), poi tutte le %K (puntini): in legenda stanno vicine,
    # e i pulsanti sotto il grafico le accendono/spengono tutte insieme.
    stoc = [(i, n, x, *stocastico(s, n, p['stoc_d'])) for i, (n, x)
            in enumerate(zip(p['stoc'], p['stoc_x']))]
    gruppi = {'%D': [], '%K': []}
    if p['stoc_d'] > 1:
        for i, n, x, k, dd in stoc:
            gruppi['%D'].append(len(fig.data))
            linea(v(dd), f'%D {n} {x}'.strip(), 2, _COLORI[i % 4], 1.4)
    for i, n, x, k, dd in stoc:
        gruppi['%K'].append(len(fig.data))
        linea(v(k), f'%K {n} {x}'.strip(), 2, _COLORI[i % 4], 1.6, 'dot')
    _soglia(fig, x0, x1, p['stoc_hi'], 2, _ROSSO)
    _soglia(fig, x0, x1, p['stoc_lo'], 2, _VERDE)

    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(gridcolor='#eef2f7', zeroline=False)
    fig.update_xaxes(gridcolor='#eef2f7')
    altezza, basso = 1000, 64
    area = altezza - 64 - basso
    # Un menu per pulsante: in un menu solo, args2 (secondo clic = riaccendi) segue
    # l'ultimo pulsante premuto e premere %D dopo %K rompeva l'alternanza di %K.
    # active=-1: altrimenti il pulsante nasce «attivo» e il primo clic esegue args2.
    menu = [dict(type='buttons', showactive=False, active=-1, x=j, xanchor=('left', 'right')[j], y=-30 / area,
                 yanchor='top', pad=dict(t=0), bgcolor='#f0f4fa', bordercolor='#b8c4d4',
                 font=dict(size=11, color='#1a3a5c'),
                 buttons=[dict(label=f'{g} tutte on/off', method='restyle',
                               args=[{'visible': 'legendonly'}, idx],
                               args2=[{'visible': True}, idx])])
            for j, (g, idx) in enumerate((g, i) for g, i in gruppi.items() if i)]   # %D a sinistra, %K a destra
    legende = {}
    for r in range(1, 5):
        alto = fig.layout[f"yaxis{'' if r == 1 else r}"].domain[1]
        legende[leg(r)] = dict(
            orientation='h', x=0, xanchor='left', y=alto + 4 / altezza, yanchor='bottom',
            font=dict(size=11), bgcolor='rgba(0,0,0,0)', itemclick='toggle',
            # titolo del pannello sopra le voci, non accanto
            title=dict(text=(f"<span style='font-size:14px'><b>{asset}</b> · {_NOME_PER[p['per']]}</span><br>"
                             if r == 1 else '') + f'<b>{pannelli[r - 1]}</b>', side='top left',
                       font=dict(size=11, color='#1a3a5c')))
    fig.update_layout(
        height=altezza, template='plotly_white', hovermode='x unified',
        margin=dict(l=50, r=12, t=120, b=basso), updatemenus=menu,
        **legende,
    )
    return fig, altezza


def _vuota(msg):
    f = go.Figure()
    f.add_annotation(text=msg, x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False,
                     font=dict(size=13, color='#64748b'))
    f.update_layout(template='plotly_white', xaxis=dict(visible=False), yaxis=dict(visible=False),
                    margin=dict(l=10, r=10, t=10, b=10))
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
def _campo(etichetta, comp):
    return html.Div([html.Span(etichetta, style=_LAB), comp],
                    style={'display': 'flex', 'align-items': 'center', 'gap': '6px'})


def _soglie(id_lo, lo, id_hi, hi):
    return _campo('Soglie', html.Div(title='Ipervenduto (verde) / ipercomprato (rosso)', children=[
        dcc.Input(id=id_lo, type='number', value=lo, min=0, max=100, persistence=True,
                  persistence_type='local', style=_NUM),
        html.Span('/', style=_LAB),
        dcc.Input(id=id_hi, type='number', value=hi, min=0, max=100, persistence=True,
                  persistence_type='local', style=_NUM),
    ], style={'display': 'flex', 'align-items': 'center', 'gap': '4px'}))


def _numero(id_, valore, passo=1, lo=2, hi=500, larg='52px'):
    return dcc.Input(id=id_, type='number', value=valore, min=lo, max=hi, step=passo, debounce=True,
                     persistence=True, persistence_type='local', style={**_NUM, 'width': larg})


def _finestra_e_molt(pref):
    """Finestra base + tre moltiplicatori (RSI e stocastico)."""
    return html.Div([
        _campo('Finestra', _numero(f'{pref}-base', _BASE_DEF)),
        html.Div(_campo('Moltiplicatori', html.Div(
            [_numero(f'{pref}-m{i + 1}', m, 0.5, 0.5, 50, '50px') for i, m in enumerate(_MOLT_DEF)],
            style={'display': 'flex', 'gap': '4px'})),
            title='Altre tre finestre = finestra × moltiplicatore (vuoto = non usata)'),
    ], style={'display': 'flex', 'gap': '10px', 'flex-wrap': 'wrap'})


def _sezione(titolo, corpo):
    """Riquadro dei parametri che sul cellulare diventa una sezione espandibile (chiusa)."""
    return html.Div(className='nb-espandi ind-sez', style=_BOX, children=[
        html.Div([titolo, html.I(className='fa-solid fa-chevron-down nb-espandi-ico')],
                 className='nb-espandi-btn', style=_TIT),
        html.Div(corpo, className='nb-espandi-corpo'),
    ])


def layout():
    return html.Div([
        html.Div([
            html.Button('▶ Calcola indicatori', id='ind-calcola', n_clicks=0,
                        className='nb-nascondi-cell',   # sul cellulare: doppione in alto (at-calcola-cell)
                        style={'background-color': '#1a3a5c', 'color': 'white', 'border': 'none',
                               'padding': '8px 16px', 'border-radius': '4px', 'cursor': 'pointer',
                               'font-weight': 'bold', 'font-size': '12px'}),
            html.Span(id='ind-stato', style={'font-size': '11px', 'color': '#b45309'}),   # solo avvisi
        ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'wrap', 'gap': '8px',
                  'margin-bottom': '10px'}),

        # ── Parametri ────────────────────────────────────────────────────
        html.Div([
            # Prima la Finestra Temporale, poi la Periodicità: sul cellulare sono due
            # sezioni chiuse all'avvio (nb-espandi), sul computer due riquadri come gli altri.
            _sezione('Finestra Temporale', html.Div([
                _campo('Da', dcc.DatePickerSingle(id='ind-da', first_day_of_week=1, clearable=False,
                                                  display_format='DD/MM/YYYY')),
                _campo('A', dcc.DatePickerSingle(id='ind-a', first_day_of_week=1, clearable=False,
                                                 display_format='DD/MM/YYYY')),
            ], className='ind-date', style={'display': 'flex', 'gap': '10px', 'flex-wrap': 'wrap'})),
            _sezione('Periodicità', dcc.RadioItems(
                id='ind-per', options=_PERIODICITA, value='D', inline=True,   # sempre giornaliera all'apertura
                labelStyle={'margin-right': '10px', 'font-size': '12px'})),
            _sezione('Parametri', html.Div([
            html.Div([
                html.Span('Stocastico', style=_SOTTO),
                _finestra_e_molt('ind-stoc'),
                html.Div([
                    _campo('%D', dcc.Input(id='ind-stoc-d', type='number', value=3, min=1, max=50,
                                           persistence=True, persistence_type='local', style=_NUM)),
                    _soglie('ind-stoc-lo', 20, 'ind-stoc-hi', 80),
                ], style={'display': 'flex', 'gap': '10px', 'flex-wrap': 'wrap'}),
            ], style=_GRUPPO),
            html.Div([
                html.Span('RSI', style=_SOTTO),
                _finestra_e_molt('ind-rsi'),
                _soglie('ind-rsi-lo', 30, 'ind-rsi-hi', 70),
            ], style=_GRUPPO),
            html.Div([
                html.Span('Medie mobili sui prezzi', style=_SOTTO),
                _campo('Periodi', html.Div(
                    [_numero(f'ind-mm{i + 1}', n, 1, 2, 500, '58px') for i, n in enumerate(_MM_DEF)],
                    style={'display': 'flex', 'gap': '4px'})),
                dcc.RadioItems(id='ind-mm-tipo', options=[{'label': 'Semplice (SMA)', 'value': 'SMA'},
                                                          {'label': 'Esponenziale (EMA)', 'value': 'EMA'}],
                               value='SMA', inline=True, persistence=True, persistence_type='local',
                               labelStyle={'margin-right': '10px', 'font-size': '12px'}),
            ], style=_GRUPPO),
            ], className='ind-parametri', style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '12px 24px'})),
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '8px', 'margin-bottom': '8px'}),

        dcc.Store(id='ind-par'),
        dcc.Store(id='ind-nuovo'),          # nome dell'asset appena aggiunto: ricalcola e lo mostra
        # ── Asset a tutta larghezza, sotto i Grafici (computer e cellulare) ──
        html.Div([
            html.Div([_sezione('Asset', [dcc.Loading(type='circle', color='#1a3a5c', className='ind-tab', children=dash_table.DataTable(
                id='ind-tabella', data=[], columns=[{'name': 'Asset', 'id': 'asset'}],
                # Asset e Prezzo fermi con position: sticky (CSS `.ind-tab`), non con fixed_columns:
                # dash_table misura le colonne fisse al disegno e, nato dentro una sezione chiusa
                # (larghezza 0), sballava le intestazioni. Così la riga d'intestazione scorre di lato
                # insieme ai valori, salvo le prime due celle.
                sort_action='native',
                style_table={'overflow': 'auto', 'maxHeight': '70vh'},
                style_header={'backgroundColor': '#f0f4fa', 'color': '#1a3a5c', 'fontWeight': '700',
                              'fontSize': '11px', 'whiteSpace': 'normal', 'textAlign': 'center', 'height': '34px',
                              'border': '1px solid #e2e8f0'},
                style_cell={'fontSize': '11.5px', 'padding': '4px 6px', 'border': '1px solid #eef2f7',
                            'width': '66px', 'minWidth': '66px', 'maxWidth': '66px', 'cursor': 'pointer'},
                style_cell_conditional=[{'if': {'column_id': 'asset'}, 'textAlign': 'left',
                                         'fontWeight': '600', 'color': '#1a3a5c', 'overflow': 'hidden',
                                         'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'},
                                        ],
                tooltip_delay=300, tooltip_duration=None,
            )),
                # ── Aggiungi asset al file personale (come in Analisi di Portafoglio) ──
                html.Div([
                    html.Span('Aggiungi asset al file personale', style=_SOTTO),
                    html.Div([
                        dcc.Input(id='ind-add-ticker', placeholder='Ticker Yahoo (es. AAPL)',
                                  debounce=False, style={'width': '150px'}),
                        dcc.Input(id='ind-add-desc', placeholder='Descrizione (facoltativa)',
                                  style={'flex': '1 1 160px', 'min-width': '0'}),
                        dcc.Dropdown(id='ind-add-valuta', options=_VALUTE, value='EUR',
                                     clearable=False, searchable=False,
                                     style={'width': '80px', 'font-size': '12px'}),
                        html.Button('➕ Aggiungi', id='ind-add-btn', n_clicks=0,
                                    style={'background': '#e8f5e9', 'border': '1px solid #a5d6a7',
                                           'color': '#1b5e20', 'border-radius': '4px',
                                           'padding': '5px 12px', 'font-size': '12px',
                                           'font-weight': '600', 'cursor': 'pointer'}),
                    ], className='ind-add',
                        style={'display': 'flex', 'gap': '6px', 'flex-wrap': 'wrap',
                               'align-items': 'center'}),
                    dcc.Loading(type='dot', color='#1a3a5c', children=html.Div(
                        id='ind-add-stato', style={'font-size': '11px', 'color': '#555',
                                                   'min-height': '14px'})),
                ], style={**_GRUPPO, 'margin-top': '10px', 'padding-top': '8px',
                          'border-top': '1px solid #e2e8f0'}),
            ]),
            ], className='nb-piena',
                style={'width': '100%', 'min-width': '0'}),
            html.Div([_sezione('Grafici', [
                dcc.Dropdown(id='ind-asset', options=[], placeholder='Asset da mostrare…',
                             persistence=True, persistence_type='local', clearable=False,
                             style={'font-size': '12px', 'margin-bottom': '6px'}),
                dcc.Loading(type='circle', color='#1a3a5c', children=dcc.Graph(
                    id='ind-grafico', figure=_vuota('Premi «Calcola indicatori»'),
                    responsive=True, style={'width': '100%', 'height': '1000px'},
                    config={'displaylogo': False, 'scrollZoom': False})),
            ])], className='nb-piena', style={'width': '100%', 'min-width': '0'}),
        ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '12px'}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────────────────
def register_callbacks(app):

    @app.callback(
        Output('ind-tabella', 'data'), Output('ind-tabella', 'columns'),
        Output('ind-tabella', 'style_data_conditional'), Output('ind-tabella', 'tooltip_data'),
        Output('ind-par', 'data'), Output('ind-asset', 'options'), Output('ind-asset', 'value'),
        Output('ind-stato', 'children'),
        Input('ind-calcola', 'n_clicks'), Input('ind-nuovo', 'data'),
        State('ind-per', 'value'),
        State('ind-rsi-base', 'value'), State('ind-rsi-m1', 'value'), State('ind-rsi-m2', 'value'),
        State('ind-rsi-m3', 'value'), State('ind-rsi-lo', 'value'), State('ind-rsi-hi', 'value'),
        State('ind-stoc-base', 'value'), State('ind-stoc-m1', 'value'), State('ind-stoc-m2', 'value'),
        State('ind-stoc-m3', 'value'), State('ind-stoc-d', 'value'), State('ind-stoc-lo', 'value'),
        State('ind-stoc-hi', 'value'),
        State('ind-mm1', 'value'), State('ind-mm2', 'value'), State('ind-mm3', 'value'),
        State('ind-mm-tipo', 'value'), State('ind-asset', 'value'), State('ind-a', 'date'),
    )
    def _ind_calcola(n, nuovo, per, rb, r1, r2, r3, rlo, rhi, sb, s1, s2, s3, sd, slo, shi,
                     m1, m2, m3, mmt, scelto, fine):
        p = parametri(per, rb, (r1, r2, r3), rlo, rhi, sb, (s1, s2, s3), sd, slo, shi,
                      (m1, m2, m3), mmt)
        prezzi = dc.build_prices()
        if prezzi is None or prezzi.empty:
            return ([], no_update, [], [], p, [], None,
                    '⚠ Nessun asset: carica i dati in Analisi di Portafoglio.')
        if not fine:                          # prima apertura: le date non sono ancora arrivate
            fine = dc.finestra_iniziale(prezzi.index.min(), prezzi.index.max())[1]
        p['a'] = str(fine)[:10]
        righe, colonne, u = tabella(taglia(prezzi, p['a']), p)
        nomi = [x['asset'] for x in righe]
        if callback_context.triggered_id == 'ind-nuovo' and nuovo in nomi:
            scelto = nuovo                    # appena aggiunto: lo mostra nel grafico
        valore = scelto if scelto in nomi else (nomi[0] if nomi else None)
        sel = [{'if': {'filter_query': f'{{asset}} = "{valore}"'}, 'backgroundColor': '#e8f0fb'}] \
            if valore and '"' not in valore else []
        stato = ''
        return (righe, colonne, sel + _stili(p), [{'asset': x['asset']} for x in righe], p,
                [{'label': x, 'value': x} for x in nomi], valore, stato)

    @app.callback(
        Output('ind-asset', 'value', allow_duplicate=True),
        Output('ind-tabella', 'active_cell'),
        Output('ind-tabella', 'selected_cells'),       # niente cella rosa: la riga la colora _ind_grafico
        Input('ind-tabella', 'active_cell'),
        State('ind-tabella', 'derived_viewport_data'),
        prevent_initial_call=True,
    )
    def _ind_clic(cella, vista):
        if not cella or not vista or cella.get('row') is None or cella['row'] >= len(vista):
            raise PreventUpdate
        return vista[cella['row']]['asset'], None, []

    @app.callback(
        Output('ind-grafico', 'figure'), Output('ind-grafico', 'style'),
        Output('ind-tabella', 'style_data_conditional', allow_duplicate=True),
        Input('ind-asset', 'value'), Input('ind-da', 'date'), Input('ind-a', 'date'),   # Finestra Temporale
        Input('ind-par', 'data'),
        prevent_initial_call=True,
    )
    def _ind_grafico(asset, da, a, p):
        if not p:
            raise PreventUpdate
        stili = _stili(p)
        if asset and '"' not in asset:
            stili = [{'if': {'filter_query': f'{{asset}} = "{asset}"'}, 'backgroundColor': '#e8f0fb'}] + stili
        prezzi = dc.build_prices()
        if not asset or prezzi is None or asset not in prezzi.columns:
            return _vuota('Scegli un asset dalla tabella'), {'width': '100%', 'height': '400px'}, stili
        if not da or not a:
            da, a = dc.finestra_iniziale(prezzi.index.min(), prezzi.index.max())
        fig, h = figura(prezzi[asset], asset, p, da, a)
        return fig, {'width': '100%', 'height': f'{h}px'}, stili

    # ── Finestra Temporale condivisa (finestra.json) ────────────────────────
    @app.callback(
        Output('ind-da', 'date'), Output('ind-a', 'date'),
        Output('ind-da', 'min_date_allowed'), Output('ind-da', 'max_date_allowed'),
        Output('ind-a', 'min_date_allowed'), Output('ind-a', 'max_date_allowed'),
        Input('at-tabs', 'value'),
    )
    def _ind_date_condivise(tab):
        # Rilette all'apertura e a ogni cambio di sotto-tab: un'altra pagina può averle cambiate.
        prezzi = dc.build_prices()
        if prezzi is None or prezzi.empty:
            raise PreventUpdate
        dmin, dmax = prezzi.index.min(), prezzi.index.max()
        da, a = dc.finestra_iniziale(dmin, dmax)
        lo, hi = dmin.strftime('%Y-%m-%d'), dmax.strftime('%Y-%m-%d')
        return da, a, lo, hi, lo, hi

    @app.callback(
        Input('ind-da', 'date'), Input('ind-a', 'date'),
        prevent_initial_call=True,
    )
    def _ind_salva_finestra(da, a):
        if not da or not a or str(da)[:10] > str(a)[:10]:
            return
        prezzi = dc.build_prices()
        dc.scrivi_finestra(da, a, dmax=prezzi.index.max() if prezzi is not None and len(prezzi) else None)

    # ── Aggiungi asset al file personale ────────────────────────────────────
    @app.callback(
        Output('ind-add-stato', 'children'), Output('ind-nuovo', 'data'),
        Output('ind-add-ticker', 'value'), Output('ind-add-desc', 'value'),
        Input('ind-add-btn', 'n_clicks'), Input('ind-add-ticker', 'n_submit'),
        State('ind-add-ticker', 'value'), State('ind-add-desc', 'value'),
        State('ind-add-valuta', 'value'),
        running=[(Output('ind-add-btn', 'disabled'), True, False)],
        prevent_initial_call=True,
    )
    def _ind_aggiungi(n, invio, ticker, desc, valuta):
        ticker = (ticker or '').strip().upper()
        if not ticker:
            return '⚠ Scrivi il ticker Yahoo (es. AAPL, SWDA.MI, BTC-USD)', no_update, no_update, no_update
        desc = (desc or '').strip() or ticker
        attuali = dc.read_current()
        if desc in attuali:
            return f'⚠ «{desc}» è già nel file', no_update, no_update, no_update
        doppio = next((k for k, v in attuali.items() if (v.get('ticker') or '').upper() == ticker), None)
        if doppio:
            return f'⚠ {ticker} è già nel file come «{doppio}»', no_update, no_update, no_update
        ok, msg = dc.add_asset_to_current(ticker, desc, valuta or 'EUR')
        if not ok:
            return msg, no_update, no_update, no_update
        return msg, desc, '', ''
