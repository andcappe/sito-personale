"""Calendario economico — i principali dati macro USA che precedono le riunioni FED.

Due sotto-tab: questo calendario macro e quello delle trimestrali delle società
(`trimestrali.py`, con il suo aggiornamento ogni 3 mesi).

Aggrega su un'unica linea temporale:
  • Riunioni FOMC (Federal Reserve) + periodo di blackout comunicazioni intorno a ognuna;
  • Sussidi di disoccupazione settimanali (Department of Labor);
  • Inflazione CPI e occupazione — buste paga non agricole/NFP (Bureau of Labor Statistics);
  • Inflazione PCE — reddito e spesa personale (Bureau of Economic Analysis).

Le date delle riunioni FOMC 2026-2027 sono quelle ufficiali (federalreserve.gov); il
blackout è calcolato secondo la regola FED (dal 2° sabato precedente l'inizio della
riunione fino al giorno successivo alla riunione). Le date di pubblicazione dei dati
macro sono lette in tempo reale dal calendario FRED (release/dates), con un fallback
locale se l'API non risponde. Ogni ente emittente ha un colore per leggere le cadenze.
"""
import json
import os
import sys
from datetime import date, datetime, timedelta

import plotly.graph_objects as go
import requests
from dash import Dash, html, dcc, Input, Output, State, ctx, ALL
from dash.exceptions import PreventUpdate

# La cartella superiore ospita navbar.py e settings/ (import condivisi del sito)
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
from navbar import make_navbar, tendina_cell         # noqa: E402
from settings.browser_css import SITE_CSS            # noqa: E402
from calendario import trimestrali as tr            # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Enti emittenti (uffici) → colore per leggere le cadenze
# ─────────────────────────────────────────────────────────────────────────────
_OFFICES = ['Federal Reserve', 'BLS', 'BEA', 'Dept. of Labor']
_OFFICE_COLOR = {
    'Federal Reserve': '#1a3a6b',   # blu istituzionale
    'BLS':             '#c0392b',   # rosso
    'BEA':             '#1e8449',   # verde
    'Dept. of Labor':  '#d97706',   # arancione
}
_OFFICE_FULL = {
    'Federal Reserve': 'Federal Reserve — FOMC',
    'BLS':             'Bureau of Labor Statistics (BLS)',
    'BEA':             'Bureau of Economic Analysis (BEA)',
    'Dept. of Labor':  'Department of Labor (DoL)',
}
_OFFICE_CADENCE = {
    'Federal Reserve': '8 riunioni all’anno (~ogni 6-7 settimane)',
    'BLS':             'mensile — CPI a metà mese, NFP il 1º venerdì',
    'BEA':             'mensile — a fine mese',
    'Dept. of Labor':  'settimanale — ogni giovedì',
}

# Metadati per tipo di evento
_KIND_META = {
    'FOMC':     dict(office='Federal Reserve', label='Riunione FOMC',                    symbol='star',         cad='~8/anno'),
    'BLACKOUT': dict(office='Federal Reserve', label='Blackout comunicazioni FED',       symbol='line-ew-open', cad='~10 gg per riunione'),
    'CPI':      dict(office='BLS',             label='Inflazione CPI',                    symbol='circle',       cad='mensile'),
    'NFP':      dict(office='BLS',             label='Occupazione — buste paga (NFP)',    symbol='diamond',      cad='mensile (1º ven.)'),
    'PCE':      dict(office='BEA',             label='Inflazione PCE',                    symbol='square',       cad='mensile'),
    'CLAIMS':   dict(office='Dept. of Labor',  label='Sussidi di disoccupazione',         symbol='triangle-up',  cad='settimanale (giov.)'),
}

_GIORNI = ['lun', 'mar', 'mer', 'gio', 'ven', 'sab', 'dom']


# ─────────────────────────────────────────────────────────────────────────────
# Riunioni FOMC ufficiali (federalreserve.gov) — (anno, mese, g_inizio, g_fine, SEP)
# SEP = riunione con proiezioni economiche (Summary of Economic Projections) + grafico a punti
# ─────────────────────────────────────────────────────────────────────────────
_FOMC = [
    (2026,  1, 27, 28, False),
    (2026,  3, 17, 18, True),
    (2026,  4, 28, 29, False),
    (2026,  6, 16, 17, True),
    (2026,  7, 28, 29, False),
    (2026,  9, 15, 16, True),
    (2026, 10, 27, 28, False),
    (2026, 12,  8,  9, True),
    (2027,  1, 26, 27, False),
    (2027,  3, 16, 17, True),
    (2027,  4, 27, 28, False),
    (2027,  6,  8,  9, True),
    (2027,  7, 27, 28, False),
    (2027,  9, 14, 15, True),
    (2027, 10, 26, 27, False),
    (2027, 12,  7,  8, True),
]


def _blackout(mtg_start, mtg_end):
    """Regola FED: dal 2º sabato precedente l'inizio della riunione fino al giorno
    successivo alla fine della riunione."""
    d = mtg_start
    while d.weekday() != 5:                # 5 = sabato → 1º sabato ≤ inizio
        d -= timedelta(days=1)
    if d == mtg_start:                     # se inizia di sabato, prendi quello prima
        d -= timedelta(days=7)
    start = d - timedelta(days=7)          # secondo sabato precedente
    end = mtg_end + timedelta(days=1)      # giorno dopo la riunione
    return start, end


# ─────────────────────────────────────────────────────────────────────────────
# Date di pubblicazione dati macro — FRED release/dates (in tempo reale) + fallback
# ─────────────────────────────────────────────────────────────────────────────
_FRED_KEY = os.environ.get('FRED_API_KEY', '65061ed1fa4c47d53b1d644e1cd858d3')
_FRED_URL = 'https://api.stlouisfed.org/fred/release/dates'
# release_id FRED → tipo evento
_FRED_RELEASES = {10: 'CPI', 50: 'NFP', 54: 'PCE', 180: 'CLAIMS'}
_RANGE_START = '2026-01-01'
_RANGE_END   = '2027-12-31'

# Fallback (usato solo se FRED non risponde): date 2026 note + giovedì per i sussidi
_FALLBACK = {
    'CPI': ['2026-01-13', '2026-02-11', '2026-03-11', '2026-04-10', '2026-05-13',
            '2026-06-10', '2026-07-14', '2026-08-12', '2026-09-11', '2026-10-14',
            '2026-11-10', '2026-12-10'],
    'NFP': ['2026-01-09', '2026-02-06', '2026-03-06', '2026-04-03', '2026-05-08',
            '2026-06-05', '2026-07-02', '2026-08-07', '2026-09-04', '2026-10-02',
            '2026-11-06', '2026-12-04'],
    'PCE': ['2026-01-30', '2026-02-27', '2026-03-27', '2026-04-30', '2026-05-29',
            '2026-06-26', '2026-07-30', '2026-08-26', '2026-09-30', '2026-10-29',
            '2026-11-25', '2026-12-23'],
}


def _fred_dates(release_id):
    """Date di pubblicazione (anche future) per una release FRED, entro il range."""
    params = dict(release_id=release_id, api_key=_FRED_KEY, file_type='json',
                  include_release_dates_with_no_data='true',
                  realtime_start=_RANGE_START, realtime_end=_RANGE_END,
                  sort_order='asc', limit=1000)
    r = requests.get(_FRED_URL, params=params, timeout=12)
    r.raise_for_status()
    out = []
    for rd in r.json().get('release_dates', []):
        d = rd.get('date', '')
        if _RANGE_START <= d <= _RANGE_END:
            out.append(d)
    return out


def _thursdays():
    """Fallback sussidi: ogni giovedì nel range."""
    d = datetime.strptime(_RANGE_START, '%Y-%m-%d').date()
    end = datetime.strptime(_RANGE_END, '%Y-%m-%d').date()
    while d.weekday() != 3:                 # 3 = giovedì
        d += timedelta(days=1)
    out = []
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=7)
    return out


def _build_events():
    """Costruisce la lista completa di eventi (FOMC + blackout + dati macro)."""
    events = []

    # FOMC + blackout
    for y, m, g1, g2, sep in _FOMC:
        s, e = date(y, m, g1), date(y, m, g2)
        events.append(dict(kind='FOMC', office='Federal Reserve', d=s, end=None,
                           label='Riunione FOMC' + (' + proiezioni (SEP)' if sep else ''),
                           note='Decisione sui tassi' + (' · dot plot e proiezioni' if sep else '')))
        bs, be = _blackout(s, e)
        events.append(dict(kind='BLACKOUT', office='Federal Reserve', d=bs, end=be,
                           label='Blackout comunicazioni FED',
                           note='Silenzio pre-riunione dei membri FOMC'))

    # Dati macro: FRED in tempo reale, fallback locale in caso di errore
    macro = {}                              # kind → lista di date iso
    fred_ok = True
    for rid, kind in _FRED_RELEASES.items():
        try:
            macro[kind] = _fred_dates(rid)
        except Exception as _e:             # noqa: BLE001 — degradazione morbida
            fred_ok = False
            macro[kind] = []
    # riempi con il fallback ogni serie rimasta vuota
    if not macro.get('CLAIMS'):
        macro['CLAIMS'] = _thursdays()
    for kind in ('CPI', 'NFP', 'PCE'):
        if not macro.get(kind):
            macro[kind] = _FALLBACK.get(kind, [])

    for kind, dates in macro.items():
        meta = _KIND_META[kind]
        for iso in dates:
            try:
                d = datetime.strptime(iso, '%Y-%m-%d').date()
            except (TypeError, ValueError):
                continue
            events.append(dict(kind=kind, office=meta['office'], d=d, end=None,
                               label=meta['label'], note=''))

    events.sort(key=lambda ev: ev['d'])
    return events, fred_ok


# Cache in memoria: caricata pigramente alla prima richiesta (evita di bloccare il boot)
_EVENTS = None
_FRED_OK = True


def _load_events():
    global _EVENTS, _FRED_OK
    if _EVENTS is None:
        _EVENTS, _FRED_OK = _build_events()
    return _EVENTS


# ─────────────────────────────────────────────────────────────────────────────
# Finestra temporale
# ─────────────────────────────────────────────────────────────────────────────
_PERIODI = [('3m', 'Prossimi 3 mesi'), ('6m', 'Prossimi 6 mesi'),
            ('2026', 'Anno 2026'), ('2027', 'Anno 2027'), ('all', 'Tutto')]


def _window(period):
    oggi = date.today()
    if period == '3m':
        return oggi, oggi + timedelta(days=92)
    if period == '2026':
        return date(2026, 1, 1), date(2026, 12, 31)
    if period == '2027':
        return date(2027, 1, 1), date(2027, 12, 31)
    if period == 'all':
        return date(2026, 1, 1), date(2027, 12, 31)
    return oggi, oggi + timedelta(days=183)     # default 6 mesi


def _in_window(ev, w0, w1):
    if ev['kind'] == 'BLACKOUT':
        return ev['d'] <= w1 and (ev['end'] or ev['d']) >= w0
    return w0 <= ev['d'] <= w1


def _in_blackout(d, blackouts):
    return any(b0 <= d <= b1 for b0, b1 in blackouts)


def _fmt(d):
    return f'{_GIORNI[d.weekday()]} {d.day:02d}/{d.month:02d}/{d.year}'


# ─────────────────────────────────────────────────────────────────────────────
# Grafico timeline
# ─────────────────────────────────────────────────────────────────────────────
def timeline_fig(offices=None, period='6m'):
    offices = offices or list(_OFFICES)
    w0, w1 = _window(period)
    events = [e for e in _load_events() if e['office'] in offices and _in_window(e, w0, w1)]

    fig = go.Figure()
    # ordine corsie: Federal Reserve in alto (in Plotly la 1ª categoria è in basso)
    lane_order = [o for o in ['Dept. of Labor', 'BEA', 'BLS', 'Federal Reserve'] if o in offices]

    show_fed = 'Federal Reserve' in offices
    blackouts = []
    if show_fed:
        for e in events:
            if e['kind'] == 'BLACKOUT':
                b0 = max(e['d'], w0)
                b1 = min(e['end'] or e['d'], w1)
                blackouts.append((b0, b1))
                fig.add_vrect(x0=b0, x1=b1 + timedelta(days=1), layer='below',
                              fillcolor=_OFFICE_COLOR['Federal Reserve'], opacity=0.07,
                              line_width=0)

    # linea "oggi"
    oggi = date.today()
    if w0 <= oggi <= w1:
        fig.add_vline(x=oggi, line=dict(color='#16a34a', width=1.5, dash='dot'))
        # in basso dentro il grafico: in alto si sovrapponeva alle date FOMC
        fig.add_annotation(x=oggi, y=0, yref='paper', yanchor='bottom', xanchor='left',
                           xshift=2, text='oggi', showarrow=False,
                           bgcolor='rgba(255,255,255,0.8)',
                           font=dict(size=10, color='#16a34a'))

    # un trace per tipo di evento (legenda per tipo, colore per ufficio, simbolo per tipo)
    span = (w1 - w0).days
    for kind, meta in _KIND_META.items():
        if kind == 'BLACKOUT' or meta['office'] not in offices:
            continue
        pts = [e for e in events if e['kind'] == kind]
        if not pts:
            continue
        xs = [e['d'] for e in pts]
        ys = [meta['office']] * len(pts)
        cds = [[_fmt(e['d']), e['label'], _OFFICE_FULL[e['office']],
                '⚠︎ in blackout' if _in_blackout(e['d'], blackouts) else ''] for e in pts]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers', name=meta['label'],
            marker=dict(symbol=meta['symbol'], size=12 if kind == 'FOMC' else 9,
                        color=_OFFICE_COLOR[meta['office']],
                        line=dict(width=1, color='white')),
            customdata=cds,
            hovertemplate='<b>%{customdata[1]}</b><br>%{customdata[0]}'
                          '<br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>',
        ))

    # linee verticali sulle riunioni FOMC (con data se la finestra non è troppo ampia)
    if show_fed:
        for e in events:
            if e['kind'] != 'FOMC':
                continue
            fig.add_vline(x=e['d'], line=dict(color=_OFFICE_COLOR['Federal Reserve'],
                                              width=1, dash='dash'))
            if span <= 210:
                fig.add_annotation(x=e['d'], y=1.10, yref='paper',
                                   text=f"FOMC<br>{e['d'].day:02d}/{e['d'].month:02d}",
                                   showarrow=False, align='center',
                                   font=dict(size=9, color=_OFFICE_COLOR['Federal Reserve']))

    fig.update_layout(
        height=380, paper_bgcolor='white', plot_bgcolor='#f8f9fb',
        margin=dict(l=10, r=20, t=46, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.14, xanchor='left', x=0,
                    font=dict(size=10)),
        hoverlabel=dict(bgcolor='white', font_size=12),
        xaxis=dict(range=[w0 - timedelta(days=2), w1 + timedelta(days=2)],
                   showgrid=True, gridcolor='#eef1f5', tickformat='%d %b\n%Y',
                   tickfont=dict(size=10)),
        yaxis=dict(categoryorder='array', categoryarray=lane_order,
                   showgrid=False, tickfont=dict(size=11, color='#333')),
    )
    if not events:
        fig.add_annotation(text='Nessun evento con i filtri scelti', showarrow=False,
                           font=dict(size=13, color='#888'))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tabella cronologica riga per riga
# ─────────────────────────────────────────────────────────────────────────────
def build_table(offices=None, period='6m'):
    offices = offices or list(_OFFICES)
    w0, w1 = _window(period)
    events = [e for e in _load_events() if e['office'] in offices and _in_window(e, w0, w1)]
    blackouts = [(e['d'], e['end'] or e['d']) for e in events if e['kind'] == 'BLACKOUT']

    header = html.Tr([
        html.Th('Data', style={'textAlign': 'left', 'padding': '7px 10px', 'width': '190px'}),
        html.Th('Ente emittente', style={'textAlign': 'left', 'padding': '7px 10px', 'width': '230px'}),
        html.Th('Evento', style={'textAlign': 'left', 'padding': '7px 10px'}),
        html.Th('Cadenza', style={'textAlign': 'left', 'padding': '7px 10px', 'width': '150px'}),
    ], style={'background': '#1a3a6b', 'color': 'white', 'fontSize': '11px',
              'textTransform': 'uppercase', 'letterSpacing': '0.04em'})

    rows = []
    for i, e in enumerate(events):
        col = _OFFICE_COLOR[e['office']]
        meta = _KIND_META[e['kind']]
        if e['kind'] == 'BLACKOUT':
            d_txt = f"{e['d'].day:02d}/{e['d'].month:02d} → {_fmt(e['end'])}"
        else:
            d_txt = _fmt(e['d'])
        in_bo = e['kind'] not in ('FOMC', 'BLACKOUT') and _in_blackout(e['d'], blackouts)
        bg = '#fbfcfe' if i % 2 else 'white'
        if e['kind'] == 'FOMC':
            bg = '#eef3fb'
        elif e['kind'] == 'BLACKOUT':
            bg = '#f4f6fa'
        rows.append(html.Tr([
            html.Td(d_txt, style={'padding': '6px 10px', 'fontSize': '12px',
                                  'fontWeight': '700' if e['kind'] == 'FOMC' else '400',
                                  'color': '#222', 'whiteSpace': 'nowrap'}),
            html.Td(html.Span(_OFFICE_FULL[e['office']], style={
                'background': col, 'color': 'white', 'padding': '2px 9px',
                'borderRadius': '10px', 'fontSize': '10.5px', 'fontWeight': '600',
                'whiteSpace': 'nowrap'}), style={'padding': '6px 10px'}),
            html.Td([
                html.Span(e['label'], style={'fontSize': '12px', 'color': '#222'}),
                html.Span('  ⚠︎ ricade nel blackout', style={
                    'fontSize': '10.5px', 'color': '#c0392b', 'fontWeight': '600'}) if in_bo else '',
            ], style={'padding': '6px 10px'}),
            html.Td(meta['cad'], style={'padding': '6px 10px', 'fontSize': '11px',
                                        'color': '#777'}),
        ], style={'background': bg, 'borderBottom': '1px solid #eef1f5'}))

    if not rows:
        rows = [html.Tr(html.Td('Nessun evento con i filtri scelti',
                                colSpan=4, style={'padding': '18px', 'textAlign': 'center',
                                                  'color': '#888', 'fontSize': '13px'}))]

    return html.Table([html.Thead(header), html.Tbody(rows)], style={
        'width': '100%', 'borderCollapse': 'collapse', 'fontFamily': 'Inter, sans-serif'})


# ─────────────────────────────────────────────────────────────────────────────
# Serie storiche FRED (grafici sotto al calendario)
# ─────────────────────────────────────────────────────────────────────────────
_FRED_OBS_URL = 'https://api.stlouisfed.org/fred/series/observations'
_Y = date.today().year
_START_3Y = f'{_Y - 3}-01-01'
_START_6Y = f'{_Y - 6}-01-01'
# series_id → (unità FRED, data inizio osservazioni)
_SERIES_SPECS = {
    'ICSA':     ('lin', _START_3Y),   # richieste iniziali sussidi (settimanali)
    'IC4WSA':   ('lin', _START_3Y),   # media mobile 4 settimane
    'CPIAUCNS': ('pc1', _START_6Y),   # CPI variazione anno su anno (%)
    'PCEPI':    ('pc1', _START_6Y),   # PCE variazione anno su anno (%)
    'PAYEMS':   ('chg', _START_3Y),   # occupati non agricoli — variazione mensile (migliaia)
}
_SERIES = None


def _fred_series(series_id, units, start):
    params = dict(series_id=series_id, api_key=_FRED_KEY, file_type='json',
                  observation_start=start, units=units, sort_order='asc')
    r = requests.get(_FRED_OBS_URL, params=params, timeout=12)
    r.raise_for_status()
    ds, vs = [], []
    for o in r.json().get('observations', []):
        v = o.get('value', '.')
        if v in ('.', '', None):
            continue
        try:
            vs.append(float(v))
            ds.append(o['date'])
        except (TypeError, ValueError):
            continue
    return ds, vs


def _load_series():
    """Scarica (una sola volta) le serie storiche; fallback a vuoto in caso di errore."""
    global _SERIES
    if _SERIES is None:
        out = {}
        for sid, (units, start) in _SERIES_SPECS.items():
            try:
                out[sid] = _fred_series(sid, units, start)
            except Exception:                # noqa: BLE001 — degradazione morbida
                out[sid] = ([], [])
        _SERIES = out
    return _SERIES


def _series_layout(fig, ytitle, pct=False, tickfmt=None):
    fig.update_layout(
        height=280, paper_bgcolor='white', plot_bgcolor='#f8f9fb',
        margin=dict(l=12, r=16, t=10, b=28),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0,
                    font=dict(size=10)),
        hovermode='x unified', hoverlabel=dict(bgcolor='white', font_size=12),
        xaxis=dict(showgrid=True, gridcolor='#eef1f5', tickfont=dict(size=10)),
        yaxis=dict(title=dict(text=ytitle, font=dict(size=11, color='#555')),
                   showgrid=True, gridcolor='#eef1f5', tickfont=dict(size=10),
                   ticksuffix='%' if pct else '', tickformat=tickfmt or ''),
    )
    return fig


def _no_data(fig):
    fig.add_annotation(text='Dati FRED non disponibili', showarrow=False,
                       font=dict(size=13, color='#888'))
    return fig


def fig_sussidi():
    s = _load_series()
    d1, v1 = s.get('ICSA', ([], []))
    d2, v2 = s.get('IC4WSA', ([], []))
    fig = go.Figure()
    if not d1 and not d2:
        return _series_layout(_no_data(fig), 'richieste')
    fig.add_trace(go.Scatter(x=d1, y=v1, name='Richieste iniziali (settimanali)',
                             mode='lines', line=dict(color='#f0b27a', width=1),
                             hovertemplate='%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=d2, y=v2, name='Media mobile 4 settimane',
                             mode='lines', line=dict(color='#d97706', width=2.4),
                             hovertemplate='%{y:,.0f}<extra></extra>'))
    return _series_layout(fig, 'richieste settimanali', tickfmt=',.0f')


def fig_inflazione():
    s = _load_series()
    dc, vc = s.get('CPIAUCNS', ([], []))
    dp, vp = s.get('PCEPI', ([], []))
    fig = go.Figure()
    if not dc and not dp:
        return _series_layout(_no_data(fig), 'a/a %', pct=True)
    fig.add_trace(go.Scatter(x=dc, y=vc, name='CPI (BLS) a/a',
                             mode='lines', line=dict(color='#c0392b', width=2.2),
                             hovertemplate='%{y:.2f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=dp, y=vp, name='PCE (BEA) a/a',
                             mode='lines', line=dict(color='#1e8449', width=2.2),
                             hovertemplate='%{y:.2f}%<extra></extra>'))
    _series_layout(fig, 'variazione anno su anno', pct=True, tickfmt='.1f')
    fig.add_hline(y=2, line=dict(color='#888', width=1, dash='dot'),
                  annotation_text='obiettivo FED 2%', annotation_position='top left',
                  annotation_font=dict(size=10, color='#888'))
    return fig


def fig_occupazione():
    s = _load_series()
    d, v = s.get('PAYEMS', ([], []))
    fig = go.Figure()
    if not d:
        return _series_layout(_no_data(fig), 'migliaia')
    colors = ['#1a3a6b' if x >= 0 else '#c0392b' for x in v]
    fig.add_trace(go.Bar(x=d, y=v, marker_color=colors, name='Variazione occupati',
                         hovertemplate='%{y:+,.0f} mila<extra></extra>'))
    fig.add_hline(y=0, line=dict(color='#ccc', width=1))
    return _series_layout(fig, 'migliaia di posti (var. mensile)', tickfmt=',.0f')


def _grafici_storici():
    """Sezione con i 3 grafici delle serie storiche, sotto al calendario."""
    def blocco(titolo, sottotitolo, graph_id, figura):
        return html.Div([
            html.H3(titolo, style={'color': '#1a3a6b', 'fontSize': '14px',
                                   'margin': '0 0 2px'}),
            html.Div(sottotitolo, style={'color': '#888', 'fontSize': '11px',
                                         'marginBottom': '4px'}),
            dcc.Graph(id=graph_id, figure=figura, config={'displayModeBar': False},
                      responsive=True, style={'height': f"{figura.layout.height or 320}px"}),
        ], className='cal-graf', style={'marginBottom': '12px'})

    return html.Div([
        html.Div('Serie effettive dai dati ufficiali (fonte FRED — St. Louis Fed).',
                 className='cal-nota', style={'marginBottom': '10px'}),
        blocco('Sussidi di disoccupazione', 'Richieste iniziali settimanali (Dept. of '
               'Labor) e media mobile a 4 settimane.', 'cal-fig-sussidi', fig_sussidi()),
        blocco('Inflazione — CPI e PCE (anno su anno)', 'CPI (BLS) e PCE (BEA), variazione '
               'percentuale sui 12 mesi; linea tratteggiata = obiettivo FED del 2%.',
               'cal-fig-inflazione', fig_inflazione()),
        blocco('Occupazione — buste paga non agricole (NFP)', 'Variazione mensile degli '
               'occupati non agricoli (BLS), in migliaia di posti.',
               'cal-fig-occupazione', fig_occupazione()),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
_EXT = [
    'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=_EXT,
           requests_pathname_prefix='/calendario/',
           routes_pathname_prefix='/calendario/')
app.title = 'Calendario economico — Andrea Cappelletti'
server = app.server

app.index_string = '''<!DOCTYPE html><html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>''' + SITE_CSS + '''
  .cal-box { border: 1px solid #e2e8f0; border-radius: 6px; background: #f8fafc;
             padding: 8px 10px; margin: 0 0 10px; }
  .cal-box > .nb-espandi-btn { font-size: 13px; font-weight: 700; color: #1a3a6b; margin-bottom: 6px; }
  .cal-nota { color: #666; font-size: 12px; }
  .cal-graf { background: #fff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 6px 4px; }
  .cal-enti { display: flex; flex-wrap: wrap; gap: 8px 22px; }
  .cal-ente { display: inline-flex; align-items: flex-start; gap: 7px; }
  .cal-ente-col { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
                  margin-top: 2px; flex-shrink: 0; }
  .cal-ente-nome { display: block; font-size: 12px; color: #1a3a6b; }
  .cal-ente-cad { display: block; font-size: 10.5px; color: #777; }
  @media (max-width: 900px) {
    .cal-pagina { padding-left: 8px !important; padding-right: 8px !important; }
    .cal-box { padding: 8px; }
    .cal-enti { flex-direction: column; gap: 8px; }
  }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''


def _sezione(titolo, corpo, aperto=True):
    """Riquadro a tutta larghezza; sul cellulare sezione espandibile (aperto=False:
    chiusa all'avvio sul cellulare; sul computer è sempre aperta)."""
    return html.Div(className='nb-espandi cal-box' + (' nb-aperto' if aperto else ''), children=[
        html.Div([titolo, html.I(className='fa-solid fa-chevron-down nb-espandi-ico')],
                 className='nb-espandi-btn'),
        html.Div(corpo, className='nb-espandi-corpo'),
    ])


def _ente_etichetta(off):
    """Voce della scelta degli enti: colore, nome e cadenza (fa anche da legenda)."""
    return html.Span([
        html.Span(className='cal-ente-col', style={'background': _OFFICE_COLOR[off]}),
        html.Span([html.B(_OFFICE_FULL[off], className='cal-ente-nome'),
                   html.Span(_OFFICE_CADENCE[off], className='cal-ente-cad')]),
    ], className='cal-ente')


# Sotto-tab del calendario, nell'ordine della pagina (anche nella tendina del cellulare)
_CAL_SOTTOTAB = [('🏛 Calendario economico — riunioni FED', 'macro'),
                 ('📊 Trimestrali delle società', 'trimestrali')]


def serve_layout():
    return html.Div([
        make_navbar('Calendario'),
        html.Div([
            dcc.Tabs(id='cal-sezione', value='macro', parent_className='nb-nascondi-cell',
                     children=[dcc.Tab(label=lab, value=v) for lab, v in _CAL_SOTTOTAB],
                     style={'marginBottom': '14px'}),
            tendina_cell('cal-tendina', _CAL_SOTTOTAB),
            html.Div(id='cal-sez-macro', children=_sezione_macro()),
            html.Div(id='cal-sez-trimestrali', children=_sezione_trimestrali(),
                     style={'display': 'none'}),
        ], className='cal-pagina',
           style={'padding': '112px 5% 32px', 'fontFamily': 'Inter, sans-serif'}),
    ])


def _sezione_macro():
    return html.Div([
        html.Div('Riunioni FOMC e blackout della Federal Reserve, sussidi di disoccupazione, '
                 'inflazione (CPI e PCE) e occupazione (NFP). Date dei dati macro dal calendario '
                 'FRED; date FOMC ufficiali. Ogni ente emittente ha un colore per leggerne la cadenza.',
                 className='cal-nota', style={'marginBottom': '10px'}),

        _sezione('Periodi', dcc.RadioItems(
            id='cal-periodo',
            options=[{'label': f' {lbl}', 'value': v} for v, lbl in _PERIODI],
            value='6m', inline=True,
            inputStyle={'marginRight': '3px'},
            labelStyle={'marginRight': '12px', 'fontSize': '12px', 'cursor': 'pointer'},
        ), aperto=False),

        _sezione('Ente Emittente', dcc.Checklist(
            id='cal-enti',
            options=[{'label': _ente_etichetta(o), 'value': o} for o in _OFFICES],
            value=list(_OFFICES), className='cal-enti',
            inputStyle={'marginRight': '6px', 'marginTop': '2px'},
            labelStyle={'display': 'flex', 'alignItems': 'flex-start', 'cursor': 'pointer'},
        ), aperto=False),

        _sezione('Grafici', html.Div(
            dcc.Graph(id='cal-timeline', figure=timeline_fig(), responsive=True,
                      config={'displayModeBar': False}, style={'height': '380px'}),
            className='cal-graf')),

        _sezione('Elenco Cronologico', [
            html.Div(id='cal-table', children=build_table(),
                     style={'maxHeight': '520px', 'overflowY': 'auto', 'background': '#fff',
                            'border': '1px solid #e8edf5', 'borderRadius': '10px'}),
            html.Div('Le date future dei dati macro sono programmate e possono variare; '
                     'il blackout FED va dal 2º sabato precedente la riunione al giorno '
                     'successivo. Fonte dati: FRED (St. Louis Fed) e Federal Reserve Board.',
                     className='cal-nota', style={'marginTop': '8px'}),
        ]),

        _sezione('Grafici Storici Dati', _grafici_storici()),
    ])


app.layout = serve_layout


@app.callback(
    Output('cal-timeline', 'figure'),
    Output('cal-table', 'children'),
    Input('cal-periodo', 'value'),
    Input('cal-enti', 'value'),
    prevent_initial_call=True,
)
def _update(period, enti):
    enti = enti or list(_OFFICES)
    return timeline_fig(enti, period or '6m'), build_table(enti, period or '6m')


# Le sezioni si nascondono invece di smontarsi: tornando indietro filtri e
# finestra restano dove li si era lasciati.
@app.callback(
    Output('cal-sez-macro', 'style'),
    Output('cal-sez-trimestrali', 'style'),
    Input('cal-sezione', 'value'),
)
def _cambia_sezione(sezione):
    nascosto = {'display': 'none'}
    return (nascosto, {}) if sezione == 'trimestrali' else ({}, nascosto)


# Sul cellulare i sotto-tab sono la tendina «Tabs Strumenti» (come Portafoglio e Analisi Tattica)
@app.callback(
    Output('cal-sezione', 'value'),
    Input({'type': 'cal-tendina-voce', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def _cal_tendina_scegli(clicks):
    tid = ctx.triggered_id
    if not tid or not any(clicks):
        raise PreventUpdate
    return tid['index']


app.clientside_callback(
    """function (val) {
        var voci = %s, lab = voci[0][0];
        voci.forEach(function (v) { if (v[1] === val) lab = v[0]; });
        return [lab, voci.map(function (v) {
            return v[1] === val ? 'nb-tendina-voce nb-attiva' : 'nb-tendina-voce'; })];
    }""" % json.dumps(_CAL_SOTTOTAB),
    Output('cal-tendina-att', 'children'),
    Output({'type': 'cal-tendina-voce', 'index': ALL}, 'className'),
    Input('cal-sezione', 'value'),
)


# ─────────────────────────────────────────────────────────────────────────────
# Sotto-tab «Trimestrali delle società»
# ─────────────────────────────────────────────────────────────────────────────
_FINESTRE = [('sett', 'Settimana'), ('2sett', 'Due settimane'), ('mese', 'Mese')]
_FONTE = {   # fonte → (etichetta, colore)
    'confermata': ('Confermata', '#1a3a6b'),
    'yahoo':      ('Prevista (Yahoo)', '#2e86c1'),
    'stimata':    ('Stimata +13 sett.', '#aab7c4'),
    'pubblicata': ('Già pubblicata', '#1e8449'),
}
_MERCATO_COL = {'USA': '#1a3a6b', 'Italia': '#1e8449', 'Europa': '#d97706'}
_MESI = ['gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno', 'luglio',
         'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre']


def _finestra_trim(tipo, scarto):
    """(inizio, fine, etichetta) della finestra `scarto` passi avanti/indietro da oggi."""
    oggi = date.today()
    if tipo == 'mese':
        m = oggi.month - 1 + scarto
        w0 = date(oggi.year + m // 12, m % 12 + 1, 1)
        w1 = (date(w0.year + (w0.month == 12), w0.month % 12 + 1, 1) - timedelta(days=1))
        return w0, w1, f'{_MESI[w0.month - 1]} {w0.year}'
    giorni = 14 if tipo == '2sett' else 7
    w0 = oggi - timedelta(days=oggi.weekday()) + timedelta(days=giorni * scarto)
    w1 = w0 + timedelta(days=giorni - 1)
    return w0, w1, f'{_fmt(w0)} → {_fmt(w1)}'


def _sezione_trimestrali():
    lbl = {'fontSize': '11px', 'fontWeight': '700', 'color': '#1a3a6b', 'marginRight': '8px'}
    radio = dict(inline=True, inputStyle={'marginRight': '3px'},
                 labelStyle={'marginRight': '12px', 'fontSize': '11px', 'cursor': 'pointer'})
    bottone = {'border': '1px solid #1a3a6b', 'background': 'white', 'color': '#1a3a6b',
               'borderRadius': '6px', 'padding': '3px 10px', 'fontSize': '12px',
               'cursor': 'pointer', 'marginRight': '4px'}
    return html.Div([
        html.H2('Trimestrali in arrivo', style={'color': '#1a3a6b', 'fontSize': '20px',
                                                'margin': '0 0 4px'}),
        html.Div('Le date di pubblicazione dei conti trimestrali, per finestra di una '
                 'settimana, due settimane o un mese. Le società principali di USA, Italia '
                 'ed Europa hanno sempre la prossima data (annunciata o stimata); con '
                 '«Tutte le società USA» si aggiungono le date già annunciate da tutto il '
                 'mercato americano.',
                 style={'color': '#666', 'fontSize': '12px', 'marginBottom': '14px'}),
        html.Div([
            html.Label('Finestra:', style=lbl),
            dcc.RadioItems(id='tr-finestra', value='sett',
                           options=[{'label': f' {t}', 'value': v} for v, t in _FINESTRE],
                           **radio),
            html.Button('◀', id='tr-prec', n_clicks=0, style=bottone),
            html.Button('Oggi', id='tr-oggi', n_clicks=0, style=bottone),
            html.Button('▶', id='tr-succ', n_clicks=0, style=bottone),
            html.B(id='tr-etichetta', style={'fontSize': '13px', 'color': '#1a3a6b',
                                             'marginLeft': '10px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap',
                  'gap': '4px', 'marginBottom': '8px'}),
        html.Div([
            html.Label('Società:', style=lbl),
            dcc.RadioItems(id='tr-universo', value='principali', options=[
                {'label': f' Principali USA, Italia, Europa ({len(tr._UNIVERSO)})',
                 'value': 'principali'},
                {'label': ' Tutte le società USA (date annunciate)', 'value': 'tutte'},
            ], **radio),
            dcc.Checklist(id='tr-stimate', value=['si'],
                          options=[{'label': ' mostra le date stimate', 'value': 'si'}],
                          **radio),
            dcc.Input(id='tr-cerca', type='text', debounce=True, value='',
                      placeholder='Cerca ticker o società…',
                      style={'fontSize': '12px', 'padding': '4px 8px', 'width': '200px',
                             'border': '1px solid #ccd', 'borderRadius': '6px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap',
                  'gap': '6px', 'marginBottom': '12px', 'paddingTop': '6px',
                  'borderTop': '1px solid #eee'}),
        dcc.Store(id='tr-scarto', data=0),
        dcc.Interval(id='tr-attesa', interval=15000, disabled=True),
        # responsive=True: la sezione nasce nascosta e senza questo Plotly
        # fissa la larghezza misurata in quel momento.
        dcc.Graph(id='tr-grafico', config={'displayModeBar': False}, responsive=True,
                  style={'height': '230px'}),
        html.Div(id='tr-tabella', style={'maxHeight': '620px', 'overflowY': 'auto',
                                         'border': '1px solid #e8edf5',
                                         'borderRadius': '10px', 'marginTop': '8px'}),
        html.Div(id='tr-nota', style={'color': '#999', 'fontSize': '10.5px',
                                      'marginTop': '10px'}),
    ])


@app.callback(
    Output('tr-scarto', 'data'),
    Input('tr-prec', 'n_clicks'),
    Input('tr-oggi', 'n_clicks'),
    Input('tr-succ', 'n_clicks'),
    Input('tr-finestra', 'value'),
    State('tr-scarto', 'data'),
    prevent_initial_call=True,
)
def _sposta_finestra(_p, _o, _s, _f, scarto):
    tid = ctx.triggered_id
    if tid == 'tr-prec':
        return (scarto or 0) - 1
    if tid == 'tr-succ':
        return (scarto or 0) + 1
    return 0                                # Oggi, o cambio di finestra


_GIORNI_BREVI = ['lun', 'mar', 'mer', 'gio', 'ven', 'sab', 'dom']


def _grafico_trim(ev, w0, w1):
    fig = go.Figure()
    giorni = [w0 + timedelta(days=i) for i in range((w1 - w0).days + 1)]
    for fonte, (nome, colore) in _FONTE.items():
        conta = {g: 0 for g in giorni}
        for e in ev:
            if e['fonte'] == fonte:
                conta[e['d']] += 1
        if any(conta.values()):
            fig.add_trace(go.Bar(x=giorni, y=[conta[g] for g in giorni], name=nome,
                                 marker_color=colore,
                                 hovertemplate='%{x|%d/%m}: %{y}<extra>' + nome + '</extra>'))
    oggi = date.today()
    if w0 <= oggi <= w1:
        # Traccia e non add_vline: la shape, ridisegnata mentre la sezione è
        # nascosta (larghezza 0), finisce in un path "MNaN" e sporca la console.
        alto = max([sum(1 for e in ev if e['d'] == g) for g in giorni] + [1])
        fig.add_trace(go.Scatter(x=[oggi, oggi], y=[0, alto * 1.05], mode='lines',
                                 line=dict(color='#16a34a', width=1.5, dash='dot'),
                                 showlegend=False, hoverinfo='skip'))
    # Etichette in italiano (Plotly scriverebbe Mon/Tue); sul mese una ogni 7 giorni.
    tick = giorni if len(giorni) <= 14 else giorni[::7]
    fig.update_layout(
        barmode='stack', height=230, paper_bgcolor='white', plot_bgcolor='#f8f9fb',
        margin=dict(l=10, r=16, t=30, b=30), bargap=0.25,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                    font=dict(size=10)),
        xaxis=dict(range=[w0 - timedelta(hours=14), w1 + timedelta(hours=14)],
                   tickvals=tick, ticktext=[f"{_GIORNI_BREVI[g.weekday()]} {g:%d/%m}"
                                            for g in tick],
                   tickfont=dict(size=10), showgrid=False),
        yaxis=dict(title=dict(text='società', font=dict(size=11, color='#555')),
                   showgrid=True, gridcolor='#eef1f5', tickfont=dict(size=10)),
    )
    if not ev:
        fig.add_annotation(text='Nessuna trimestrale in questa finestra', showarrow=False,
                           font=dict(size=13, color='#888'))
    return fig


def _tabella_trim(ev):
    th = {'textAlign': 'left', 'padding': '7px 10px'}
    testa = html.Tr([html.Th(t, style=th) for t in
                     ('Momento', 'Ticker', 'Società', 'Mercato', 'Trimestre', 'EPS stimato',
                      'Fonte')],
                    style={'background': '#1a3a6b', 'color': 'white', 'fontSize': '11px',
                           'textTransform': 'uppercase', 'letterSpacing': '0.04em',
                           'position': 'sticky', 'top': '0'})
    td = {'padding': '5px 10px', 'fontSize': '12px', 'color': '#222'}
    righe, giorno = [], None
    for e in ev[:tr._MAX_RIGHE]:
        if e['d'] != giorno:
            giorno = e['d']
            n = sum(1 for x in ev if x['d'] == giorno)
            righe.append(html.Tr(html.Td(f"{_fmt(giorno)} — {n} società",
                                         colSpan=7, style={
                'padding': '6px 10px', 'fontSize': '12px', 'fontWeight': '700',
                'color': '#1a3a6b', 'background': '#eef3fb'})))
        nome_f, col_f = _FONTE[e['fonte']]
        momento = {'pre-market': 'prima dell’apertura',
                   'post-market': 'dopo la chiusura'}.get(e['momento'], '')
        eps = '' if e['eps'] is None else f"{e['eps']:.2f} {e['valuta']}".strip()
        stimata = e['fonte'] == 'stimata'
        righe.append(html.Tr([
            html.Td(momento, style={**td, 'color': '#777', 'fontSize': '11px'}),
            html.Td(e['ticker'], style={**td, 'fontWeight': '700'}),
            html.Td(e['nome'], style=td),
            html.Td(html.Span(e['mercato'], style={
                'background': _MERCATO_COL[e['mercato']], 'color': 'white',
                'padding': '1px 8px', 'borderRadius': '10px', 'fontSize': '10.5px'}),
                style=td),
            html.Td(e['trimestre'] and f"chiuso il {e['trimestre'][8:10]}/"
                    f"{e['trimestre'][5:7]}/{e['trimestre'][:4]}", style={**td, 'color': '#555'}),
            html.Td(eps, style={**td, 'textAlign': 'right'}),
            html.Td(html.Span(nome_f, style={'color': col_f, 'fontWeight': '600',
                                             'fontSize': '11px'}), style=td),
        ], style={'borderBottom': '1px solid #eef1f5',
                  'fontStyle': 'italic' if stimata else 'normal'}))
    if not righe:
        righe = [html.Tr(html.Td('Nessuna trimestrale in questa finestra', colSpan=7,
                                 style={'padding': '18px', 'textAlign': 'center',
                                        'color': '#888', 'fontSize': '13px'}))]
    return html.Table([html.Thead(testa), html.Tbody(righe)], style={
        'width': '100%', 'borderCollapse': 'collapse', 'fontFamily': 'Inter, sans-serif'})


@app.callback(
    Output('tr-grafico', 'figure'),
    Output('tr-tabella', 'children'),
    Output('tr-etichetta', 'children'),
    Output('tr-nota', 'children'),
    Output('tr-attesa', 'disabled'),
    Input('tr-finestra', 'value'),
    Input('tr-scarto', 'data'),
    Input('tr-universo', 'value'),
    Input('tr-stimate', 'value'),
    Input('tr-cerca', 'value'),
    Input('tr-attesa', 'n_intervals'),
)
def _aggiorna_trimestrali(finestra, scarto, universo, stimate, cerca, _n):
    w0, w1, etichetta = _finestra_trim(finestra or 'sett', scarto or 0)
    ev = tr.eventi(w0, w1, universo or 'principali', cerca)
    if 'si' not in (stimate or []):
        ev = [e for e in ev if e['fonte'] != 'stimata']
    dati_ = tr.dati()
    attesa = tr.in_corso()

    nota = []
    if dati_:
        agg = datetime.fromisoformat(dati_['aggiornato'])
        nota.append(f"Dati aggiornati il {agg.strftime('%d/%m/%Y')}; prossimo "
                    f"aggiornamento automatico il "
                    f"{tr.prossimo_aggiornamento(dati_).strftime('%d/%m/%Y')}. ")
        if dati_.get('av_errore'):
            nota.append(f"⚠ Alpha Vantage non ha risposto all'ultimo aggiornamento "
                        f"({dati_['av_errore'][:90]}): restano le date precedenti. ")
    if attesa:
        nota.append('⏳ Aggiornamento del calendario in corso (circa un minuto): '
                    'la pagina si completa da sola. ')
    if len(ev) > tr._MAX_RIGHE:
        nota.append(f"Mostrate le prime {tr._MAX_RIGHE} righe su {len(ev)}: restringi "
                    "la finestra o usa la ricerca. ")
    nota.append('«Confermata» = data annunciata dalla società (Alpha Vantage); «Prevista» = '
                'prossima data secondo Yahoo; «Stimata» = ultima data nota + 13 settimane, '
                'in corsivo. Le date possono cambiare fino all’annuncio.')
    return (_grafico_trim(ev, w0, w1), _tabella_trim(ev), etichetta, ''.join(nota),
            not attesa)


# Job trimestrale: parte all'import, come lo scheduler di portafoglio
# (gunicorn ha un solo worker, quindi un solo job).
_SCHEDULER_TRIM = tr.avvia_scheduler()


if __name__ == '__main__':
    app.run(debug=True, port=8066)
