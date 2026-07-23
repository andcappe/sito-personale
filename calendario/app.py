"""Calendario economico — i principali dati macro USA che precedono le riunioni FED.

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
import os
import sys
from datetime import date, datetime, timedelta

import plotly.graph_objects as go
import requests
from dash import Dash, html, dcc, Input, Output

# La cartella superiore ospita navbar.py e settings/ (import condivisi del sito)
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
from navbar import make_navbar                       # noqa: E402
from settings.browser_css import SITE_CSS            # noqa: E402

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
        fig.add_annotation(x=oggi, y=1.03, yref='paper', text='oggi', showarrow=False,
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
            dcc.Graph(id=graph_id, figure=figura, config={'displayModeBar': False}),
        ], style={'marginBottom': '18px'})

    return html.Div([
        html.H2('Andamento storico dei dati', style={'color': '#1a3a6b',
                'fontSize': '17px', 'margin': '26px 0 4px'}),
        html.Div('Serie effettive dai dati ufficiali (fonte FRED — St. Louis Fed).',
                 style={'color': '#666', 'fontSize': '12px', 'marginBottom': '14px'}),
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
<style>''' + SITE_CSS + '''</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''


def _legenda_enti():
    """Riquadro colori/cadenze per ente emittente."""
    items = []
    for off in _OFFICES:
        items.append(html.Div([
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px',
                             'borderRadius': '3px', 'background': _OFFICE_COLOR[off],
                             'marginRight': '8px', 'flexShrink': '0'}),
            html.Div([
                html.Span(_OFFICE_FULL[off], style={'fontSize': '11.5px', 'fontWeight': '700',
                                                    'color': '#1a3a6b'}),
                html.Span(_OFFICE_CADENCE[off], style={'fontSize': '10.5px', 'color': '#777',
                                                       'display': 'block'}),
            ]),
        ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '8px'}))
    return html.Div(items, style={'padding': '12px 14px', 'background': '#f8fafd',
                                  'border': '1px solid #e8edf5', 'borderRadius': '10px'})


def serve_layout():
    return html.Div([
        make_navbar('Calendario'),
        html.Div([
            html.H2('Calendario economico — dati chiave prima delle riunioni FED',
                    style={'color': '#1a3a6b', 'fontSize': '20px', 'margin': '0 0 4px'}),
            html.Div('Riunioni FOMC e blackout della Federal Reserve, sussidi di '
                     'disoccupazione, inflazione (CPI e PCE) e occupazione (NFP). '
                     'Date dei dati macro dal calendario FRED; date FOMC ufficiali. '
                     'Ogni ente emittente ha un colore per leggerne la cadenza.',
                     style={'color': '#666', 'fontSize': '12px', 'marginBottom': '14px'}),

            # ── Controlli ──────────────────────────────────────────────────────
            html.Div([
                html.Label('Periodo:', style={'fontSize': '11px', 'fontWeight': '700',
                                              'color': '#1a3a6b', 'marginRight': '8px'}),
                dcc.RadioItems(
                    id='cal-periodo',
                    options=[{'label': f' {lbl}', 'value': v} for v, lbl in _PERIODI],
                    value='6m', inline=True,
                    inputStyle={'marginRight': '3px'},
                    labelStyle={'marginRight': '12px', 'fontSize': '11px', 'cursor': 'pointer'},
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
            html.Div([
                html.Label('Ente emittente:', style={'fontSize': '11px', 'fontWeight': '700',
                                                     'color': '#1a3a6b', 'marginRight': '8px'}),
                dcc.Checklist(
                    id='cal-enti',
                    options=[{'label': f' {_OFFICE_FULL[o]}', 'value': o} for o in _OFFICES],
                    value=list(_OFFICES), inline=True,
                    inputStyle={'marginRight': '3px'},
                    labelStyle={'marginRight': '14px', 'fontSize': '11px', 'cursor': 'pointer'},
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap',
                      'gap': '6px', 'marginBottom': '12px', 'paddingTop': '6px',
                      'borderTop': '1px solid #eee'}),

            # ── Timeline + legenda affiancate ──────────────────────────────────
            html.Div([
                html.Div(dcc.Graph(id='cal-timeline', figure=timeline_fig(),
                                   config={'displayModeBar': False}),
                         style={'flex': '1 1 640px', 'minWidth': '0'}),
                html.Div(_legenda_enti(), style={'flex': '0 0 240px'}),
            ], style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap',
                      'alignItems': 'flex-start', 'marginBottom': '20px'}),

            # ── Tabella cronologica ────────────────────────────────────────────
            html.H3('Elenco cronologico', style={'color': '#1a3a6b', 'fontSize': '15px',
                                                 'margin': '0 0 8px'}),
            html.Div(id='cal-table', children=build_table(),
                     style={'maxHeight': '520px', 'overflowY': 'auto',
                            'border': '1px solid #e8edf5', 'borderRadius': '10px'}),
            html.Div('Le date future dei dati macro sono programmate e possono variare; '
                     'il blackout FED va dal 2º sabato precedente la riunione al giorno '
                     'successivo. Fonte dati: FRED (St. Louis Fed) e Federal Reserve Board.',
                     style={'color': '#999', 'fontSize': '10.5px', 'marginTop': '10px'}),

            # ── Grafici storici delle serie ────────────────────────────────────
            _grafici_storici(),
        ], style={'padding': '112px 5% 32px', 'fontFamily': 'Inter, sans-serif'}),
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


if __name__ == '__main__':
    app.run(debug=True, port=8066)
