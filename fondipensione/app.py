"""Fondi Pensione — confronto rischio/rendimento di comparti Negoziali, PIP (FIP) e
Aperti su dati ufficiali COVIP (elenco rendimenti per comparto, fine 2025).

L'asse X è la CATEGORIA di rischio (Garantito → Obbligazionario → Bilanciato →
Azionario): COVIP pubblica i rendimenti ma non la volatilità, quindi il rischio è
rappresentato dalla categoria del comparto. L'asse Y è il rendimento medio annuo
sull'orizzonte scelto (1/3/5/10/20 anni).
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, html, dcc, Input, Output, State

# La cartella superiore ospita navbar.py e settings/ (import condivisi del sito)
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
from navbar import make_navbar                       # noqa: E402
from settings.browser_css import SITE_CSS            # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Dati COVIP — parsing dei 3 elenchi rendimenti
# ─────────────────────────────────────────────────────────────────────────────
_DATI_DIR = Path(__file__).parent / 'dati'

# Categoria COVIP → rango di rischio crescente + etichetta leggibile
CAT_RANK  = {'GAR': 1, 'OBB PURO': 2, 'OBB MISTO': 3, 'BIL': 4, 'AZN': 5}
CAT_LABEL = {'GAR': 'Garantito', 'OBB PURO': 'Obblig. puro',
             'OBB MISTO': 'Obblig. misto', 'BIL': 'Bilanciato', 'AZN': 'Azionario'}
# etichetta asse X in ordine di rischio
RANK_LABEL = {1: 'Garantito', 2: 'Obblig.\npuro', 3: 'Obblig.\nmisto',
              4: 'Bilanciato', 5: 'Azionario'}

# indici colonna (0-based) per ciascun file COVIP
_SPEC = {
    'Negoziale': dict(file='FPN_Rendimenti_fine2025.xlsx',  soc=None, fondo=2, comp=3, cat=5, rend=[6, 7, 8, 9, 10]),
    'PIP (FIP)': dict(file='PIP_Rendimenti_fine2025_0.xlsx', soc=1,    fondo=2, comp=5, cat=7, rend=[9, 10, 11, 12, 13]),
    'Aperto':    dict(file='FPA_Rendimenti_fine2025.xlsx',   soc=1,    fondo=2, comp=5, cat=6, rend=[8, 9, 10, 11, 12]),
}
REND_COLS = ['rend_1a', 'rend_3a', 'rend_5a', 'rend_10a', 'rend_20a']
ORIZZONTI = [('rend_1a', '1 anno'), ('rend_3a', '3 anni'), ('rend_5a', '5 anni'),
             ('rend_10a', '10 anni'), ('rend_20a', '20 anni')]

# colore per tipologia (il confronto principale)
_TIPO_COLOR = {'Negoziale': '#1a3a6b', 'PIP (FIP)': '#c0392b', 'Aperto': '#1e8449'}
_TIPI = list(_TIPO_COLOR.keys())
# categorie in ordine di rischio crescente (per i filtri)
_CATS = [CAT_LABEL[k] for k in sorted(CAT_RANK, key=CAT_RANK.get)]


def _num(v):
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def parse_covip():
    """Legge i 3 file COVIP e restituisce un DataFrame unico normalizzato."""
    out = []
    for tipo, s in _SPEC.items():
        fpath = _DATI_DIR / s['file']
        if not fpath.exists():
            continue
        df = pd.read_excel(fpath, sheet_name=0, header=None)
        for c in filter(lambda x: x is not None, (s['soc'], s['fondo'])):
            df[c] = df[c].ffill()            # società/fondo solo sulla 1ª riga del gruppo
        for _, row in df.iterrows():
            cat = str(row[s['cat']]).strip() if pd.notna(row[s['cat']]) else ''
            if cat not in CAT_RANK:          # riga-comparto valida solo con categoria nota
                continue
            rec = {
                'tipologia': tipo,
                'societa':  str(row[s['soc']]).strip() if s['soc'] is not None and pd.notna(row[s['soc']]) else '',
                'fondo':    str(row[s['fondo']]).strip() if pd.notna(row[s['fondo']]) else '',
                'comparto': str(row[s['comp']]).strip() if pd.notna(row[s['comp']]) else '',
                'categoria': CAT_LABEL[cat],
                'rischio':   CAT_RANK[cat],
            }
            for name, ci in zip(REND_COLS, s['rend']):
                rec[name] = _num(row[ci])
            out.append(rec)
    return pd.DataFrame(out)


# Cache in memoria: i dati COVIP sono statici (aggiornati a fine 2025)
_DF = parse_covip()


# ─────────────────────────────────────────────────────────────────────────────
# Grafico
# ─────────────────────────────────────────────────────────────────────────────
def scatter_fig(orizzonte='rend_5a', tipologie=None, categorie=None, search=''):
    """Scatter rischio (categoria) × rendimento medio annuo sull'orizzonte scelto,
    filtrato per tipologia (Negoziale/PIP/Aperto) e categoria di comparto.
    `search`: se valorizzato, i comparti il cui nome contiene il testo vengono
    cerchiati in rosso e ingranditi, mentre gli altri restano attenuati."""
    oriz_lbl = dict(ORIZZONTI).get(orizzonte, orizzonte)
    fig = go.Figure()
    if _DF.empty:
        fig.add_annotation(text='Dati COVIP non disponibili', showarrow=False,
                           font=dict(size=14, color='#c0392b'))
        return fig
    tipologie = tipologie or _TIPI
    categorie = categorie or _CATS
    sub = _DF[_DF[orizzonte].notna()
              & _DF['tipologia'].isin(tipologie)
              & _DF['categoria'].isin(categorie)]
    if sub.empty:
        fig.add_annotation(text='Nessun comparto con i filtri scelti', showarrow=False,
                           font=dict(size=13, color='#888'))
        fig.update_layout(height=560, paper_bgcolor='white', plot_bgcolor='#f8f9fb')
        return fig
    search = (search or '').strip().lower()
    n_match = 0
    rng = np.random.default_rng(42)          # jitter deterministico per separare i punti
    for tipo, color in _TIPO_COLOR.items():
        d = sub[sub['tipologia'] == tipo]
        if d.empty:
            continue
        x = d['rischio'].values + rng.uniform(-0.28, 0.28, len(d))
        if search:
            hay = (d['societa'].fillna('') + ' ' + d['fondo'].fillna('') + ' '
                   + d['comparto'].fillna('')).str.lower()
            hit = hay.str.contains(re.escape(search), na=False).values
            n_match += int(hit.sum())
            sizes = np.where(hit, 14, 8)
            lcol  = np.where(hit, '#d00000', 'white')       # anello rosso sui match
            lwid  = np.where(hit, 3.0, 0.5)
            opac  = np.where(hit, 1.0, 0.45)                # attenua i non-match
        else:
            sizes, lcol, lwid, opac = 8, 'white', 0.5, 0.72
        fig.add_trace(go.Scatter(
            x=x, y=d[orizzonte].values, mode='markers', name=tipo,
            marker=dict(size=sizes, color=color, opacity=opac,
                        line=dict(width=lwid, color=lcol)),
            customdata=np.stack([d['fondo'], d['comparto'], d['categoria'],
                                 d['tipologia'], d[orizzonte]], axis=-1),
            hovertemplate=('<b>%{customdata[0]}</b><br>%{customdata[1]}'
                           '<br>%{customdata[3]} · %{customdata[2]}'
                           '<br>Rendimento ' + oriz_lbl + ': %{customdata[4]:.2f}%<extra></extra>'),
        ))
    fig.add_hline(y=0, line_color='#999', line_dash='dot', line_width=1)
    if search:
        fig.add_annotation(
            xref='paper', yref='paper', x=0.99, y=0.99, xanchor='right', yanchor='top',
            showarrow=False,
            text=(f'🔴 {n_match} comparti corrispondono a «{search}»' if n_match
                  else f'Nessun comparto corrisponde a «{search}» (verifica i filtri)'),
            font=dict(size=11, color='#d00000' if n_match else '#888'),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#d00000' if n_match else '#ccc', borderwidth=1, borderpad=4,
        )
    fig.update_layout(
        title=dict(text=f'Rischio × Rendimento medio annuo ({oriz_lbl}) — dati COVIP fine 2025',
                   font=dict(size=14, color='#1a3a6b'), x=0.02),
        xaxis=dict(title='Categoria di rischio →', tickmode='array',
                   tickvals=list(RANK_LABEL.keys()),
                   ticktext=[RANK_LABEL[k].replace('\n', ' ') for k in RANK_LABEL],
                   range=[0.5, 5.5], gridcolor='#eee'),
        yaxis=dict(title=f'Rendimento medio annuo % ({oriz_lbl})', gridcolor='#eee', zeroline=False),
        plot_bgcolor='#f8f9fb', paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=11),
        legend=dict(orientation='h', y=1.08, x=0), height=560,
        margin=dict(t=70, b=50, l=60, r=20), hovermode='closest',
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Assistente AI — interroga i dati COVIP in linguaggio naturale (Google Gemini,
# piano gratuito: nessun credito a pagamento, chiave da aistudio.google.com/apikey)
# ─────────────────────────────────────────────────────────────────────────────
_AI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-flash-latest')
_GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
_AI_SYSTEM = (
    "Sei un assistente esperto di previdenza complementare e fondi pensione italiani, "
    "al servizio di un consulente finanziario. Rispondi in italiano in modo professionale, "
    "chiaro e sintetico. Per qualsiasi dato numerico sui comparti (rendimenti) usa "
    "ESCLUSIVAMENTE i dati COVIP forniti nel messaggio (rendimenti medi annui netti a fine "
    "2025) e non inventare numeri; se un dato non è presente, dillo. Puoi aggiungere "
    "spiegazioni e contesto generale sui fondi pensione. Metti sempre i rendimenti in "
    "relazione alla categoria di rischio del comparto e all'orizzonte temporale."
)
_REND_LBL = '1a | 3a | 5a | 10a | 20a'
# parole da ignorare nella ricerca dei comparti: interrogative comuni + termini di
# dominio troppo generici (compaiono in quasi tutti i nomi e non discriminano)
_AI_STOPWORDS = {
    'come', 'mai', 'cosi', 'così', 'poco', 'molto', 'tanto', 'reso', 'rende', 'rendere',
    'rendimento', 'rendimenti', 'perche', 'perché', 'quanto', 'quale', 'quali', 'questo',
    'questa', 'questi', 'queste', 'con', 'per', 'del', 'della', 'dei', 'delle', 'dal',
    'dalla', 'nel', 'nella', 'tra', 'fra', 'che', 'chi', 'non', 'più', 'meno', 'hanno',
    'sono', 'essere', 'gli', 'una', 'uno', 'sul', 'sulla', 'suoi', 'loro', 'anche',
    'meglio', 'peggio', 'migliore', 'peggiore', 'migliori', 'peggiori', 'confronta',
    'confronto', 'dimmi', 'spiega', 'quali', 'ultimi', 'ultimo', 'anni', 'anno',
    'fondo', 'fondi', 'pensione', 'pensioni', 'pension', 'comparto', 'comparti',
    'previdenza', 'categoria', 'categorie', 'rischio', 'negoziale', 'negoziali',
    'aperto', 'aperti', 'garantito', 'garantiti', 'azionario', 'bilanciato',
}


def _fmt_row(r):
    vals = ' | '.join('n.d.' if pd.isna(r[c]) else f'{r[c]:+.2f}' for c in REND_COLS)
    return f"{r['tipologia']} | {r['societa']} | {r['fondo']} | {r['comparto']} | {r['categoria']} | {vals}"


def _build_ai_context(question):
    """Costruisce un contesto compatto dai dati COVIP: medie per categoria +
    comparti pertinenti alle parole chiave della domanda (o migliori/peggiori a 5a)."""
    parts = [
        f"DATI COVIP — {len(_DF)} comparti di fondi pensione. Valori = rendimento medio annuo netto (%), fine 2025.",
        "Categorie in ordine di rischio crescente: Garantito < Obblig. puro < Obblig. misto < Bilanciato < Azionario.",
        "",
        f"RENDIMENTO MEDIO PER CATEGORIA ({_REND_LBL}):",
    ]
    agg = _DF.groupby('categoria')[REND_COLS].mean()
    for k in sorted(CAT_RANK, key=CAT_RANK.get):
        lab = CAT_LABEL[k]
        if lab in agg.index:
            row = agg.loc[lab]
            vals = ' | '.join('n.d.' if pd.isna(row[c]) else f'{row[c]:+.2f}' for c in REND_COLS)
            parts.append(f"  {lab}: {vals}")

    header = f"tipologia | società | fondo | comparto | categoria | {_REND_LBL}"
    toks = re.findall(r'[a-zA-Zàèéìòùç0-9]{3,}', (question or '').lower())
    toks = [t for t in toks if t not in _AI_STOPWORDS]
    hay = (_DF['societa'].fillna('') + ' ' + _DF['fondo'].fillna('') + ' '
           + _DF['comparto'].fillna('')).str.lower()
    mask = pd.Series(False, index=_DF.index)
    _generic = max(1, int(len(_DF) * 0.25))       # un token che matcha >25% è troppo generico
    for t in set(toks):
        tm = hay.str.contains(re.escape(t), na=False)
        if tm.sum() and tm.sum() <= _generic:     # tieni solo i termini discriminanti
            mask |= tm
    matches = _DF[mask]

    parts.append("")
    if not matches.empty:
        parts.append(f"COMPARTI PERTINENTI ALLA DOMANDA ({len(matches)}):")
        parts.append(header)
        parts += [_fmt_row(r) for _, r in matches.head(40).iterrows()]
    else:
        d5 = _DF[_DF['rend_5a'].notna()]
        parts.append("Nessun comparto corrisponde a parole chiave specifiche; ecco i 10 migliori "
                     "e i 10 peggiori per rendimento a 5 anni come riferimento:")
        parts.append(header)
        parts += [_fmt_row(r) for _, r in d5.nlargest(10, 'rend_5a').iterrows()]
        parts += [_fmt_row(r) for _, r in d5.nsmallest(10, 'rend_5a').iterrows()]
    return '\n'.join(parts)


def _ask_ai(question):
    """Interroga Google Gemini (piano gratuito) sui dati COVIP; ritorna testo markdown
    o un messaggio d'errore. Nessun credito a pagamento consumato."""
    question = (question or '').strip()
    if not question:
        return "Scrivi una domanda per interrogare l'assistente."
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return ("⚠️ **Servizio AI non configurato.** Ottieni una chiave **gratuita** su "
                "[aistudio.google.com/apikey](https://aistudio.google.com/apikey) (senza "
                "carta di credito) e impostala nella variabile d'ambiente `GEMINI_API_KEY` "
                "(in locale nel file `.env`, su Digital Ocean tra le variabili dell'app). "
                "Il piano gratuito non consuma crediti a pagamento.")
    try:
        context = _build_ai_context(question)
        payload = {
            'system_instruction': {'parts': [{'text': _AI_SYSTEM}]},
            'contents': [{'role': 'user', 'parts': [{
                'text': f"{context}\n\n---\nDOMANDA DEL CONSULENTE: {question}"}]}],
            'generationConfig': {
                'temperature': 0.3,
                # Gemini 3.x "pensa" di default e quei token contano dentro maxOutputTokens:
                # budget ampio (gratis) + thinking basso = risposte complete, mai troncate.
                'maxOutputTokens': 8192,
                'thinkingConfig': {'thinkingLevel': 'low'},
            },
        }
        r = requests.post(
            _GEMINI_URL.format(model=_AI_MODEL),
            headers={'x-goog-api-key': api_key, 'Content-Type': 'application/json'},
            json=payload, timeout=60,
        )
        if r.status_code != 200:
            return f"⚠️ Errore AI (HTTP {r.status_code}): {r.text[:300]}"
        data = r.json()
        cands = data.get('candidates') or []
        if not cands:
            return f"⚠️ Nessuna risposta dal modello. {data.get('promptFeedback', '')}"
        cand = cands[0]
        parts = cand.get('content', {}).get('parts', [])
        txt = ''.join(p.get('text', '') for p in parts).strip()
        if not txt:
            return "(nessuna risposta — riprova a riformulare la domanda)"
        if cand.get('finishReason') == 'MAX_TOKENS':
            txt += "\n\n*(risposta interrotta per lunghezza — prova a fare una domanda più specifica)*"
        return txt
    except Exception as e:                       # noqa: BLE001 — degradazione morbida in UI
        return f"⚠️ Errore nella richiesta all'AI: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
_EXT = [
    'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=_EXT,
           requests_pathname_prefix='/fondipensione/',
           routes_pathname_prefix='/fondipensione/')
app.title = 'Fondi Pensione — Andrea Cappelletti'
server = app.server

app.index_string = '''<!DOCTYPE html><html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>''' + SITE_CSS + '''</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''


def serve_layout():
    n_tot = len(_DF)
    return html.Div([
        make_navbar('Fondi Pensione'),
        html.Div([
            html.H2('Fondi Pensione — confronto rischio / rendimento',
                    style={'color': '#1a3a6b', 'fontSize': '20px', 'margin': '0 0 4px'}),
            html.Div(f'{n_tot} comparti da fonte COVIP (Negoziali, PIP/FIP, Aperti) — '
                     f'rendimenti medi annui netti, dati a fine 2025.',
                     style={'color': '#666', 'fontSize': '12px', 'marginBottom': '14px'}),
            html.Div([
                html.Label('Orizzonte temporale:',
                           style={'fontSize': '11px', 'fontWeight': '700',
                                  'color': '#1a3a6b', 'marginRight': '8px'}),
                dcc.RadioItems(
                    id='fp-orizzonte',
                    options=[{'label': f' {lbl}', 'value': col} for col, lbl in ORIZZONTI],
                    value='rend_5a', inline=True,
                    inputStyle={'marginRight': '3px'},
                    labelStyle={'marginRight': '12px', 'fontSize': '11px', 'cursor': 'pointer'},
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
            html.Div([
                html.Div([
                    html.Label('Tipologia:', style={'fontSize': '11px', 'fontWeight': '700',
                                                    'color': '#1a3a6b', 'marginRight': '8px'}),
                    dcc.Checklist(
                        id='fp-tipologia',
                        options=[{'label': f' {t}', 'value': t} for t in _TIPI],
                        value=list(_TIPI), inline=True,
                        inputStyle={'marginRight': '3px'},
                        labelStyle={'marginRight': '12px', 'fontSize': '11px', 'cursor': 'pointer'},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '24px'}),
                html.Div([
                    html.Label('Categoria:', style={'fontSize': '11px', 'fontWeight': '700',
                                                    'color': '#1a3a6b', 'marginRight': '8px'}),
                    dcc.Checklist(
                        id='fp-categoria',
                        options=[{'label': f' {c}', 'value': c} for c in _CATS],
                        value=list(_CATS), inline=True,
                        inputStyle={'marginRight': '3px'},
                        labelStyle={'marginRight': '12px', 'fontSize': '11px', 'cursor': 'pointer'},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap',
                      'gap': '6px', 'marginBottom': '10px', 'paddingTop': '6px',
                      'borderTop': '1px solid #eee'}),
            html.Div([
                html.Label('Cerca comparto:', style={'fontSize': '11px', 'fontWeight': '700',
                                                      'color': '#1a3a6b', 'marginRight': '8px'}),
                dcc.Input(id='fp-search', type='text', value='', debounce=False,
                          placeholder='es. Core, Cometa, Fonchim…',
                          style={'padding': '5px 10px', 'fontSize': '12px', 'width': '220px',
                                 'border': '1px solid #ccd9ee', 'borderRadius': '6px',
                                 'fontFamily': 'Inter, sans-serif'}),
                html.Span('I comparti corrispondenti vengono cerchiati in rosso sul grafico.',
                          style={'color': '#666', 'fontSize': '11px', 'marginLeft': '10px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap',
                      'gap': '4px', 'marginBottom': '8px'}),
            dcc.Graph(id='fp-scatter', figure=scatter_fig('rend_5a'),
                      config={'displayModeBar': False}),

            # ── Zona AI: interroga i dati COVIP in linguaggio naturale ──────────
            html.Div([
                html.Div([
                    html.I(className='fa-solid fa-robot',
                           style={'color': '#1a3a6b', 'marginRight': '8px'}),
                    html.Span('Chiedi all’intelligenza artificiale',
                              style={'color': '#1a3a6b', 'fontSize': '15px', 'fontWeight': '700'}),
                ], style={'marginBottom': '4px'}),
                html.Div('Domande sui rendimenti dei comparti (es. “come mai il comparto '
                         'Core Pension ha reso così poco?” o “confronta i garantiti '
                         'negoziali con i PIP”). Risposte basate sui dati COVIP a fine 2025, '
                         'tramite Google Gemini (gratuito).',
                         style={'color': '#666', 'fontSize': '11px', 'marginBottom': '8px'}),
                dcc.Textarea(
                    id='fp-ai-question', placeholder='Scrivi qui la tua domanda…',
                    style={'width': '100%', 'minHeight': '60px', 'padding': '8px 10px',
                           'fontSize': '12px', 'fontFamily': 'Inter, sans-serif',
                           'border': '1px solid #ccd9ee', 'borderRadius': '6px',
                           'boxSizing': 'border-box', 'resize': 'vertical'},
                ),
                html.Button('Chiedi', id='fp-ai-btn', n_clicks=0, style={
                    'marginTop': '8px', 'padding': '8px 22px', 'background': '#1a3a6b',
                    'color': '#fff', 'border': 'none', 'borderRadius': '6px',
                    'fontSize': '12px', 'fontWeight': '700', 'letterSpacing': '0.03em',
                    'textTransform': 'uppercase', 'cursor': 'pointer',
                    'fontFamily': 'Inter, sans-serif'}),
                dcc.Loading(type='dot', color='#1a3a6b', children=dcc.Markdown(
                    id='fp-ai-answer', children='', style={
                        'marginTop': '12px', 'fontSize': '13px', 'lineHeight': '1.55',
                        'color': '#222', 'fontFamily': 'Inter, sans-serif'})),
            ], style={'marginTop': '26px', 'padding': '16px 18px', 'background': '#f8fafd',
                      'border': '1px solid #e8edf5', 'borderRadius': '10px'}),
        ], style={'padding': '112px 5% 32px', 'fontFamily': 'Inter, sans-serif'}),
    ])


app.layout = serve_layout


@app.callback(
    Output('fp-scatter', 'figure'),
    Input('fp-orizzonte', 'value'),
    Input('fp-tipologia', 'value'),
    Input('fp-categoria', 'value'),
    Input('fp-search', 'value'),
    prevent_initial_call=True,
)
def _update(orizzonte, tipologie, categorie, search):
    return scatter_fig(orizzonte or 'rend_5a', tipologie, categorie, search)


@app.callback(
    Output('fp-ai-answer', 'children'),
    Input('fp-ai-btn', 'n_clicks'),
    State('fp-ai-question', 'value'),
    prevent_initial_call=True,
)
def _ai_answer(_n_clicks, question):
    return _ask_ai(question)


if __name__ == '__main__':
    app.run(debug=True, port=8065)
