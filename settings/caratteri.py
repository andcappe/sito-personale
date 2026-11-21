"""
Caratteri di tutto il sito andreacappelletti.app — si cambiano SOLO qui.

Vale per: le nove dashboard (testo, pulsanti, tabelle, tendine, grafici Plotly),
la homepage (profilo/index.html), le pagine di login/registrazione/admin.
Restano fuori le email (i programmi di posta non caricano i caratteri web).

Come funziona
- `installa(app)` (chiamato da wsgi._load per ogni dashboard) aggiunge al <head>
  il carattere di Google Fonts e una regola CSS con !important: vince sui
  `fontFamily` scritti a mano nei singoli componenti e sul testo SVG dei grafici,
  quindi non serve toccare le centinaia di stili sparsi nei file app.py.
- Il testo in `monospace` (tabelle numeriche di Macro e Valutazione) prende
  CARATTERE_NUMERI; le icone Font Awesome restano intatte.
- `imposta_plotly()` mette lo stesso carattere come predefinito dei grafici.
- `in_html(testo)` fa lo stesso per le pagine HTML scritte a mano (homepage, login).

Per cambiare carattere: modificare CARATTERE (e, se serve, il nome su Google
Fonts in GOOGLE_FONTS) e riavviare il sito.
"""
import re

# ─── Famiglie ────────────────────────────────────────────────────────────────
# Testo di tutto il sito (dashboard comprese, titoli delle dashboard compresi).
CARATTERE = "'Inter', -apple-system, 'Segoe UI', Arial, sans-serif"
# Solo i grandi titoli della homepage (nome, titoli di sezione, piè di pagina).
CARATTERE_TITOLI = "'Playfair Display', Georgia, serif"
# Cifre allineate in colonna (dove il codice chiede 'monospace').
CARATTERE_NUMERI = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

# Famiglie e pesi da scaricare da Google Fonts (devono comprendere quelle sopra).
GOOGLE_FONTS = ("https://fonts.googleapis.com/css2?"
                "family=Inter:wght@300;400;500;600;700"
                "&family=Playfair+Display:wght@400;600;700&display=swap")

# ─── Dimensioni (px) ─────────────────────────────────────────────────────────
# Usate dai componenti che importano FONT (settings.browser_css.FONT è questo
# stesso dizionario). Molte dimensioni nelle dashboard sono ancora scritte a mano.
FONT = {
    'xs':   '11px',   # etichette griglia, badge
    'sm':   '12px',   # testo secondario, descrizioni
    'md':   '13px',   # testo corpo standard
    'lg':   '14px',   # bottoni, label principali
    'xl':   '16px',   # titoli sezione
    'xxl':  '20px',   # titoli pagina
    'icon': '12px',   # icone/emoji nelle celle
}

# Nomi degli asset nelle griglie (Analisi Portafoglio, Frontiera, Matrice
# Correlazioni): stessa dimensione e peso ovunque; il colore lo decide ogni pagina.
NOME_ASSET = {'fontSize': '12px', 'fontWeight': '600'}


# ─── Da qui in giù non serve toccare niente ──────────────────────────────────
_ICONE = ':not(.fa):not([class*="fa-"])'
_MONO = '[style*="monospace"]'

CSS = f"""
  :root {{
    --car-testo: {CARATTERE};
    --car-titoli: {CARATTERE_TITOLI};
    --car-numeri: {CARATTERE_NUMERI};
  }}
  html body, html body *{_ICONE}:not({_MONO}) {{
    font-family: var(--car-testo) !important;
  }}
  html body {_MONO}, html body {_MONO} *{_ICONE},
  html body code, html body pre, html body kbd, html body samp {{
    font-family: var(--car-numeri) !important;
  }}
  html body .car-titoli {{ font-family: var(--car-titoli) !important; }}
"""

LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        f'<link rel="stylesheet" href="{GOOGLE_FONTS}">')

_SEGNO = "car-testo"   # per non installare due volte


def in_html(testo: str) -> str:
    """Aggiunge carattere e regole a una pagina HTML (prima di </head>)."""
    if _SEGNO in testo or "</head>" not in testo:
        return testo
    return testo.replace("</head>", f"{LINK}<style>{CSS}</style></head>", 1)


def in_homepage(testo: str) -> str:
    """Homepage (profilo/index.html): lì il titolo in Playfair è una scelta di stile,
    quindi niente regola generale; si sostituiscono i nomi dei caratteri scritti nel
    file con quelli di questo modulo."""
    if _SEGNO in testo or "</head>" not in testo:
        return testo
    testo = re.sub(r"'Playfair Display'\s*,\s*serif", "var(--car-titoli)", testo)
    testo = re.sub(r"'Inter'\s*,\s*sans-serif", "var(--car-testo)", testo)
    radice = (f":root {{ --car-testo: {CARATTERE}; --car-titoli: {CARATTERE_TITOLI};"
              f" --car-numeri: {CARATTERE_NUMERI}; }}"
              # pulsanti e campi del modulo contatti: senza questo il browser usa Arial
              " button, input, select, textarea { font-family: inherit; }")
    return testo.replace("</head>", f"{LINK}<style>{radice}</style></head>", 1)


def installa(app) -> None:
    """Aggiunge carattere e regole all'index_string di una app Dash."""
    app.index_string = in_html(app.index_string)


def imposta_plotly() -> None:
    """Carattere predefinito dei grafici Plotly (il CSS lo impone comunque)."""
    try:
        import plotly.io as pio
        for nome in ("plotly", "plotly_white", "simple_white"):
            if nome in pio.templates:
                pio.templates[nome].layout.font.family = CARATTERE.replace("'", "")
    except Exception:
        pass
