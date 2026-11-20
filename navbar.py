"""Navbar condivisa per tutte le dashboard — andreacappelletti.app
   Solo la barra dei tab strumenti, con «Sito» in testa (la barra del sito
   pubblico e il pulsante Esci sono stati tolti il 19/11/2026).
"""
from dash import html

_BASE = "https://andreacappelletti.app"

# Nome del file di lavoro dell'utente, uguale in tutte le dashboard
# (selettore del Portafoglio, Rendimenti, Frontiera): si cambia solo qui.
ETICHETTA_PERSONALE = "👤 File personale"

_TABS = [
    ("Dashboard Finanziaria Strumenti", "/portafoglio/"),
    ("Strategie Opzioni",    "/opzioni/"),
    ("Analisi Tattica",      "/analisitattica/"),
    ("Macro Economia",       "/fred/"),
    ("Clienti",              "/fondipensione/"),
    ("Calendario",           "/calendario/"),
]


# ─── CSS iniettato in ogni dashboard ──────────────────────────────────────────
# La navbar ha stili inline, che una media query non può battere senza
# !important. wsgi.py inietta questo CSS nell'index_string di ogni dashboard
# (installa_css). Prima la navbar era alta 64 + 38 px e le dashboard lasciano
# sotto di lei margini inline diversi (102–112 px): invece di toccarle una per
# una, il body sale di 64 px, l'altezza della riga tolta (il margine negativo si
# somma al primo margine positivo dei figli anche quando collassano). Non in
# embed (Frontiera/Rendimenti dentro Portafoglio): lì la navbar è nascosta e il
# margine già portato a 8 px, NAVBAR_JS mette la classe nb-embed su <html>.
# ≤900 px (cellulare) la barra dei tab lascia il posto a una riga della stessa
# altezza (38 px, così i margini sotto restano giusti) col nome della dashboard
# a sinistra e il ☰ a destra, come sulla homepage: il ☰ apre il menu verticale
# Sito → Portafoglio → gli altri tab.
NAVBAR_CSS = """
  html:not(.nb-embed) body { margin-top: -64px !important; }
  #site-navbar .nb-mobile, #site-navbar .nb-menu { display: none !important; }
  /* Computer stretto (901–1180 px): i sette tab non ci stavano e «Calendario» usciva
     dalla pagina. Spazi e carattere più piccoli (gli stili dei tab sono inline). */
  @media (min-width: 901px) and (max-width: 1180px) {
    #site-navbar .nb-tabbar .nb-tab, #site-navbar .nb-tabbar .nb-sito {
      padding: 0 8px !important; font-size: 0.66rem !important; letter-spacing: 0.02em !important; }
  }
  @media (max-width: 900px) {
    #site-navbar .nb-tabbar { display: none !important; }
    #site-navbar .nb-mobile { display: flex !important; }
    #site-navbar .nb-menu.nb-aperto { display: flex !important; }
    .nb-nascondi-cell { display: none !important; }
    .nb-tendina { display: block !important; }
    .nb-espandi-btn { cursor: pointer; flex: 1 1 100%; display: flex !important;
      justify-content: space-between; align-items: center; padding: 4px 0;
      margin-right: 0 !important; }
    .nb-solo-cell { display: inline !important; }
    html .nb-solo-flex { display: flex !important; }   /* batte la regola base, che viene dopo */
    /* Riga etichetta + menu a tendina + pulsante: il menu prende lo spazio che resta. */
    .nb-riga-cresci { margin-right: 0 !important; }
    .nb-riga-cresci > .dash-dropdown { flex: 1 1 auto; min-width: 0; width: auto !important; }
    .nb-etic-data { text-align: left !important; }
    /* Griglia 2×2 del range temporale: le righe si «sciolgono» nella griglia,
       così etichette e date stanno in due colonne allineate. */
    .nb-griglia-date { display: grid !important; grid-template-columns: max-content max-content;
      gap: 6px 10px !important; align-items: center; }
    .nb-espandi:not(.nb-aperto) .nb-griglia-date { display: none !important; }
    .nb-riga-data { display: contents !important; }
    .nb-espandi-ico { display: inline-block !important; transition: transform 0.2s; }
    /* Stessa rientranza a destra della tendina (14 px): le frecce stanno in colonna. */
    .nb-espandi { padding-right: 14px !important; box-sizing: border-box; }
    .nb-espandi:not(.nb-aperto) .nb-espandi-corpo { display: none !important; }
    .nb-espandi.nb-aperto .nb-espandi-ico { transform: rotate(180deg); }
    .nb-espandi.nb-aperto .nb-espandi-corpo { margin-top: 6px; }
    /* Colonne affiancate sul computer, impilate sul cellulare. */
    .nb-impila { flex-direction: column !important; align-items: flex-start !important; }
    .nb-impila-inv { flex-direction: column-reverse !important; }
    .nb-piena { width: 100% !important; justify-content: flex-start !important; }
    .nb-a-capo { flex-wrap: wrap !important; }
  }
  /* Sezioni espandibili solo sul cellulare: sul computer restano aperte e
     l'intestazione non si clicca. */
  .nb-espandi-ico { display: none; }
  /* Intestazione che esiste solo sul cellulare (es. «Asset», «Grafici» nel
     Portafoglio): sul computer sparisce; ≤900 px la mostra .nb-espandi-btn. */
  .nb-testa-cell { display: none; }
  /* Blocco flex che esiste solo sul cellulare (es. etichette colonne nella sezione «Asset»). */
  .nb-solo-flex { display: none !important; }
  /* Testo che compare solo sul cellulare (il gemello da PC ha nb-nascondi-cell). */
  .nb-solo-cell { display: none; }
  #site-navbar .nb-menu a:hover { color: #1a3a6b !important; }
  /* Tendina dei sotto-tab (solo cellulare): il pulsante mostra il sotto-tab
     attivo, il menu verticale sostituisce i dcc.Tabs nascosti con nb-nascondi-cell. */
  .nb-tendina { display: none; position: relative; margin-bottom: 6px; }
  .nb-tendina-btn { width: 100%; display: flex; align-items: center;
    justify-content: space-between; gap: 10px; padding: 10px 14px; cursor: pointer;
    background: #f0f4fa; border: 1px solid #dee2e6; border-top: 3px solid #1a3a5c;
    font-family: Inter, sans-serif; color: #1a3a5c; text-align: left; }
  .nb-tendina-riga { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; font-size: 13px; }
  .nb-tendina-tit { font-size: 11px; font-weight: 700;
    letter-spacing: 0.05em; text-transform: uppercase; color: #5a7099; }
  .nb-tendina-sep { color: #9baabf; }
  .nb-tendina-att { font-weight: 700; }
  .nb-tendina-menu { display: none; position: absolute; left: 0; right: 0; top: 100%;
    z-index: 900; flex-direction: column; background: #ffffff;
    border: 1px solid #dee2e6; box-shadow: 0 8px 24px rgba(26,58,92,0.18); }
  .nb-tendina.nb-aperto .nb-tendina-menu { display: flex; }
  .nb-tendina-voce { text-align: left; padding: 12px 16px; cursor: pointer;
    border: none; border-left: 3px solid transparent; border-bottom: 1px solid #eef1f6;
    background: #ffffff; font-family: Inter, sans-serif; font-size: 14px; color: #1a3a5c; }
  .nb-tendina-voce.nb-attiva { font-weight: 700; border-left-color: #1a3a5c;
    background: #f0f4fa; }
"""

# Il ☰ apre/chiude il menu (lo stesso fanno le tendine nb-tendina dei
# sotto-tab); un tocco fuori lo chiude. Delega sul document,
# perché Dash può ridisegnare la navbar dopo il caricamento.
NAVBAR_JS = """
if (location.search.indexOf('embed=1') !== -1)
  document.documentElement.classList.add('nb-embed');
document.addEventListener('click', function (e) {
  // Tendine dei sotto-tab: il pulsante apre/chiude, una voce o un tocco fuori chiude.
  // Doppioni da cellulare: cliccano il componente vero (data-bersaglio = id).
  var cl = e.target.closest && e.target.closest('.nb-clicca');
  if (cl) {
    var ber = document.getElementById(cl.getAttribute('data-bersaglio'));
    if (ber) ber.click();
    return;
  }
  var eb = e.target.closest && e.target.closest('.nb-espandi-btn');
  if (eb && window.matchMedia('(max-width: 900px)').matches) {
    var sez = eb.closest('.nb-espandi');
    if (sez) sez.classList.toggle('nb-aperto');
  }
  var tb = e.target.closest && e.target.closest('.nb-tendina-btn');
  document.querySelectorAll('.nb-tendina.nb-aperto').forEach(function (t) {
    if (!tb || !t.contains(tb)) t.classList.remove('nb-aperto');
  });
  if (tb) {
    var ap1 = tb.parentNode.classList.toggle('nb-aperto');
    tb.setAttribute('aria-expanded', ap1 ? 'true' : 'false');
    return;
  }
  var menu = document.querySelector('#site-navbar .nb-menu');
  if (!menu) return;
  var btn = e.target.closest && e.target.closest('#site-navbar .nb-burger');
  if (btn) {
    var ap = menu.classList.toggle('nb-aperto');
    btn.setAttribute('aria-expanded', ap ? 'true' : 'false');
    var ic = btn.querySelector('i');
    if (ic) ic.className = ap ? 'fa-solid fa-xmark' : 'fa-solid fa-bars';
    return;
  }
  if (menu.classList.contains('nb-aperto') && !menu.contains(e.target)) {
    menu.classList.remove('nb-aperto');
    var b = document.querySelector('#site-navbar .nb-burger');
    if (b) { b.setAttribute('aria-expanded', 'false');
             var i2 = b.querySelector('i'); if (i2) i2.className = 'fa-solid fa-bars'; }
  }
});
"""


def installa_css(app) -> None:
    """Aggiunge NAVBAR_CSS (e il viewport, se manca) all'index_string di una app Dash."""
    idx = app.index_string
    if "nb-tabbar" in idx or "</head>" not in idx:
        return
    extra = "<style>" + NAVBAR_CSS + "</style><script>" + NAVBAR_JS + "</script>"
    if 'name="viewport"' not in idx and "{%metas%}" not in idx:
        extra = '<meta name="viewport" content="width=device-width, initial-scale=1">' + extra
    app.index_string = idx.replace("</head>", extra + "</head>", 1)


def make_navbar(current: str = "", margine: str = "5%") -> html.Div:
    """margine: spazio ai lati della barra dei tab, uguale a quello del contenuto della
    dashboard, così la barra va da bordo a bordo della pagina (i tab si allargano per
    riempirla). Il Portafoglio ha contenuto a 1% + 8 px del body."""

    _url = {lab: u.strip("/") for lab, u in _TABS}

    def _attivo(label: str) -> bool:
        # Basta una parola: «Strategie Opzioni» arriva come 'Opzioni'. Vale
        # anche il percorso: «Dashboard Finanziaria» arriva come 'portafoglio'.
        cur = current.lower()
        return (_url.get(label, "") in cur and bool(_url.get(label))) or \
            any(w in cur for w in label.lower().split())

    def _tab_style(label: str) -> dict:
        active = _attivo(label)
        return {
            "padding": "0 20px",
            "height": "38px",
            "lineHeight": "38px",
            "fontSize": "0.75rem",
            "fontWeight": "700",
            "letterSpacing": "0.05em",
            "textTransform": "uppercase",
            "textDecoration": "none",
            "fontFamily": "Inter, sans-serif",
            "borderBottom": "2px solid #1a3a6b" if active else "2px solid transparent",
            "color": "#1a3a6b" if active else "#5a7099",
            "background": "transparent",
            "transition": "all 0.2s",
            "whiteSpace": "nowrap",
            "display": "inline-block",
            # i tab si dividono lo spazio: la barra arriva al bordo destro della pagina
            "flex": "1 1 auto",
            "textAlign": "center",
        }

    # ── Sito + tab strumenti ─────────────────────────────────────────────────
    _stile_primo = {
        "padding": "0 20px",
        "height": "38px",
        "lineHeight": "38px",
        "fontSize": "0.75rem",
        "fontWeight": "700",
        "letterSpacing": "0.05em",
        "textTransform": "uppercase",
        "textDecoration": "none",
        "fontFamily": "Inter, sans-serif",
        "display": "inline-flex",
        "alignItems": "center",
        "borderRight": "1px solid #e8edf5",
        "marginRight": "8px",
        "flexShrink": "0",
    }
    tabbar = html.Div([
        html.A([
            html.I(className="fa-solid fa-house", style={"marginRight": "6px"}),
            "Sito",
        ], href=f"{_BASE}/", className="nb-sito",
           style={**_stile_primo, "color": "#f37021"}),
        *[html.A(label, href=url, style=_tab_style(label),
                 className="nb-tab nb-attivo" if _attivo(label) else "nb-tab")
          for label, url in _TABS],
    ], className="nb-tabbar", style={
        "display": "flex",
        "alignItems": "center",
        "padding": f"0 {margine}",
        "height": "38px",
        "borderBottom": "1px solid #e8edf5",
        "background": "#f8fafd",
    })

    # ── Cellulare: nome della dashboard + ☰, e il menu verticale ─────────────
    _attiva = next((lab for lab, _ in _TABS if _attivo(lab)), "")
    mobile = html.Div([
        html.Span(_attiva, style={
            "fontSize": "0.8rem", "fontWeight": "700", "letterSpacing": "0.05em",
            "textTransform": "uppercase", "color": "#1a3a6b",
            "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis",
        }),
        html.Button(html.I(className="fa-solid fa-bars"), className="nb-burger",
                    title="Menu", **{"aria-label": "Menu", "aria-expanded": "false"},
                    style={
            "background": "transparent", "border": "none", "cursor": "pointer",
            "color": "#1a3a6b", "fontSize": "1.4rem", "padding": "0 4px",
            "height": "38px", "lineHeight": "38px", "flexShrink": "0",
        }),
    ], className="nb-mobile", style={
        "alignItems": "center", "justifyContent": "space-between",
        "padding": "0 14px", "height": "38px",
        "borderBottom": "1px solid #e8edf5", "background": "#f8fafd",
    })

    def _voce(label, href, icona=None, colore=None, attiva=False):
        figli = [html.I(className=icona, style={"width": "22px"}), label] if icona else [label]
        return html.A(figli, href=href, style={
            "fontSize": "1rem", "fontWeight": "700" if attiva else "600",
            "color": colore or ("#1a3a6b" if attiva else "#5a7099"),
            "padding": "0.75rem 0", "borderBottom": "1px solid #ccd9ee",
            "textDecoration": "none", "fontFamily": "Inter, sans-serif",
            "display": "flex", "alignItems": "center",
            "borderLeft": "3px solid #1a3a6b" if attiva else "3px solid transparent",
            "paddingLeft": "10px",
        })

    menu = html.Div([
        _voce("Sito", f"{_BASE}/", "fa-solid fa-house", colore="#f37021"),
        *[_voce(lab, url, attiva=_attivo(lab)) for lab, url in _TABS],
    ], className="nb-menu", style={
        "position": "absolute", "top": "38px", "left": "0", "right": "0",
        "flexDirection": "column", "padding": "0.5rem 5% 1rem",
        "background": "#ffffff",
        "borderBottom": "1px solid #ccd9ee",
        "boxShadow": "0 8px 20px rgba(26,58,107,0.12)",
        "maxHeight": "calc(100vh - 38px)", "overflowY": "auto",
    })

    return html.Div([tabbar, mobile, menu], id="site-navbar", style={
        "position": "fixed",
        "top": "0", "left": "0", "right": "0",
        "zIndex": "1000",
        "background": "rgba(255,255,255,0.97)",
        "backdropFilter": "blur(14px)",
        "boxShadow": "0 2px 12px rgba(26,58,107,0.08)",
        "fontFamily": "Inter, sans-serif",
    })


def tendina_cell(id_prefisso: str, voci, titolo: str = "Tabs Strumenti") -> html.Div:
    """Tendina dei sotto-tab per il cellulare (≤900 px; sul computer è nascosta).

    voci = [(etichetta, valore)]. I componenti hanno id
    f"{id_prefisso}-att" (etichetta del sotto-tab attivo) e
    {"type": f"{id_prefisso}-voce", "index": valore} (le voci): i callback che
    li collegano ai dcc.Tabs stanno nella dashboard. Apertura/chiusura in NAVBAR_JS.
    """
    return html.Div(className="nb-tendina", children=[
        html.Button(className="nb-tendina-btn", n_clicks=0,
                    **{"aria-expanded": "false"}, children=[
            html.Span(className="nb-tendina-riga", children=[
                html.Span(titolo, className="nb-tendina-tit"),
                html.Span(" – ", className="nb-tendina-sep"),
                html.Span(voci[0][0], id=f"{id_prefisso}-att", className="nb-tendina-att"),
            ]),
            html.I(className="fa-solid fa-chevron-down"),
        ]),
        html.Div(className="nb-tendina-menu", children=[
            html.Button(lab, id={"type": f"{id_prefisso}-voce", "index": val},
                        n_clicks=0, className="nb-tendina-voce")
            for lab, val in voci
        ]),
    ])
