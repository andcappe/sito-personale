"""Navbar condivisa per tutte le dashboard — andreacappelletti.app
   Replica esatta della navbar del sito pubblico + tab bar strumenti.
"""
from dash import html

_BASE = "https://andreacappelletti.app"

_NAV_LINKS = [
    ("Home",         f"{_BASE}/"),
    ("Chi Sono",     f"{_BASE}/#chi-sono"),
    ("Esperienza",   f"{_BASE}/#esperienza"),
    ("Strumenti",    f"{_BASE}/#dashboard"),
    ("Prenota Call", f"{_BASE}/#prenota"),
    ("Contatti",     f"{_BASE}/#contatti"),
]

_TABS = [
    ("Portafoglio",          "/portafoglio/"),
    ("Strategie Opzioni",    "/opzioni/"),
    ("Analisi Tattica",      "/analisitattica/"),
    ("Macro Economia",       "/fred/"),
    ("Clienti",              "/fondipensione/"),
    ("Calendario",           "/calendario/"),
]


# ─── Cellulare ────────────────────────────────────────────────────────────────
# La navbar ha stili inline, che una media query non può battere senza
# !important. wsgi.py inietta questo CSS nell'index_string di ogni dashboard
# (installa_css). Le altezze delle due righe (64 + 38 px) non cambiano, così
# il margine sotto la navbar fissa che ogni dashboard già lascia resta giusto.
NAVBAR_CSS = """
  #site-navbar .nb-tabbar { overflow-x: auto; scrollbar-width: none;
                            -webkit-overflow-scrolling: touch; }
  #site-navbar .nb-tabbar::-webkit-scrollbar { display: none; }
  #site-navbar .nb-tabbar > a { flex-shrink: 0; }
  @media (max-width: 900px) {
    #site-navbar .nb-links { display: none !important; }
    #site-navbar .nb-top { padding: 0 14px !important; }
    #site-navbar .nb-tabbar { padding: 0 4px !important; }
    #site-navbar .nb-tab { padding: 0 12px !important; letter-spacing: 0.02em !important; }
    #site-navbar .nb-esci { padding: 0 12px !important; margin-right: 2px !important; }
    /* sfumatura a destra: dice che la barra scorre; tolta quando si è in fondo */
    #site-navbar .nb-tabbar:not(.nb-fine) {
      -webkit-mask-image: linear-gradient(to right, #000 82%, transparent);
              mask-image: linear-gradient(to right, #000 82%, transparent); }
  }
  @media (max-width: 480px) {
    #site-navbar .nb-cta { padding: 9px 12px !important; }
    #site-navbar .nb-cta-testo { display: none; }
  }
"""

# Dash disegna la navbar dopo il caricamento: si aspetta che compaia, poi si
# porta in vista il tab attivo (su cellulare Calendario sarebbe fuori schermo).
NAVBAR_JS = """
(function () {
  var giri = 0;
  var t = setInterval(function () {
    var bar = document.querySelector('#site-navbar .nb-tabbar');
    if (!bar) { if (++giri > 100) clearInterval(t); return; }
    clearInterval(t);
    var fine = function () {
      bar.classList.toggle('nb-fine',
        bar.scrollLeft + bar.clientWidth >= bar.scrollWidth - 4);
    };
    var att = bar.querySelector('.nb-attivo');
    if (att && bar.scrollWidth > bar.clientWidth) {
      var a = att.getBoundingClientRect(), b = bar.getBoundingClientRect();
      bar.scrollLeft += a.left - b.left - (b.width - a.width) / 2;
    }
    fine();
    bar.addEventListener('scroll', fine, {passive: true});
    window.addEventListener('resize', fine);
  }, 200);
})();
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


def make_navbar(current: str = "") -> html.Div:

    def _attivo(label: str) -> bool:
        # Basta una parola: «Strategie Opzioni» arriva come 'Opzioni'.
        return any(w in current.lower() for w in label.lower().split())

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
        }

    # ── Riga 1: replica esatta navbar sito pubblico ───────────────────────────
    topbar = html.Nav([

        # Brand — identico al sito (link alla home: su cellulare è l'unica via,
        # perché i link di navigazione si nascondono)
        html.A([
            html.Span("A·C", style={
                "fontFamily": "'Playfair Display', serif",
                "fontSize": "1.1rem",
                "color": "#1a3a6b",
            }),
            html.Span("Private Banker", style={
                "fontFamily": "Inter, sans-serif",
                "fontSize": "0.62rem", "fontWeight": "700",
                "letterSpacing": "0.1em", "textTransform": "uppercase",
                "color": "#f37021",
                "background": "rgba(243,112,33,0.1)",
                "border": "1px solid rgba(243,112,33,0.3)",
                "padding": "3px 8px", "borderRadius": "4px",
            }),
        ], href=f"{_BASE}/", className="nb-brand",
           style={"display": "flex", "alignItems": "center", "gap": "10px",
                  "textDecoration": "none", "flexShrink": "0"}),

        # Link navigazione — identici al sito
        html.Ul([
            html.Li(html.A(label, href=url, style={
                "fontSize": "0.82rem", "fontWeight": "600",
                "color": "#5a7099", "letterSpacing": "0.04em",
                "textTransform": "uppercase", "textDecoration": "none",
                "transition": "color 0.2s", "fontFamily": "Inter, sans-serif",
            }))
            for label, url in _NAV_LINKS
        ], className="nb-links", style={
            "display": "flex", "gap": "2rem",
            "listStyle": "none", "margin": "0", "padding": "0",
            "alignItems": "center",
        }),

        # CTA — identico al sito
        html.A([
            html.I(className="fa-regular fa-calendar"),
            html.Span(" Prenota call", className="nb-cta-testo"),
        ], href=f"{_BASE}/#prenota", className="nb-cta", style={
            "padding": "9px 20px",
            "background": "#1a3a6b", "color": "#ffffff",
            "borderRadius": "7px", "fontSize": "0.8rem", "fontWeight": "700",
            "letterSpacing": "0.04em", "textTransform": "uppercase",
            "textDecoration": "none", "display": "inline-flex",
            "alignItems": "center", "gap": "7px",
            "fontFamily": "Inter, sans-serif",
            "transition": "background 0.2s",
        }),

    ], style={
        "display": "flex", "alignItems": "center",
        "justifyContent": "space-between",
        "padding": "0 5%", "height": "64px",
        "borderBottom": "1px solid #ccd9ee",
    }, className="nb-top")

    # ── Riga 2: esci + tab strumenti ─────────────────────────────────────────
    tabbar = html.Div([
        html.A([
            html.I(className="fa-solid fa-right-from-bracket",
                   style={"marginRight": "6px"}),
            "Esci",
        ], href="/logout", className="nb-esci", style={
            "padding": "0 20px",
            "height": "38px",
            "lineHeight": "38px",
            "fontSize": "0.75rem",
            "fontWeight": "700",
            "letterSpacing": "0.05em",
            "textTransform": "uppercase",
            "textDecoration": "none",
            "fontFamily": "Inter, sans-serif",
            "color": "#c0392b",
            "display": "inline-flex",
            "alignItems": "center",
            "borderRight": "1px solid #e8edf5",
            "marginRight": "8px",
        }),
        *[html.A(label, href=url, style=_tab_style(label),
                 className="nb-tab nb-attivo" if _attivo(label) else "nb-tab")
          for label, url in _TABS],
    ], className="nb-tabbar", style={
        "display": "flex",
        "alignItems": "center",
        "padding": "0 5%",
        "height": "38px",
        "borderBottom": "1px solid #e8edf5",
        "background": "#f8fafd",
    })

    return html.Div([topbar, tabbar], id="site-navbar", style={
        "position": "fixed",
        "top": "0", "left": "0", "right": "0",
        "zIndex": "1000",
        "background": "rgba(255,255,255,0.97)",
        "backdropFilter": "blur(14px)",
        "boxShadow": "0 2px 12px rgba(26,58,107,0.08)",
        "fontFamily": "Inter, sans-serif",
    })
