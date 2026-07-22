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
    ("Fondi Pensione",       "/fondipensione/"),
]


def make_navbar(current: str = "") -> html.Div:

    def _tab_style(label: str) -> dict:
        active = label.lower().split()[0] in current.lower()
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

        # Brand — identico al sito
        html.Div([
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
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),

        # Link navigazione — identici al sito
        html.Ul([
            html.Li(html.A(label, href=url, style={
                "fontSize": "0.82rem", "fontWeight": "600",
                "color": "#5a7099", "letterSpacing": "0.04em",
                "textTransform": "uppercase", "textDecoration": "none",
                "transition": "color 0.2s", "fontFamily": "Inter, sans-serif",
            }))
            for label, url in _NAV_LINKS
        ], style={
            "display": "flex", "gap": "2rem",
            "listStyle": "none", "margin": "0", "padding": "0",
            "alignItems": "center",
        }),

        # CTA — identico al sito
        html.A([
            html.I(className="fa-regular fa-calendar"),
            " Prenota call",
        ], href=f"{_BASE}/#prenota", style={
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
    })

    # ── Riga 2: esci + tab strumenti ─────────────────────────────────────────
    tabbar = html.Div([
        html.A([
            html.I(className="fa-solid fa-right-from-bracket",
                   style={"marginRight": "6px"}),
            "Esci",
        ], href="/logout", style={
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
        *[html.A(label, href=url, style=_tab_style(label)) for label, url in _TABS],
    ], style={
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
