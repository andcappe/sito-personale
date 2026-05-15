"""Navbar condivisa per tutte le dashboard — andreacappelletti.app"""
from dash import html

_BASE = "https://andreacappelletti.app"

_NAV_LINKS = [
    ("Home",         _BASE),
    ("Chi Sono",     f"{_BASE}#chi-sono"),
    ("Esperienza",   f"{_BASE}#esperienza"),
    ("Strumenti",    f"{_BASE}#dashboard"),
    ("Prenota Call", f"{_BASE}#prenota"),
    ("Contatti",     f"{_BASE}#contatti"),
]


def make_navbar() -> html.Nav:
    ls = {
        "fontSize": "0.82rem", "fontWeight": "600",
        "color": "#5a7099", "letterSpacing": "0.04em",
        "textTransform": "uppercase", "textDecoration": "none",
        "transition": "color 0.2s", "fontFamily": "Inter, sans-serif",
    }
    return html.Nav([
        # ── Brand ────────────────────────────────────────────────────────────
        html.A([
            html.Span("A·C", style={
                "fontFamily": "'Playfair Display', serif",
                "fontSize": "1.1rem", "fontWeight": "700",
                "color": "#1a3a6b", "marginRight": "10px",
            }),
            html.Span("Monomandatario Fineco Bank", style={
                "fontFamily": "Inter, sans-serif",
                "fontSize": "0.62rem", "fontWeight": "700",
                "letterSpacing": "0.1em", "textTransform": "uppercase",
                "color": "#f37021",
                "background": "rgba(243,112,33,0.1)",
                "border": "1px solid rgba(243,112,33,0.3)",
                "padding": "3px 8px", "borderRadius": "4px",
            }),
        ], href=_BASE, target="_blank",
           style={"textDecoration": "none", "display": "flex", "alignItems": "center"}),

        # ── Link navigazione ─────────────────────────────────────────────────
        html.Ul([
            html.Li(html.A(label, href=url, target="_blank", style=ls))
            for label, url in _NAV_LINKS
        ], style={
            "display": "flex", "gap": "2rem", "listStyle": "none",
            "margin": "0", "padding": "0", "alignItems": "center",
        }),

        # ── CTA ──────────────────────────────────────────────────────────────
        html.A([
            html.I(className="fa-regular fa-calendar", style={"marginRight": "7px"}),
            "Prenota call",
        ], href=f"{_BASE}#prenota", target="_blank", style={
            "padding": "9px 20px",
            "background": "#1a3a6b", "color": "white",
            "borderRadius": "7px", "fontSize": "0.8rem", "fontWeight": "700",
            "letterSpacing": "0.04em", "textTransform": "uppercase",
            "textDecoration": "none", "display": "inline-flex",
            "alignItems": "center", "fontFamily": "Inter, sans-serif",
        }),
    ], style={
        "position": "fixed", "top": "0", "left": "0", "right": "0",
        "zIndex": "1000",
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "padding": "0 5%", "height": "64px",
        "background": "rgba(255,255,255,0.97)",
        "backdropFilter": "blur(14px)",
        "borderBottom": "1px solid #ccd9ee",
        "boxShadow": "0 2px 12px rgba(26,58,107,0.08)",
        "fontFamily": "Inter, sans-serif",
    })
