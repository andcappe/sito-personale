"""
Configurazione CSS cross-browser per tutto il sito andreacappelletti.app.
Importa BROWSER_RESET_CSS e inseriscilo nel <style> del tuo index_string.
Importa le costanti tipografiche per font-size uniformi su tutti i browser.
"""

# ─── Font-size uniformi (px) ────────────────────────────────────────────────
# Modificare qui per aggiornare tutta la UI del sito
FONT = {
    'xs':   '11px',   # etichette griglia, badge
    'sm':   '12px',   # testo secondario, descrizioni
    'md':   '13px',   # testo corpo standard
    'lg':   '14px',   # bottoni, label principali
    'xl':   '16px',   # titoli sezione
    'xxl':  '20px',   # titoli pagina
    'icon': '12px',   # icone/emoji nelle celle
}

# ─── CSS reset cross-browser ─────────────────────────────────────────────────
# Corregge le differenze di rendering tra Chrome, Firefox e Safari.
# Aggiungere questo CSS nel <style> di index_string di ogni dashboard.
BROWSER_RESET_CSS = f"""
  *, *::before, *::after {{
    box-sizing: border-box;
  }}
  html {{
    font-size: 16px;
    -webkit-text-size-adjust: 100%;
    -moz-text-size-adjust: 100%;
    text-size-adjust: 100%;
  }}
  body {{
    margin: 0;
    font-family: 'Inter', sans-serif;
    background: #f5f8fe;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
  }}
  button {{
    font-family: inherit;
    -webkit-appearance: none;
    -moz-appearance: none;
  }}
  input, select, textarea {{
    font-family: inherit;
    font-size: inherit;
  }}
"""


# ─── CSS PAGINA UNIFICATO PER TUTTO IL SITO ──────────────────────────────────
# Reset + layout pagina identico ad Analisi di Portafoglio (font, sfondo,
# contenitore "spazio pagina" sotto la navbar fissa, intestazione pagina).
# Le app lo iniettano nel <style> del loro index_string al posto del solo reset.
SITE_CSS = BROWSER_RESET_CSS + """
  /* ── Layout pagina unificato (identico ad Analisi di Portafoglio) ── */
  body { background: #ffffff; color: #1a2a4a; }
  .page-wrap  { margin-top: 106px; padding: 0 1%; }
  .page-head  { padding: 14px 20px 12px; border-bottom: 2px solid #e2e8f0;
                background: linear-gradient(90deg, #f0f4fb 0%, #ffffff 100%);
                margin-bottom: 10px; }
  .page-head h1 { margin: 0; font-size: 1.6rem; font-weight: 700; color: #1a3a6b;
                  font-family: 'Playfair Display', serif; letter-spacing: 0.02em; }
  .page-head .sub { font-size: 1.1rem; font-weight: 400; color: #4a5d7a; }
  @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
"""
