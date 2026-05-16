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
