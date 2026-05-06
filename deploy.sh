#!/bin/bash
# Sincronizza profilo.html → portafoglio e fa il deploy di entrambi i repo

set -e

PROFILO_DIR="$(cd "$(dirname "$0")/profilo" && pwd)"
PORTAFOGLIO_DIR="$(cd "$(dirname "$0")/portafoglio" && pwd)"
SRC="$PROFILO_DIR/index.html"
DST="$PORTAFOGLIO_DIR/profilo.html"

echo "▶ Copia index.html → profilo.html"
cp "$SRC" "$DST"

# ── Deploy sito personale (andcappe.github.io) ──────────────────────────────
echo "▶ Push repo profilo..."
cd "$PROFILO_DIR"
git add index.html
if git diff --cached --quiet; then
  echo "  Nessuna modifica al sito personale."
else
  git commit -m "Aggiorna sito personale"
  git push
  echo "  ✓ Sito personale aggiornato."
fi

# ── Deploy portafoglio (analisi-portafoglio) ─────────────────────────────────
echo "▶ Push repo portafoglio..."
cd "$PORTAFOGLIO_DIR"
git add profilo.html
if git diff --cached --quiet; then
  echo "  Nessuna modifica a profilo.html."
else
  git commit -m "Sincronizza profilo.html dal sito personale"
  git push
  echo "  ✓ profilo.html aggiornato su DigitalOcean."
fi

echo ""
echo "✅ Deploy completato!"
echo "   andreacappelletti.app       → sito personale"
echo "   andreacappelletti.app/portafoglio/ → web app"
