#!/bin/bash
# Pubblica tutte le modifiche su DigitalOcean e GitHub
# Uso: ./pubblica.sh "descrizione delle modifiche"

set -e

MSG="${1:-Aggiornamento}"
PROFILO_DIR="$(cd "$(dirname "$0")/profilo" && pwd)"
PORTAFOGLIO_DIR="$(cd "$(dirname "$0")/portafoglio" && pwd)"

# Sincronizza sempre profilo.html
cp "$PROFILO_DIR/index.html" "$PORTAFOGLIO_DIR/profilo.html"

# ── Repo portafoglio (DigitalOcean) ─────────────────────────────────────────
echo "▶ Pubblico su DigitalOcean..."
cd "$PORTAFOGLIO_DIR"
git add -A -- ':!sessions/' ':!avvia.sh'
if git diff --cached --quiet; then
  echo "  Nessuna modifica."
else
  git commit -m "$MSG"
  git push
  echo "  ✓ Deploy avviato su andreacappelletti.app"
fi

# ── Repo profilo (GitHub Pages) ─────────────────────────────────────────────
echo "▶ Pubblico sito personale..."
cd "$PROFILO_DIR"
git add index.html
if git diff --cached --quiet; then
  echo "  Nessuna modifica al sito."
else
  git commit -m "$MSG"
  git push
  echo "  ✓ Sito personale aggiornato."
fi

echo ""
echo "✅ Tutto pubblicato!"
