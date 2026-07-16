#!/bin/bash
# Pubblica il sito su DigitalOcean.
# Deploy = commit + push del repo "Sito personale" (origin = andcappe/sito-personale),
# che DigitalOcean App Platform ricostruisce in automatico ad ogni push su main.
# Uso: ./pubblica.sh "descrizione delle modifiche"
#
# NOTA: le vecchie versioni di questo script pubblicavano il repo ANNIDATO in
# portafoglio/ (andcappe/analisi-portafoglio), che è il deploy legacy di quando il
# sito era solo la dashboard portafoglio: così le altre sei dashboard non arrivavano
# mai in produzione. Ora si pubblica il repo giusto e c'è una guardia che lo verifica.

set -e

MSG="${1:-Aggiornamento}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Guardia: dev'essere il repo del sito, non il repo legacy ─────────────────
ORIGIN="$(git remote get-url origin 2>/dev/null || echo '')"
case "$ORIGIN" in
  *sito-personale*) : ;;
  *)
    echo "✖ Abort: origin non è 'sito-personale' ma:"
    echo "    $ORIGIN"
    echo "  Lancia lo script dalla cartella 'SITO_WEB/Sito personale/'."
    exit 1 ;;
esac

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Aggiorna la copia di profilo.html usata dal deploy legacy standalone (innocua qui,
# tiene solo il file allineato alla homepage vera in profilo/index.html).
cp "$ROOT/profilo/index.html" "$ROOT/portafoglio/profilo.html"

# I market_data*.pkl sono cache riscritta dallo scheduler notturno: restano fuori dai
# commit automatici (su DO li ripristina R2). Per aggiornare i dati seed, committali a
# mano prima di lanciare lo script.
echo "▶ Pubblico su andreacappelletti.app (repo sito-personale, branch $BRANCH)..."
git add -A -- ':!portafoglio/sessions/market_data*.pkl' ':!sessions/'

if git diff --cached --quiet; then
  echo "  Nessuna modifica da pubblicare."
else
  git commit -m "$MSG"
  git push origin "$BRANCH"
  echo "  ✓ Push effettuato: DigitalOcean sta ricostruendo (un paio di minuti)."
fi

echo ""
echo "✅ Fatto. Verifica su https://andreacappelletti.app/ quando il build è finito."
