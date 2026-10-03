"""Pannello amministratore — i dati salvati dagli utenti registrati.

SOLA LETTURA, e niente impersonificazione: le pagine leggono le cartelle
`sessions/<utente>/`, calcolano le statistiche sul posto e non scrivono un byte.
Non toccano mai `session['username']`: se l'amministratore "diventasse" il
cliente, un suo giro su una dashboard (che salva in automatico su
`current.json` e `working.pkl`) sovrascriverebbe il lavoro di quel cliente.

Le fonti dei dati di un utente sono quattro, e vanno lette tutte perché
coesistono due modelli di salvataggio (vedi CLAUDE.md):
  1. `sessions/<u>/current.json`   — modello `data_core` (titoli, prezzi,
     rendimenti, spunte, pesi P1/P2/P3). È l'unica fonte che hanno TUTTI.
  2. `sessions/<u>/*.pkl`          — modello `sessions_manager`: `working.pkl`
     (sessione aperta) e i salvataggi nominati, ognuno con i propri pesi.
  3. `sessions/<u>/portfolios.json`— profili di portafogli (asset → peso).
  4. `sessions/<u>/analyses.json`  — analisi salvate (un portafoglio ciascuna).

Le statistiche escono da un solo motore (`_statistiche`), così i numeri delle
pagine sono confrontabili fra dati correnti e salvataggi.
"""

import html as _html
import math
import os
import urllib.parse
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import redirect, session, Response, abort

import auth
import data_core
import sessions_manager as sm

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
_SESSIONS = ROOT / 'sessions'
GIORNI_ANNO = 252

# Cartelle che stanno sotto sessions/ ma non sono utenti: i prefissi di
# servizio ('_alphavantage', '_mercato', ...) e la sessione anonima.
_NON_UTENTI = {'anon'}

_HEAD = ''          # riempito da registra_rotte(): l'HTML_HEAD di wsgi.py


# ═════════════════════════════════════════════════════════════════════════════
# Formattazione (italiana: virgola decimale, punto per le migliaia)
# ═════════════════════════════════════════════════════════════════════════════
def _e(x):
    return _html.escape(str(x if x is not None else ''), quote=True)


def _vuoto(x):
    if x is None:
        return True
    try:
        return isinstance(x, float) and (math.isnan(x) or math.isinf(x))
    except Exception:
        return False


def _n(x, dec=2, suff=''):
    """1234.5 → '1.234,50'."""
    if _vuoto(x):
        return '—'
    try:
        s = f'{float(x):,.{dec}f}'
    except Exception:
        return _e(x)
    return s.replace(',', '§').replace('.', ',').replace('§', '.') + suff


def _pc(x, dec=2, segno=False):
    """Frazione → percentuale. segno=True mette il + davanti ai positivi."""
    if _vuoto(x):
        return '—'
    v = float(x) * 100
    testo = _n(v, dec, '%')
    return ('+' + testo) if (segno and v > 0) else testo


def _cls(x):
    if _vuoto(x):
        return ''
    return 'pos' if float(x) > 0 else ('neg' if float(x) < 0 else '')


def _data(x):
    """Timestamp, date o stringa ISO → 'gg/mm/aaaa'."""
    if x is None or x == '':
        return '—'
    try:
        return pd.Timestamp(x).strftime('%d/%m/%Y')
    except Exception:
        return _e(str(x)[:10])


def _f(x):
    try:
        v = float(x)
        return 0.0 if math.isnan(v) else v
    except Exception:
        return 0.0


def _q(x):
    return urllib.parse.quote(str(x), safe='')


# ═════════════════════════════════════════════════════════════════════════════
# Motore delle statistiche (matematica pura)
# ═════════════════════════════════════════════════════════════════════════════
def _statistiche(rend):
    """Statistiche di una serie di rendimenti giornalieri.

    Serve meno di un mese e mezzo di dati per dire qualcosa: sotto i 30 giorni
    restituisce None e la pagina scrive '—' invece di un numero inventato.
    """
    try:
        s = pd.Series(rend).dropna()
    except Exception:
        return None
    if len(s) < 30:
        return None
    eq = (1.0 + s).cumprod()
    finale = float(eq.iloc[-1])
    anni = len(s) / GIORNI_ANNO
    cagr = (finale ** (1 / anni) - 1) if (anni > 0 and finale > 0) else None
    sd = float(s.std()) * math.sqrt(GIORNI_ANNO)
    mdd = float((eq / eq.cummax() - 1).min())
    return {
        'n': len(s), 'anni': anni,
        'dal': s.index[0], 'al': s.index[-1],
        'cagr': cagr, 'sd': sd,
        'sharpe': (cagr / sd) if (cagr is not None and sd) else None,
        'mdd': mdd, 'tot': finale - 1,
    }


def _rend_portafoglio(cr, pesi):
    """Serie del portafoglio e pesi effettivamente usati.

    Ribilanciamento giornaliero (somma pesata dei rendimenti) e finestra
    comune: si tengono solo i giorni in cui TUTTI i titoli con peso hanno un
    dato, altrimenti un titolo nato nel 2023 falserebbe gli anni precedenti.
    I pesi si normalizzano a 100: l'utente non li fa quasi mai quadrare.
    """
    if cr is None or getattr(cr, 'empty', True):
        return None, {}, []
    usati, fuori = {}, []
    for a, p in (pesi or {}).items():
        if _f(p) <= 0:
            continue
        if a in cr.columns:
            usati[a] = _f(p)
        else:
            fuori.append(a)
    if not usati:
        return None, {}, fuori
    tot = sum(usati.values())
    w = pd.Series({a: p / tot for a, p in usati.items()})
    sub = cr[list(usati)].dropna()
    if sub.empty:
        return None, usati, fuori
    return sub.mul(w, axis=1).sum(axis=1), usati, fuori


# ═════════════════════════════════════════════════════════════════════════════
# Lettura dei dati di un utente
# ═════════════════════════════════════════════════════════════════════════════
def _cartelle_dati():
    """Gli utenti che hanno una cartella sotto sessions/ (prefissi di servizio
    e sessione anonima esclusi)."""
    if not _SESSIONS.is_dir():
        return []
    out = []
    for d in sorted(_SESSIONS.iterdir()):
        if d.is_dir() and not d.name.startswith('_') and d.name not in _NON_UTENTI:
            out.append(d.name)
    return out


def _valido(utente):
    """Difesa dal path traversal: il nome deve essere una cartella esistente
    sotto sessions/, non un pezzo di percorso."""
    u = (utente or '').strip()
    if not u or '/' in u or '\\' in u or u.startswith('.') or u.startswith('_'):
        return False
    return (_SESSIONS / u).is_dir()


_CACHE_RIEP = {}        # utente → (firma_file, riepilogo)


def _riepilogo(utente):
    """Riga di sintesi per l'elenco. current.json pesa qualche MB a utente,
    quindi il risultato sta in cache finché il file non cambia (firma =
    data di modifica + dimensione)."""
    d = _SESSIONS / utente
    cur = d / 'current.json'
    try:
        st = cur.stat()
        # Nella firma entra anche la data della cartella: cambia quando l'utente
        # aggiunge o toglie un salvataggio, che current.json non registra.
        firma = (st.st_mtime, st.st_size, d.stat().st_mtime)
    except Exception:
        firma = None
    vecchio = _CACHE_RIEP.get(utente)
    if vecchio and vecchio[0] == firma:
        return vecchio[1]

    r = {'titoli': 0, 'spuntati': 0, 'pesi': {'P1': 0, 'P2': 0, 'P3': 0},
         'ultimo': None, 'aggiornato': None, 'mb': 0.0, 'tipo': '',
         'salvataggi': 0, 'working': False, 'profili': 0, 'portafogli': 0,
         'analisi': 0, 'errore': ''}
    if firma:
        r['mb'] = round(firma[1] / 1048576, 1)
        r['aggiornato'] = datetime.fromtimestamp(firma[0])
        try:
            with open(cur) as f:
                import json as _json
                raw = _json.load(f)
            r['tipo'] = str(raw.get('_tipo', '') or '')
            for k, v in raw.items():
                if not isinstance(v, dict):
                    continue
                r['titoli'] += 1
                if v.get('checked'):
                    r['spuntati'] += 1
                for p in ('P1', 'P2', 'P3'):
                    if _f(v.get(p)) > 0:
                        r['pesi'][p] += 1
                dd = v.get('dates') or []
                if dd and (r['ultimo'] is None or dd[-1] > r['ultimo']):
                    r['ultimo'] = dd[-1]
        except Exception as e:
            r['errore'] = str(e)[:120]

    # I lettori di sessions_manager passano da user_dir(), che CREA la cartella:
    # su una pagina di sola lettura si riempirebbe sessions/ (e il bucket R2) di
    # cartelle vuote per ogni iscritto che non ha mai aperto una dashboard.
    if d.is_dir():
        try:
            r['salvataggi'] = len(sm.list_user_files(utente))
            r['working'] = (d / 'working.pkl').exists()
        except Exception:
            pass
        try:
            prof = sm.load_profiles(utente) or {}
            r['profili'] = len(prof)
            r['portafogli'] = sum(len((v or {}).get('portfolios', {}))
                                  for v in prof.values())
        except Exception:
            pass
        try:
            r['analisi'] = len(sm.list_analyses(utente))
        except Exception:
            pass

    _CACHE_RIEP[utente] = (firma, r)
    return r


def _carica(utente):
    """Un solo parse di current.json → (assets, rendimenti, prezzi, meta).

    Fa il lavoro di `data_core.read_current` + `build_dataset` leggendo il file
    una volta sola: sono 3 MB a utente e la pagina li userebbe entrambi.
    """
    import json as _json
    try:
        with open(_SESSIONS / utente / 'current.json') as f:
            raw = _json.load(f)
    except Exception:
        return {}, None, None, {}
    meta = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    assets = {k: v for k, v in raw.items() if isinstance(v, dict)}
    rcols, pcols = {}, {}
    for a, v in assets.items():
        dates = v.get('dates')
        if not dates:
            continue
        try:
            idx = pd.to_datetime(dates)
        except Exception:
            continue
        if v.get('returns') and len(v['returns']) == len(dates):
            rcols[a] = pd.Series(v['returns'], index=idx)
        if v.get('prices') and len(v['prices']) == len(dates):
            pcols[a] = pd.Series(v['prices'], index=idx)
    cr = data_core.clip_returns(pd.DataFrame(rcols).sort_index()) if rcols else None
    op = pd.DataFrame(pcols).sort_index() if pcols else None
    return assets, cr, op, meta


def _pesi_correnti(assets):
    """{'P1': {asset: peso}, ...} dai campi P1/P2/P3 di current.json."""
    return {p: {a: _f(v.get(p)) for a, v in assets.items() if _f(v.get(p)) > 0}
            for p in ('P1', 'P2', 'P3')}


def _da_pkl(d):
    """(rendimenti, ticker_map, pesi, info) da un salvataggio pickle."""
    if not isinstance(d, dict):
        return None, {}, {}, {}
    cr = d.get('close_returns')
    if not isinstance(cr, pd.DataFrame) or cr.empty:
        cr = None
    cr = data_core.clip_returns(cr)
    st = d.get('_stores') or {}
    pesi = {p: (st.get('weights-store-' + p) or {}) for p in ('P1', 'P2', 'P3')}
    info = {
        'nome':     d.get('_saved_name') or '',
        'salvato':  (d.get('_saved_at') or d.get('saved_at') or
                     d.get('_working_saved_at') or ''),
        'da':       d.get('_saved_by') or '',
        'fonte':    d.get('_source') or '',
        'in_sospeso': bool(d.get('_has_unsaved_changes')),
        'ultimo_salvataggio': d.get('_last_named_save_name') or '',
    }
    return cr, (d.get('ticker_map') or {}), pesi, info


# ═════════════════════════════════════════════════════════════════════════════
# HTML
# ═════════════════════════════════════════════════════════════════════════════
_CSS = """
    .container { max-width: 1400px; }
    .bread { font-size: 12px; color: #90b8d8; margin-bottom: 18px; }
    .bread a { color: #90d8f8; text-decoration: none; }
    .bread a:hover { color: #fff; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    .pos { color: #8ce39a; }
    .neg { color: #f2938f; }
    .muted { color: #7fa0bd; font-size: 11px; }
    h3 { font-size: 13px; color: #cfe3f5; margin: 26px 0 8px; font-weight: 600; }
    .comp { font-size: 11px; color: #9fc0dc; padding: 2px 2px 14px; }
    .btn-link { display: inline-block; padding: 4px 10px; border-radius: 4px;
                background: #254e7a; color: #e0e8f0; font-size: 12px;
                text-decoration: none; }
    .btn-link:hover { background: #336295; color: #fff; }
    .nota { background: #15304d; border-left: 3px solid #336295;
            padding: 10px 14px; border-radius: 4px; font-size: 12px;
            color: #b9d2e8; margin: 8px 0 20px; }
    table + h3 { margin-top: 30px; }
"""

_INTEST_STAT = ('<th class="num">Dal</th><th class="num">Al</th>'
                '<th class="num">Rend. annuo</th><th class="num">Volatilità</th>'
                '<th class="num">Sharpe</th><th class="num">Perdita max</th>'
                '<th class="num">Cumulato</th>')


def _celle_stat(s):
    """Le sette colonne di statistica, uguali in tutte le tabelle."""
    if not s:
        return '<td class="num muted" colspan="7">storico troppo corto</td>'
    return (f'<td class="num">{_data(s["dal"])}</td>'
            f'<td class="num">{_data(s["al"])}</td>'
            f'<td class="num {_cls(s["cagr"])}">{_pc(s["cagr"], 2, True)}</td>'
            f'<td class="num">{_pc(s["sd"])}</td>'
            f'<td class="num">{_n(s["sharpe"])}</td>'
            f'<td class="num neg">{_pc(s["mdd"])}</td>'
            f'<td class="num {_cls(s["tot"])}">{_pc(s["tot"], 1, True)}</td>')


def _composizione(pesi, fuori=None):
    """«Az. ACWI 30,0% · OB. Global 20,0%» + i titoli non più in archivio."""
    if not pesi:
        return ''
    tot = sum(pesi.values()) or 1
    voci = ' · '.join(f'{_e(a)} <b>{_n(p / tot * 100, 1, "%")}</b>'
                      for a, p in sorted(pesi.items(), key=lambda x: -x[1]))
    extra = ''
    if fuori:
        extra = (f'<br><span class="neg">Non più in archivio: '
                 f'{_e(", ".join(sorted(fuori)))}</span>')
    return f'<div class="comp">{voci}{extra}</div>'


def _pagina(titolo, briciole, corpo, u):
    testa = (_HEAD.replace('</style>', _CSS + '  </style>', 1)
             if '</style>' in _HEAD else _HEAD + f'<style>{_CSS}</style>')
    return testa + f"""
  <div class="topbar">
    <h1>{titolo}</h1>
    <div>
      <a href="/admin">Gestione utenti</a>
      &nbsp;&nbsp;<a href="/">← Dashboard</a>
      &nbsp;&nbsp;<a href="/logout">Esci ({_e(u)})</a>
    </div>
  </div>
  <div class="container">
    <div class="bread">{briciole}</div>
    {corpo}
  </div>
</body>
</html>"""


def _tabella_portafogli(cr, gruppi):
    """Tabella «un portafoglio per riga» + composizione sotto ciascuna.
    gruppi = [(etichetta, {asset: peso})]."""
    righe = ''
    for etichetta, pesi in gruppi:
        serie, usati, fuori = _rend_portafoglio(cr, pesi)
        if not usati and not fuori:
            continue
        s = _statistiche(serie) if serie is not None else None
        righe += (f'<tr><td><b>{_e(etichetta)}</b>'
                  f'{_composizione(usati, fuori)}</td>'
                  f'<td class="num">{len(usati)}</td>'
                  f'<td class="num">{_n(sum(usati.values()), 1, "%")}</td>'
                  f'{_celle_stat(s)}</tr>')
    if not righe:
        return '<div class="nota">Nessun portafoglio con pesi assegnati.</div>'
    return (f'<table><thead><tr><th>Portafoglio</th>'
            f'<th class="num">Titoli</th><th class="num">Somma pesi</th>'
            f'{_INTEST_STAT}</tr></thead><tbody>{righe}</tbody></table>')


def _tabella_titoli(assets, cr, ticker_map=None):
    """Un titolo per riga, con le sue statistiche. Prima gli spuntati."""
    if not assets:
        return '<div class="nota">Nessun titolo in archivio.</div>'
    tm = ticker_map or {}
    righe = ''
    ordinati = sorted(assets.items(),
                      key=lambda kv: (not kv[1].get('checked'), kv[0].lower()))
    for a, v in ordinati:
        s = _statistiche(cr[a]) if (cr is not None and a in cr.columns) else None
        prezzi = v.get('prices') or []
        ultimo = next((p for p in reversed(prezzi) if p is not None), None)
        spunta = '✓' if v.get('checked') else ''
        pesi = ''.join(f'<td class="num">{_n(v.get(p), 1) if _f(v.get(p)) else ""}</td>'
                       for p in ('P1', 'P2', 'P3'))
        righe += (f'<tr><td>{_e(a)}</td>'
                  f'<td>{_e(v.get("ticker") or tm.get(a) or "")}</td>'
                  f'<td>{_e(v.get("currency") or "")}</td>'
                  f'<td class="num">{_n(ultimo)}</td>'
                  f'<td class="num">{spunta}</td>{pesi}{_celle_stat(s)}</tr>')
    return (f'<table><thead><tr><th>Descrizione</th><th>Ticker</th>'
            f'<th>Valuta</th><th class="num">Ultimo prezzo</th>'
            f'<th class="num">Usato</th><th class="num">P1</th>'
            f'<th class="num">P2</th><th class="num">P3</th>'
            f'{_INTEST_STAT}</tr></thead><tbody>{righe}</tbody></table>')


def _assets_da_pkl(d, cr, tm, pesi):
    """Pseudo-`assets` (lo stesso formato di current.json) da un salvataggio,
    così la tabella dei titoli è una sola per tutte le pagine."""
    op = d.get('original_prices') if isinstance(d, dict) else None
    vm = (d.get('valuta_map') or {}) if isinstance(d, dict) else {}
    sel = ((d.get('_stores') or {}).get('global-assets-selected') or []
           if isinstance(d, dict) else [])
    colonne = list(cr.columns) if cr is not None else list(tm.keys())
    out = {}
    for a in colonne:
        ultimo = None
        if isinstance(op, pd.DataFrame) and a in op.columns:
            s = op[a].dropna()
            if not s.empty:
                try:
                    ultimo = float(s.iloc[-1])
                except Exception:
                    ultimo = None
        out[a] = {'ticker': tm.get(a, ''), 'currency': vm.get(a, ''),
                  'checked': a in sel,
                  'prices': [ultimo] if ultimo is not None else [],
                  'P1': pesi.get('P1', {}).get(a),
                  'P2': pesi.get('P2', {}).get(a),
                  'P3': pesi.get('P3', {}).get(a)}
    return out


def _card(num, lbl):
    return f'<div class="stat-card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>'


# ═════════════════════════════════════════════════════════════════════════════
# Pagina 1 — elenco degli utenti
# ═════════════════════════════════════════════════════════════════════════════
def _pagina_elenco(u):
    try:
        registrati = {x['username']: x for x in auth.list_users()}
    except Exception:
        registrati = {}
    cartelle = _cartelle_dati()
    nomi = sorted(set(cartelle) | set(registrati), key=str.lower)

    righe, tot_titoli, tot_salv, con_dati = '', 0, 0, 0
    for nome in nomi:
        ha_cartella = nome in cartelle
        r = _riepilogo(nome) if ha_cartella else None
        reg = registrati.get(nome)
        if r and r['titoli']:
            con_dati += 1
        tot_titoli += (r['titoli'] if r else 0)
        tot_salv += ((r['salvataggi'] + (1 if r['working'] else 0)) if r else 0)

        if reg:
            stato = reg.get('status', '')
            badge = {'active': 'badge-active', 'pending': 'badge-pending',
                     'suspended': 'badge-suspended'}.get(stato, 'badge-active')
            cella_stato = (f'<span class="badge {badge}">{_e(stato)}</span>'
                           f' <span class="muted">{_e(reg.get("plan", ""))}</span>')
        else:
            # Cartella rimasta da un account cancellato: i dati restano, e vanno
            # visti, altrimenti si perdono senza che nessuno lo sappia.
            cella_stato = '<span class="muted">non registrato</span>'

        if not r:
            righe += (f'<tr><td><b>{_e(nome)}</b></td><td>{cella_stato}</td>'
                      f'<td class="num muted" colspan="10">nessun dato salvato</td>'
                      f'<td></td></tr>')
            continue
        azione = (f'<a class="btn-link" href="/admin/dati/{_q(nome)}">Apri</a>'
                  if (r['titoli'] or r['salvataggi'] or r['working']) else '')
        righe += (
            f'<tr><td><b>{_e(nome)}</b>'
            + (f'<div class="muted">{_e(r["tipo"])}</div>' if r['tipo'] else '')
            + f'</td><td>{cella_stato}</td>'
            f'<td class="num">{r["titoli"] or ""}</td>'
            f'<td class="num">{r["spuntati"] or ""}</td>'
            f'<td class="num">{r["pesi"]["P1"] or ""}</td>'
            f'<td class="num">{r["pesi"]["P2"] or ""}</td>'
            f'<td class="num">{r["pesi"]["P3"] or ""}</td>'
            f'<td class="num">{r["portafogli"] or ""}</td>'
            f'<td class="num">{r["analisi"] or ""}</td>'
            f'<td class="num">{r["salvataggi"] or ""}'
            + ('<span class="muted"> +sessione</span>' if r['working'] else '')
            + f'</td><td class="num">{_data(r["ultimo"])}</td>'
            f'<td class="num muted">{_data(r["aggiornato"])}</td>'
            f'<td>{azione}</td></tr>')

    corpo = f"""
    <div class="stats">
      {_card(len(registrati), 'Utenti registrati')}
      {_card(con_dati, 'Con titoli in archivio')}
      {_card(tot_titoli, 'Titoli salvati in tutto')}
      {_card(tot_salv, 'Salvataggi (file)')}
    </div>
    <div class="nota">
      Sola lettura: queste pagine leggono le cartelle degli utenti e calcolano
      le statistiche sul posto, non scrivono niente e non entrano nelle
      dashboard al posto loro. <b>Titoli</b> e <b>pesi P1/P2/P3</b> vengono da
      <i>current.json</i> (l'archivio che hanno tutti), <b>portafogli</b> e
      <b>analisi</b> dai file salvati a parte.
    </div>
    <h2>Dati degli utenti</h2>
    <table>
      <thead><tr>
        <th>Utente</th><th>Stato</th>
        <th class="num">Titoli</th><th class="num">Usati</th>
        <th class="num">P1</th><th class="num">P2</th><th class="num">P3</th>
        <th class="num">Portaf.</th><th class="num">Analisi</th>
        <th class="num">Salvataggi</th><th class="num">Prezzi al</th>
        <th class="num">Modificato</th><th></th>
      </tr></thead>
      <tbody>{righe}</tbody>
    </table>"""
    return _pagina('A·C Dashboard — Dati utenti',
                   '<a href="/admin">Admin</a> › Dati utenti', corpo, u)


# ═════════════════════════════════════════════════════════════════════════════
# Pagina 2 — un utente
# ═════════════════════════════════════════════════════════════════════════════
def _pagina_utente(u, utente):
    assets, cr, op, meta = _carica(utente)
    pesi = _pesi_correnti(assets)
    r = _riepilogo(utente)

    # ── portafogli correnti (i pesi dentro current.json)
    gruppi = [(f'{p} — portafoglio {p[-1]}', pesi[p]) for p in ('P1', 'P2', 'P3')]
    html_port = _tabella_portafogli(cr, gruppi)

    # ── portafogli dei profili salvati, misurati sui prezzi di oggi
    blocchi_prof = ''
    try:
        profili = sm.load_profiles(utente) or {}
    except Exception:
        profili = {}
    for nome_prof, corpo_prof in sorted(profili.items()):
        ports = (corpo_prof or {}).get('portfolios', {}) or {}
        if not ports:
            continue
        blocchi_prof += (
            f'<h3>Profilo «{_e(nome_prof)}» '
            f'<span class="muted">salvato il {_data((corpo_prof or {}).get("saved_at"))}'
            f' — {len(ports)} portafogli</span></h3>'
            + _tabella_portafogli(cr, sorted(ports.items())))

    # ── analisi salvate (un portafoglio ciascuna)
    try:
        analisi = sm.load_analyses(utente) or {}
    except Exception:
        analisi = {}
    html_analisi = ''
    if analisi:
        voci = [(f'{k} ({_data((v or {}).get("saved_at"))})',
                 (v or {}).get('weights', {}) or {})
                for k, v in sorted(analisi.items(),
                                   key=lambda kv: (kv[1] or {}).get('saved_at', ''),
                                   reverse=True)]
        html_analisi = ('<h3>Analisi salvate</h3>'
                        + _tabella_portafogli(cr, voci))

    # ── file salvati
    try:
        files = sm.list_user_files(utente)
    except Exception:
        files = []
    righe_file = ''
    wp = _SESSIONS / utente / 'working.pkl'
    if wp.exists():
        st = wp.stat()
        righe_file += (
            f'<tr><td><b>Sessione aperta</b><div class="muted">working.pkl</div></td>'
            f'<td class="num muted">—</td>'
            f'<td class="num">{_data(datetime.fromtimestamp(st.st_mtime))}</td>'
            f'<td class="num">{_n(st.st_size / 1024, 0, " kB")}</td>'
            f'<td><a class="btn-link" href="/admin/dati/{_q(utente)}'
            f'/salvataggio/working.pkl">Analizza</a></td></tr>')
    for f in files:
        righe_file += (
            f'<tr><td><b>{_e(f["saved_name"])}</b>'
            f'<div class="muted">{_e(f["filename"])}</div></td>'
            f'<td class="num">{_data(f["saved_at"])}</td>'
            f'<td class="num">{_e(f["modified"])}</td>'
            f'<td class="num">{_n(f["size_kb"], 0, " kB")}</td>'
            f'<td><a class="btn-link" href="/admin/dati/{_q(utente)}'
            f'/salvataggio/{_q(f["filename"])}">Analizza</a></td></tr>')
    if righe_file:
        html_file = (
            '<h3>Salvataggi dell\'utente</h3>'
            '<table><thead><tr><th>Nome</th><th class="num">Salvato il</th>'
            '<th class="num">File modificato</th><th class="num">Peso</th>'
            f'<th></th></tr></thead><tbody>{righe_file}</tbody></table>'
            f'<p style="margin-top:10px"><a class="btn-link" '
            f'href="/admin/dati/{_q(utente)}/salvataggi">'
            f'▶ Analizza tutti i salvataggi insieme</a></p>')
    else:
        html_file = ('<h3>Salvataggi dell\'utente</h3>'
                     '<div class="nota">Nessun file salvato: '
                     'questo utente lavora solo sull\'archivio corrente.</div>')

    nota_vuoto = ''
    if not assets:
        nota_vuoto = ('<div class="nota neg">Archivio corrente vuoto o non '
                      'leggibile' + (f': {_e(r["errore"])}' if r['errore'] else '')
                      + '.</div>')

    corpo = f"""
    <div class="stats">
      {_card(r['titoli'], 'Titoli in archivio')}
      {_card(r['spuntati'], 'Usati nel piano')}
      {_card(r['portafogli'], 'Portafogli salvati')}
      {_card(r['analisi'], 'Analisi salvate')}
      {_card(r['salvataggi'] + (1 if r['working'] else 0), 'File salvati')}
      {_card(_data(r['ultimo']), 'Prezzi aggiornati al')}
    </div>
    {nota_vuoto}
    <div class="nota">
      Rendimento, volatilità e perdita massima sono calcolati sui rendimenti
      giornalieri salvati nell'archivio dell'utente (già in euro e tagliati a
      ±50% al giorno), con <b>ribilanciamento giornaliero</b> dei pesi
      normalizzati a 100 e sulla <b>finestra comune</b> a tutti i titoli del
      portafoglio. <a class="btn-link" href="/admin/dati/{_q(utente)}/export.xlsx">
      ⬇ Scarica Excel (titoli + prezzi)</a>
    </div>
    <h2>Portafogli dell'archivio corrente</h2>
    {html_port}
    {blocchi_prof}
    {html_analisi}
    {html_file}
    <h3>Titoli in archivio</h3>
    {_tabella_titoli(assets, cr)}"""
    return _pagina(f'Dati di {_e(utente)}',
                   f'<a href="/admin">Admin</a> › '
                   f'<a href="/admin/dati">Dati utenti</a> › {_e(utente)}',
                   corpo, u)


# ═════════════════════════════════════════════════════════════════════════════
# Pagina 3 — tutti i salvataggi di un utente, a confronto
# ═════════════════════════════════════════════════════════════════════════════
def _elenco_file(utente):
    """I file analizzabili: la sessione aperta per prima, poi i salvataggi
    nominati dal più recente."""
    out = []
    if (_SESSIONS / utente / 'working.pkl').exists():
        out.append(('working.pkl', 'Sessione aperta', ''))
    try:
        for f in sm.list_user_files(utente):
            out.append((f['filename'], f['saved_name'], f['saved_at']))
    except Exception:
        pass
    return out


def _pagina_salvataggi(u, utente):
    righe = ''
    for nome_file, etichetta, salvato in _elenco_file(utente):
        try:
            d = sm.load_user_as_admin(utente, nome_file)
        except Exception as e:
            righe += (f'<tr><td><b>{_e(etichetta)}</b></td>'
                      f'<td class="neg" colspan="10">illeggibile: {_e(e)}</td></tr>')
            continue
        cr, tm, pesi, info = _da_pkl(d)
        link = (f'<a href="/admin/dati/{_q(utente)}/salvataggio/{_q(nome_file)}">'
                f'{_e(etichetta)}</a>')
        n_titoli = len(cr.columns) if cr is not None else 0
        vuoto = True
        for p in ('P1', 'P2', 'P3'):
            serie, usati, fuori = _rend_portafoglio(cr, pesi[p])
            if not usati:
                continue
            vuoto = False
            s = _statistiche(serie) if serie is not None else None
            righe += (f'<tr><td><b>{link}</b>'
                      f'<div class="muted">{_data(info["salvato"] or salvato)}'
                      f' — {n_titoli} titoli</div></td>'
                      f'<td>{p}{_composizione(usati, fuori)}</td>'
                      f'<td class="num">{len(usati)}</td>{_celle_stat(s)}</tr>')
        if vuoto:
            righe += (f'<tr><td><b>{link}</b>'
                      f'<div class="muted">{_data(info["salvato"] or salvato)}'
                      f' — {n_titoli} titoli</div></td>'
                      f'<td class="muted" colspan="9">nessun peso assegnato '
                      f'in questo salvataggio</td></tr>')
    if not righe:
        righe = '<tr><td colspan="11" class="muted">Nessun salvataggio.</td></tr>'

    corpo = f"""
    <div class="nota">
      Ogni riga è un portafoglio (P1, P2, P3) dentro un salvataggio, misurato
      sui prezzi <b>contenuti in quel file</b> — non su quelli di oggi: così si
      vede il portafoglio come l'utente lo aveva davanti quando ha salvato.
      Ribilanciamento giornaliero, pesi normalizzati a 100, finestra comune ai
      titoli del portafoglio.
    </div>
    <h2>Salvataggi di {_e(utente)}</h2>
    <table>
      <thead><tr><th>Salvataggio</th><th>Portafoglio</th>
      <th class="num">Titoli</th>{_INTEST_STAT}</tr></thead>
      <tbody>{righe}</tbody>
    </table>"""
    return _pagina(f'Salvataggi di {_e(utente)}',
                   f'<a href="/admin">Admin</a> › '
                   f'<a href="/admin/dati">Dati utenti</a> › '
                   f'<a href="/admin/dati/{_q(utente)}">{_e(utente)}</a> › Salvataggi',
                   corpo, u)


# ═════════════════════════════════════════════════════════════════════════════
# Pagina 4 — un singolo salvataggio
# ═════════════════════════════════════════════════════════════════════════════
def _pagina_salvataggio(u, utente, nome_file):
    try:
        d = sm.load_user_as_admin(utente, nome_file)
    except Exception as e:
        d = None
        errore = str(e)
    else:
        errore = '' if d else 'file vuoto o illeggibile'
    if not d:
        corpo = f'<div class="nota neg">{_e(nome_file)}: {_e(errore)}.</div>'
        return _pagina(f'Salvataggio di {_e(utente)}', '', corpo, u)

    cr, tm, pesi, info = _da_pkl(d)
    assets = _assets_da_pkl(d, cr, tm, pesi)
    gruppi = [(f'{p} — portafoglio {p[-1]}', pesi[p]) for p in ('P1', 'P2', 'P3')]
    etichetta = info['nome'] or nome_file
    sospeso = ('<div class="nota">Questa sessione ha <b>modifiche non '
               'salvate</b> rispetto all\'ultimo salvataggio nominato'
               + (f' («{_e(info["ultimo_salvataggio"])}»)'
                  if info['ultimo_salvataggio'] else '') + '.</div>'
               ) if info['in_sospeso'] else ''

    corpo = f"""
    <div class="stats">
      {_card(len(assets), 'Titoli nel file')}
      {_card(sum(1 for p in ('P1','P2','P3') if pesi[p]), 'Portafogli con pesi')}
      {_card(_data(info['salvato']), 'Salvato il')}
      {_card(_data(cr.index[-1]) if cr is not None and len(cr) else '—',
             'Prezzi fino al')}
    </div>
    {sospeso}
    <div class="nota">
      Fonte: <i>{_e(nome_file)}</i>
      {('— salvato da ' + _e(info['da'])) if info['da'] else ''}
      {('— origine dati: ' + _e(info['fonte'])) if info['fonte'] else ''}.
      I numeri sono calcolati sui prezzi contenuti nel file.
    </div>
    <h2>Portafogli del salvataggio «{_e(etichetta)}»</h2>
    {_tabella_portafogli(cr, gruppi)}
    <h3>Titoli del salvataggio</h3>
    {_tabella_titoli(assets, cr, tm)}"""
    return _pagina(f'«{_e(etichetta)}» — {_e(utente)}',
                   f'<a href="/admin">Admin</a> › '
                   f'<a href="/admin/dati">Dati utenti</a> › '
                   f'<a href="/admin/dati/{_q(utente)}">{_e(utente)}</a> › '
                   f'<a href="/admin/dati/{_q(utente)}/salvataggi">Salvataggi</a> › '
                   f'{_e(etichetta)}',
                   corpo, u)


# ═════════════════════════════════════════════════════════════════════════════
# Rotte
# ═════════════════════════════════════════════════════════════════════════════
def _file_consentito(utente, nome_file):
    """Whitelist: solo la sessione aperta o un salvataggio nominato di QUESTO
    utente. Senza questo controllo il nome del file sarebbe un percorso
    arbitrario da aprire con pickle."""
    if not nome_file or '/' in nome_file or '\\' in nome_file or '..' in nome_file:
        return False
    if nome_file == 'working.pkl':
        return (_SESSIONS / utente / nome_file).exists()
    try:
        return nome_file in {f['filename'] for f in sm.list_user_files(utente)}
    except Exception:
        return False


def registra_rotte(flask_server, html_head=''):
    """Aggancia le pagine al server Flask che serve già /admin."""
    global _HEAD
    _HEAD = html_head or _HEAD

    def _chi():
        """L'amministratore collegato, None se chi chiede non lo è."""
        nome = session.get('username')
        utente = auth.get_user(nome) if nome else None
        if not utente or utente.get('role') != 'admin':
            return None
        return nome

    @flask_server.route('/admin/dati')
    def _admin_dati():
        u = _chi()
        if not u:
            return redirect('/')
        return _pagina_elenco(u)

    @flask_server.route('/admin/dati/<utente>')
    def _admin_dati_utente(utente):
        u = _chi()
        if not u:
            return redirect('/')
        if not _valido(utente):
            abort(404)
        return _pagina_utente(u, utente)

    @flask_server.route('/admin/dati/<utente>/salvataggi')
    def _admin_dati_salvataggi(utente):
        u = _chi()
        if not u:
            return redirect('/')
        if not _valido(utente):
            abort(404)
        return _pagina_salvataggi(u, utente)

    @flask_server.route('/admin/dati/<utente>/salvataggio/<nome_file>')
    def _admin_dati_salvataggio(utente, nome_file):
        u = _chi()
        if not u:
            return redirect('/')
        if not _valido(utente) or not _file_consentito(utente, nome_file):
            abort(404)
        return _pagina_salvataggio(u, utente, nome_file)

    @flask_server.route('/admin/dati/<utente>/export.xlsx')
    def _admin_dati_export(utente):
        u = _chi()
        if not u:
            return redirect('/')
        if not _valido(utente):
            abort(404)
        try:
            dati = data_core.export_bytes(username=utente)
        except Exception as e:
            return f'Export non riuscito: {_e(e)}', 500
        sicuro = ''.join(c if (c.isalnum() or c in '-_.') else '_'
                         for c in utente)
        return Response(
            dati,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition':
                     f'attachment; filename="dati_{sicuro}.xlsx"'})

    print('• [admin] pagine dati utenti su /admin/dati', flush=True)
