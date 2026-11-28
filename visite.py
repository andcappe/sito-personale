"""
Contatore delle visite al sito e alle dashboard, con la pagina /admin/visite.

Cosa conta
  - le pagine aperte: homepage (anche chi non ha fatto il login), pagine di login e
    registrazione, ogni dashboard (solo utenti col login: gli altri vengono mandati
    al login e contano lì). Una pagina = una richiesta GET di HTML all'indirizzo
    della pagina; i callback di Dash, i file statici e le figure non contano;
  - gli accessi (login con password, Google, Facebook) e le registrazioni.
  Bot, anteprime dei link (WhatsApp, Telegram…) e controlli automatici si scartano
  dallo User-Agent.

Privacy
  Nessun indirizzo IP viene salvato. Un visitatore anonimo è un'impronta di 12
  caratteri = HMAC(SECRET_KEY, giorno + IP + browser): cambia ogni giorno, quindi
  identifica la stessa persona solo dentro la giornata. Nessun cookie in più.
  Gli utenti col login si riconoscono dal nome utente.

Dove
  sessions/_visite/AAAA-MM-GG.jsonl (un evento per riga) e
  sessions/_visite/utenti.json (ultimo segnale di ogni utente), sincronizzati su R2.
  Gli eventi stanno in memoria e si scrivono ogni 5 minuti (e all'uscita del
  processo): un riavvio brusco ne perde al massimo 5 minuti. «Online adesso» è
  solo in memoria (gunicorn ha un worker solo, quindi è completo).
"""
import atexit
import hashlib
import hmac
import html
import json
import os
import re
import threading
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
CARTELLA = ROOT / 'sessions' / '_visite'
_FILE_UTENTI = CARTELLA / 'utenti.json'
_TZ = ZoneInfo('Europe/Rome')

_SCRIVI_OGNI = 300        # secondi fra due scritture su disco/R2
_ONLINE_MIN = 15          # «online adesso» = un segnale negli ultimi 15 minuti
_PERIODI = (1, 7, 30, 90, 365)

# Pagine contate: percorso → nome. Le dashboard contano a «/prefisso» e «/prefisso/».
_PAGINE = {
    '/': 'Homepage', '/sito': 'Homepage',
    '/login': 'Login', '/register': 'Registrazione',
}
_DASHBOARD = {
    '/portafoglio':    'Analisi Strategica',
    '/analisitattica': 'Analisi Tattica',
    '/opzioni':        'Strategie Opzioni',
    '/calendario':     'Calendario',
    '/fondipensione':  'Clienti',
    '/fred':           'Macro Economia',
    '/frontiera':      'Frontiera',
    '/rendimenti':     'Rendimenti',
    '/macro':          'Macro (vecchia)',
}
_BOT = re.compile(
    r'bot|crawl|spider|slurp|preview|monitor|uptime|probe|health|curl|wget|python|'
    r'go-http|java/|okhttp|httpclient|headless|lighthouse|facebookexternalhit|'
    r'whatsapp|telegram|discord|slack|skype|bingpreview|scan|fetch', re.I)

_SEGRETO = os.environ.get('SECRET_KEY', 'cambia-questa-chiave-in-produzione').encode()

_LOCK = threading.Lock()
_BUFFER = []              # eventi non ancora scritti
_SEGNALI = {}             # chiave → {'t': epoch, 'chi', 'sez', 'disp'}: per «online adesso»
_ULTIMI_UTENTI = {}       # utente → ultimo segnale (iso), da fondere in utenti.json
_avviato = False


# ─── Raccolta ─────────────────────────────────────────────────────────────────

def _adesso():
    return datetime.now(_TZ)


def _sezione(path):
    """(nome, è_dashboard) se il percorso è una pagina da contare, altrimenti None."""
    if path in _PAGINE:
        return _PAGINE[path], False
    p = path.rstrip('/')
    if p in _DASHBOARD:
        return _DASHBOARD[p], True
    return None


def _sezione_di(path):
    """La dashboard a cui appartiene una richiesta qualsiasi (anche un callback)."""
    for pref, nome in _DASHBOARD.items():
        if path == pref or path.startswith(pref + '/'):
            return nome
    return _PAGINE.get(path, 'Sito')


def _dispositivo(ua):
    if re.search(r'iPad|Tablet', ua):
        return 'tablet'
    if re.search(r'Mobi|Android|iPhone', ua):
        return 'cellulare'
    return 'computer'


def _ip(request):
    xff = request.headers.get('X-Forwarded-For', '')
    return (xff.split(',')[0].strip() if xff else '') or (request.remote_addr or '')


def _impronta(request, giorno):
    dato = f"{giorno}|{_ip(request)}|{request.headers.get('User-Agent', '')}".encode()
    return hmac.new(_SEGRETO, dato, hashlib.sha256).hexdigest()[:12]


def _provenienza(request):
    """Dominio del sito da cui si arriva; vuoto se diretto o interno."""
    ref = request.referrer or ''
    if not ref:
        return ''
    host = (urllib.parse.urlsplit(ref).hostname or '').lower()
    proprio = (request.host or '').split(':')[0].lower()
    if not host or host == proprio:
        return ''
    return host[4:] if host.startswith('www.') else host


def _evento(request, tipo, sez, utente, **extra):
    ora = _adesso()
    ua = request.headers.get('User-Agent', '')
    ev = {
        't': ora.strftime('%Y-%m-%dT%H:%M:%S'),
        'tipo': tipo,
        'sez': sez,
        'u': utente or '',
        'v': _impronta(request, ora.strftime('%Y-%m-%d')),
        'disp': _dispositivo(ua),
    }
    ref = _provenienza(request)
    if ref:
        ev['ref'] = ref
    paese = request.headers.get('CF-IPCountry', '')
    if paese and paese not in ('XX', 'T1'):
        ev['paese'] = paese
    ev.update(extra)
    with _LOCK:
        _BUFFER.append(ev)
    return ev


def registra(request, session):
    """Da chiamare in before_request di ogni server montato. Non solleva mai."""
    try:
        if request.method != 'GET':
            return
        ua = request.headers.get('User-Agent', '')
        if not ua or _BOT.search(ua):
            return
        utente = session.get('username') or ''
        path = request.path

        # segnale di vita (anche i callback di Dash): serve a «online adesso»
        if utente or path in _PAGINE:
            chiave = f"u:{utente}" if utente else f"v:{_impronta(request, _adesso().strftime('%Y-%m-%d'))}"
            with _LOCK:
                _SEGNALI[chiave] = {'t': time.time(), 'chi': utente, 'sez': _sezione_di(path),
                                    'disp': _dispositivo(ua)}
                if utente:
                    _ULTIMI_UTENTI[utente] = _adesso().strftime('%Y-%m-%dT%H:%M:%S')

        trovata = _sezione(path)
        if not trovata:
            return
        sez, dashboard = trovata
        if dashboard and not utente:
            return                    # verrà mandato al login: conta lì
        if 'text/html' not in request.headers.get('Accept', ''):
            return
        extra = {}
        if request.args.get('embed'):
            extra['embed'] = 1        # Frontiera/Rendimenti aperti come tab del Portafoglio
        _evento(request, 'pagina', sez, utente, **extra)
    except Exception as e:
        print(f"⚠ [visite] registrazione non riuscita: {e}", flush=True)


def login(request, utente, metodo):
    """Accesso riuscito (metodo: password, google, facebook)."""
    try:
        _evento(request, 'login', 'Login', utente, metodo=metodo)
    except Exception as e:
        print(f"⚠ [visite] login non registrato: {e}", flush=True)


def registrazione(request, utente, metodo='password'):
    """Nuovo utente registrato."""
    try:
        _evento(request, 'registrazione', 'Registrazione', utente, metodo=metodo)
    except Exception as e:
        print(f"⚠ [visite] registrazione utente non registrata: {e}", flush=True)


# ─── Scrittura ────────────────────────────────────────────────────────────────

def _push(path):
    try:
        import cloud_storage
        cloud_storage.push(path)
    except Exception:
        pass


def _leggi_utenti():
    try:
        return json.loads(_FILE_UTENTI.read_text())
    except Exception:
        return {}


def scrivi():
    """Aggiunge gli eventi in memoria ai file del giorno e aggiorna utenti.json."""
    with _LOCK:
        eventi, _BUFFER[:] = list(_BUFFER), []
        ultimi = dict(_ULTIMI_UTENTI)
        _ULTIMI_UTENTI.clear()
        # «online adesso»: via i segnali vecchi di un giorno
        limite = time.time() - 86400
        for k in [k for k, s in _SEGNALI.items() if s['t'] < limite]:
            del _SEGNALI[k]
    if not eventi and not ultimi:
        return
    try:
        CARTELLA.mkdir(parents=True, exist_ok=True)
        per_giorno = defaultdict(list)
        for ev in eventi:
            per_giorno[ev['t'][:10]].append(ev)
        for giorno, evs in per_giorno.items():
            f = CARTELLA / f'{giorno}.jsonl'
            with open(f, 'a', encoding='utf-8') as fh:
                for ev in evs:
                    fh.write(json.dumps(ev, ensure_ascii=False) + '\n')
            _push(f)
        if ultimi:
            dati = _leggi_utenti()
            for u, t in ultimi.items():
                if t > dati.get(u, ''):
                    dati[u] = t
            tmp = str(_FILE_UTENTI) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(dati, fh, ensure_ascii=False)
            os.replace(tmp, _FILE_UTENTI)
            _push(_FILE_UTENTI)
    except Exception as e:
        # rimette gli eventi in coda: ci riprova al prossimo giro
        with _LOCK:
            _BUFFER[:0] = eventi
            for u, t in ultimi.items():
                if t > _ULTIMI_UTENTI.get(u, ''):
                    _ULTIMI_UTENTI[u] = t
        print(f"⚠ [visite] scrittura non riuscita: {e}", flush=True)


def avvia():
    """Thread che scrive ogni _SCRIVI_OGNI secondi, più una scrittura all'uscita."""
    global _avviato
    if _avviato:
        return
    _avviato = True

    def _giro():
        while True:
            time.sleep(_SCRIVI_OGNI)
            scrivi()

    threading.Thread(target=_giro, daemon=True, name='visite').start()
    atexit.register(scrivi)
    print("✓ [visite] contatore attivo — /admin/visite", flush=True)


# ─── Lettura ──────────────────────────────────────────────────────────────────

def eventi(giorni):
    """Eventi degli ultimi `giorni` giorni (oggi compreso), dal disco e dalla memoria."""
    oggi = _adesso().date()
    primo = (oggi - timedelta(days=giorni - 1)).isoformat()
    out = []
    for i in range(giorni):
        f = CARTELLA / f'{(oggi - timedelta(days=i)).isoformat()}.jsonl'
        if not f.exists():
            continue
        try:
            with open(f, encoding='utf-8') as fh:
                for riga in fh:
                    try:
                        out.append(json.loads(riga))
                    except ValueError:
                        pass
        except OSError:
            pass
    with _LOCK:
        out += [ev for ev in _BUFFER if ev['t'][:10] >= primo]
    out.sort(key=lambda ev: ev['t'])
    return out


def online():
    """Chi ha dato segni di vita negli ultimi _ONLINE_MIN minuti, il più recente prima."""
    limite = time.time() - _ONLINE_MIN * 60
    with _LOCK:
        vivi = [dict(s) for s in _SEGNALI.values() if s['t'] >= limite]
    return sorted(vivi, key=lambda s: -s['t'])


def ultimi_accessi():
    """utente → ultimo segnale (iso), dal file più quelli ancora in memoria."""
    dati = _leggi_utenti()
    with _LOCK:
        for u, t in _ULTIMI_UTENTI.items():
            if t > dati.get(u, ''):
                dati[u] = t
    return dati


# ─── Pagina /admin/visite ─────────────────────────────────────────────────────

_CSS = """
<style>
  .vi-filtri { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 24px; }
  .vi-filtri a { padding: 5px 12px; border-radius: 14px; background: #1a3a5c; color: #90b8d8;
                 text-decoration: none; font-size: 12px; font-weight: 600; }
  .vi-filtri a.on { background: #90d8f8; color: #0d2137; }
  .vi-filtri .sep { width: 12px; }
  .vi-sez { margin-bottom: 32px; }
  .vi-due { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }
  .vi-graf { background: #1a3a5c; border-radius: 8px; padding: 14px 14px 8px; }
  .vi-graf .tit { font-size: 12px; color: #90b8d8; margin-bottom: 8px; }
  .vi-barre { display: flex; align-items: flex-end; gap: 2px; height: 140px;
              border-bottom: 1px solid #254e7a; }
  .vi-barre div { flex: 1; background: #4a9fd8; border-radius: 3px 3px 0 0; min-height: 0; }
  .vi-barre div:hover { background: #90d8f8; }
  .vi-assi { display: flex; justify-content: space-between; font-size: 10px; color: #6f94b8;
             margin-top: 4px; }
  .vi-nota { font-size: 11px; color: #6f94b8; margin-top: 6px; line-height: 1.5; }
  .vi-scorri { overflow-x: auto; }
  .vi-scorri.largo table { min-width: 560px; }
  .vi-num { text-align: right; font-variant-numeric: tabular-nums; }
  .vi-mute { color: #6f94b8; }
  .vi-pallino { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                background: #66bb6a; margin-right: 6px; }
  @media (max-width: 600px) {
    .container { padding: 20px 12px !important; }
    .topbar { flex-wrap: wrap; gap: 6px; }
    .stat-card { padding: 12px 14px !important; }
    .stat-card .num { font-size: 22px !important; }
    th, td { padding: 8px 8px !important; }
  }
</style>
"""


def _e(x):
    return html.escape(str(x))


def _quando(iso):
    """«oggi 14:05», «ieri 09:12», «12/11 18:40», «12/11/2025»."""
    if not iso:
        return '—'
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return _e(iso)
    oggi = _adesso().date()
    if t.date() == oggi:
        return f"oggi {t:%H:%M}"
    if t.date() == oggi - timedelta(days=1):
        return f"ieri {t:%H:%M}"
    if t.year == oggi.year:
        return f"{t:%d/%m %H:%M}"
    return f"{t:%d/%m/%Y}"


def _barre(conteggi, giorni, titolo, unita):
    """Grafico a barre per giorno in HTML puro (una serie: niente legenda)."""
    oggi = _adesso().date()
    date = [oggi - timedelta(days=i) for i in range(giorni - 1, -1, -1)]
    valori = [conteggi.get(d.isoformat(), 0) for d in date]
    mx = max(valori) or 1
    barre = ''.join(
        f'<div style="height:{v / mx * 100:.1f}%" title="{d:%a %d/%m}: {v} {unita}"></div>'
        for d, v in zip(date, valori))
    return f"""
    <div class="vi-graf">
      <div class="tit">{_e(titolo)} · massimo {mx if any(valori) else 0} al giorno</div>
      <div class="vi-barre">{barre}</div>
      <div class="vi-assi"><span>{date[0]:%d/%m}</span><span>{date[-1]:%d/%m}</span></div>
    </div>"""


def _tabella(intestazioni, righe, vuoto='Nessun dato nel periodo.', largo=True):
    if not righe:
        return f'<p class="vi-mute" style="font-size:13px">{vuoto}</p>'
    th = ''.join(f'<th class="{"vi-num" if num else ""}">{_e(h)}</th>' for h, num in intestazioni)
    corpo = ''
    for r in righe:
        corpo += '<tr>' + ''.join(
            f'<td class="{"vi-num" if num else ""}">{c}</td>'
            for c, (_, num) in zip(r, intestazioni)) + '</tr>'
    return f'<div class="vi-scorri{" largo" if largo else ""}"><table><thead><tr>{th}</tr></thead><tbody>{corpo}</tbody></table></div>'


def pagina(admin, giorni=30, includi_admin=False, admin_utenti=()):
    giorni = giorni if giorni in _PERIODI else 30
    tutti = eventi(giorni)
    escludi = set() if includi_admin else set(admin_utenti)
    evs = [ev for ev in tutti if ev.get('u', '') not in escludi]

    home = [ev for ev in evs if ev['tipo'] == 'pagina' and ev['sez'] == 'Homepage']
    dash = [ev for ev in evs if ev['tipo'] == 'pagina' and ev['u']
            and ev['sez'] not in ('Homepage', 'Login', 'Registrazione')]
    logins = [ev for ev in evs if ev['tipo'] == 'login']
    nuovi = [ev for ev in evs if ev['tipo'] == 'registrazione']
    pag_login = [ev for ev in evs if ev['tipo'] == 'pagina' and ev['sez'] == 'Login']

    visitatori_home = len({(ev['t'][:10], ev['u'] or ev['v']) for ev in home})
    utenti_attivi = {ev['u'] for ev in dash + logins}
    vivi = [s for s in online() if s['chi'] not in escludi]

    # ── filtri ──
    def _link(g, inc):
        q = f"?giorni={g}" + ('&admin=1' if inc else '')
        return f"/admin/visite{q}"
    nomi_periodo = {1: 'Oggi', 7: '7 giorni', 30: '30 giorni', 90: '90 giorni', 365: '1 anno'}
    filtri = ''.join(
        f'<a class="{"on" if g == giorni else ""}" href="{_link(g, includi_admin)}">{nomi_periodo[g]}</a>'
        for g in _PERIODI)
    filtri += ('<span class="sep"></span>'
               f'<a class="{"on" if includi_admin else ""}" href="{_link(giorni, not includi_admin)}">'
               f'{"✓ " if includi_admin else ""}Includi gli amministratori</a>')

    # ── riquadri ──
    def _card(n, lbl, tit=''):
        return (f'<div class="stat-card" title="{_e(tit)}"><div class="num">{n}</div>'
                f'<div class="lbl">{_e(lbl)}</div></div>')
    cards = ''.join([
        _card(len(vivi), 'Online adesso', f'Segnali negli ultimi {_ONLINE_MIN} minuti'),
        _card(len(home), 'Visite homepage'),
        _card(visitatori_home, 'Visitatori homepage',
              'Persone diverse giorno per giorno, sommate sul periodo: chi torna in due giorni diversi conta due volte'),
        _card(len(dash), 'Aperture dashboard'),
        _card(len(utenti_attivi), 'Utenti attivi', 'Utenti col login che hanno aperto una dashboard o fatto un accesso'),
        _card(len(logins), 'Accessi (login)'),
        _card(len(nuovi), 'Nuove registrazioni'),
    ])

    # ── online adesso ──
    ora = time.time()
    righe_on = [[
        (f'<span class="vi-pallino"></span>{_e(s["chi"])}' if s['chi']
         else '<span class="vi-pallino"></span><span class="vi-mute">visitatore anonimo</span>'),
        _e(s['sez']), _e(s['disp']), f"{int((ora - s['t']) // 60)} min fa",
    ] for s in vivi]
    sez_online = _tabella([('Chi', False), ('Dove', False), ('Dispositivo', False), ('Ultimo segnale', True)],
                          righe_on, 'Nessuno collegato negli ultimi 15 minuti.')

    # ── grafici per giorno ──
    grafici = ''
    if giorni > 1:
        per_g_home = Counter(ev['t'][:10] for ev in home)
        per_g_dash = Counter(ev['t'][:10] for ev in dash)
        grafici = (f'<div class="vi-sez vi-due">'
                   f'{_barre(per_g_home, giorni, "Visite alla homepage per giorno", "visite")}'
                   f'{_barre(per_g_dash, giorni, "Aperture delle dashboard per giorno", "aperture")}'
                   f'</div>')
    else:
        per_h_home = Counter(int(ev['t'][11:13]) for ev in home)
        per_h_dash = Counter(int(ev['t'][11:13]) for ev in dash)
        def _ore(c, tit):
            mx = max(c.values(), default=0) or 1
            barre = ''.join(f'<div style="height:{c.get(h, 0) / mx * 100:.1f}%" '
                            f'title="ore {h:02d}: {c.get(h, 0)}"></div>' for h in range(24))
            return (f'<div class="vi-graf"><div class="tit">{tit} · per ora</div>'
                    f'<div class="vi-barre">{barre}</div>'
                    f'<div class="vi-assi"><span>00</span><span>12</span><span>23</span></div></div>')
        grafici = (f'<div class="vi-sez vi-due">{_ore(per_h_home, "Visite alla homepage")}'
                   f'{_ore(per_h_dash, "Aperture delle dashboard")}</div>')

    # ── dashboard ──
    per_sez = defaultdict(list)
    for ev in dash:
        nome = ev['sez'] + (' (tab del Portafoglio)' if ev.get('embed') else '')
        per_sez[nome].append(ev)
    righe_dash = sorted(([
        _e(nome), len(lst), len({ev['u'] for ev in lst}), _quando(lst[-1]['t']),
    ] for nome, lst in per_sez.items()), key=lambda r: -r[1])
    sez_dash = _tabella([('Dashboard', False), ('Aperture', True), ('Utenti diversi', True),
                         ('Ultima apertura', True)], righe_dash)

    # ── utenti ──
    ultimi = ultimi_accessi()
    per_utente = defaultdict(lambda: {'aperture': 0, 'login': 0, 'sez': Counter(), 'disp': Counter()})
    for ev in dash + logins:
        d = per_utente[ev['u']]
        if ev['tipo'] == 'login':
            d['login'] += 1
        else:
            d['aperture'] += 1
            d['sez'][ev['sez']] += 1
        d['disp'][ev['disp']] += 1
    righe_ut = []
    for u, d in sorted(per_utente.items(), key=lambda kv: ultimi.get(kv[0], ''), reverse=True):
        preferite = ', '.join(f'{_e(s)} ({n})' for s, n in d['sez'].most_common(3)) or '—'
        righe_ut.append([
            f'<a href="/admin/dati/{urllib.parse.quote(u, safe="")}" style="color:#90d8f8">{_e(u)}</a>',
            _quando(ultimi.get(u, '')), d['login'], d['aperture'], preferite,
            _e(d['disp'].most_common(1)[0][0]) if d['disp'] else '—',
        ])
    sez_ut = _tabella([('Utente', False), ('Ultimo segnale', True), ('Accessi', True),
                       ('Aperture', True), ('Dashboard più usate', False), ('Dispositivo', False)], righe_ut)

    # ── homepage: provenienza, dispositivi, paesi ──
    prov = Counter(ev.get('ref') or 'diretto / sconosciuto' for ev in home)
    disp = Counter(ev['disp'] for ev in home)
    paesi = Counter(ev['paese'] for ev in home if ev.get('paese'))
    tot_h = len(home) or 1
    sez_prov = _tabella([('Provenienza', False), ('Visite', True), ('%', True)],
                        [[_e(k), n, f'{n / tot_h * 100:.0f}%'] for k, n in prov.most_common(12)], largo=False)
    sez_disp = _tabella([('Dispositivo', False), ('Visite', True), ('%', True)],
                        [[_e(k), n, f'{n / tot_h * 100:.0f}%'] for k, n in disp.most_common()], largo=False)
    sez_paesi = ''
    if paesi:
        sez_paesi = ('<div><h2>Paesi</h2>' + _tabella(
            [('Paese', False), ('Visite', True)], [[_e(k), n] for k, n in paesi.most_common(12)], largo=False) + '</div>')
    imbuto = (f'<p class="vi-nota">Pagina di login aperta {len(pag_login)} volte, '
              f'{len(logins)} accessi riusciti, {len(nuovi)} nuove registrazioni.</p>')

    # ── ultimi eventi ──
    nomi_tipo = {'pagina': 'pagina', 'login': 'accesso', 'registrazione': 'registrazione'}
    righe_ev = [[
        _quando(ev['t']), _e(nomi_tipo.get(ev['tipo'], ev['tipo'])),
        _e(ev['sez'] + (' (tab)' if ev.get('embed') else '')
           + (f" · {ev['metodo']}" if ev.get('metodo') else '')),
        _e(ev['u']) if ev['u'] else f'<span class="vi-mute">anonimo {_e(ev["v"][:6])}</span>',
        _e(ev['disp']), _e(ev.get('ref', '')),
    ] for ev in reversed(evs[-150:])]
    sez_ev = _tabella([('Quando', False), ('Tipo', False), ('Pagina', False), ('Chi', False),
                       ('Dispositivo', False), ('Provenienza', False)], righe_ev)

    nota_admin = ('' if includi_admin else
                  f'<p class="vi-nota">Esclusi gli amministratori ({_e(", ".join(sorted(escludi)) or "nessuno")}).</p>')
    return _CSS + f"""
  <div class="topbar">
    <h1>A·C Dashboard — Visite</h1>
    <div>
      <a href="/admin">Utenti</a>&nbsp;&nbsp;
      <a href="/admin/dati">Dati utenti</a>&nbsp;&nbsp;
      <a href="/">← Sito</a>
    </div>
  </div>
  <div class="container">
    <div class="vi-filtri">{filtri}</div>
    {nota_admin}
    <div class="stats">{cards}</div>
    <div class="vi-sez"><h2>Online adesso</h2>{sez_online}</div>
    {grafici}
    <div class="vi-sez"><h2>Dashboard</h2>{sez_dash}</div>
    <div class="vi-sez"><h2>Utenti</h2>{sez_ut}</div>
    <div class="vi-sez vi-due">
      <div><h2>Homepage — provenienza</h2>{sez_prov}</div>
      <div><h2>Homepage — dispositivi</h2>{sez_disp}{imbuto}</div>
      {sez_paesi}
    </div>
    <div class="vi-sez"><h2>Ultimi eventi</h2>{sez_ev}</div>
    <p class="vi-nota">Si contano le pagine aperte (non i clic dentro le dashboard), gli accessi e le
      registrazioni; bot e anteprime dei link sono esclusi. Nessun indirizzo IP viene salvato: un
      visitatore anonimo è riconosciuto solo dentro la stessa giornata. I dati degli ultimi minuti
      si scrivono su disco ogni {_SCRIVI_OGNI // 60} minuti, ma compaiono qui subito.</p>
  </div>
</body>
</html>"""


def registra_rotte(flask_server, html_head):
    """Monta /admin/visite sul server che serve già /admin."""
    from flask import request, session, redirect
    import auth

    @flask_server.route('/admin/visite')
    def _admin_visite():
        u = session.get('username')
        utente = auth.get_user(u) if u else None
        if not utente or utente.get('role') != 'admin':
            return redirect('/')
        try:
            giorni = int(request.args.get('giorni', 30))
        except ValueError:
            giorni = 30
        admin_utenti = [x['username'] for x in auth.list_users() if x.get('role') == 'admin']
        corpo = pagina(u, giorni, includi_admin=bool(request.args.get('admin')),
                       admin_utenti=admin_utenti)
        return (html_head.replace('<title>Admin – Dashboard Finanziaria</title>',
                                  '<title>Visite – Dashboard Finanziaria</title>') + corpo,
                200, {'Cache-Control': 'no-store'})
