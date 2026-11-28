"""Calendario delle trimestrali — chi pubblica i conti, giorno per giorno.

Due fonti, perché nessuna basta da sola:
  • Alpha Vantage EARNINGS_CALENDAR: una sola richiesta restituisce tutte le
    società USA, con stima dell'EPS e momento (pre/post mercato). Contiene però
    soltanto le date **già annunciate**: subito dopo una stagione di trimestrali
    il mese seguente è pieno e i successivi quasi vuoti (il 03/11/2026 AAPL,
    MSFT e META non c'erano, avendo appena pubblicato).
  • Yahoo (`Ticker.calendar`) per un elenco di società principali USA, Italia ed
    Europa: dà la prossima data anche quando non è ancora annunciata. Sui titoli
    europei a volte restituisce la trimestrale appena passata invece della
    prossima (ENI 23/10, ASML 14/10 visti il 03/11).
Le trimestrali successive delle società principali si stimano a +13 settimane
dall'ultima data nota: 91 giorni tengono lo stesso giorno della settimana e la
stessa settimana del trimestre, che è l'abitudine di quasi tutte le società.
In tabella la fonte di ogni data è dichiarata (confermata / Yahoo / stimata).

Aggiornamento: ogni 3 mesi (1º gennaio, aprile, luglio, ottobre alle 03:00), e
comunque alla prima apertura se il file ha più di `_OGNI_GIORNI` giorni o manca
— un deploy o un job saltato non lasciano il calendario fermo. Costa una
richiesta Alpha Vantage (sulle 25 giornaliere) e ~100 chiamate Yahoo, in un
thread in background. Il file sta in `sessions/_trimestrali/`, prefisso
sincronizzato su R2 (l'underscore non viene scambiato per un utente dai job
notturni, che cercano un `current.json`).
"""
import csv
import io
import json
import os
import threading
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

_DIR  = Path(__file__).resolve().parent.parent / 'sessions' / '_trimestrali'
_FILE = _DIR / 'calendario.json'
_OGNI_GIORNI = 92                    # 3 mesi: età oltre la quale si riscarica
_PASSO = timedelta(days=91)          # una trimestrale dopo l'altra (13 settimane)
_ORIZZONTE = timedelta(days=200)     # fin dove si proiettano le date stimate
_MAX_RIGHE = 800                     # oltre, la tabella diventa pesante nel browser

_AV_URL = 'https://www.alphavantage.co/query'
_AV_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', 'ZX9YB88WV3EYUBTT')

# Società principali: ticker Yahoo → nome (il mercato si ricava dal suffisso).
# Fuori: Moncler, LVMH, L'Oréal, Schneider, Nestlé e Roche — Yahoo non dà loro
# nessuna data (semestrali o solo fatturato trimestrale; verificato il 03/11/2026).
_UNIVERSO = {
    # USA
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'AMZN': 'Amazon',
    'GOOGL': 'Alphabet', 'META': 'Meta Platforms', 'TSLA': 'Tesla',
    'BRK-B': 'Berkshire Hathaway', 'AVGO': 'Broadcom', 'JPM': 'JPMorgan Chase',
    'LLY': 'Eli Lilly', 'V': 'Visa', 'MA': 'Mastercard', 'UNH': 'UnitedHealth',
    'XOM': 'Exxon Mobil', 'JNJ': 'Johnson & Johnson', 'WMT': 'Walmart',
    'PG': 'Procter & Gamble', 'HD': 'Home Depot', 'COST': 'Costco',
    'ORCL': 'Oracle', 'NFLX': 'Netflix', 'ABBV': 'AbbVie', 'BAC': 'Bank of America',
    'KO': 'Coca-Cola', 'PEP': 'PepsiCo', 'CRM': 'Salesforce', 'AMD': 'AMD',
    'ADBE': 'Adobe', 'CSCO': 'Cisco', 'MRK': 'Merck', 'CVX': 'Chevron',
    'WFC': 'Wells Fargo', 'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley',
    'C': 'Citigroup', 'INTC': 'Intel', 'QCOM': 'Qualcomm', 'TXN': 'Texas Instruments',
    'IBM': 'IBM', 'CAT': 'Caterpillar', 'BA': 'Boeing', 'GE': 'GE Aerospace',
    'DIS': 'Disney', 'MCD': "McDonald's", 'NKE': 'Nike', 'PFE': 'Pfizer',
    'TMO': 'Thermo Fisher', 'UBER': 'Uber', 'PLTR': 'Palantir', 'MU': 'Micron',
    'BLK': 'BlackRock',
    # Italia
    'ENI.MI': 'Eni', 'ENEL.MI': 'Enel', 'ISP.MI': 'Intesa Sanpaolo',
    'UCG.MI': 'UniCredit', 'RACE.MI': 'Ferrari', 'STLAM.MI': 'Stellantis',
    'G.MI': 'Generali', 'STMMI.MI': 'STMicroelectronics', 'PRY.MI': 'Prysmian',
    'LDO.MI': 'Leonardo', 'MB.MI': 'Mediobanca', 'BMPS.MI': 'Monte dei Paschi',
    'BAMI.MI': 'Banco BPM', 'TIT.MI': 'Telecom Italia',
    'PST.MI': 'Poste Italiane', 'SRG.MI': 'Snam', 'TRN.MI': 'Terna',
    'TEN.MI': 'Tenaris', 'CPR.MI': 'Campari', 'A2A.MI': 'A2A', 'FBK.MI': 'FinecoBank',
    # Europa
    'ASML.AS': 'ASML', 'SAP.DE': 'SAP', 'SIE.DE': 'Siemens',
    'TTE.PA': 'TotalEnergies', 'SAN.PA': 'Sanofi',
    'ALV.DE': 'Allianz', 'AIR.PA': 'Airbus',
    'DTE.DE': 'Deutsche Telekom', 'NOVO-B.CO': 'Novo Nordisk',
    'NOVN.SW': 'Novartis', 'SHEL.L': 'Shell',
    'AZN.L': 'AstraZeneca', 'HSBA.L': 'HSBC', 'BNP.PA': 'BNP Paribas',
    'SAN.MC': 'Santander', 'IBE.MC': 'Iberdrola', 'ADS.DE': 'Adidas',
    'BMW.DE': 'BMW', 'MBG.DE': 'Mercedes-Benz',
}


def _mercato(tk):
    if tk.endswith('.MI'):
        return 'Italia'
    return 'Europa' if '.' in tk else 'USA'


# ─────────────────────────────────────────────────────────────────────────────
# Scaricamento
# ─────────────────────────────────────────────────────────────────────────────
def _scarica_av():
    """Righe del calendario Alpha Vantage: [simbolo, nome, data, fine trimestre,
    stima EPS, valuta, momento]. Solleva se la risposta non è il CSV atteso."""
    url = f'{_AV_URL}?function=EARNINGS_CALENDAR&horizon=12month&apikey={_AV_KEY}'
    testo = urllib.request.urlopen(url, timeout=30).read().decode()
    righe = list(csv.reader(io.StringIO(testo)))
    # Limite raggiunto o chiave rifiutata: arriva un JSON, non il CSV.
    if not righe or righe[0][:3] != ['symbol', 'name', 'reportDate']:
        raise ValueError(testo[:160])
    return [r[:7] + [''] * (7 - len(r)) for r in righe[1:] if len(r) >= 3]


def _num(v):
    try:
        v = float(v)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _scarica_yahoo():
    """ticker → {data, eps, ricavi} da `Ticker.calendar` (un titolo che fallisce
    resta senza data, non ferma gli altri)."""
    import yfinance as yf
    out = {}
    for tk in _UNIVERSO:
        try:
            c = yf.Ticker(tk).calendar or {}
            date_ = [d for d in (c.get('Earnings Date') or []) if d]
            out[tk] = {'data': min(date_).isoformat() if date_ else None,
                       'eps': _num(c.get('Earnings Average')),
                       'ricavi': _num(c.get('Revenue Average'))}
        except Exception as e:                       # noqa: BLE001
            out[tk] = {'data': None, 'eps': None, 'ricavi': None,
                       'errore': str(e)[:120]}
        time.sleep(0.3)
    return out


def aggiorna():
    """Riscarica entrambe le fonti e salva. Se una fonte fallisce si tiene la
    versione precedente di quella fonte: meglio date vecchie che un buco."""
    vecchio = _leggi_file() or {}
    nuovo = {'aggiornato': datetime.now().isoformat(timespec='seconds')}
    try:
        nuovo['av'] = _scarica_av()
        nuovo['av_errore'] = None
    except Exception as e:                           # noqa: BLE001
        nuovo['av'] = vecchio.get('av') or []
        nuovo['av_errore'] = str(e)[:200]
        print(f"⚠ [trimestrali] Alpha Vantage: {e}", flush=True)
    try:
        nuovo['yahoo'] = _scarica_yahoo()
    except Exception as e:                           # noqa: BLE001
        nuovo['yahoo'] = vecchio.get('yahoo') or {}
        print(f"⚠ [trimestrali] Yahoo: {e}", flush=True)

    _DIR.mkdir(parents=True, exist_ok=True)
    tmp = _FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(nuovo))
    tmp.replace(_FILE)
    try:
        import data_core
        data_core.cloud_push(_FILE)
    except Exception:                                # noqa: BLE001
        pass
    global _CACHE
    _CACHE = nuovo
    n_y = sum(1 for v in nuovo['yahoo'].values() if v.get('data'))
    print(f"✓ [trimestrali] aggiornato: {len(nuovo['av'])} date Alpha Vantage, "
          f"{n_y}/{len(_UNIVERSO)} società principali da Yahoo", flush=True)
    return nuovo


# ─────────────────────────────────────────────────────────────────────────────
# Lettura (con aggiornamento pigro in background)
# ─────────────────────────────────────────────────────────────────────────────
_CACHE = None
_IN_CORSO = threading.Lock()


def _leggi_file():
    try:
        return json.loads(_FILE.read_text())
    except Exception:                                # noqa: BLE001
        return None


def _eta_giorni(dati):
    try:
        return (datetime.now() - datetime.fromisoformat(dati['aggiornato'])).days
    except Exception:                                # noqa: BLE001
        return 10 ** 6


def _aggiorna_in_background():
    """Un solo aggiornamento alla volta: chi arriva mentre è in corso non ne
    fa partire un secondo."""
    if not _IN_CORSO.acquire(blocking=False):
        return

    def _lavoro():
        try:
            aggiorna()
        except Exception as e:                       # noqa: BLE001
            print(f"⚠ [trimestrali] aggiornamento fallito: {e}", flush=True)
        finally:
            _IN_CORSO.release()
    threading.Thread(target=_lavoro, daemon=True).start()


def in_corso():
    return _IN_CORSO.locked()


def dati():
    """I dati in cache (memoria → disco). Se mancano o sono più vecchi di
    `_OGNI_GIORNI` parte un aggiornamento in background e intanto si usa
    quello che c'è (anche niente)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _leggi_file()
    if _CACHE is None or _eta_giorni(_CACHE) >= _OGNI_GIORNI:
        _aggiorna_in_background()
    return _CACHE


def prossimo_aggiornamento(dati_):
    """Il primo fra il prossimo 1º di gennaio/aprile/luglio/ottobre e la
    scadenza per età."""
    oggi = date.today()
    m = ((oggi.month - 1) // 3 + 1) * 3 + 1
    cron = date(oggi.year + (m > 12), (m - 1) % 12 + 1, 1)
    try:
        eta = datetime.fromisoformat(dati_['aggiornato']).date() + timedelta(days=_OGNI_GIORNI)
        return min(cron, eta)
    except Exception:                                # noqa: BLE001
        return cron


def avvia_scheduler():
    """Job trimestrale. Separato dallo scheduler di portafoglio: quello va
    lun-sab di notte e il calendario non deve dipendere da lui."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        print("⚠ [trimestrali] apscheduler assente: solo aggiornamento all'apertura")
        return None
    s = BackgroundScheduler(timezone='Europe/Rome')
    s.add_job(_aggiorna_in_background, 'cron', month='1,4,7,10', day=1, hour=3,
              id='trimestrali', misfire_grace_time=6 * 3600, coalesce=True,
              max_instances=1)
    s.start()
    print("✓ [trimestrali] aggiornamento programmato il 1º di gen/apr/lug/ott alle 03:00")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Eventi in una finestra
# ─────────────────────────────────────────────────────────────────────────────
def _d(iso):
    try:
        return date.fromisoformat((iso or '')[:10])
    except ValueError:
        return None

_VOLO_CACHE = {}                     # testo → (istante, risultati) della ricerca al volo
_VOLO_TTL = 6 * 3600


def _yahoo_al_volo(testo):
    """[(ticker, nome, {data, eps})] delle azioni che Yahoo trova per `testo` (nome
    o ticker), al massimo 3, con la prossima trimestrale da `Ticker.calendar`.
    Cache di 6 ore, anche dei risultati vuoti: il campo si ridisegna spesso."""
    chiave = testo.strip().lower()
    c = _VOLO_CACHE.get(chiave)
    if c and time.time() - c[0] < _VOLO_TTL:
        return c[1]
    out = []
    try:
        import yfinance as yf
        quotes = yf.Search(testo, max_results=12, news_count=0).quotes or []
        tutte = [(q['symbol'], q.get('shortname') or q.get('longname') or q['symbol'])
                 for q in quotes if q.get('symbol') and q.get('quoteType') == 'EQUITY']
        # Ticker scelto dai suggerimenti (anche se Yahoo lo mette oltre il terzo posto)
        esatto = [x for x in tutte if x[0].upper() == testo.strip().upper()]
        simboli = esatto or tutte[:3]
        for tk, nome in simboli:
            try:
                cal = yf.Ticker(tk).calendar or {}
            except Exception:                        # noqa: BLE001
                continue
            date_ = [d for d in (cal.get('Earnings Date') or []) if d]
            if date_:
                out.append((tk, nome, {'data': min(date_).isoformat(),
                                       'eps': _num(cal.get('Earnings Average'))}))
    except Exception as e:                           # noqa: BLE001
        print(f'[trimestrali] ricerca Yahoo «{testo}» non riuscita: {e}')
        return []
    if len(_VOLO_CACHE) > 300:
        _VOLO_CACHE.clear()
    _VOLO_CACHE[chiave] = (time.time(), out)
    return out


_SUGG_CACHE = {}                     # testo → (istante, suggerimenti di Yahoo)


def suggerimenti(testo, max_locali=6, max_yahoo=8):
    """Suggerimenti per la casella di ricerca, come un motore di ricerca: prima le
    società già nel calendario (universo + Alpha Vantage) il cui ticker o nome
    comincia col testo, poi le azioni che Yahoo trova per nome o ticker.
    Lista di (ticker Yahoo, nome, borsa). Nessuna chiamata a `Ticker.calendar`:
    la data si cerca solo quando si sceglie un titolo."""
    t = (testo or '').strip().lower()
    if len(t) < 2:
        return []
    out, visti = [], set()

    def _ok(tk, nome):
        base = tk.lower().split('.')[0]
        return (tk.lower().startswith(t) or base.startswith(t)
                or any(p.startswith(t) for p in nome.lower().split()))

    for tk, nome in _UNIVERSO.items():
        if len(out) < max_locali and _ok(tk, nome):
            out.append((tk, nome, _mercato(tk))); visti.add(tk)
    for r in (dati() or {}).get('av') or []:
        if len(out) >= max_locali:
            break
        tk = r[0].replace('.', '-')
        if tk not in visti and _ok(tk, r[1]):
            out.append((tk, r[1].title(), 'USA')); visti.add(tk)

    c = _SUGG_CACHE.get(t)
    if c and time.time() - c[0] < _VOLO_TTL:
        yh = c[1]
    else:
        yh = []
        try:
            import yfinance as yf
            for q in yf.Search(testo, max_results=12, news_count=0).quotes or []:
                if q.get('symbol') and q.get('quoteType') == 'EQUITY':
                    yh.append((q['symbol'], (q.get('shortname') or q.get('longname') or q['symbol']).strip(),
                               q.get('exchDisp') or q.get('exchange') or ''))
        except Exception as e:                       # noqa: BLE001
            print(f'[trimestrali] suggerimenti Yahoo «{testo}» non riusciti: {e}')
        if yh:
            if len(_SUGG_CACHE) > 500:
                _SUGG_CACHE.clear()
            _SUGG_CACHE[t] = (time.time(), yh)
    for tk, nome, borsa in yh[:max_yahoo]:
        if tk not in visti:
            out.append((tk, nome, borsa)); visti.add(tk)
    return out


def eventi(w0, w1, universo='principali', cerca='', yahoo_al_volo=False):
    """Trimestrali fra w0 e w1 (incluse). Ogni evento: dict con d, ticker, nome,
    mercato, momento, trimestre, eps, valuta, fonte ('confermata' | 'yahoo' |
    'stimata')."""
    dati_ = dati() or {}
    oggi = date.today()
    av = dati_.get('av') or []
    yahoo = dati_.get('yahoo') or {}

    # Alpha Vantage scrive BRK.B, Yahoo BRK-B
    av_per_tk = {}
    for r in av:
        tk = r[0].replace('.', '-')
        d = _d(r[2])
        if d:
            av_per_tk.setdefault(tk, []).append((d, r))

    out = []

    def _add(d, tk, nome, fonte, momento='', trimestre='', eps=None, valuta=''):
        if w0 <= d <= w1:
            out.append(dict(d=d, ticker=tk, nome=nome, mercato=_mercato(tk),
                            momento=momento, trimestre=trimestre, eps=eps,
                            valuta=valuta, fonte=fonte))

    def _societa(tk, nome, y):
        """Data nota (AV prima, poi Yahoo) + proiezioni a +13 settimane."""
        noti = sorted(av_per_tk.get(tk, []), key=lambda x: x[0])
        base = None
        for d, r in noti:
            _add(d, tk, nome, 'confermata', r[6], r[3], _num(r[4]), r[5])
            base = d
        dy = _d((y or {}).get('data'))
        if dy and not any(abs((dy - d).days) <= 10 for d, _ in noti):
            # data già passata = trimestrale appena pubblicata: vale solo come base
            _add(dy, tk, nome, 'yahoo' if dy >= oggi else 'pubblicata',
                 eps=y.get('eps') if dy >= oggi else None)
            base = max(base, dy) if base else dy
        if base:
            d = base + _PASSO
            while d <= min(w1, oggi + _ORIZZONTE):
                if d >= oggi:
                    _add(d, tk, nome, 'stimata')
                d += _PASSO

    for tk, nome in _UNIVERSO.items():
        _societa(tk, nome, yahoo.get(tk))

    if universo == 'tutte':
        for tk, righe in av_per_tk.items():
            if tk in _UNIVERSO:
                continue
            for d, r in righe:
                _add(d, tk, r[1].title(), 'confermata', r[6], r[3], _num(r[4]), r[5])

    testo = (cerca or '').strip()
    cerca = testo.lower()
    if cerca:
        # Prima il ticker esatto (ENI → ENI.MI), poi i nomi con una parola che
        # comincia col testo, solo in mancanza la sottostringa («eni» è anche
        # dentro «Fresenius»).
        esatti = [e for e in out if cerca in (e['ticker'].lower(),
                                               e['ticker'].lower().split('.')[0])]
        parola = [e for e in out if any(p.startswith(cerca)
                                        for p in e['nome'].lower().split())]
        out = esatti or parola or [e for e in out if cerca in e['ticker'].lower()
                                   or cerca in e['nome'].lower()]
        # Nessuna società nota con quel nome: la si chiede a Yahoo al volo
        # (ricerca per nome o ticker, poi `Ticker.calendar`).
        if not out and yahoo_al_volo:
            trovati = _yahoo_al_volo(testo)
            # Ticker scelto dai suggerimenti: solo lui, non gli omonimi di altre borse
            esatto = [x for x in trovati if x[0].upper() == testo.upper()]
            for tk, nome, y in esatto or trovati:
                _societa(tk, nome, y)
    out.sort(key=lambda e: (e['d'], e['mercato'] != 'Italia', e['ticker']))
    return out
