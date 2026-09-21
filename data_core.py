"""
data_core.py — Logica dati CONDIVISA (un solo posto, richiamato da più app).

Tutto ruota attorno al file UNICO per utente: sessions/<utente>/current.json
(dataset + pesi P1/P2/P3) e sessions/<utente>/analyses.json (analisi salvate).

Espone: lettura/scrittura current.json, download prezzi con conversione valuta
in EUR, aggiungi-asset (accoda a current.json), template/esporta Excel, e
salva/carica "tutto il lavoro" (snapshot di current.json + analyses.json).

NON contiene componenti o callback Dash: è una libreria pura, importabile da
qualunque app (Analisi Tattica, Portafoglio, …) senza effetti collaterali.
"""
import os
import io
import json
import re
import hashlib
import threading
from pathlib import Path

import pandas as pd

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Tetto ai rendimenti giornalieri: oltre ±50%/giorno è quasi sempre un tick
# corrotto (prezzo sbagliato a inizio serie o buco → pct_change esplode) che fa
# schizzare volatilità e covarianze. Vale per TUTTA la dashboard.
MAX_DAILY_RET = 0.5


def clip_returns(obj):
    """Limita i rendimenti anomali a ±MAX_DAILY_RET (preserva i NaN)."""
    if obj is None:
        return obj
    try:
        return obj.clip(-MAX_DAILY_RET, MAX_DAILY_RET)
    except Exception:
        return obj


# ─── Storage persistente (S3/R2) ──────────────────────────────────────────────
def cloud_push(path):
    """Replica il file sullo storage persistente (R2) se configurato. Best-effort."""
    try:
        import cloud_storage
        cloud_storage.push(path)
    except Exception:
        pass


# ─── Utente / percorsi ────────────────────────────────────────────────────────
def get_username():
    try:
        from flask import session as _fs
        return _fs.get('username') or 'anon'
    except Exception:
        return 'anon'


def current_path(username=None):
    u = username or get_username()
    d = ROOT / 'sessions' / u
    d.mkdir(parents=True, exist_ok=True)
    return d / 'current.json'


def analyses_path(username=None):
    u = username or get_username()
    return ROOT / 'sessions' / u / 'analyses.json'


# ─── current.json: lettura / scrittura ───────────────────────────────────────
def read_current(username=None):
    try:
        with open(current_path(username)) as f:
            raw = json.load(f)
        # current.json contiene solo voci-asset (dict). Eventuali chiavi meta
        # (es. "_tipo": "personale"|"default:ETF") vengono ignorate qui, così
        # ogni iterazione a valle vede solo asset.
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    except Exception:
        return {}


def read_meta(username=None):
    """Legge le chiavi meta (non-asset) di current.json, es. {'_tipo': ...}."""
    try:
        with open(current_path(username)) as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not isinstance(v, dict)}
    except Exception:
        return {}


def write_current(data, username=None):
    """Scrittura atomica di current.json + replica su storage persistente."""
    path = current_path(username)
    try:
        tmp = str(path) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)
        cloud_push(path)
        return True
    except Exception as e:
        print(f"⚠ [data_core] scrittura current.json fallita: {e}", flush=True)
        return False


def write_json_atomic(path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(path) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)
        cloud_push(path)
        return True
    except Exception as e:
        print(f"⚠ [data_core] scrittura {getattr(path, 'name', path)} fallita: {e}", flush=True)
        return False


def read_analyses(username=None):
    try:
        with open(analyses_path(username)) as f:
            return json.load(f)
    except Exception:
        return {}


# ─── Viste sui dati ───────────────────────────────────────────────────────────
def asset_options(username=None):
    return [{'label': a, 'value': a} for a in read_current(username).keys()]


def build_dataset(username=None):
    """
    Ricostruisce (close_returns, original_prices, ticker_map) dal file UNICO
    current.json. Usato per mostrare i dati senza dipendere dai buffer in memoria.
    """
    data = read_current(username)
    pcols, rcols, tm = {}, {}, {}
    for a, v in data.items():
        if not isinstance(v, dict):
            continue
        dates = v.get('dates')
        if not dates:
            continue
        try:
            idx = pd.to_datetime(dates)
        except Exception:
            continue
        if v.get('prices') and len(v['prices']) == len(dates):
            pcols[a] = pd.Series(v['prices'], index=idx)
        if v.get('returns') and len(v['returns']) == len(dates):
            rcols[a] = pd.Series(v['returns'], index=idx)
        tm[a] = v.get('ticker') or a
    op = pd.DataFrame(pcols).sort_index() if pcols else None
    cr = pd.DataFrame(rcols).sort_index() if rcols else None
    cr = clip_returns(cr)   # neutralizza tick corrotti (±50%/giorno)
    return cr, op, tm


def build_prices(username=None):
    """DataFrame dei PREZZI da current.json (chiave 'prices' per asset)."""
    data = read_current(username)
    cols = {}
    for asset, v in data.items():
        if not isinstance(v, dict):
            continue
        dates, prices = v.get('dates'), v.get('prices')
        if not dates or not prices or len(dates) != len(prices):
            continue
        try:
            cols[asset] = pd.Series(prices, index=pd.to_datetime(dates))
        except Exception:
            continue
    if not cols:
        return None
    try:
        return pd.DataFrame(cols).sort_index()
    except Exception:
        return None


# ─── Download + conversione valuta (come Analisi di Portafoglio) ──────────────
def fx_series(name, start):
    """Serie del cambio (es. EURUSD=X) per la conversione in EUR."""
    import yfinance as yf
    try:
        fx = yf.download(name, start=start, auto_adjust=True, progress=False)
        if fx is None or len(fx) == 0:
            return None
        c = fx['Close']
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        return c.ffill()
    except Exception:
        return None


# ─── Valute: tutto riportato in euro ─────────────────────────────────────────
# Yahoo quota il cambio come EUR<valuta>=X (quante unita' di valuta vale 1 euro):
# per passare a euro si DIVIDE il prezzo per quel cambio.
#
# Alcune piazze quotano in centesimi: Londra in penny ('GBp', 'GBX'), Tel Aviv in
# agorot ('ILA'), Johannesburg in cent ('ZAc'), molti future agricoli del CME in
# cent di dollaro ('USX'). Il prezzo va diviso per 100 e poi cambiato con la
# valuta piena. Il codice si confronta COSI' COM'E': 'GBp'.upper() diventa 'GBP'
# e i penny sparirebbero dentro le sterline.
CENTESIMI = {'GBp': 'GBP', 'GBX': 'GBP', 'ILA': 'ILS', 'ZAc': 'ZAR', 'USX': 'USD'}

_VALUTA_YAHOO = {}
_VALUTA_ASSENTE = set()   # Yahoo ha risposto, ma senza valuta


def e_cambio(ticker):
    """True per i tassi di cambio (EURUSD=X, JPY=X). Sono gia' il rapporto fra due
    monete e non vanno convertiti: EURUSD=X diviso EURUSD fa sempre 1."""
    return str(ticker or '').strip().upper().endswith('=X')


def valuta_yahoo(ticker):
    """Valuta di quotazione secondo Yahoo, nella forma esatta che usa Yahoo
    ('GBp' e 'GBP' non sono la stessa cosa). None se non si riesce a saperlo.

    E' una chiamata di rete per ticker e la valuta di quotazione non cambia, quindi
    le risposte buone restano in cache finche' il processo vive. Gli errori no: un
    problema di rete momentaneo non deve pesare per tutta la giornata.
    """
    if not ticker:
        return None
    if ticker in _VALUTA_YAHOO:
        return _VALUTA_YAHOO[ticker]
    try:
        import yfinance as yf
        fi = yf.Ticker(ticker).fast_info
        val = (fi.get('currency') if isinstance(fi, dict) else getattr(fi, 'currency', None))
    except Exception:
        return None
    val = str(val).strip() if val else ''
    if not val:
        _VALUTA_ASSENTE.add(ticker)
        return None
    _VALUTA_YAHOO[ticker] = val
    return val


def valute_yahoo(tickers, timeout=30):
    """valuta_yahoo per molti ticker in parallelo, con un tetto di tempo complessivo:
    chi non risponde in tempo resta None e il download va avanti."""
    from concurrent.futures import ThreadPoolExecutor, wait
    da_chiedere = [t for t in dict.fromkeys(tickers) if t and t not in _VALUTA_YAHOO]
    if da_chiedere:
        ex = ThreadPoolExecutor(max_workers=min(8, len(da_chiedere)))
        wait([ex.submit(valuta_yahoo, t) for t in da_chiedere], timeout=timeout)
        ex.shutdown(wait=False, cancel_futures=True)
    return {t: _VALUTA_YAHOO.get(t) for t in tickers}


def valute_note(tickers):
    """Le valute che Yahoo ha gia' dato in questo processo, senza chiedere niente
    alla rete: {ticker: valuta o None}. Serve ad allineare un file appena
    riscritto senza rifare il giro dei ticker."""
    return {t: _VALUTA_YAHOO.get(t) for t in dict.fromkeys(tickers) if t}


def valuta_dichiarata(v):
    """Valuta scritta a mano (Excel, tendina) ripulita: '' se manca."""
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    c = str(v).strip()
    if c in CENTESIMI:
        return c
    c = c.upper()
    return '' if c in ('NAN', 'NONE') else c


def valuta_piena(codice):
    """Da un codice Yahoo alla coppia (valuta piena, divisore): 'GBp' -> ('GBP', 100),
    'USD' -> ('USD', 1). Il divisore riporta i centesimi all'unita' di valuta."""
    c = (codice or '').strip()
    if c in CENTESIMI:
        return CENTESIMI[c], 100.0
    return (c.upper() or 'EUR'), 1.0


def valuta_valida(codice):
    """True se e' un codice valuta plausibile: tre lettere (USD) o un codice in
    centesimi di Yahoo (GBp). 'EIUR' o '' non lo sono."""
    c = (codice or '').strip()
    return c in CENTESIMI or bool(re.fullmatch(r'[A-Z]{3}', c))


def valuta_da_salvare(dichiarata, codice):
    """La valuta che finisce nel file dell'utente: quella con cui i prezzi sono
    stati davvero convertiti, cioe' quella di Yahoo. Quella scritta a mano vale
    solo se Yahoo non risponde. Prima comandava la scritta a mano e il download
    si limitava a segnalare la differenza: un titolo in dollari restava marcato
    'EUR' finche' non lo si correggeva riga per riga, ora si corregge da se'."""
    c = (codice or '').strip()
    if valuta_valida(c):
        return c
    return valuta_dichiarata(dichiarata) or 'EUR'


def problema_valuta(ticker, dichiarata, yahoo):
    """Confronta la valuta del file con quella di Yahoo → (tipo, messaggio) o None.
    GBP e GBp sono la stessa valuta (i centesimi li gestisce la conversione)."""
    if e_cambio(ticker):
        return None                     # un cambio non si converte: niente da confrontare
    d = valuta_dichiarata(dichiarata)
    y_txt = f": Yahoo lo quota in {yahoo}" if yahoo else ''
    if not d:
        return 'valuta_mancante', f"Valuta non indicata{y_txt}"
    if not valuta_valida(d):
        return 'valuta_sconosciuta', f"Valuta «{d}» non riconosciuta{y_txt}"
    if yahoo and valuta_piena(yahoo)[0] != valuta_piena(d)[0]:
        return 'valuta_diversa', (f"Nel file è {d}, ma Yahoo lo quota in {yahoo}: i prezzi "
                                  f"sono convertiti da {yahoo}")
    return None


def problema(asset, ticker, tipo, messaggio, gravita='errore', **extra):
    """Una riga del Controllo asset."""
    return {'asset': asset, 'ticker': ticker, 'tipo': tipo, 'gravita': gravita,
            'messaggio': messaggio, **extra}


def _colonna_close(raw, simbolo):
    """Close di un simbolo da una risposta di yf.download (una o piu' colonne)."""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            # con group_by='ticker' le colonne sono (simbolo, 'Close'),
            # senza (anche per un solo ticker) sono ('Close', simbolo)
            if (simbolo, 'Close') in raw.columns:
                c = raw[(simbolo, 'Close')]
            elif ('Close', simbolo) in raw.columns:
                c = raw[('Close', simbolo)]
            else:
                c = raw['Close']
        else:
            c = raw['Close']
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        return c.dropna()
    except Exception:
        return None


class CambiEuro:
    """Riporta in euro i prezzi scaricati da Yahoo. Un'istanza per download.

    La conversione usa la valuta che dice Yahoo, perche' e' quella in cui sono
    davvero i numeri che Yahoo restituisce; la valuta scritta a mano serve solo se
    Yahoo non risponde. Quella di Yahoo e' anche la valuta che finisce nel file
    (valuta_da_salvare), cosi' l'etichetta dice sempre da dove sono stati
    convertiti i prezzi. Prima veniva usata solo quella scritta a mano, e un
    titolo in dollari salvato come 'EUR' restava in dollari senza che nessuno se
    ne accorgesse.

    Se il cambio che serve non arriva, `converti` NON restituisce la serie: un
    asset lasciato in dollari passerebbe per euro e falserebbe ogni confronto.
    Le chiamate di rete stanno in thread con timeout, come nel resto del sito.
    """

    def __init__(self, tickers, dichiarate, start, timeout=30):
        self.start   = start
        self.timeout = timeout
        self._codici = {}          # {ticker: codice valuta usato}
        self._fx     = {}          # {'USD': serie EURUSD=X}
        self._fx_chiesti = set()
        self.prepara(tickers, dichiarate)

    def prepara(self, tickers, dichiarate):
        """Stabilisce la valuta dei ticker nuovi e scarica i cambi che mancano."""
        coppie = [(t, d) for t, d in zip(tickers, dichiarate) if t and t not in self._codici]
        if not coppie:
            return
        yahoo = valute_yahoo([t for t, _ in coppie if not e_cambio(t)], self.timeout)
        for t, d in coppie:
            dichiarata = valuta_dichiarata(d)
            if not valuta_valida(dichiarata):
                dichiarata = ''          # 'EIUR' non e' una valuta: nessun cambio da cercare
            # un cambio (=X) non si converte: resta nella sua unita', e lo si
            # registra come 'EUR' perche' nessuno lo riconverta dopo
            self._codici[t] = ('EUR' if e_cambio(t)
                               else yahoo.get(t) or dichiarata or 'EUR')
        servono = {valuta_piena(c)[0] for t, c in self._codici.items() if not e_cambio(t)}
        servono -= {'EUR'} | self._fx_chiesti
        if servono:
            self._scarica_cambi(sorted(servono))

    def _scarica_cambi(self, valute):
        self._fx_chiesti.update(valute)
        for _ in range(2):                     # un secondo tentativo se Yahoo torna vuoto
            mancanti = [v for v in valute if v not in self._fx]
            if not mancanti:
                return
            simboli  = [f'EUR{v}=X' for v in mancanti]
            risposta = [None]

            def _scarica():
                try:
                    import yfinance as yf
                    risposta[0] = yf.download(simboli, start=self.start, group_by='ticker',
                                              auto_adjust=True, progress=False)
                except Exception as e:
                    print(f"⚠ cambi {', '.join(simboli)}: {e}")

            th = threading.Thread(target=_scarica, daemon=True)
            th.start()
            th.join(timeout=self.timeout)
            if th.is_alive():
                print(f"⚠ cambi {', '.join(simboli)}: timeout {self.timeout}s")
                return                         # rete ferma: riprovare costerebbe altro tempo
            raw = risposta[0]
            if raw is None or raw.empty:
                continue
            for v, simbolo in zip(mancanti, simboli):
                c = _colonna_close(raw, simbolo)
                if c is not None and not c.empty:
                    self._fx[v] = c.sort_index()

    def codice(self, ticker, dichiarata=None):
        """Codice valuta usato per un ticker (lo chiede a Yahoo se e' nuovo)."""
        if ticker not in self._codici:
            self.prepara([ticker], [dichiarata])
        return self._codici.get(ticker) or 'EUR'

    def converti(self, px, ticker, dichiarata=None):
        """→ (serie in euro o None, codice valuta, messaggio d'errore o None)."""
        codice = self.codice(ticker, dichiarata)
        if px is None or e_cambio(ticker):
            return px, codice, None
        piena, divisore = valuta_piena(codice)
        if divisore != 1.0:
            px = px / divisore
        if piena == 'EUR':
            return px, codice, None
        fx = self._fx.get(piena)
        if fx is None:
            return None, codice, (f"{ticker}: cambio EUR/{piena} non disponibile — escluso "
                                  f"per non lasciarlo in {piena}, riprova più tardi")
        # Unione delle date: nei giorni in cui la borsa del titolo e' aperta ma il
        # cambio non ha il dato vale quello del giorno prima.
        fx = fx.reindex(fx.index.union(px.index)).ffill().reindex(px.index)
        return px / fx, codice, None


def download_series_eur(ticker, currency='EUR'):
    """Scarica i prezzi (Close adj) da Yahoo e li riporta in EUR.
    → (serie o None, codice valuta usato, messaggio d'errore o None)."""
    import yfinance as yf
    start = (pd.Timestamp.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    try:
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        return None, None, None
    px = _colonna_close(df, ticker)
    if px is None or px.empty:
        return None, None, None
    px, codice, errore = CambiEuro([ticker], [currency], start).converti(px, ticker, currency)
    return (px.dropna() if px is not None else None), codice, errore


def download_series(ticker, currency='EUR'):
    """Come download_series_eur, ma solo la serie in EUR (None se non disponibile)."""
    return download_series_eur(ticker, currency)[0]


def add_asset_to_current(ticker, description, currency='EUR', username=None):
    """Scarica un ticker e lo ACCODA a current.json (file unico). → (ok, messaggio)."""
    ticker      = (ticker or '').strip()
    description = (description or '').strip() or ticker
    currency    = valuta_dichiarata(currency)
    if not ticker:
        return False, "⚠ Inserisci un ticker"
    px, codice, errore = download_series_eur(ticker, currency)
    if errore:
        return False, f"⚠ {errore}"
    if px is None or len(px) < 30:
        return False, f"⚠ Nessun dato per '{ticker}' (ticker corretto?)"
    rets = clip_returns(px.pct_change(fill_method=None))   # ±50%/giorno
    dates = [d.strftime('%Y-%m-%d') for d in px.index]
    entry = {
        'ticker':   ticker,
        # La valuta e' quella con cui i prezzi sono stati convertiti (Yahoo):
        # se quella scritta era sbagliata la si corregge senza chiedere.
        'currency': valuta_da_salvare(currency, codice),
        'dates':    dates,
        'prices':   [round(float(v), 4) if pd.notna(v) else None for v in px],
        'returns':  [round(float(v), 6) if pd.notna(v) else None for v in rets],
        'checked':  False, 'P1': 0, 'P2': 0, 'P3': 0,
    }
    data = read_current(username)
    is_new = description not in data
    data[description] = entry
    if not write_current(data, username):
        return False, "⚠ Errore salvataggio current.json"
    valuta_txt = '' if (codice or 'EUR') == 'EUR' else f" — da {codice} in EUR"
    p = problema_valuta(ticker, entry['currency'], valuta_yahoo(ticker))
    avviso = f" — ⚠ {p[1]}" if p else ''
    return True, (f"✓ {description} ({ticker}) {'aggiunto' if is_new else 'aggiornato'} "
                  f"— {len(px)} prezzi{valuta_txt}{avviso}")


# ─── Controllo asset ──────────────────────────────────────────────────────────
# Un file per utente, sessions/<u>/controllo_asset.json. Lo scrivono i download
# (caricamento file, Gestisci, aggiornamento notturno) e il pulsante del
# Portafoglio. Contiene due tipi di problemi:
#  - quelli che si vedono guardando current.json (valuta diversa da Yahoo,
#    valuta mancante o inesistente, asset fermo da giorni), ricalcolati ogni volta;
#  - quelli che si vedono solo mentre si scarica (ticker che Yahoo non trova,
#    cambio mancante, riga del file scartata): li passa chi scarica. Restano
#    finche' non si ricarica la lista (azzera=True) o si riscarica quell'asset
#    (riscaricati), altrimenti sparirebbero al primo controllo senza download.
GIORNI_FERMI = 10
# Versione delle regole del controllo. Si alza quando cambia il modo di
# giudicare, cosi' i resoconti gia' salvati (che direbbero cose non piu' vere)
# vengono rifatti invece di essere mostrati com'erano.
#   2 = la valuta non e' piu' un errore da correggere a mano: si allinea da sola
VERSIONE_CONTROLLO = 2
_TIPI_DA_DOWNLOAD = {'introvabile', 'cambio_mancante', 'ticker_mancante',
                     'descrizione_duplicata', 'non_scaricato', 'ticker_sostituito'}
# quelli che un nuovo download dello stesso asset rifà da capo
_TIPI_RISCARICO = {'introvabile', 'cambio_mancante', 'non_scaricato'}


def controllo_path(username=None):
    return current_path(username).parent / 'controllo_asset.json'


def firma_asset(assets):
    """Impronta di (descrizione, ticker, valuta): se cambia, il controllo e' vecchio."""
    righe = sorted((d, str(v.get('ticker', '')), str(v.get('currency', '')))
                   for d, v in assets.items())
    return hashlib.md5(json.dumps(righe).encode('utf-8')).hexdigest()


def leggi_controllo(username=None):
    try:
        with open(controllo_path(username), encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def controllo_attuale(username=None):
    """Il controllo salvato se riguarda ancora la lista di adesso ed e' stato
    scritto con le regole di adesso, altrimenti None."""
    rep = leggi_controllo(username)
    if (rep and rep.get('versione') == VERSIONE_CONTROLLO
            and rep.get('firma') == firma_asset(read_current(username))):
        return rep
    return None


def controlla_asset(username=None, problemi_download=None, origine='controllo',
                    azzera=False, riscaricati=(), timeout=30):
    """Controlla gli asset dell'utente, salva e restituisce il resoconto.
    azzera: la lista e' nuova, i vecchi problemi di download non valgono piu'.
    riscaricati: asset appena riscaricati, per cui vale solo l'esito di adesso."""
    assets  = read_current(username)
    tickers = [str(v.get('ticker') or '').strip() for v in assets.values()]
    yahoo   = valute_yahoo([t for t in tickers if t and not e_cambio(t)], timeout)
    # La valuta giusta la sa Yahoo ed e' quella con cui i prezzi sono stati
    # convertiti: si scrive nel file invece di chiedere all'utente di correggerla.
    corrette = allinea_valute(username, assets, yahoo)

    ultimi = {}
    for d, v in assets.items():
        date = [g for g, p in zip(v.get('dates') or [], v.get('prices') or []) if p is not None]
        if date:
            try:
                ultimi[d] = pd.Timestamp(date[-1])
            except Exception:
                pass
    riferimento = max(ultimi.values()) if ultimi else None

    problemi = []
    for d, v in assets.items():
        t = str(v.get('ticker') or '').strip()
        dich = valuta_dichiarata(v.get('currency'))
        y = yahoo.get(t)
        extra = {'valuta_file': dich, 'valuta_yahoo': y or ''}
        if not t:
            problemi.append(problema(d, '', 'ticker_mancante', 'Ticker mancante', **extra))
            continue
        if d in corrette:
            prima, dopo = corrette[d]
            problemi.append(problema(
                d, t, 'valuta_corretta',
                (f"Nel file era «{prima}», Yahoo lo quota in {dopo}: valuta corretta "
                 f"da sola, i prezzi erano già convertiti in euro" if prima else
                 f"Valuta non indicata nel file: presa da Yahoo ({dopo}), "
                 f"prezzi convertiti in euro"),
                gravita='avviso', valuta_file=prima, valuta_yahoo=dopo))
        pv = problema_valuta(t, dich, y)
        if pv:
            problemi.append(problema(d, t, pv[0], pv[1], **extra))
        elif not y and t in _VALUTA_ASSENTE and not e_cambio(t):
            # solo se Yahoo ha risposto senza valuta: un timeout non e' colpa del ticker
            problemi.append(problema(
                d, t, 'valuta_non_verificata',
                'Yahoo non dà la valuta di questo ticker: controlla che sia giusto',
                gravita='avviso', **extra))
        if d not in ultimi:
            problemi.append(problema(d, t, 'senza_dati', 'Nessun prezzo salvato', **extra))
        elif (riferimento - ultimi[d]).days > GIORNI_FERMI:
            giorni = (riferimento - ultimi[d]).days
            problemi.append(problema(
                d, t, 'dati_fermi',
                f"Ultimo prezzo del {ultimi[d]:%d/%m/%Y}, {giorni} giorni prima degli altri: "
                f"ticker cambiato o non più quotato?", **extra))

    visti = {(p['asset'], p['tipo']) for p in problemi}
    for p in problemi_download or []:
        if (p.get('asset'), p.get('tipo')) not in visti:
            problemi.append(p)
            visti.add((p.get('asset'), p.get('tipo')))
    if not azzera:
        rifatti = set(riscaricati or ())
        for p in leggi_controllo(username).get('problemi', []):
            a, tipo = p.get('asset'), p.get('tipo')
            if tipo not in _TIPI_DA_DOWNLOAD or (a, tipo) in visti:
                continue
            if tipo in _TIPI_RISCARICO and a in rifatti:
                continue                # appena riscaricato: vale l'esito di adesso
            if tipo == 'ticker_sostituito' and not (
                    a in assets and assets[a].get('ticker') == p.get('ticker_usato')):
                continue                # nel file non c'e' piu' il ticker sostituto
            problemi.append(p)
            visti.add((a, tipo))

    problemi.sort(key=lambda p: (p.get('gravita') != 'errore', str(p.get('asset'))))
    resoconto = {
        'aggiornato': pd.Timestamp.now().strftime('%d/%m/%Y %H:%M'),
        'origine':    origine,
        'versione':   VERSIONE_CONTROLLO,
        'firma':      firma_asset(assets),
        'n_asset':    len(assets),
        'problemi':   problemi,
    }
    write_json_atomic(controllo_path(username), resoconto)
    n_err = sum(p.get('gravita') == 'errore' for p in problemi)
    print(f"🔎 Controllo asset [{username or get_username()}] ({origine}): "
          f"{n_err} errori, {len(problemi) - n_err} avvisi su {len(assets)} asset", flush=True)
    return resoconto


def allinea_valute(username=None, assets=None, yahoo=None, timeout=30):
    """Scrive nel file la valuta con cui Yahoo quota ogni asset, dove nel file
    manca, non esiste o e' diversa. I prezzi sono gia' in euro — il download li
    converte con quella stessa valuta — quindi qui si allinea solo l'etichetta,
    che altrimenti continuerebbe a dire 'EUR' su un titolo in dollari.
    assets: il dizionario gia' letto dal chiamante, aggiornato anche lui.
    yahoo:  {ticker: valuta} gia' chiesto a Yahoo, per non richiederlo.
    → {descrizione: (valuta_prima, valuta_dopo)} degli asset corretti."""
    path = current_path(username)
    try:
        with open(path) as f:
            raw = json.load(f)               # tutto il file: le chiavi _meta restano
    except Exception:
        return {}
    righe = {d: v for d, v in raw.items() if isinstance(v, dict)}
    if yahoo is None:
        yahoo = valute_yahoo([str(v.get('ticker') or '').strip() for v in righe.values()],
                             timeout)
    corrette = {}
    for d, v in righe.items():
        t = str(v.get('ticker') or '').strip()
        y = yahoo.get(t)
        # senza la valuta di Yahoo non si tocca niente: un timeout non e' una
        # smentita di quello che c'e' scritto nel file
        if not (t and y and valuta_valida(y)):
            continue
        prima = valuta_dichiarata(v.get('currency'))
        if not problema_valuta(t, prima, y):
            continue
        v['currency'] = y
        corrette[d] = (prima, y)
        if isinstance(assets, dict) and isinstance(assets.get(d), dict):
            assets[d]['currency'] = y
    if corrette:
        write_json_atomic(path, raw)
    return corrette


def correggi_valute(username=None, timeout=30):
    """Il pulsante «correggi le valute» del Controllo asset: ormai l'allineamento
    lo fa gia' ogni controllo, questo lo rifa' a richiesta. → asset corretti."""
    return len(allinea_valute(username, timeout=timeout))


# ─── Template / Esporta Excel ─────────────────────────────────────────────────
def template_bytes():
    """Template Excel ticker (TICKER/DESCRIZIONE/VALUTA/Peso %) formattato."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Portafoglio'
    headers    = ['TICKER', 'DESCRIZIONE', 'VALUTA', 'Peso %']
    col_widths = [14, 32, 10, 10]
    examples   = [
        ['ISAC.L',   'Az. ACWI',            'USD', ''],
        ['SWDA.MI',  'Az. World',           'EUR', ''],
        ['CSSPX.MI', 'Az. USA SP500',       'EUR', ''],
        ['EIMI.MI',  'Az. Emerging Market', 'EUR', ''],
        ['NVDA',     'NVIDIA Corporation',  'USD', ''],
    ]
    hdr_fill = PatternFill('solid', fgColor='1A3A5C')
    hdr_font = Font(bold=True, color='FFFFFF', size=10)
    hdr_aln  = Alignment(horizontal='center', vertical='center')
    alt_fill = PatternFill('solid', fgColor='EEF4FF')
    whi_fill = PatternFill('solid', fgColor='FFFFFF')
    thin     = Side(style='thin', color='C0D0E8')
    border   = Border(left=thin, right=thin, top=thin, bottom=thin)
    for ci, (h, w) in enumerate(zip(headers, col_widths), 1):
        c = ws.cell(row=1, column=ci, value=h)
        c.font, c.fill, c.alignment, c.border = hdr_font, hdr_fill, hdr_aln, border
        ws.column_dimensions[c.column_letter].width = w
    ws.row_dimensions[1].height = 18
    for ri, row_data in enumerate(examples, 2):
        fill = alt_fill if ri % 2 == 0 else whi_fill
        for ci, val in enumerate(row_data, 1):
            c = ws.cell(row=ri, column=ci, value=val)
            c.fill, c.border = fill, border
            c.alignment = Alignment(horizontal='left', vertical='center')
        ws.row_dimensions[ri].height = 16
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return out.read()


def export_bytes(username=None):
    data = read_current(username)
    rows = [{'DESCRIZIONE': k, 'TICKER': v.get('ticker', ''),
             'VALUTA': v.get('currency', 'EUR'),
             'N_PREZZI': len(v.get('prices', [])),
             'ULTIMO_PREZZO': (v.get('prices') or [None])[-1]}
            for k, v in data.items() if isinstance(v, dict)]
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        pd.DataFrame(rows).to_excel(w, index=False, sheet_name='Asset')
        prices = build_prices(username)
        if prices is not None:
            prices.to_excel(w, sheet_name='Prezzi')
    out.seek(0)
    return out.read()


# ─── "File": salva/ricarica TUTTO il lavoro (current.json + analyses.json) ────
def profili_dir(username=None):
    u = username or get_username()
    d = ROOT / 'sessions' / u / 'tattica_profili'
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_profili(username=None):
    out = []
    for p in sorted(profili_dir(username).glob('*.json'),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            kb = max(1, round(p.stat().st_size / 1024))
        except Exception:
            kb = 0
        out.append({'name': p.name, 'label': p.stem, 'kb': kb})
    return out


def save_profilo(name, username=None):
    import re as _re
    name = (name or '').strip()
    if not name:
        return False, "⚠ Scrivi un nome per il lavoro"
    current = read_current(username)
    if not current:
        return False, "⚠ Nessun dato da salvare"
    snapshot = {'_format': 'tattica_v1',
                'current': current,
                'analyses': read_analyses(username)}
    safe = _re.sub(r'[^A-Za-z0-9_\- ]+', '_', name).strip()[:40] or 'lavoro'
    if write_json_atomic(profili_dir(username) / f"{safe}.json", snapshot):
        return True, f"✓ Salvato tutto il lavoro: '{safe}'"
    return False, "⚠ Errore salvataggio"


def load_profilo(filename, username=None):
    if not filename:
        return False, "⚠ Scegli un lavoro salvato"
    try:
        snap = json.load(open(profili_dir(username) / filename))
    except Exception:
        return False, "⚠ File non leggibile"
    if isinstance(snap, dict) and '_format' in snap:
        current  = snap.get('current', {}) or {}
        analyses = snap.get('analyses', None)
    else:
        current, analyses = (snap or {}), None
    ok = write_current(current, username)
    if analyses is not None:
        write_json_atomic(analyses_path(username), analyses)
    if ok:
        extra = f", {len(analyses)} analisi" if analyses else ""
        return True, f"✓ Caricato tutto: {len(current)} asset{extra}"
    return False, "⚠ Errore caricamento"


def delete_profilo(filename, username=None):
    if not filename:
        return False, "⚠ Niente da cancellare"
    try:
        (profili_dir(username) / filename).unlink()
        return True, "🗑 Lavoro cancellato"
    except Exception as e:
        return False, f"⚠ Errore: {e}"
