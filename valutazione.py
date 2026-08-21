"""
Valutazione di un titolo azionario — modulo condiviso.

Sei modelli su dati fondamentali Yahoo Finance (yfinance): DCF, DDM, formula di
Graham, P/E relativo, EV/EBITDA e heatmap di sensitività, più una scheda
SaaS & Growth e il tab Bilanci (conto economico da Alpha Vantage).

Ogni variabile dei modelli è uno slider della barra laterale e nessun calcolo
usa un valore diverso da quello che si vede: gli slider sono tutti `Input` del
callback di disegno, così muoverne uno rifà i conti sul dato già in memoria
senza riscaricare niente. Il pulsante ▶ scarica e basta.

Era un tab di `fred/app.py` (Macro Economia); vive qui perché valutare un
singolo titolo è analisi tattica, non macroeconomia. Il blocco non usava nulla
di quel modulo (solo `app` per registrare i callback), quindi è stato spostato
senza modifiche alla logica: cambiano solo le altezze, adattate alla pagina che
lo ospita ora.

Uso da un'app Dash:
    import valutazione
    ...  valutazione.layout()              # contenuto del tab
    valutazione.register_callbacks(app)    # callback (una volta sola)
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import plotly.graph_objects as go                       # noqa: F401  (usato nei callback)
from dash import html, dcc, Input, Output, State, no_update

# ── Bilanci Alpha Vantage ────────────────────────────────────────────────────
# Il piano gratuito concede 25 richieste al giorno e una al secondo. I tre
# prospetti di un titolo (conto economico, stato patrimoniale, rendiconto) sono
# tre richieste distinte: si scaricano una volta sola e restano su disco per 30
# giorni. La cartella sta sotto `sessions/` perché è uno dei prefissi replicati
# su R2 (il disco di DO è effimero) e il nome con l'underscore non viene
# scambiato per un utente dai job notturni, che cercano un `current.json`.
_AV_URL   = "https://www.alphavantage.co/query"
_AV_KEY   = os.environ.get("ALPHA_VANTAGE_API_KEY", "ZX9YB88WV3EYUBTT")
_AV_DIR   = Path(__file__).resolve().parent / "sessions" / "_alphavantage"
_AV_TTL   = 30 * 24 * 3600          # 30 giorni
_AV_MEM   = {}                      # (ticker, prospetto) → payload: gli slider non riscaricano
_AV_ERR   = {}                      # (ticker, prospetto) → (messaggio, quando): vedi sotto
_AV_ERR_TTL = 15 * 60               # 15 minuti
_AV_PAUSA  = 1.2                    # secondi fra due chiamate: il limite è 1/s
_AV_ULTIMA = [0.0]                  # istante dell'ultima chiamata di rete

# prospetto → (funzione Alpha Vantage, suffisso del file di cache, nome esteso).
# Il conto economico resta senza suffisso: i file già scaricati (e già su R2) si
# chiamano `<TICKER>.json` e non vanno riscaricati per un cambio di nome.
_AV_PROSPETTI = {
    "ce": ("INCOME_STATEMENT", "",    "conto economico"),
    "sp": ("BALANCE_SHEET",    "_SP", "stato patrimoniale"),
    "cf": ("CASH_FLOW",        "_CF", "rendiconto finanziario"),
}

# ── Orizzonte del DCF ────────────────────────────────────────────────────────
# Fase 1 = anni 1..ANNI_FASE1 allo slider g1, fase 2 = fino ad ANNI_DCF allo
# slider g2, poi il tasso finale (perpetuità di Gordon). Le costanti stanno qui
# perché etichette, tabella delle ipotesi e colori del grafico le leggono da
# qui: cambiare l'orizzonte resta una riga sola.
ANNI_FASE1 = 3
ANNI_FASE2 = 3
ANNI_DCF   = ANNI_FASE1 + ANNI_FASE2      # ultimo anno esplicito, base del TV


def _av_num(v):
    """I valori di Alpha Vantage sono stringhe, e i buchi sono 'None'/'-'."""
    try:
        if v in (None, "None", "none", "-", ""):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def av_prospetto(ticker, prospetto="ce", forza=False):
    """Un prospetto di bilancio (annuale + trimestrale) da Alpha Vantage.

    `prospetto` è 'ce' (conto economico), 'sp' (stato patrimoniale) o 'cf'
    (rendiconto finanziario). Ritorna (payload, provenienza): `provenienza` dice
    da dove arriva il dato — memoria, disco con la data dello scarico, oppure
    rete — e in caso di errore è un messaggio da mostrare a schermo con payload
    None.
    """
    funzione, suffisso, nome = _AV_PROSPETTI[prospetto]
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None, "nessun ticker"
    ch = (ticker, prospetto)

    if not forza and ch in _AV_MEM:
        return _AV_MEM[ch], "memoria"

    # Anche i fallimenti restano in memoria per un quarto d'ora: il tab si
    # ridisegna a ogni slider e senza questo un titolo non coperto brucerebbe
    # una richiesta per movimento, sulle 25 al giorno del piano gratuito.
    if not forza and ch in _AV_ERR:
        msg, quando = _AV_ERR[ch]
        if time.time() - quando < _AV_ERR_TTL:
            return None, msg
        del _AV_ERR[ch]

    def _fallito(msg):
        _AV_ERR[ch] = (msg, time.time())
        return None, msg

    f = _AV_DIR / f"{ticker}{suffisso}.json"
    if not forza and f.exists() and (time.time() - f.stat().st_mtime) < _AV_TTL:
        try:
            payload = json.loads(f.read_text())
            _AV_MEM[ch] = payload
            giorni = (time.time() - f.stat().st_mtime) / 86400
            return payload, (f"scaricato {giorni:.0f} giorni fa"
                             if giorni >= 1 else "scaricato oggi")
        except Exception:
            pass

    # I tre prospetti si scaricano in fila: senza questa pausa la seconda
    # richiesta torna indietro con il messaggio "1 request per second".
    attesa = _AV_PAUSA - (time.time() - _AV_ULTIMA[0])
    if attesa > 0:
        time.sleep(attesa)
    _AV_ULTIMA[0] = time.time()

    url = f"{_AV_URL}?function={funzione}&symbol={ticker}&apikey={_AV_KEY}"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read().decode()
        payload = json.loads(raw)
    except Exception as e:
        return _fallito(f"scaricamento fallito: {e}")

    # Alpha Vantage risponde 200 anche quando non ha il dato: l'errore sta nel
    # corpo, come 'Note' (limite di 25 richieste al giorno) o 'Information'.
    if payload.get("Note") or payload.get("Information"):
        return _fallito(str(payload.get("Note") or payload.get("Information")))
    if not payload.get("quarterlyReports") and not payload.get("annualReports"):
        return _fallito(f"Alpha Vantage non ha il {nome} di {ticker} "
                        "(copre soprattutto i titoli USA, senza suffisso di borsa)")

    try:
        _AV_DIR.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(raw)
        tmp.replace(f)
        try:
            import data_core
            data_core.cloud_push(f)
        except Exception:
            pass
    except Exception:
        pass

    _AV_MEM[ch] = payload
    return payload, "appena scaricato"


def av_conto_economico(ticker, forza=False):
    """Scorciatoia storica: il conto economico è il prospetto di partenza."""
    return av_prospetto(ticker, "ce", forza)


def av_serie(payload, modo="ttm"):
    """Serie di fatturato, utile netto, utile lordo e margine netto.

    `modo='ttm'` somma i 4 trimestri scorrevoli (trailing twelve months),
    `modo='annuale'` usa gli esercizi come li pubblica l'azienda.
    """
    vuoto = {"date": [], "fatturato": [], "utile": [], "lordo": [],
             "margine": [], "valuta": "USD"}
    if not payload:
        return vuoto

    chiave = "annualReports" if modo == "annuale" else "quarterlyReports"
    reports = payload.get(chiave) or []
    if not reports:
        return vuoto
    # Alpha Vantage li manda dal più recente: qui servono in ordine di tempo.
    reports = sorted(reports, key=lambda r: r.get("fiscalDateEnding", ""))

    date   = [r.get("fiscalDateEnding", "") for r in reports]
    ricavi = [_av_num(r.get("totalRevenue")) for r in reports]
    utili  = [_av_num(r.get("netIncome")) for r in reports]
    # L'utile lordo si ricalcola come ricavi − costo del venduto: dove i due
    # dati ci sono coincide con il `grossProfit` dichiarato, ma su qualche
    # azienda Alpha Vantage pubblica un lordo che ignora i ricavi (ACHR 2022:
    # ricavi 10,8 mln, costo 7,7 mln, lordo dichiarato −7,7 invece di +3,1).
    lordi  = [(_av_num(r.get("totalRevenue")) - _av_num(r.get("costOfRevenue")))
              if (_av_num(r.get("totalRevenue")) is not None
                  and _av_num(r.get("costOfRevenue")) is not None)
              else _av_num(r.get("grossProfit"))
              for r in reports]
    valuta = next((r.get("reportedCurrency") for r in reports
                   if r.get("reportedCurrency")), "USD")

    if modo != "annuale":
        # Somma scorrevole di 4 trimestri: i primi 3 punti non hanno un anno
        # completo alle spalle e vanno scartati, altrimenti la serie parte da
        # un "anno" fatto di un trimestre solo.
        def _ttm(serie):
            out = []
            for i in range(len(serie)):
                fin = serie[i - 3:i + 1] if i >= 3 else []
                out.append(sum(fin) if len(fin) == 4 and None not in fin else None)
            return out

        date, ricavi, utili, lordi = (date[3:], _ttm(ricavi)[3:],
                                      _ttm(utili)[3:], _ttm(lordi)[3:])

    margine = [(u / r * 100) if (u is not None and r) else None
               for u, r in zip(utili, ricavi)]
    return {"date": date, "fatturato": ricavi, "utile": utili, "lordo": lordi,
            "margine": margine, "valuta": valuta}


# ── Prospetti riclassificati ─────────────────────────────────────────────────
# Le tre funzioni `_ricl_*` sono matematica pura: ricevono i periodi già estratti
# e restituiscono le righe dello schema, che il tab si limita a impaginare.
# Ogni riga è (stile, etichetta, valori, nota):
#   voce  = riga normale      sub  = "di cui", rientrata
#   tot   = subtotale         totf = totale di sezione, in evidenza
#   sez   = intestazione di sezione (senza valori)
#   memo  = riga di richiamo, fuori dalla cascata (corsivo)

def _av_v(p, *chiavi):
    """Primo campo valorizzato fra quelli indicati: Alpha Vantage cambia nome
    alla stessa voce da azienda ad azienda e lascia buchi con disinvoltura."""
    for k in chiavi:
        if p.get(k) is not None:
            return p[k]
    return None


def _z(x):
    """None come zero: serve solo dentro i calcoli, mai in ciò che si stampa
    (una voce assente resta un trattino, non uno zero inventato)."""
    return 0.0 if x is None else x


def av_periodi(payload, modo="ttm", flusso=True, n=4):
    """Ultimi `n` periodi di un prospetto, dal più recente. Ritorna (periodi, valuta).

    `flusso=True` (conto economico, rendiconto) in modo TTM somma i 4 trimestri
    scorrevoli. `flusso=False` è lo stato patrimoniale: una situazione
    patrimoniale è la fotografia a una data, sommare quattro trimestri
    quadruplicherebbe il capitale — lì il TTM prende il trimestre così com'è.
    """
    if not payload:
        return [], "USD"
    chiave  = "annualReports" if modo == "annuale" else "quarterlyReports"
    reports = sorted(payload.get(chiave) or [],
                     key=lambda r: r.get("fiscalDateEnding", ""))
    valuta = next((r.get("reportedCurrency") for r in reports
                   if r.get("reportedCurrency")), "USD")

    out = []
    for i in range(len(reports) - 1, -1, -1):
        if len(out) >= n:
            break
        if modo == "annuale" or not flusso:
            voci = {k: _av_num(v) for k, v in reports[i].items()}
        else:
            if i < 3:
                break                      # meno di un anno completo alle spalle
            fin  = reports[i - 3:i + 1]
            voci = {}
            for k in reports[i]:
                val = [_av_num(r.get(k)) for r in fin]
                voci[k] = sum(val) if None not in val else None
        out.append((reports[i].get("fiscalDateEnding", ""), voci))
    return out, valuta


def av_scala(righe):
    """Unità della tabella, scelta una volta sola sul valore più grande: i
    colossi in miliardi, gli altri in milioni. Colonne tutte sulla stessa scala."""
    val = [v for _, _, valori, _ in righe for v in valori if v is not None]
    m = max((abs(v) for v in val), default=0)
    if m >= 1e10:
        return 1e9, "mld", 2
    if m >= 1e7:
        return 1e6, "mln", 1
    return 1e3, "migliaia", 0


def _ricl_stato_patrimoniale(periodi):
    """Stato patrimoniale riclassificato secondo il criterio funzionale.

    Non è la riclassificazione per liquidità (attivo/passivo corrente): qui si
    risponde a "quanto capitale è investito nell'attività" e "chi lo finanzia".
    Impieghi e fonti quadrano **sempre**, perché il capitale investito netto è
    ricavato dai totali di bilancio (attivo − passivo) e non da una somma di
    voci scelte a mano: le poste non dettagliate finiscono nelle righe "altre",
    calcolate per differenza, invece di sparire lasciando la tabella scoperta.
    """
    calc = []
    for _, p in periodi:
        # Cassa: `cashAndShortTermInvestments` a volte contiene già i titoli a
        # breve e a volte no (è uguale alla sola cassa) — si tiene il maggiore.
        liq   = _z(_av_v(p, "cashAndCashEquivalentsAtCarryingValue",
                         "cashAndCashEquivalentsAtCarrying"))
        tit   = _z(_av_v(p, "shortTermInvestments"))
        cassa = max(_z(_av_v(p, "cashAndShortTermInvestments")), liq + tit)

        att_c = _z(_av_v(p, "totalCurrentAssets"))
        pas_c = _z(_av_v(p, "totalCurrentLiabilities"))
        tot_a = _z(_av_v(p, "totalAssets"))
        tot_p = _z(_av_v(p, "totalLiabilities"))
        att_n = _av_v(p, "totalNonCurrentAssets")
        att_n = tot_a - att_c if att_n is None else att_n
        pas_n = _av_v(p, "totalNonCurrentLiabilities")
        pas_n = tot_p - pas_c if pas_n is None else pas_n

        deb_c = _z(_av_v(p, "shortTermDebt", "currentDebt")) + \
            _z(_av_v(p, "currentLongTermDebt"))
        deb_n = _z(_av_v(p, "longTermDebt", "longTermDebtNoncurrent")) + \
            _z(_av_v(p, "capitalLeaseObligations"))

        crediti  = _z(_av_v(p, "currentNetReceivables"))
        rimanenze = _z(_av_v(p, "inventory"))
        fornitori = _z(_av_v(p, "currentAccountsPayable"))
        avv      = _z(_av_v(p, "goodwill"))
        imm      = _z(_av_v(p, "intangibleAssetsExcludingGoodwill"))
        if not imm:
            imm = max(_z(_av_v(p, "intangibleAssets")) - avv, 0)

        ccc  = crediti + rimanenze - fornitori
        alt_ac = att_c - crediti - rimanenze - cassa      # per differenza
        alt_pc = pas_c - fornitori - deb_c
        ccn  = ccc + alt_ac - alt_pc
        mat  = att_n - avv - imm                          # materiali e altre
        alt_pn = pas_n - deb_n
        cin  = ccn + avv + imm + mat - alt_pn

        pfn = deb_c + deb_n - cassa
        pn  = _z(_av_v(p, "totalShareholderEquity"))
        calc.append(dict(
            crediti=crediti, rimanenze=rimanenze, fornitori=-fornitori, ccc=ccc,
            alt_ac=alt_ac, alt_pc=-alt_pc, ccn=ccn, avv=avv, imm=imm, mat=mat,
            alt_pn=-alt_pn, cin=cin, deb_c=deb_c, deb_n=deb_n, cassa=-cassa,
            pfn=pfn, pn=pn, terzi=cin - pfn - pn, fonti=cin))

    def c(k):
        return [x[k] for x in calc]

    righe = [
        ("sez",  "Impieghi — dove è investito il capitale", [], ""),
        ("voce", "Crediti commerciali", c("crediti"), ""),
        ("voce", "Rimanenze", c("rimanenze"), ""),
        ("voce", "Debiti verso fornitori", c("fornitori"), ""),
        ("tot",  "Capitale circolante commerciale", c("ccc"), ""),
        ("voce", "Altre attività correnti", c("alt_ac"), ""),
        ("voce", "Altre passività correnti", c("alt_pc"), ""),
        ("tot",  "Capitale circolante netto (CCN)", c("ccn"), ""),
        ("voce", "Avviamento", c("avv"), ""),
        ("voce", "Altre immobilizzazioni immateriali", c("imm"), ""),
        ("voce", "Immobilizzazioni materiali e altre attività non correnti",
         c("mat"), ""),
        ("voce", "Altre passività non correnti", c("alt_pn"), ""),
        ("totf", "CAPITALE INVESTITO NETTO (CIN)", c("cin"), ""),
        ("sez",  "Fonti — chi lo finanzia", [], ""),
        ("voce", "Debiti finanziari correnti", c("deb_c"), ""),
        ("voce", "Debiti finanziari non correnti e leasing", c("deb_n"), ""),
        ("voce", "Disponibilità liquide e titoli a breve", c("cassa"), ""),
        ("tot",  "Posizione finanziaria netta (PFN)", c("pfn"), ""),
        ("voce", "Patrimonio netto", c("pn"), ""),
    ]
    # Attivo − passivo non fa il patrimonio netto quando ci sono soci di
    # minoranza: la differenza è una voce vera, non un errore da nascondere.
    if any(abs(x) > 1e6 for x in c("terzi")):
        righe.append(("voce", "Interessi di minoranza e altre differenze",
                      c("terzi"), ""))
    righe.append(("totf", "TOTALE FONTI (PFN + patrimonio netto)", c("fonti"), ""))
    return righe


def _ricl_conto_economico(periodi):
    """Conto economico riclassificato a costo del venduto, in forma scalare."""
    calc = []
    for _, p in periodi:
        ricavi = _av_v(p, "totalRevenue")
        costo  = _av_v(p, "costOfRevenue", "costofGoodsAndServicesSold")
        dich   = _av_v(p, "grossProfit")          # lordo come lo pubblica AV
        # In una riclassificazione la cascata deve chiudere: il margine lordo è
        # ricavi − costo del venduto. Il dato dichiarato serve solo se manca il
        # costo, e quando i due divergono lo si dice sotto la tabella invece di
        # far comparire un subtotale che non torna con le righe sopra.
        if None not in (ricavi, costo):
            lordo = ricavi - costo
        else:
            lordo = dich
            if costo is None and None not in (ricavi, lordo):
                costo = ricavi - lordo
        rs   = _av_v(p, "researchAndDevelopment")
        sga  = _av_v(p, "sellingGeneralAndAdministrative")
        ebit = _av_v(p, "operatingIncome")
        if ebit is None and lordo is not None:
            ebit = lordo - _z(_av_v(p, "operatingExpenses"))
        # Quanto resta fra margine lordo e risultato operativo dopo R&S e SG&A:
        # ammortamenti non allocati, accantonamenti, oneri una tantum.
        altri = _z(lordo) - _z(rs) - _z(sga) - _z(ebit)
        amm   = _av_v(p, "depreciationAndAmortization", "depreciation")
        ante  = _av_v(p, "incomeBeforeTax")
        netto = _av_v(p, "netIncome", "netIncomeFromContinuingOperations")
        imp   = _av_v(p, "incomeTaxExpense")
        if ante is None and None not in (netto, imp):
            ante = netto + imp
        # Le imposte mancano spesso in un trimestre, e nel TTM basta quel buco a
        # farle sparire dalla colonna: sono comunque la differenza fra risultato
        # ante imposte e utile netto, così la cascata si chiude a vista.
        if imp is None and None not in (ante, netto):
            imp = ante - netto
        calc.append(dict(
            ricavi=ricavi, costo=None if costo is None else -costo, lordo=lordo,
            dich=dich,
            rs=None if rs is None else -rs, sga=None if sga is None else -sga,
            altri=-altri, ebit=ebit, amm=amm,
            ebitda=None if ebit is None else ebit + _z(amm),
            fin=None if None in (ante, ebit) else ante - ebit, ante=ante,
            imp=None if imp is None else -imp,
            resid=None if None in (netto, ante, imp) else netto - (ante - imp),
            netto=netto))

    def c(k):
        return [x[k] for x in calc]

    # La colonna delle percentuali guarda il periodo più recente: è lì che si
    # legge la struttura di costo attuale, il resto della riga dà la tendenza.
    ric0 = calc[0]["ricavi"] if calc else None

    def q(k):
        v = calc[0][k] if calc else None
        if v is None or not ric0:
            return ""
        pct = v / ric0 * 100
        # Su un'azienda che i ricavi non li ha ancora (una biotech, un
        # costruttore prima delle consegne) l'incidenza è un numero a cinque
        # cifre che non dice niente: meglio dichiararlo non significativo.
        return "n.s." if abs(pct) > 999 else f"{pct:,.1f}%"

    righe = [
        ("voce", "Ricavi", c("ricavi"), q("ricavi")),
        ("voce", "Costo del venduto", c("costo"), q("costo")),
        ("tot",  "Margine lordo", c("lordo"), q("lordo")),
        ("voce", "Ricerca e sviluppo", c("rs"), q("rs")),
        ("voce", "Costi commerciali, generali e amministrativi", c("sga"), q("sga")),
        ("voce", "Altri costi operativi netti", c("altri"), q("altri")),
        ("tot",  "Risultato operativo (EBIT)", c("ebit"), q("ebit")),
        ("memo", "Ammortamenti e svalutazioni", c("amm"), q("amm")),
        ("memo", "EBITDA (EBIT + ammortamenti)", c("ebitda"), q("ebitda")),
        ("voce", "Gestione finanziaria e proventi/oneri vari", c("fin"), q("fin")),
        ("tot",  "Risultato ante imposte", c("ante"), q("ante")),
        ("voce", "Imposte sul reddito", c("imp"), q("imp")),
    ]
    if any(v is not None and abs(v) > 1e6 for v in c("resid")):
        righe.append(("voce", "Attività cessate e altre componenti",
                      c("resid"), q("resid")))
    righe.append(("totf", "UTILE NETTO", c("netto"), q("netto")))

    # Se il margine lordo pubblicato non coincide con ricavi − costo, la
    # tabella resta coerente e lo scarto si dichiara: è un difetto della fonte.
    scarti = [(d, abs(x["lordo"] - x["dich"]))
              for (d, _), x in zip(periodi, calc)
              if None not in (x["lordo"], x["dich"], x["ricavi"])
              and abs(x["lordo"] - x["dich"]) > max(abs(x["ricavi"]) * 0.01, 1e6)]
    if scarti:
        date_s = ", ".join(d for d, _ in scarti)
        righe.append(("nota", "Il margine lordo è ricalcolato come ricavi − costo "
                              f"del venduto: su {date_s} Alpha Vantage pubblica un "
                              "utile lordo che non torna con le due voci.", [], ""))
    return righe


def _ricl_rendiconto(periodi):
    """Rendiconto finanziario riclassificato fino al free cash flow."""
    calc = []
    for _, p in periodi:
        utile = _av_v(p, "netIncome", "profitLoss")
        amm   = _av_v(p, "depreciationDepletionAndAmortization")
        sbc   = _av_v(p, "stockBasedCompensation")
        cfo   = _av_v(p, "operatingCashflow")
        # Capitale circolante e rettifiche non monetarie per differenza: Alpha
        # Vantage lascia quasi sempre vuoti i campi di dettaglio, ma il totale
        # del flusso operativo c'è, e da lì si torna indietro.
        altre = None if cfo is None else cfo - _z(utile) - _z(amm) - _z(sbc)
        capex = _av_v(p, "capitalExpenditures")
        cfi   = _av_v(p, "cashflowFromInvestment")
        cff   = _av_v(p, "cashflowFromFinancing")
        div   = _av_v(p, "dividendPayout", "dividendPayoutCommonStock")
        riac  = _av_v(p, "proceedsFromRepurchaseOfEquity",
                      "paymentsForRepurchaseOfCommonStock",
                      "paymentsForRepurchaseOfEquity")
        var   = _av_v(p, "changeInCashAndCashEquivalents")
        if var is None and cfo is not None:
            var = cfo + _z(cfi) + _z(cff)
        calc.append(dict(
            utile=utile, amm=amm, sbc=sbc, altre=altre, cfo=cfo,
            capex=None if capex is None else -abs(capex),
            fcf=None if cfo is None else cfo - abs(_z(capex)),
            cfi=cfi, cff=cff,
            div=None if div is None else -abs(div),
            riac=None if riac is None else -abs(riac), var=var))

    def c(k):
        return [x[k] for x in calc]

    return [
        ("sez",  "Gestione operativa", [], ""),
        ("voce", "Utile netto", c("utile"), ""),
        ("voce", "Ammortamenti e svalutazioni", c("amm"), ""),
        ("voce", "Compensi in azioni", c("sbc"), ""),
        ("voce", "Capitale circolante e altre rettifiche", c("altre"), ""),
        ("tot",  "Flusso di cassa operativo (CFO)", c("cfo"), ""),
        ("voce", "Investimenti in immobilizzazioni (capex)", c("capex"), ""),
        ("totf", "FREE CASH FLOW (CFO − capex)", c("fcf"), ""),
        ("sez",  "Flussi complessivi del periodo", [], ""),
        ("voce", "Flusso da attività di investimento", c("cfi"), ""),
        ("sub",  "di cui investimenti in immobilizzazioni", c("capex"), ""),
        ("voce", "Flusso da attività di finanziamento", c("cff"), ""),
        ("sub",  "di cui dividendi pagati", c("div"), ""),
        ("sub",  "di cui riacquisto di azioni proprie", c("riac"), ""),
        ("tot",  "Variazione delle disponibilità liquide", c("var"), ""),
    ]


def layout():
    """Tab valutazione titolo azionario — 6 modelli + heatmap sensitività."""

    def _inp(id_, placeholder, value="", width="100%", type_="text"):
        return dcc.Input(id=id_, type=type_, placeholder=placeholder, value=value,
                         debounce=True,
                         style={"width": width, "padding": "5px 8px",
                                "border": "1px solid #ccc", "border-radius": "4px",
                                "font-size": "12px"})

    def _lbl(text):
        return html.Label(text, style={"font-size": "10px", "color": "#555",
                                       "margin-top": "8px", "display": "block"})

    def _sl(id_, mn, mx, step, val, label):
        return html.Div([
            html.Label(label, style={"font-size": "10px", "color": "#555"}),
            dcc.Slider(id=id_, min=mn, max=mx, step=step, value=val,
                       tooltip={"placement": "bottom", "always_visible": True},
                       marks={}),
            html.Div(style={"height": "6px"}),
        ])

    def _grp(text):
        """Intestazione di un gruppo di slider: un gruppo per modello."""
        return html.B(text, style={"font-size": "10px", "color": "#1a5276",
                                   "background": "#eaf4fb", "display": "block",
                                   "padding": "4px 8px", "border-radius": "3px",
                                   "margin-bottom": "8px"})

    sidebar = html.Div([
        html.B("🔍 Ticker", style={"font-size": "10px", "color": "#1a5276",
                                    "background": "#eaf4fb", "display": "block",
                                    "padding": "4px 8px", "border-radius": "3px",
                                    "margin-bottom": "8px"}),
        _lbl("Simbolo (es. AAPL, ENI.MI, MC.PA)"),
        _inp("val-ticker", "Ticker Yahoo Finance", "AAPL"),
        html.Button("▶ Carica & Valuta", id="btn-run-valuation", n_clicks=0,
                    style={"width": "100%", "margin-top": "8px",
                           "background": "#1a3a5c", "color": "white",
                           "border": "none", "border-radius": "6px",
                           "padding": "8px", "font-size": "12px",
                           "cursor": "pointer"}),
        html.Div(id="val-fetch-status",
                 style={"font-size": "10px", "color": "#555",
                        "margin-top": "6px", "white-space": "pre-wrap"}),

        html.Hr(style={"margin": "10px 0"}),

        _grp("📉 DCF — flussi di cassa"),
        _sl("val-wacc",    4.0, 20.0, 0.5,  9.0, "WACC — tasso di sconto (%)"),
        _sl("val-g1", 0.0, 40.0, 0.5, 12.0,
            f"Crescita anni 1-{ANNI_FASE1} (%)"),
        _sl("val-g2", 0.0, 20.0, 0.5, 6.0,
            f"Crescita anni {ANNI_FASE1 + 1}-{ANNI_DCF} (%)"),
        _sl("val-gterm",   0.0,  6.0, 0.25, 2.5,
            f"Tasso finale — da anno {ANNI_DCF + 1} in poi (%)"),
        _sl("val-fcf-margin", 1.0, 50.0, 0.5, 15.0, "Margine FCF/Revenue (%)"),
        html.Div(id="val-fcf-nota",
                 style={"font-size": "9px", "color": "#888", "margin": "-2px 0 4px",
                        "line-height": "1.45"}),

        html.Hr(style={"margin": "10px 0"}),

        _grp("💰 DDM — dividendi"),
        _sl("val-ke",     4.0, 20.0, 0.5, 10.0, "Ke — costo del capitale proprio (%)"),
        _sl("val-ddm-g",  0.0, 10.0, 0.25, 2.5, "Crescita del dividendo (%)"),

        html.Hr(style={"margin": "10px 0"}),

        _grp("📐 Graham"),
        _sl("val-graham-g",   0.0, 25.0, 0.5,  8.0, "Crescita EPS attesa (%)"),
        _sl("val-bond-yield", 1.0, 10.0, 0.25, 4.5, "Rendimento AAA bond (%)"),

        html.Hr(style={"margin": "10px 0"}),

        _grp("📈 Multipli"),
        _sl("val-pe-sector",    5.0, 60.0, 1.0, 22.0, "P/E settore (multiplo)"),
        _sl("val-ev-ebitda",    3.0, 30.0, 0.5, 12.0, "EV/EBITDA settore (multiplo)"),

        html.Hr(style={"margin": "10px 0"}),

        _grp("🏢 Bilanci — base di calcolo"),
        dcc.RadioItems(
            id="val-bilanci-modo",
            options=[{"label": " Trimestrali TTM (4 trimestri scorrevoli)",
                      "value": "ttm"},
                     {"label": " Annuali (esercizi pubblicati)",
                      "value": "annuale"}],
            value="ttm",
            labelStyle={"display": "block", "font-size": "11px",
                        "color": "#333", "margin-bottom": "4px"}),
        html.Div("Vale per i 4 grafici del tab Bilanci.",
                 style={"font-size": "9px", "color": "#888",
                        "line-height": "1.45", "margin-top": "2px"}),

    ], style={"width": "270px", "min-width": "270px", "padding": "14px",
              "background": "#fafafa", "border-right": "1px solid #ddd",
              "overflow-y": "auto", "height": "calc(100vh - 250px)",
              "min-height": "520px"})

    results = html.Div([
        dcc.Tabs(id="val-result-tabs", value="val-tab-summary",
                 children=[
                     dcc.Tab(label="📊 Riepilogo",       value="val-tab-summary"),
                     dcc.Tab(label="🏢 Bilanci",          value="val-tab-bilanci"),
                     dcc.Tab(label="📉 DCF",              value="val-tab-dcf"),
                     dcc.Tab(label="💰 DDM",              value="val-tab-ddm"),
                     dcc.Tab(label="📐 Graham",           value="val-tab-graham"),
                     dcc.Tab(label="📈 P/E & EV/EBITDA",  value="val-tab-multiples"),
                     dcc.Tab(label="🔥 Sensitività",      value="val-tab-heatmap"),
                     dcc.Tab(label="📱 SaaS & Growth",   value="val-tab-saas"),
                 ],
                 style={"font-size": "12px"}),

        dcc.Loading(
            id="val-loading", type="circle", color="#1a3a5c",
            children=html.Div(id="val-tab-content",
                              style={"padding": "10px",
                                     "height": "calc(100vh - 310px)", "min-height": "460px",
                                     "overflow-y": "auto"})),

        dcc.Store(id="store-valuation", storage_type="session"),
    ], style={"flex": "1", "overflow": "hidden"})

    return html.Div([
        html.Div([
            html.H3("Valutazione Titolo Azionario",
                    style={"margin": "0 20px 0 0", "font-size": "15px",
                           "color": "#1a3a5c", "white-space": "nowrap"}),
            html.Span("DCF · DDM · Graham · P/E relativo · EV/EBITDA · Heatmap sensitività — "
                      "dati fondamentali da Yahoo Finance (yfinance)",
                      style={"font-size": "11px", "color": "#666"}),
        ], style={"display": "flex", "align-items": "center",
                  "padding": "8px 16px", "background": "#f0f4fa",
                  "border-bottom": "1px solid #dee2e6",
                  "flex-wrap": "wrap", "gap": "8px"}),
        html.Div([sidebar, results],
                 style={"display": "flex", "height": "calc(100vh - 250px)",
                        "min-height": "520px"}),
    ])


def register_callbacks(app):
    """Registra i callback del tab Valutazione sull'app passata."""
    # ── scarica i fondamentali e posiziona gli slider sui dati del titolo ─────
    @app.callback(
        Output("store-valuation",  "data"),
        Output("val-fetch-status", "children"),
        Output("val-fcf-margin",   "value"),
        Output("val-fcf-nota",     "children"),
        Output("val-graham-g",     "value"),
        Input("btn-run-valuation", "n_clicks"),
        State("val-ticker",        "value"),
        prevent_initial_call=True,
    )
    def run_valuation(n_clicks, ticker):
        import json, traceback
        import yfinance as yf

        if not ticker:
            return (no_update, "⚠ Inserisci un ticker.", no_update, no_update,
                    no_update)

        ticker = ticker.strip().upper()
        try:
            t    = yf.Ticker(ticker)
            info = t.info

            def _g(key, default=None):
                v = info.get(key)
                return default if (v is None or v != v) else v  # NaN check

            name          = _g("shortName", ticker)
            sector        = _g("sector", "N/D")
            industry      = _g("industry", "N/D")
            currency      = _g("currency", "USD")
            price         = _g("currentPrice") or _g("regularMarketPrice", 0)
            market_cap    = _g("marketCap", 0)
            shares        = _g("sharesOutstanding", 0)
            eps_ttm       = _g("trailingEps", 0)
            eps_fwd       = _g("forwardEps")  or eps_ttm
            revenue       = _g("totalRevenue", 0)
            ebitda        = _g("ebitda", 0)
            fcf_yf        = _g("freeCashflow", 0)
            total_debt    = _g("totalDebt", 0)
            cash          = _g("totalCash", 0)
            net_debt      = total_debt - cash
            dividend      = _g("dividendRate", 0) or 0
            beta          = _g("beta", 1.0) or 1.0
            pe_trailing   = _g("trailingPE")
            pe_forward    = _g("forwardPE")
            book_val      = _g("bookValue", 0)
            revenue_growth  = _g("revenueGrowth", 0) or 0   # YoY
            gross_margins   = _g("grossMargins", 0) or 0     # %
            gross_profits   = _g("grossProfits", 0) or 0
            ebitda_margins  = _g("ebitdaMargins", 0) or 0    # %
            operating_margins = _g("operatingMargins", 0) or 0
            ps_trailing     = _g("priceToSalesTrailing12Months")
            ev              = _g("enterpriseValue", 0) or 0
            # R&D: non sempre in info, proviamo financials
            rd_expense = 0
            try:
                fin = t.financials
                if fin is not None and not fin.empty:
                    rd_keys = [k for k in fin.index
                               if "research" in k.lower() or "development" in k.lower()]
                    if rd_keys:
                        rd_series = fin.loc[rd_keys[0]]
                        rd_vals   = rd_series.dropna().values
                        rd_expense = abs(float(rd_vals[0])) if len(rd_vals) > 0 else 0
            except Exception:
                pass

            # FCF dal rendiconto finanziario: 'info["freeCashflow"]' è spesso un
            # TTM sballato (MSFT: 16.5B contro i 67B del rendiconto) e siccome è
            # la base di tutto il DCF si preferisce il dato di bilancio.
            fcf_bilancio, fcf_esercizio = 0, ""
            try:
                cf = t.cashflow
                if cf is not None and not cf.empty:
                    k_fcf = [k for k in cf.index if "free cash flow" in k.lower()]
                    if k_fcf:
                        serie = cf.loc[k_fcf[0]].dropna()
                        if len(serie) > 0:
                            fcf_bilancio  = float(serie.iloc[0])
                            fcf_esercizio = str(serie.index[0].date())
            except Exception:
                pass

            fcf_margin_actual = (fcf_yf / revenue) if (revenue > 0 and fcf_yf) else 0

            d = {
                "ticker": ticker, "name": name, "sector": sector,
                "industry": industry, "currency": currency,
                "price": price, "market_cap": market_cap, "shares": shares,
                "eps_ttm": eps_ttm, "eps_fwd": eps_fwd,
                "revenue": revenue, "ebitda": ebitda, "fcf_yf": fcf_yf,
                "fcf_margin_actual": fcf_margin_actual,
                "fcf_bilancio": fcf_bilancio, "fcf_esercizio": fcf_esercizio,
                "total_debt": total_debt, "cash": cash, "net_debt": net_debt,
                "dividend": dividend, "beta": beta,
                "pe_trailing": pe_trailing, "pe_forward": pe_forward,
                "book_val": book_val, "revenue_growth": revenue_growth,
                "gross_margins": gross_margins, "gross_profits": gross_profits,
                "ebitda_margins": ebitda_margins, "operating_margins": operating_margins,
                "ps_trailing": ps_trailing, "ev": ev, "rd_expense": rd_expense,
            }

            # Gli slider si posizionano sul dato reale del titolo e da lì restano
            # tuoi: nessun modello usa un valore diverso da quello che vedi.
            marg = (fcf_bilancio / revenue * 100) if (revenue > 0 and fcf_bilancio) \
                else (fcf_margin_actual * 100 if fcf_margin_actual else no_update)
            if marg is not no_update:
                marg = round(min(50.0, max(1.0, marg)) * 2) / 2      # step 0.5

            gr_g = round(min(25.0, max(0.0, revenue_growth * 100)) * 2) / 2 \
                if revenue_growth else no_update

            nota = []
            if fcf_bilancio and revenue > 0:
                nota.append(f"rendiconto {fcf_esercizio}: {fcf_bilancio/1e9:,.1f} mld "
                            f"({fcf_bilancio/revenue*100:.1f}%)")
            if fcf_yf and revenue > 0:
                nota.append(f"yfinance TTM: {fcf_yf/1e9:,.1f} mld "
                            f"({fcf_yf/revenue*100:.1f}%)")
            if not nota:
                nota.append("nessun FCF disponibile: imposta il margine a mano")

            status = f"✅ {name} ({ticker}) — {sector} | {currency} | prezzo: {price:.2f}"
            return json.dumps(d), status, marg, " · ".join(nota), gr_g

        except Exception as e:
            tb = traceback.format_exc()
            print(f"=== VALUATION ERROR ===\n{tb}")
            return no_update, f"❌ {e}", no_update, no_update, no_update

    # ── ricalcolo: ogni slider rifà i conti sul dato già in memoria ───────────
    @app.callback(
        Output("val-tab-content", "children"),
        Input("store-valuation",  "data"),
        Input("val-result-tabs",  "value"),
        Input("val-wacc",         "value"),
        Input("val-g1",           "value"),
        Input("val-g2",           "value"),
        Input("val-gterm",        "value"),
        Input("val-fcf-margin",   "value"),
        Input("val-pe-sector",    "value"),
        Input("val-ev-ebitda",    "value"),
        Input("val-ke",           "value"),
        Input("val-ddm-g",        "value"),
        Input("val-graham-g",     "value"),
        Input("val-bond-yield",   "value"),
        Input("val-bilanci-modo", "value"),
    )
    def _val_render(stored, active_tab, wacc, g1, g2, gterm, fcf_margin,
                    pe_sector, ev_ebitda_mult, ke, ddm_g, graham_g, bond_yield,
                    bilanci_modo):
        import json

        if not stored:
            return html.Div("Inserisci un ticker e clicca ▶ Carica & Valuta.",
                            style={"padding": "40px", "color": "#888",
                                   "text-align": "center", "font-size": "14px"})
        try:
            d = json.loads(stored) if isinstance(stored, str) else stored
        except Exception:
            return html.Div("Dati non leggibili: ricarica il titolo.",
                            style={"padding": "40px", "color": "#888",
                                   "text-align": "center", "font-size": "14px"})

        # `or` non va bene: 0 è un valore legittimo per le crescite.
        def _n(v, dflt):
            return float(dflt if v is None else v)

        return _val_build_content(
            d, active_tab,
            _n(wacc, 9.0) / 100, _n(g1, 12.0) / 100,
            _n(g2, 6.0) / 100, _n(gterm, 2.5) / 100,
            _n(fcf_margin, 15.0) / 100, _n(pe_sector, 22.0),
            _n(ev_ebitda_mult, 12.0), _n(ke, 10.0) / 100,
            _n(ddm_g, 2.5) / 100, _n(graham_g, 8.0),
            _n(bond_yield, 4.5) / 100, bilanci_modo or "ttm")


    def _val_fmt_num(v, decimals=2, suffix=""):
        """Formatta numero grande in M/B."""
        if v is None or v != v: return "N/D"
        if abs(v) >= 1e12: return f"{v/1e12:.{decimals}f}T{suffix}"
        if abs(v) >= 1e9:  return f"{v/1e9:.{decimals}f}B{suffix}"
        if abs(v) >= 1e6:  return f"{v/1e6:.{decimals}f}M{suffix}"
        return f"{v:.{decimals}f}{suffix}"


    def _val_saas_tab(d):
        """Tab metriche SaaS & Growth: Rule of 40, ARR, P/S, EV/Rev, Gross Margin, R&D."""
        price          = d.get("price", 0) or 0
        revenue        = d.get("revenue", 0) or 0
        ebitda         = d.get("ebitda", 0) or 0
        market_cap     = d.get("market_cap", 0) or 0
        ev             = d.get("ev", 0) or 0
        shares         = d.get("shares", 0) or 1
        fcf_yf         = d.get("fcf_yf", 0) or 0
        rev_growth     = d.get("revenue_growth", 0) or 0        # decimale
        gross_margins  = d.get("gross_margins", 0) or 0         # decimale
        ebitda_margins = d.get("ebitda_margins", 0) or 0        # decimale
        op_margins     = d.get("operating_margins", 0) or 0
        ps_trailing    = d.get("ps_trailing")
        rd_expense     = d.get("rd_expense", 0) or 0
        currency       = d.get("currency", "USD")
        name           = d.get("name", d.get("ticker", ""))

        # ── calcoli ────────────────────────────────────────────────────────────
        # ARR proxy: per aziende non-SaaS = revenue TTM; per SaaS ideale sarebbe MRR×12
        arr_proxy = revenue  # yfinance non distingue ARR da revenue

        # Rule of 40
        rule40_val = rev_growth * 100 + ebitda_margins * 100
        rule40_ok  = rule40_val >= 40

        # P/S
        ps_calc = (market_cap / revenue) if revenue > 0 else None

        # EV/Revenue
        ev_rev = (ev / revenue) if (ev > 0 and revenue > 0) else None

        # EV/ARR (= EV/Revenue per proxy)
        ev_arr = ev_rev

        # FCF margin
        fcf_margin_act = (fcf_yf / revenue) if revenue > 0 else None

        # R&D as % of revenue
        rd_pct = (rd_expense / revenue) if revenue > 0 else None

        # Gross margin %
        gm_pct = gross_margins * 100

        td  = {"padding": "5px 10px", "borderBottom": "1px solid #eee", "fontSize": "12px"}
        tbl = {"width": "100%", "borderCollapse": "collapse",
               "border": "1px solid #ddd", "marginBottom": "12px"}
        th  = {"background": "#f0f0f0", "padding": "6px 10px",
               "fontSize": "12px", "textAlign": "left"}

        def _badge(val, good_thresh, bad_thresh, fmt, higher_is_better=True):
            """Pill colorato: verde se buono, arancio se medio, rosso se scarso."""
            if val is None: return html.Span("N/D", style={"color": "#888"})
            txt = fmt.format(val)
            if higher_is_better:
                col = "#2ca02c" if val >= good_thresh else "#ff7f0e" if val >= bad_thresh else "#d62728"
            else:
                col = "#2ca02c" if val <= good_thresh else "#ff7f0e" if val <= bad_thresh else "#d62728"
            return html.Span(txt, style={"background": col, "color": "white",
                                          "padding": "2px 10px", "borderRadius": "12px",
                                          "fontWeight": "bold", "fontSize": "12px"})

        # ── Rule of 40 gauge ─────────────────────────────────────────────────
        r40_col = "#2ca02c" if rule40_ok else "#d62728"
        fig_r40 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rule40_val,
            delta={"reference": 40, "valueformat": ".1f",
                   "increasing": {"color": "#2ca02c"},
                   "decreasing": {"color": "#d62728"}},
            title={"text": "Rule of 40", "font": {"size": 14}},
            gauge={
                "axis": {"range": [-20, 100], "tickwidth": 1},
                "bar":  {"color": r40_col},
                "steps": [
                    {"range": [-20, 0],  "color": "#ffebee"},
                    {"range": [0,  40],  "color": "#fff8e1"},
                    {"range": [40, 100], "color": "#e8f5e9"},
                ],
                "threshold": {"line": {"color": "#333", "width": 3},
                               "thickness": 0.8, "value": 40},
            },
            number={"suffix": "", "valueformat": ".1f"}
        ))
        fig_r40.update_layout(margin=dict(t=40, b=10, l=20, r=20),
                               paper_bgcolor="white", height=200)

        # ── Radar dei margini ─────────────────────────────────────────────────
        categories = ["Gross Margin", "EBITDA Margin", "Op. Margin",
                      "FCF Margin", "Rev. Growth"]
        values_radar = [
            gross_margins  * 100,
            ebitda_margins * 100,
            op_margins     * 100,
            (fcf_margin_act * 100) if fcf_margin_act else 0,
            rev_growth     * 100,
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=values_radar + [values_radar[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line_color="#1f77b4",
            fillcolor="rgba(31,119,180,0.2)",
            name="Profilo"))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-10, 100])),
            title=dict(text="Profilo dei margini (%)", font=dict(size=11)),
            margin=dict(t=40, b=10, l=40, r=40),
            paper_bgcolor="white", height=280)

        # ── P/S & EV/Rev benchmark ────────────────────────────────────────────
        # Benchmarks SaaS 2025-2026 (normalizzati dopo il de-rating post-2021)
        ps_bench     = {"Alto (>10x)": 10, "Medio (5-10x)": 5, "Basso (<5x)": 2}
        evrev_bench  = {"Alto (>8x)": 8,  "Medio (4-8x)": 4, "Basso (<4x)": 2}

        # ── tabella metriche SaaS ─────────────────────────────────────────────
        metric_rows = [
            ("ARR (proxy Revenue TTM)",
             _val_fmt_num(arr_proxy, suffix=f" {currency}"),
             "Per SaaS puro = ricavi annui ricorrenti contrattualizzati. "
             "Qui usato Revenue TTM come proxy (yfinance non separa ARR)."),

            ("Crescita Revenue YoY",
             _badge(rev_growth*100, 30, 15, "{:.1f}%"),
             "Crescita robusta >30% è ottimale per SaaS in espansione; "
             ">15% accettabile; <15% indica maturità o rallentamento."),

            ("Gross Margin",
             _badge(gm_pct, 70, 50, "{:.1f}%"),
             "Misura la scalabilità: margine lordo >70% è il benchmark SaaS. "
             "Indica quanto rimane dopo i costi diretti di erogazione del servizio."),

            ("EBITDA Margin",
             _badge(ebitda_margins*100, 20, 5, "{:.1f}%"),
             "Redditività operativa. Valori negativi sono normali per SaaS in crescita "
             "che investe in S&M e R&D. Positivo >20% = azienda matura profittevole."),

            ("FCF Margin",
             _badge((fcf_margin_act or 0)*100, 15, 0, "{:.1f}%")
             if fcf_margin_act is not None else html.Span("N/D", style={"color":"#888"}),
             "Il free cash flow margin è la metrica più importante per valutare "
             "la sostenibilità della crescita. >15% = eccellente; >0% = autofinanziante."),

            ("Rule of 40",
             _badge(rule40_val, 40, 20, "{:.1f}"),
             f"Rev Growth {rev_growth*100:.1f}% + EBITDA Margin {ebitda_margins*100:.1f}% = "
             f"{rule40_val:.1f}. {'✓ Sopra 40: bilancio crescita/redditività sano.' if rule40_ok else '⚠ Sotto 40: l azienda non compensa il rallentamento con la redditività.'}"),

            ("P/S Ratio (Price/Sales)",
             _badge(ps_calc or 0, 0, 20, "{:.1f}x", higher_is_better=False)
             if ps_calc else html.Span("N/D", style={"color":"#888"}),
             "Valutazione rispetto ai ricavi. Post de-rating 2022-2024, SaaS ad alta crescita "
             "tratta tipicamente 5-15x. >20x richiede crescita >40% per giustificarsi."),

            ("EV/Revenue",
             _badge(ev_rev or 0, 0, 15, "{:.1f}x", higher_is_better=False)
             if ev_rev else html.Span("N/D", style={"color":"#888"}),
             "Capital-structure neutral. Benchmark 2025: SaaS alta crescita 6-12x, "
             "media crescita 3-6x, matura 1-3x."),

            ("EV/ARR (proxy)",
             _badge(ev_arr or 0, 0, 15, "{:.1f}x", higher_is_better=False)
             if ev_arr else html.Span("N/D", style={"color":"#888"}),
             "Come EV/Revenue ma normalizzato sull ARR. Per SaaS puri con alta retention, "
             "multipli EV/ARR più alti sono giustificati da Net Revenue Retention (NRR) elevata."),

            ("R&D / Revenue",
             _badge((rd_pct or 0)*100, 10, 5, "{:.1f}%")
             if rd_pct else html.Span("N/D", style={"color":"#888"}),
             "Intensità di innovazione. SaaS maturi investono 10-25% in R&D. "
             "Molto alto (>30%) può essere aggressivo; molto basso (<5%) segnala "
             "possibile commodity del prodotto."),
        ]

        rows_html = [
            html.Tr([
                html.Td(k, style={**td, "fontWeight": "bold", "width": "22%",
                                   "color": "#1a3a5c"}),
                html.Td(v, style={**td, "width": "13%", "textAlign": "center"}),
                html.Td(note, style={**td, "color": "#555", "fontSize": "11px",
                                      "lineHeight": "1.5"}),
            ], style={"background": "#fafafa" if i % 2 == 0 else "white"})
            for i, (k, v, note) in enumerate(metric_rows)
        ]

        return html.Div([
            html.H4("SaaS & Growth Metrics",
                    style={"fontSize": "14px", "margin": "0 0 6px", "color": "#1a3a5c",
                           "borderBottom": "2px solid #1a3a5c", "paddingBottom": "6px"}),
            html.P("Metriche specifiche per aziende growth e SaaS. "
                   "I semafori (verde/arancio/rosso) usano benchmark di settore 2025-2026.",
                   style={"fontSize": "11px", "color": "#666", "margin": "0 0 14px"}),

            html.Div([
                # Gauge Rule of 40
                html.Div([
                    dcc.Graph(figure=fig_r40, config={"displayModeBar": False}),
                    html.P(
                        f"{'✓ SANO' if rule40_ok else '⚠ SOTTO SOGLIA'}  "
                        f"({rev_growth*100:.1f}% crescita + {ebitda_margins*100:.1f}% EBITDA margin)",
                        style={"textAlign": "center", "color": r40_col,
                               "fontWeight": "bold", "fontSize": "12px",
                               "marginTop": "-8px"}),
                ], style={"flex": "1", "minWidth": "220px"}),

                # Radar margini
                html.Div([
                    dcc.Graph(figure=fig_radar, config={"displayModeBar": False}),
                ], style={"flex": "1", "minWidth": "280px"}),

                # Card multipli di valutazione
                html.Div([
                    html.H5("Multipli di valutazione growth",
                            style={"fontSize": "12px", "margin": "0 0 10px",
                                   "color": "#1a3a5c"}),
                    html.Table([html.Tbody([
                        html.Tr([
                            html.Td(k, style={**td, "color": "#555"}),
                            html.Td(f"{v:.1f}x" if v else "N/D",
                                    style={**td, "fontWeight": "bold"}),
                        ])
                        for k, v in [
                            ("P/S trailing",   ps_calc),
                            ("EV / Revenue",   ev_rev),
                            ("EV / ARR proxy", ev_arr),
                            ("Market Cap",     None),
                        ] if k != "Market Cap"
                    ] + [
                        html.Tr([
                            html.Td("Market Cap", style={**td, "color": "#555"}),
                            html.Td(_val_fmt_num(market_cap, suffix=f" {currency}"),
                                    style={**td, "fontWeight": "bold"}),
                        ]),
                        html.Tr([
                            html.Td("Enterprise Value", style={**td, "color": "#555"}),
                            html.Td(_val_fmt_num(ev, suffix=f" {currency}"),
                                    style={**td, "fontWeight": "bold"}),
                        ]),
                    ])], style=tbl),
                ], style={"flex": "1", "minWidth": "220px", "paddingLeft": "8px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "16px",
                       "marginBottom": "20px", "alignItems": "flex-start"}),

            html.H5("Dettaglio metriche con benchmark",
                    style={"fontSize": "12px", "margin": "0 0 8px", "color": "#1a3a5c"}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Metrica", style=th),
                    html.Th("Valore", style={**th, "textAlign": "center"}),
                    html.Th("Interpretazione", style=th),
                ])),
                html.Tbody(rows_html),
            ], style=tbl),

            html.Div([
                html.P([html.B("Come usare queste metriche insieme: ")],
                       style={"fontSize": "12px", "marginBottom": "4px"}),
                html.Ul([
                    html.Li("Rule of 40 ≥ 40 + Gross Margin ≥ 70% → azienda SaaS di qualità",
                            style={"fontSize": "11px", "lineHeight": "1.7"}),
                    html.Li("P/S basso (< 5x) + crescita alta (> 25%) → opportunità di valutazione",
                            style={"fontSize": "11px", "lineHeight": "1.7"}),
                    html.Li("EV/Rev in calo YoY con crescita stabile → de-rating ingiustificato",
                            style={"fontSize": "11px", "lineHeight": "1.7"}),
                    html.Li("FCF Margin negativo + R&D > 20% → azienda in fase di investimento "
                            "aggressivo, non necessariamente un problema se la crescita è alta",
                            style={"fontSize": "11px", "lineHeight": "1.7"}),
                    html.Li("Gross Margin < 50% per SaaS → possibile problema di architettura "
                            "o alta dipendenza da cloud/infrastruttura",
                            style={"fontSize": "11px", "lineHeight": "1.7"}),
                ], style={"paddingLeft": "18px", "margin": "0"}),
            ], style={"background": "#f0f6ff", "padding": "12px 14px", "borderRadius": "6px",
                       "borderLeft": "4px solid #1f77b4", "marginTop": "16px"}),

        ], style={"padding": "14px 16px 30px"})


    def _val_dcf(revenue, fcf_margin, wacc, g1, g2, gterm, shares):
        """DCF a 2 fasi: anni 1-ANNI_FASE1 a g1, poi fino ad ANNI_DCF a g2,
        infine il tasso finale in perpetuità (Gordon)."""
        if revenue <= 0 or shares <= 0:
            return None, [], []
        fcf0   = revenue * fcf_margin
        pv_sum = 0.0
        fcf_rows = []
        fcf_t = fcf0
        for yr in range(1, ANNI_FASE1 + 1):
            fcf_t *= (1 + g1)
            pv = fcf_t / (1 + wacc) ** yr
            pv_sum += pv
            fcf_rows.append((yr, f"Anni 1-{ANNI_FASE1} (g={g1*100:.1f}%)", fcf_t, pv))
        for yr in range(ANNI_FASE1 + 1, ANNI_DCF + 1):
            fcf_t *= (1 + g2)
            pv = fcf_t / (1 + wacc) ** yr
            pv_sum += pv
            fcf_rows.append((yr, f"Anni {ANNI_FASE1+1}-{ANNI_DCF} (g={g2*100:.1f}%)",
                             fcf_t, pv))
        # Tasso finale: Gordon sull'ultimo flusso esplicito
        if wacc <= gterm:
            tv = 0
        else:
            tv = fcf_t * (1 + gterm) / (wacc - gterm)
        pv_tv = tv / (1 + wacc) ** ANNI_DCF
        pv_sum += pv_tv
        fair_price = pv_sum / shares
        return fair_price, fcf_rows, pv_tv


    def _val_ddm(dividend, ke, g):
        """Gordon Growth Model: P = D1 / (ke - g), g = crescita del dividendo."""
        if dividend <= 0 or ke <= g:
            return None
        d1 = dividend * (1 + g)
        return d1 / (ke - g)


    def _val_graham(eps, g_pct, bond_yield):
        """Formula di Graham aggiornata: P = EPS × (8.5 + 2g) × 4.4 / Y."""
        if eps <= 0 or bond_yield <= 0:
            return None
        return eps * (8.5 + 2 * g_pct) * 4.4 / (bond_yield * 100)


    def _val_build_content(d, active_tab, wacc, g1, g2, gterm,
                            fcf_margin, pe_sector, ev_mult, ke,
                            ddm_g, graham_g_pct, bond_yield,
                            bilanci_modo="ttm"):
        """Renderizza il tab attivo con i dati fondamentali d."""
        import plotly.graph_objects as go

        price    = d.get("price", 0) or 0
        shares   = d.get("shares", 0) or 1
        revenue  = d.get("revenue", 0) or 0
        ebitda   = d.get("ebitda", 0) or 0
        net_debt = d.get("net_debt", 0) or 0
        eps_ttm  = d.get("eps_ttm", 0) or 0
        eps_fwd  = d.get("eps_fwd", 0) or eps_ttm
        dividend = d.get("dividend", 0) or 0
        currency = d.get("currency", "USD")
        name     = d.get("name", d.get("ticker", ""))
        fcf_margin_actual = d.get("fcf_margin_actual", fcf_margin)
        fcf_bilancio      = d.get("fcf_bilancio", 0) or 0
        rev_growth = d.get("revenue_growth", 0) or 0

        # Comanda lo slider, sempre: nessun valore "reale" che lo scavalca di
        # nascosto (era il motivo per cui il margine sembrava fermo al 5%).
        fcf_m = fcf_margin

        # ── calcola tutti i modelli ───────────────────────────────────────────
        dcf_price, fcf_rows, pv_tv = _val_dcf(revenue, fcf_m, wacc, g1, g2, gterm, shares)
        ddm_price  = _val_ddm(dividend, ke, ddm_g)
        g_est_pct  = graham_g_pct                          # slider crescita EPS
        graham_price = _val_graham(eps_ttm, g_est_pct, bond_yield)
        pe_price     = eps_fwd * pe_sector if eps_fwd > 0 else None
        ev_fair      = ebitda * ev_mult if ebitda > 0 else None
        ev_price     = (ev_fair - net_debt) / shares if (ev_fair and shares > 0) else None

        # raccoglie prezzi validi
        model_prices = {
            "DCF 2-fasi":    dcf_price,
            "DDM Gordon":    ddm_price,
            "Graham":        graham_price,
            "P/E relativo":  pe_price,
            "EV/EBITDA":     ev_price,
        }
        valid = {k: v for k, v in model_prices.items() if v and v > 0}

        # Cosa sta usando ogni modello, con i valori degli slider: si legge
        # subito nel Riepilogo, senza aprire i singoli tab.
        model_params = {
            "DCF 2-fasi": (
                f"FCF₀ {_val_fmt_num(revenue * fcf_m)} (revenue × {fcf_m*100:.1f}%) · "
                f"WACC {wacc*100:.1f}% · g anni 1-{ANNI_FASE1} {g1*100:.1f}% · "
                f"g anni {ANNI_FASE1+1}-{ANNI_DCF} {g2*100:.1f}% · "
                f"finale {gterm*100:.2f}%"),
            "DDM Gordon": (
                f"dividendo {dividend:.2f} {currency} · Ke {ke*100:.1f}% · "
                f"crescita {ddm_g*100:.2f}%"),
            "Graham": (
                f"EPS {eps_ttm:.2f} {currency} · crescita EPS {g_est_pct:.1f}% · "
                f"bond AAA {bond_yield*100:.2f}%"),
            "P/E relativo": (
                f"EPS forward {eps_fwd:.2f} {currency} × P/E settore {pe_sector:.1f}x"),
            "EV/EBITDA": (
                f"EBITDA {_val_fmt_num(ebitda)} × {ev_mult:.1f}x − debito netto "
                f"{_val_fmt_num(net_debt)} ÷ {_val_fmt_num(shares, 0)} azioni"),
        }

        def _updown(fv):
            if not fv or price <= 0: return ""
            pct = (fv - price) / price * 100
            col = "#2ca02c" if pct >= 0 else "#d62728"
            arrow = "▲" if pct >= 0 else "▼"
            return html.Span(f" {arrow}{abs(pct):.1f}%",
                             style={"color": col, "font-weight": "bold"})

        def _verdict(fv):
            if not fv or price <= 0: return ("N/D", "#888")
            pct = (fv - price) / price * 100
            if pct > 20:   return ("SOTTOVALUTATO", "#2ca02c")
            if pct > 5:    return ("LEGGERMENTE SOTTO", "#8bc34a")
            if pct > -5:   return ("A FAIR VALUE", "#ff7f0e")
            if pct > -20:  return ("LEGGERMENTE SOPRA", "#e67e22")
            return ("SOPRAVVALUTATO", "#d62728")

        tbl_style = {"width": "100%", "border-collapse": "collapse",
                     "font-size": "12px", "border": "1px solid #ddd"}
        th_style  = {"background": "#f0f0f0", "font-size": "12px",
                     "padding": "6px 8px", "text-align": "left"}
        td_style  = {"padding": "5px 8px", "border-bottom": "1px solid #eee"}

        # ── TAB RIEPILOGO ─────────────────────────────────────────────────────
        if active_tab == "val-tab-summary":
            # Fundamentals card
            fund_rows = [
                ("Prezzo corrente",    f"{price:.2f} {currency}"),
                ("Market Cap",         _val_fmt_num(d.get("market_cap"), suffix=f" {currency}")),
                ("EPS TTM",            f"{eps_ttm:.2f} {currency}"),
                ("EPS Forward",        f"{eps_fwd:.2f} {currency}"),
                ("Revenue (TTM)",      _val_fmt_num(revenue, suffix=f" {currency}")),
                ("EBITDA",             _val_fmt_num(ebitda, suffix=f" {currency}")),
                ("FCF (yfinance TTM)", _val_fmt_num(d.get("fcf_yf"), suffix=f" {currency}")),
                ("FCF (rendiconto)",   _val_fmt_num(fcf_bilancio, suffix=f" {currency}")
                                       if fcf_bilancio else "N/D"),
                ("Margine FCF di bilancio",
                 f"{fcf_bilancio/revenue*100:.1f}%" if (fcf_bilancio and revenue > 0)
                 else f"{fcf_margin_actual*100:.1f}%"),
                ("Margine FCF usato (slider)", f"{fcf_m*100:.1f}%"),
                ("Debito netto",       _val_fmt_num(net_debt, suffix=f" {currency}")),
                ("Dividendo/azione",   f"{dividend:.2f} {currency}" if dividend else "N/D"),
                ("P/E trailing",       f"{d.get('pe_trailing'):.1f}x" if d.get('pe_trailing') else "N/D"),
                ("P/E forward",        f"{d.get('pe_forward'):.1f}x" if d.get('pe_forward') else "N/D"),
                ("Beta",               f"{d.get('beta', 1.0):.2f}"),
                ("Crescita Rev. YoY",  f"{rev_growth*100:+.1f}%"),
                ("Settore",            d.get("sector", "N/D")),
                ("Industria",          d.get("industry", "N/D")),
            ]
            fund_table = html.Table([
                html.Tbody([
                    html.Tr([
                        html.Td(k, style={**td_style, "color": "#555", "width": "45%"}),
                        html.Td(v, style={**td_style, "font-weight": "bold"}),
                    ]) for k, v in fund_rows
                ])
            ], style=tbl_style)

            # Summary valuation table
            sum_rows = []
            for model, fv in model_prices.items():
                verdict, vcol = _verdict(fv)
                sum_rows.append(html.Tr([
                    html.Td([
                        html.Div(model, style={"font-weight": "bold"}),
                        html.Div(model_params.get(model, ""),
                                 style={"font-size": "10px", "color": "#888",
                                        "line-height": "1.45", "margin-top": "2px"}),
                    ], style={**td_style, "width": "42%"}),
                    html.Td(f"{fv:.2f} {currency}" if fv else "N/D",
                            style={**td_style, "font-weight": "bold",
                                   "white-space": "nowrap"}),
                    html.Td([_updown(fv)] if fv else "—", style=td_style),
                    html.Td(verdict, style={**td_style, "color": vcol,
                                            "font-weight": "bold"}),
                ]))

            # Media ponderata (escludi None)
            if valid:
                avg_price = sum(valid.values()) / len(valid)
                avg_verdict, avg_col = _verdict(avg_price)
            else:
                avg_price = None
                avg_col = "#888"

            # Waterfall chart
            labels = list(valid.keys()) + (["Media modelli", "Prezzo corrente"] if valid else [])
            values = list(valid.values()) + ([avg_price, price] if valid else [])
            colors = []
            for v in values[:-1]:
                colors.append("#2ca02c" if v and v > price else "#d62728")
            colors.append("#1f77b4")

            fig_bar = go.Figure(go.Bar(
                x=labels, y=values,
                marker_color=colors,
                text=[f"{v:.1f}" if v else "" for v in values],
                textposition="outside",
            ))
            fig_bar.add_hline(y=price, line_color="#333", line_dash="dash",
                               line_width=2,
                               annotation_text=f"Prezzo corrente {price:.2f}")
            fig_bar.update_layout(
                title=dict(text=f"{name} — Fair Value per modello vs prezzo corrente ({currency})",
                           font=dict(size=12)),
                yaxis_title=f"Prezzo ({currency})",
                margin=dict(t=50, b=40, l=55, r=20),
                paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                showlegend=False)

            return html.Div([
                html.Div([
                    # colonna sinistra: fondamentali
                    html.Div([
                        html.H4(f"Fondamentali — {name}",
                                style={"font-size": "13px", "margin": "0 0 10px",
                                       "color": "#1a3a5c"}),
                        fund_table,
                    ], style={"flex": "1", "min-width": "260px",
                               "padding-right": "20px"}),

                    # colonna destra: valutazioni
                    html.Div([
                        html.H4("Riepilogo valutazioni",
                                style={"font-size": "13px", "margin": "0 0 10px",
                                       "color": "#1a3a5c"}),
                        html.Table([
                            html.Thead(html.Tr([
                                html.Th("Modello", style=th_style),
                                html.Th("Fair Value", style=th_style),
                                html.Th("vs Prezzo", style=th_style),
                                html.Th("Verdetto", style=th_style),
                            ])),
                            html.Tbody(sum_rows),
                        ], style=tbl_style),
                        html.Div([
                            html.Span("Media modelli: ",
                                      style={"font-size": "13px", "color": "#555"}),
                            html.Span(f"{avg_price:.2f} {currency}" if avg_price else "N/D",
                                      style={"font-size": "16px", "font-weight": "bold",
                                             "color": avg_col}),
                            _updown(avg_price),
                        ], style={"margin": "14px 0 6px",
                                   "background": "#f8f8f8", "padding": "10px",
                                   "border-radius": "6px",
                                   "border-left": f"4px solid {avg_col}"}),
                        html.P(f"Prezzo corrente: {price:.2f} {currency}  |  "
                               f"Modelli calcolati: {len(valid)}/5",
                               style={"font-size": "11px", "color": "#888"}),
                    ], style={"flex": "1", "min-width": "300px"}),
                ], style={"display": "flex", "flex-wrap": "wrap", "gap": "20px",
                           "margin-bottom": "20px"}),

                html.Hr(),
                dcc.Graph(figure=fig_bar, style={"height": "320px"},
                          config={"displayModeBar": False}),
            ], style={"padding": "14px 16px 30px"})

        # ── TAB BILANCI (Alpha Vantage) ───────────────────────────────────────
        elif active_tab == "val-tab-bilanci":
            ticker  = d.get("ticker", "")
            payload, fonte = av_conto_economico(ticker)
            if not payload:
                return html.Div([
                    html.H4(f"Bilanci — {name}",
                            style={"font-size": "14px", "margin": "0 0 8px",
                                   "color": "#1a3a5c"}),
                    html.P(f"⚠ {fonte}",
                           style={"font-size": "12px", "color": "#b0413e"}),
                    html.P("Alpha Vantage pubblica il conto economico dei titoli "
                           "quotati negli USA usando il simbolo senza suffisso "
                           "(MSFT, AAPL, KO). Per i titoli europei il conto "
                           "economico non è disponibile.",
                           style={"font-size": "11px", "color": "#888",
                                  "line-height": "1.6"}),
                ], style={"padding": "24px 16px"})

            modo = "annuale" if bilanci_modo == "annuale" else "ttm"
            s    = av_serie(payload, modo)
            if not s["date"]:
                return html.Div(f"Nessun dato {modo} per {ticker}.",
                                style={"padding": "40px", "color": "#888",
                                       "text-align": "center"})

            val    = s["valuta"]
            annuale = (modo == "annuale")
            periodo = "esercizi" if annuale else "trimestri TTM"
            base    = ("dati annuali pubblicati dall'azienda" if annuale else
                       "somma scorrevole degli ultimi 4 trimestri")

            def _mld(v):
                return None if v is None else v / 1e9

            # Variazione rispetto a un anno prima: sugli esercizi è il periodo
            # precedente, sui TTM sono 4 trimestri indietro (stesso trimestre
            # dell'anno prima), altrimenti si confronterebbero periodi che si
            # sovrappongono per tre quarti.
            passo_anno = 1 if annuale else 4

            def _var_anno(y):
                out = []
                for i, v in enumerate(y):
                    prec = y[i - passo_anno] if i >= passo_anno else None
                    # Da una base negativa la variazione percentuale non vuole
                    # dire niente (da -6 a +5 non è "+183%"): meglio un buco.
                    out.append((v / prec - 1) * 100
                               if (v is not None and prec is not None and prec > 0)
                               else None)
                return out

            def _grafico(y, titolo, colore, unita, percentuale=False,
                         y2=None, y2_titolo="", y2_unita=""):
                """Stessa forma per tutti e quattro: barre sugli esercizi,
                area sui TTM (dove i punti sono decine). `y2` aggiunge una serie
                sul secondo asse a destra."""
                fig = go.Figure()
                if annuale:
                    fig.add_trace(go.Bar(x=s["date"], y=y, marker_color=colore,
                                         name=titolo))
                else:
                    fig.add_trace(go.Scatter(
                        x=s["date"], y=y, mode="lines", line=dict(color=colore, width=2),
                        fill="tozeroy",
                        fillcolor="rgba(" + ",".join(
                            str(int(colore[i:i + 2], 16)) for i in (1, 3, 5)) + ",0.15)",
                        name=titolo))
                if percentuale:
                    fig.add_hline(y=0, line_color="#999", line_width=1)
                if y2 is not None:
                    fig.add_trace(go.Scatter(
                        x=s["date"], y=y2, mode="lines+markers", yaxis="y2",
                        line=dict(color="#d62728", width=1.8),
                        marker=dict(size=4), name=y2_titolo,
                        hovertemplate="%{y:+.1f}%<extra></extra>"))
                    fig.add_hline(y=0, line_color="#d62728", line_width=1,
                                  line_dash="dot", opacity=0.45, yref="y2")
                fig.update_layout(
                    title=dict(text=titolo, font=dict(size=11)),
                    yaxis=dict(title=unita),
                    margin=dict(t=40, b=30, l=55, r=18 if y2 is None else 52),
                    paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                    showlegend=y2 is not None, hovermode="x unified")
                if y2 is not None:
                    fig.update_layout(
                        yaxis2=dict(title=dict(text=y2_unita,
                                               font=dict(color="#d62728")),
                                    overlaying="y", side="right", showgrid=False,
                                    tickfont=dict(color="#d62728"), ticksuffix="%"),
                        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)))
                return dcc.Graph(figure=fig, style={"height": "290px"},
                                 config={"displayModeBar": False})

            grafici = [
                _grafico([_mld(v) for v in s["fatturato"]],
                         "Fatturato", "#1f77b4", f"mld {val}",
                         y2=_var_anno(s["fatturato"]),
                         y2_titolo="Variazione su un anno prima", y2_unita="var. %"),
                _grafico([_mld(v) for v in s["utile"]],
                         "Utile netto", "#2ca02c", f"mld {val}",
                         y2=_var_anno(s["utile"]),
                         y2_titolo="Variazione su un anno prima", y2_unita="var. %"),
                _grafico(s["margine"], "Margine netto", "#ff7f0e", "%",
                         percentuale=True),
                _grafico([_mld(v) for v in s["lordo"]],
                         "Utile lordo", "#9467bd", f"mld {val}"),
            ]

            # Statistiche e ultimi periodi: le stesse quattro serie dei grafici.
            def _stat(y, dec=2):
                v = sorted(x for x in y if x is not None)
                if not v:
                    return ["—"] * 5
                n = len(v)
                mediana = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
                ultimo = next((x for x in reversed(y) if x is not None), None)
                return [f"{sum(v)/n:,.{dec}f}", f"{mediana:,.{dec}f}",
                        f"{v[0]:,.{dec}f}", f"{v[-1]:,.{dec}f}",
                        f"{ultimo:,.{dec}f}" if ultimo is not None else "—"]

            serie_tab = [
                (f"Fatturato (mld {val})", [_mld(v) for v in s["fatturato"]], 2),
                (f"Utile netto (mld {val})", [_mld(v) for v in s["utile"]], 2),
                ("Margine netto (%)", s["margine"], 2),
                (f"Utile lordo (mld {val})", [_mld(v) for v in s["lordo"]], 2),
            ]
            stat_tbl = html.Table([
                html.Thead(html.Tr([html.Th("", style=th_style)] +
                                   [html.Th(c, style=th_style) for c in
                                    ("Media", "Mediana", "Min", "Max", "Ultimo")])),
                html.Tbody([
                    html.Tr([html.Td(et, style={**td_style, "color": "#555"})] +
                            [html.Td(x, style={**td_style, "font-weight": "bold"})
                             for x in _stat(y, dec)])
                    for et, y, dec in serie_tab
                ]),
            ], style=tbl_style)

            n_ult = min(5, len(s["date"]))
            ult_tbl = html.Table([
                html.Thead(html.Tr([html.Th("Periodo", style=th_style)] +
                                   [html.Th(et.split(" (")[0], style=th_style)
                                    for et, _, _ in serie_tab])),
                html.Tbody([
                    html.Tr([html.Td(s["date"][i], style=td_style)] +
                            [html.Td(f"{y[i]:,.2f}" if y[i] is not None else "—",
                                     style={**td_style, "font-weight": "bold"})
                             for _, y, _ in serie_tab])
                    for i in range(len(s["date"]) - n_ult, len(s["date"]))
                ]),
            ], style=tbl_style)

            # ── I tre prospetti riclassificati ────────────────────────────────
            # Stato patrimoniale e rendiconto sono altre due richieste ad Alpha
            # Vantage: se mancano (limite giornaliero, titolo non coperto) il
            # tab non si rompe, mostra la nota e va avanti con il resto.
            pay_sp, fonte_sp = av_prospetto(ticker, "sp")
            pay_cf, fonte_cf = av_prospetto(ticker, "cf")

            def _riga_stile(stile):
                if stile == "sez":
                    return ({**td_style, "background": "#eaf4fb", "font-weight": "bold",
                             "color": "#1a5276", "font-size": "11px",
                             "padding": "5px 8px"}, None)
                if stile == "nota":
                    return ({**td_style, "color": "#8a6d3b", "font-size": "10px",
                             "font-style": "italic", "background": "#fdf7e6",
                             "line-height": "1.5"}, None)
                if stile == "totf":
                    return ({**td_style, "background": "#eaf4fb", "font-weight": "bold",
                             "color": "#1a3a5c", "border-top": "2px solid #1a3a5c"},
                            {"font-weight": "bold", "color": "#1a3a5c"})
                if stile == "tot":
                    return ({**td_style, "background": "#fafafa", "font-weight": "bold",
                             "border-top": "1px solid #bbb"},
                            {"font-weight": "bold"})
                if stile == "sub":
                    return ({**td_style, "padding-left": "26px", "color": "#888",
                             "font-size": "11px", "font-style": "italic"},
                            {"color": "#888", "font-size": "11px"})
                if stile == "memo":
                    return ({**td_style, "color": "#888", "font-style": "italic"},
                            {"color": "#888", "font-style": "italic"})
                return ({**td_style, "color": "#444"}, {})

            def _tab_ricl(righe, date_col, extra_tit, scala):
                div, um, dec = scala
                n_col = len(date_col) + (1 if extra_tit else 0)

                def _cella(v):
                    return "—" if v is None else f"{v / div:,.{dec}f}"

                corpo = []
                for stile, etichetta, valori, extra in righe:
                    st_lbl, st_val = _riga_stile(stile)
                    if st_val is None:            # intestazione di sezione
                        corpo.append(html.Tr([html.Td(etichetta, colSpan=n_col + 1,
                                                      style=st_lbl)]))
                        continue
                    celle = []
                    for v in valori:
                        col = dict(st_val)
                        if v is not None and v < 0 and stile in ("tot", "totf"):
                            col["color"] = "#b0413e"
                        celle.append(html.Td(_cella(v),
                                             style={**td_style, "text-align": "right",
                                                    "font-variant-numeric": "tabular-nums",
                                                    **col}))
                    if extra_tit:
                        celle.append(html.Td(extra, style={
                            **td_style, "text-align": "right", "color": "#777",
                            "font-size": "11px",
                            **({"font-weight": "bold"} if stile in ("tot", "totf") else {})}))
                    corpo.append(html.Tr([html.Td(etichetta, style=st_lbl)] + celle))

                testata = [html.Th(f"valori in {um} {val}", style=th_style)] + \
                          [html.Th(dt, style={**th_style, "text-align": "right"})
                           for dt in date_col]
                if extra_tit:
                    testata.append(html.Th(extra_tit,
                                           style={**th_style, "text-align": "right"}))
                return html.Table([html.Thead(html.Tr(testata)), html.Tbody(corpo)],
                                  style=tbl_style)

            def _prepara(titolo, sottotitolo, payload, fonte_p, flusso, builder,
                         extra_tit=""):
                per = av_periodi(payload, modo, flusso=flusso, n=4)[0] if payload else []
                return dict(titolo=titolo, sottotitolo=sottotitolo, errore=fonte_p,
                            righe=builder(per) if per else None,
                            date=[d for d, _ in per], extra=extra_tit)

            def _blocco(b, scala):
                if b["righe"] is None:
                    return html.Div([
                        html.H5(b["titolo"], style={"font-size": "12px",
                                                    "margin": "0 0 6px",
                                                    "color": "#1a3a5c"}),
                        html.P(f"⚠ non disponibile — {b['errore']}",
                               style={"font-size": "11px", "color": "#b0413e"}),
                    ], style={"margin-bottom": "18px"})
                return html.Div([
                    html.H5(b["titolo"], style={"font-size": "12px", "margin": "0 0 3px",
                                                "color": "#1a3a5c"}),
                    html.P(b["sottotitolo"], style={"font-size": "10px", "color": "#888",
                                                    "margin": "0 0 6px"}),
                    _tab_ricl(b["righe"], b["date"], b["extra"], scala),
                ], style={"margin-bottom": "22px"})

            # In modo TTM lo stato patrimoniale non si somma: è la fotografia
            # alla fine di ogni trimestre (vedi av_periodi).
            sp_nota = ("situazione alla data di chiusura di ciascun esercizio"
                       if annuale else
                       "fotografia alla fine di ciascun trimestre — una situazione "
                       "patrimoniale non si somma su quattro trimestri")
            fl_nota = ("esercizi come pubblicati" if annuale else
                       "somma scorrevole degli ultimi 4 trimestri")

            blocchi = [
                _prepara("🏛 Stato patrimoniale riclassificato (criterio funzionale)",
                         f"Capitale investito netto e sue fonti · {sp_nota}",
                         pay_sp, fonte_sp, False, _ricl_stato_patrimoniale),
                _prepara("📑 Conto economico riclassificato (a costo del venduto)",
                         f"Dai ricavi all'utile netto in forma scalare · {fl_nota}",
                         payload, fonte, True, _ricl_conto_economico,
                         extra_tit="% ricavi"),
                _prepara("💧 Rendiconto finanziario riclassificato",
                         f"Dall'utile netto al free cash flow · {fl_nota} · "
                         "il capex è già dentro il flusso da investimenti: la riga "
                         "del free cash flow è un richiamo, non si somma sotto",
                         pay_cf, fonte_cf, True, _ricl_rendiconto),
            ]
            # Un'unica unità di misura per i tre prospetti: con una scala per
            # tabella lo stesso titolo finiva in miliardi nel conto economico e
            # in milioni nel rendiconto, e i numeri non erano più confrontabili.
            scala = av_scala([r for b in blocchi if b["righe"] for r in b["righe"]])
            prospetti = html.Div([_blocco(b, scala) for b in blocchi])

            return html.Div([
                html.H4(f"Bilanci riclassificati — {name} ({ticker})",
                        style={"font-size": "14px", "margin": "0 0 6px",
                               "color": "#1a3a5c", "border-bottom": "2px solid #1a3a5c",
                               "padding-bottom": "6px"}),
                html.P(f"{len(s['date'])} {periodo} — dal {s['date'][0]} al "
                       f"{s['date'][-1]}  ·  {base}  ·  valuta di bilancio {val}  ·  "
                       f"Alpha Vantage (conto economico {fonte}, stato patrimoniale "
                       f"{fonte_sp}, rendiconto {fonte_cf})",
                       style={"font-size": "11px", "color": "#666"}),
                prospetti,

                html.Hr(),
                html.H5("Andamento storico",
                        style={"font-size": "12px", "margin": "10px 0 0",
                               "color": "#1a3a5c"}),
                html.Div([html.Div(g, style={"flex": "1 1 46%", "min-width": "320px"})
                          for g in grafici],
                         style={"display": "flex", "flex-wrap": "wrap", "gap": "10px",
                                "margin": "10px 0"}),

                html.Hr(),
                html.Div([
                    html.Div([
                        html.H5(f"Statistiche descrittive ({periodo})",
                                style={"font-size": "12px", "margin": "0 0 8px"}),
                        stat_tbl,
                    ], style={"flex": "1", "min-width": "300px"}),
                    html.Div([
                        html.H5(f"Ultimi {n_ult} periodi",
                                style={"font-size": "12px", "margin": "0 0 8px"}),
                        ult_tbl,
                    ], style={"flex": "1", "min-width": "320px"}),
                ], style={"display": "flex", "flex-wrap": "wrap", "gap": "16px"}),
            ], style={"padding": "14px 16px 30px"})

        # ── TAB DCF ───────────────────────────────────────────────────────────
        elif active_tab == "val-tab-dcf":
            if dcf_price is None:
                return html.Div("Dati insufficienti per il DCF (revenue o shares = 0).",
                                style={"padding": "30px", "color": "#888",
                                       "text-align": "center"})
            verdict, vcol = _verdict(dcf_price)

            # Tabella flussi
            flow_rows = []
            for yr, fase, fcf_t, pv in fcf_rows:
                flow_rows.append(html.Tr([
                    html.Td(f"Anno {yr}", style=td_style),
                    html.Td(fase, style={**td_style, "color": "#555"}),
                    html.Td(_val_fmt_num(fcf_t, suffix=f" {currency}"), style=td_style),
                    html.Td(_val_fmt_num(pv, suffix=f" {currency}"),
                            style={**td_style, "font-weight": "bold"}),
                ]))

            # Grafico PV per anno
            yrs  = [r[0] for r in fcf_rows]
            pvs  = [r[3] for r in fcf_rows]
            cols = ["#1f77b4" if yr <= ANNI_FASE1 else "#ff7f0e" for yr in yrs]
            fig_dcf = go.Figure()
            fig_dcf.add_trace(go.Bar(
                x=[f"Anno {y}" for y in yrs], y=pvs,
                marker_color=cols,
                name="PV FCF",
                text=[_val_fmt_num(p) for p in pvs],
                textposition="outside"))
            fig_dcf.add_trace(go.Bar(
                x=["Terminal Value"], y=[pv_tv],
                marker_color="#9467bd",
                name="PV Terminal Value",
                text=[_val_fmt_num(pv_tv)],
                textposition="outside"))
            fig_dcf.update_layout(
                title=dict(text="Valore Attuale dei FCF per anno + Terminal Value",
                           font=dict(size=11)),
                yaxis_title=currency,
                margin=dict(t=45, b=30, l=55, r=20),
                paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                legend=dict(orientation="h", y=1.04, x=0, font=dict(size=9)))

            return html.Div([
                html.H4(f"DCF — {ANNI_DCF} anni ({ANNI_FASE1} a g1 + {ANNI_FASE2} a g2) "
                        f"+ tasso finale",
                        style={"font-size": "14px", "margin": "0 0 6px",
                               "color": "#1a3a5c", "border-bottom": "2px solid #1a3a5c",
                               "padding-bottom": "6px"}),
                html.P(f"FCF₀ = Revenue × Margine FCF = "
                       f"{_val_fmt_num(revenue)} × {fcf_m*100:.1f}% = "
                       f"{_val_fmt_num(revenue*fcf_m)} {currency}",
                       style={"font-size": "11px", "color": "#666",
                              "font-family": "monospace"}),
                html.Div([
                    html.Span(f"Fair Value DCF: {dcf_price:.2f} {currency}",
                              style={"font-size": "16px", "font-weight": "bold",
                                     "color": vcol}),
                    html.Span("  "),
                    _updown(dcf_price),
                    html.Span(f"  →  {verdict}",
                              style={"color": vcol, "font-weight": "bold",
                                     "margin-left": "8px", "font-size": "13px"}),
                ], style={"margin": "10px 0", "background": "#f8f8f8",
                           "padding": "10px", "border-radius": "6px",
                           "border-left": f"4px solid {vcol}"}),

                html.Div([
                    html.Div([
                        html.H5("Flussi di cassa attualizzati",
                                style={"font-size": "12px", "margin": "0 0 8px"}),
                        html.Table([
                            html.Thead(html.Tr([
                                html.Th("Anno", style=th_style),
                                html.Th("Fase", style=th_style),
                                html.Th("FCF", style=th_style),
                                html.Th("PV", style=th_style),
                            ])),
                            html.Tbody(flow_rows + [
                                html.Tr([
                                    html.Td(f"Da anno {ANNI_DCF+1}",
                                            style={**td_style, "font-weight": "bold"}),
                                    html.Td(f"tasso finale g={gterm*100:.2f}%", style=td_style),
                                    html.Td("—", style=td_style),
                                    html.Td(_val_fmt_num(pv_tv, suffix=f" {currency}"),
                                            style={**td_style, "font-weight": "bold",
                                                   "color": "#9467bd"}),
                                ], style={"background": "#f3eaff"}),
                            ]),
                        ], style=tbl_style),
                    ], style={"flex": "1", "min-width": "280px"}),
                    html.Div([
                        html.H5("Ipotesi DCF", style={"font-size": "12px",
                                                        "margin": "0 0 8px"}),
                        html.Table([
                            html.Tbody([
                                html.Tr([html.Td(k, style={**td_style, "color": "#555"}),
                                         html.Td(v, style={**td_style, "font-weight": "bold"})])
                                for k, v in [
                                    ("WACC",           f"{wacc*100:.1f}%"),
                                    (f"Crescita anni 1-{ANNI_FASE1}", f"{g1*100:.1f}%"),
                                    (f"Crescita anni {ANNI_FASE1+1}-{ANNI_DCF}",
                                     f"{g2*100:.1f}%"),
                                    (f"Tasso finale (da anno {ANNI_DCF+1})",
                                     f"{gterm*100:.2f}%"),
                                    ("Margine FCF",     f"{fcf_m*100:.1f}%"),
                                    ("Revenue base",    _val_fmt_num(revenue, suffix=f" {currency}")),
                                    ("Azioni (shares)", _val_fmt_num(shares, 0)),
                                ]
                            ])
                        ], style=tbl_style),
                        html.Div([
                            html.P([html.B("Note: "),
                                    "Il DCF è molto sensibile a WACC e tasso finale: con "
                                    f"{ANNI_DCF} anni espliciti la perpetuità pesa da sola "
                                    "la maggior parte del fair value. Usa la tab Sensitività "
                                    "per vedere l'intervallo al variare di WACC e g."],
                                   style={"font-size": "11px", "color": "#888",
                                          "line-height": "1.5", "margin-top": "10px"}),
                        ]),
                    ], style={"flex": "1", "min-width": "240px",
                               "padding-left": "16px"}),
                ], style={"display": "flex", "flex-wrap": "wrap",
                           "gap": "12px", "margin": "14px 0"}),

                html.Hr(),
                dcc.Graph(figure=fig_dcf, style={"height": "320px"},
                          config={"displayModeBar": False}),
            ], style={"padding": "14px 16px 30px"})

        # ── TAB DDM ───────────────────────────────────────────────────────────
        elif active_tab == "val-tab-ddm":
            has_div = dividend > 0
            verdict, vcol = _verdict(ddm_price) if ddm_price else ("N/D", "#888")

            div_sensitivity = []
            if has_div and ke > ddm_g:
                g_range = [g * 0.01 for g in range(0, int(ke * 100) - 1, 1)]
                for g_t in g_range:
                    p = dividend * (1 + g_t) / (ke - g_t)
                    div_sensitivity.append((g_t * 100, p))

                fig_ddm = go.Figure()
                fig_ddm.add_trace(go.Scatter(
                    x=[x[0] for x in div_sensitivity],
                    y=[x[1] for x in div_sensitivity],
                    mode="lines", line=dict(color="#1f77b4", width=2.5),
                    name="DDM Fair Value"))
                fig_ddm.add_hline(y=price, line_color="#d62728", line_dash="dash",
                                   line_width=2,
                                   annotation_text=f"Prezzo corrente {price:.2f}")
                if ddm_price:
                    fig_ddm.add_vline(x=ddm_g * 100, line_color="#2ca02c",
                                       line_dash="dot", line_width=1.5,
                                       annotation_text=f"g={ddm_g*100:.1f}%")
                fig_ddm.update_layout(
                    title=dict(text="DDM Fair Value al variare della crescita del dividendo g",
                               font=dict(size=11)),
                    xaxis_title="g — Crescita del dividendo (%)",
                    yaxis_title=f"Fair Value ({currency})",
                    margin=dict(t=45, b=35, l=55, r=20),
                    paper_bgcolor="white", plot_bgcolor="#f8f8f8")
            else:
                fig_ddm = go.Figure()
                fig_ddm.add_annotation(text="Dividendo = 0 o ke ≤ g: DDM non applicabile",
                                        xref="paper", yref="paper", x=0.5, y=0.5,
                                        showarrow=False, font=dict(size=14, color="#888"))

            return html.Div([
                html.H4("DDM — Gordon Growth Model",
                        style={"font-size": "14px", "margin": "0 0 6px",
                               "color": "#1a3a5c", "border-bottom": "2px solid #1a3a5c",
                               "padding-bottom": "6px"}),
                html.P("P = D₁ / (Ke − g)   dove D₁ = D₀ × (1 + g)",
                       style={"font-size": "11px", "color": "#666",
                              "font-family": "monospace"}),

                html.Div([
                    html.Span(f"Dividendo annuo: {dividend:.2f} {currency}/azione  |  "
                              f"Ke = {ke*100:.1f}%  |  g = {ddm_g*100:.2f}%",
                              style={"font-size": "12px", "color": "#555"}),
                ], style={"margin": "8px 0"}),

                html.Div([
                    html.Span(
                        f"Fair Value DDM: {ddm_price:.2f} {currency}" if ddm_price
                        else "⚠ DDM non applicabile (dividendo = 0 o ke ≤ g)",
                        style={"font-size": "16px", "font-weight": "bold", "color": vcol}),
                    html.Span("  "),
                    _updown(ddm_price) if ddm_price else "",
                    html.Span(f"  →  {verdict}",
                              style={"color": vcol, "font-weight": "bold",
                                     "margin-left": "8px", "font-size": "13px"}),
                ], style={"margin": "10px 0", "background": "#f8f8f8",
                           "padding": "10px", "border-radius": "6px",
                           "border-left": f"4px solid {vcol}"}),

                html.Div([
                    html.P([html.B("Come leggere il DDM: "),
                            "Il modello Gordon è appropriato per aziende mature con dividendi "
                            "stabili e crescenti (utilities, banche, consumer staples). "
                            "Non è applicabile a società growth che reinvestono tutto il FCF "
                            "senza distribuire dividendi."],
                           style={"font-size": "12px", "line-height": "1.6"}),
                    html.P([html.B("Ke vs WACC: "),
                            "Nel DDM si usa Ke (costo equity puro), non il WACC che include "
                            "anche il debito. Ke = Rf + β × (Rm − Rf), tipicamente 8-12%."],
                           style={"font-size": "12px", "line-height": "1.6",
                                  "margin-top": "6px"}),
                ], style={"background": "#f0f6ff", "padding": "12px",
                           "border-radius": "6px", "border-left": "4px solid #1f77b4",
                           "margin": "12px 0"}),

                dcc.Graph(figure=fig_ddm, style={"height": "300px"},
                          config={"displayModeBar": False}),
            ], style={"padding": "14px 16px 30px"})

        # ── TAB GRAHAM ────────────────────────────────────────────────────────
        elif active_tab == "val-tab-graham":
            verdict, vcol = _verdict(graham_price) if graham_price else ("N/D", "#888")

            # Sensitività EPS × crescita
            g_vals   = [2, 5, 8, 10, 12, 15, 18, 20, 25]
            eps_vals = [round(eps_ttm * m, 2)
                        for m in [0.5, 0.75, 1.0, 1.25, 1.5]]
            heat_z = []
            for eps_v in eps_vals:
                row = []
                for g_v in g_vals:
                    row.append(_val_graham(eps_v, g_v, bond_yield) or 0)
                heat_z.append(row)

            fig_gr = go.Figure(go.Heatmap(
                z=heat_z,
                x=[f"g={g}%" for g in g_vals],
                y=[f"EPS={e:.2f}" for e in eps_vals],
                colorscale="RdYlGn",
                text=[[f"{v:.0f}" for v in row] for row in heat_z],
                texttemplate="%{text}",
                colorbar=dict(title=currency, tickfont=dict(size=9))))
            fig_gr.add_annotation(
                text=f"★ EPS attuale={eps_ttm:.2f},  g={g_est_pct:.1f}%  →  "
                     f"Graham={graham_price:.2f}" if graham_price else "",
                xref="paper", yref="paper", x=0.5, y=1.08,
                showarrow=False, font=dict(size=11))
            fig_gr.update_layout(
                title=dict(text=f"Graham Fair Value — sensitività EPS × crescita  "
                                f"(Y={bond_yield*100:.2f}%)",
                           font=dict(size=11)),
                margin=dict(t=55, b=40, l=80, r=20),
                paper_bgcolor="white")

            return html.Div([
                html.H4("Formula di Graham",
                        style={"font-size": "14px", "margin": "0 0 6px",
                               "color": "#1a3a5c", "border-bottom": "2px solid #1a3a5c",
                               "padding-bottom": "6px"}),
                html.P("P = EPS × (8.5 + 2g) × 4.4 / Y   "
                       "(Graham 1962, aggiornata con rendimento AAA bond Y)",
                       style={"font-size": "11px", "color": "#666",
                              "font-family": "monospace"}),

                html.Div([
                    html.Span(
                        f"Fair Value Graham: {graham_price:.2f} {currency}" if graham_price
                        else "⚠ Graham non applicabile (EPS ≤ 0)",
                        style={"font-size": "16px", "font-weight": "bold", "color": vcol}),
                    html.Span("  "),
                    _updown(graham_price) if graham_price else "",
                    html.Span(f"  →  {verdict}",
                              style={"color": vcol, "font-weight": "bold",
                                     "margin-left": "8px", "font-size": "13px"}),
                ], style={"margin": "10px 0", "background": "#f8f8f8",
                           "padding": "10px", "border-radius": "6px",
                           "border-left": f"4px solid {vcol}"}),

                html.Div([
                    html.P([html.B("Calcolo: "),
                            f"EPS = {eps_ttm:.2f}  ×  (8.5 + 2×{g_est_pct:.1f})  ×  "
                            f"4.4 / {bond_yield*100:.2f}  =  "
                            f"{graham_price:.2f}" if graham_price else "N/D"],
                           style={"font-size": "12px", "font-family": "monospace"}),
                    html.P([html.B("8.5 "), "= P/E di un'azienda a crescita zero secondo Graham. ",
                            html.B("2g "), "= ogni punto percentuale di crescita aggiunge 2x al P/E. ",
                            html.B("4.4 "), "= rendimento AAA bond nell'anno di pubblicazione (1962). ",
                            html.B("Y "), "= rendimento AAA bond corrente (normalizzazione)."],
                           style={"font-size": "12px", "line-height": "1.7",
                                  "margin-top": "8px"}),
                    html.P([html.B("Limiti: "),
                            "La formula è conservativa per aziende tech/growth con EPS basso "
                            "ma alto potenziale. Funziona bene per settori maturi (industriali, "
                            "consumer, utilities). Va usata come floor di valutazione, non come "
                            "stima precisa."],
                           style={"font-size": "11px", "color": "#888",
                                  "line-height": "1.5", "margin-top": "6px"}),
                ], style={"background": "#f0f6ff", "padding": "12px",
                           "border-radius": "6px", "border-left": "4px solid #1f77b4",
                           "margin": "12px 0"}),

                dcc.Graph(figure=fig_gr, style={"height": "340px"},
                          config={"displayModeBar": False}),
            ], style={"padding": "14px 16px 30px"})

        # ── TAB MULTIPLI ──────────────────────────────────────────────────────
        elif active_tab == "val-tab-multiples":
            v_pe,  vc_pe  = _verdict(pe_price)
            v_ev,  vc_ev  = _verdict(ev_price)

            # Tabella confronto multipli
            mult_rows = [
                ("P/E trailing",    f"{d.get('pe_trailing'):.1f}x" if d.get('pe_trailing') else "N/D",
                 f"{pe_sector:.1f}x",
                 "Sopra mercato" if d.get('pe_trailing') and d['pe_trailing'] > pe_sector
                 else "Sotto mercato"),
                ("P/E forward",     f"{d.get('pe_forward'):.1f}x" if d.get('pe_forward') else "N/D",
                 f"{pe_sector:.1f}x", "—"),
                ("EV/EBITDA impl.", f"{(d.get('market_cap',0)+net_debt)/(ebitda or 1):.1f}x"
                 if ebitda > 0 else "N/D",
                 f"{ev_mult:.1f}x",
                 "Sopra settore" if ebitda > 0 and
                 (d.get('market_cap', 0) + net_debt) / ebitda > ev_mult
                 else "Sotto settore"),
            ]

            fig_mult = go.Figure()
            models   = ["P/E relativo", "EV/EBITDA"]
            fv_vals  = [pe_price or 0, ev_price or 0]
            fig_mult.add_trace(go.Bar(
                name="Fair Value modello",
                x=models, y=fv_vals,
                marker_color=["#2ca02c" if v and v > price else "#d62728"
                               for v in [pe_price, ev_price]],
                text=[f"{v:.1f}" if v else "N/D" for v in fv_vals],
                textposition="outside"))
            fig_mult.add_hline(y=price, line_color="#333", line_dash="dash",
                                line_width=2,
                                annotation_text=f"Prezzo {price:.2f}")
            fig_mult.update_layout(
                title=dict(text="Fair Value per modelli multipli vs prezzo corrente",
                           font=dict(size=11)),
                yaxis_title=currency,
                margin=dict(t=45, b=30, l=55, r=20),
                paper_bgcolor="white", plot_bgcolor="#f8f8f8")

            return html.Div([
                html.H4("Valutazione per Multipli — P/E relativo & EV/EBITDA",
                        style={"font-size": "14px", "margin": "0 0 6px",
                               "color": "#1a3a5c", "border-bottom": "2px solid #1a3a5c",
                               "padding-bottom": "6px"}),

                html.Div([
                    html.Div([
                        html.H5("P/E Relativo", style={"font-size": "12px",
                                                         "margin": "0 0 6px",
                                                         "color": "#1a3a5c"}),
                        html.P("Fair Value = EPS_forward × P/E settore",
                               style={"font-size": "10px", "font-family": "monospace",
                                      "color": "#666"}),
                        html.P([html.B("EPS forward: "), f"{eps_fwd:.2f}  ×  ",
                                html.B("P/E settore: "), f"{pe_sector:.1f}x  =  ",
                                html.Span(f"{pe_price:.2f} {currency}" if pe_price else "N/D",
                                          style={"font-weight": "bold", "color": vc_pe})],
                               style={"font-size": "13px", "margin-top": "8px"}),
                        html.P([html.Span(f"→ {v_pe}", style={"color": vc_pe,
                                                                "font-weight": "bold"}),
                                "  ", _updown(pe_price)],
                               style={"font-size": "12px"}),
                        html.P([html.B("Come si usa: "),
                                "Il P/E relativo confronta il titolo col multiplo medio "
                                "del settore. Un P/E aziendale > P/E settore indica "
                                "premio di valutazione — giustificato solo da crescita "
                                "superiore o moat competitivo."],
                               style={"font-size": "11px", "color": "#666",
                                      "line-height": "1.5", "margin-top": "10px"}),
                    ], style={"flex": "1", "background": "#f8f8f8", "padding": "14px",
                               "border-radius": "6px", "min-width": "240px"}),

                    html.Div([
                        html.H5("EV/EBITDA", style={"font-size": "12px",
                                                      "margin": "0 0 6px",
                                                      "color": "#1a3a5c"}),
                        html.P("Fair Equity = EBITDA × multiplo − Debito Netto",
                               style={"font-size": "10px", "font-family": "monospace",
                                      "color": "#666"}),
                        html.P([html.B("EBITDA: "), f"{_val_fmt_num(ebitda)}  ×  ",
                                html.B("Multiplo: "), f"{ev_mult:.1f}x",
                                html.Br(),
                                html.B("Fair EV: "),
                                f"{_val_fmt_num(ev_fair)}  −  Debito netto "
                                f"{_val_fmt_num(net_debt)}",
                                html.Br(),
                                html.B("Fair Price: "),
                                html.Span(f"{ev_price:.2f} {currency}" if ev_price else "N/D",
                                          style={"font-weight": "bold", "color": vc_ev})],
                               style={"font-size": "12px", "margin-top": "8px",
                                      "line-height": "1.8"}),
                        html.P([html.Span(f"→ {v_ev}", style={"color": vc_ev,
                                                                "font-weight": "bold"}),
                                "  ", _updown(ev_price)],
                               style={"font-size": "12px"}),
                        html.P([html.B("Come si usa: "),
                                "EV/EBITDA è capital-structure neutral (include debito). "
                                "È preferibile al P/E per confronti cross-settoriali o "
                                "aziende con struttura finanziaria complessa."],
                               style={"font-size": "11px", "color": "#666",
                                      "line-height": "1.5", "margin-top": "10px"}),
                    ], style={"flex": "1", "background": "#f8f8f8", "padding": "14px",
                               "border-radius": "6px", "min-width": "240px"}),
                ], style={"display": "flex", "flex-wrap": "wrap",
                           "gap": "16px", "margin-bottom": "16px"}),

                html.H5("Confronto multipli azienda vs settore",
                        style={"font-size": "12px", "margin": "16px 0 8px"}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Multiplo", style=th_style),
                        html.Th("Aziendale", style=th_style),
                        html.Th("Settore (input)", style=th_style),
                        html.Th("Posizione", style=th_style),
                    ])),
                    html.Tbody([
                        html.Tr([html.Td(r[0], style=td_style),
                                 html.Td(r[1], style={**td_style, "font-weight": "bold"}),
                                 html.Td(r[2], style=td_style),
                                 html.Td(r[3], style={**td_style, "color": "#d62728"
                                          if "Sopra" in r[3] else "#2ca02c"})])
                        for r in mult_rows
                    ]),
                ], style={**tbl_style, "margin-bottom": "16px"}),

                dcc.Graph(figure=fig_mult, style={"height": "280px"},
                          config={"displayModeBar": False}),
            ], style={"padding": "14px 16px 30px"})

        # ── TAB HEATMAP SENSITIVITÀ ───────────────────────────────────────────
        elif active_tab == "val-tab-heatmap":
            wacc_range = [w / 100 for w in range(6, 17, 1)]   # 6% → 16%
            g_range    = [g / 100 for g in range(0, 7, 1)]    # 0% → 6%

            z_vals = []
            for wc in wacc_range:
                row = []
                for gt in g_range:
                    fv, _, _ = _val_dcf(revenue, fcf_m, wc, g1, g2, gt, shares)
                    row.append(round(fv, 2) if fv else 0)
                z_vals.append(row)

            # Calcola % rispetto al prezzo corrente
            z_pct = [[(v - price) / price * 100 if price > 0 else 0
                       for v in row] for row in z_vals]

            fig_heat = go.Figure(go.Heatmap(
                z=z_pct,
                x=[f"{g*100:.0f}%" for g in g_range],
                y=[f"{w*100:.0f}%" for w in wacc_range],
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"{v:+.0f}%" for v in row] for row in z_pct],
                texttemplate="%{text}",
                colorbar=dict(title="Upside/Downside %",
                              tickfont=dict(size=9))))
            # Marca il punto corrente (WACC e gterm degli slider)
            fig_heat.update_layout(
                title=dict(
                    text=f"DCF Sensitività — Upside/Downside% vs Prezzo {price:.2f} {currency}  "
                         f"[FCF margin={fcf_m*100:.1f}%, g1={g1*100:.1f}%, g2={g2*100:.1f}%]",
                    font=dict(size=11)),
                xaxis_title="Tasso finale g",
                yaxis_title="WACC",
                margin=dict(t=55, b=40, l=65, r=20),
                paper_bgcolor="white")

            # Secondo heatmap: prezzo assoluto
            fig_heat2 = go.Figure(go.Heatmap(
                z=z_vals,
                x=[f"{g*100:.0f}%" for g in g_range],
                y=[f"{w*100:.0f}%" for w in wacc_range],
                colorscale="Blues",
                text=[[f"{v:.1f}" for v in row] for row in z_vals],
                texttemplate="%{text}",
                colorbar=dict(title=f"Fair Value ({currency})",
                              tickfont=dict(size=9))))
            fig_heat2.update_layout(
                title=dict(text=f"DCF Fair Value assoluto ({currency})",
                           font=dict(size=11)),
                xaxis_title="Tasso finale g",
                yaxis_title="WACC",
                margin=dict(t=45, b=40, l=65, r=20),
                paper_bgcolor="white")

            return html.Div([
                html.H4("Analisi di Sensitività DCF — WACC × tasso finale",
                        style={"font-size": "14px", "margin": "0 0 6px",
                               "color": "#1a3a5c", "border-bottom": "2px solid #1a3a5c",
                               "padding-bottom": "6px"}),
                html.P("Verde = titolo sottovalutato rispetto al prezzo corrente. "
                       "Rosso = sopravvalutato. La cella è il % di upside/downside "
                       "del DCF per quella combinazione di WACC e g terminale.",
                       style={"font-size": "11px", "color": "#666",
                              "margin": "0 0 12px"}),
                dcc.Graph(figure=fig_heat, style={"height": "360px"},
                          config={"displayModeBar": False}),
                html.Hr(style={"margin": "16px 0"}),
                dcc.Graph(figure=fig_heat2, style={"height": "320px"},
                          config={"displayModeBar": False}),
            ], style={"padding": "14px 16px 30px"})

        elif active_tab == "val-tab-saas":
            return _val_saas_tab(d)

        return html.Div()
