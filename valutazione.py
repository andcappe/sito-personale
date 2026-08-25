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
# Un titolo nuovo scarica i tre prospetti dentro un solo callback: con un
# timeout generoso la richiesta HTTP supererebbe il limite del router di
# Digital Ocean e il browser si troverebbe senza risposta.
_AV_TIMEOUT = 15

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

    url = f"{_AV_URL}?function={funzione}&symbol={ticker}&apikey={_AV_KEY}"

    def _scarica():
        # I tre prospetti si scaricano in fila: senza questa pausa la seconda
        # richiesta torna indietro con il messaggio "1 request per second".
        attesa = _AV_PAUSA - (time.time() - _AV_ULTIMA[0])
        if attesa > 0:
            time.sleep(attesa)
        _AV_ULTIMA[0] = time.time()
        grezzo = urllib.request.urlopen(url, timeout=_AV_TIMEOUT).read().decode()
        return grezzo, json.loads(grezzo)

    try:
        raw, payload = _scarica()
    except Exception as e:
        return _fallito(f"scaricamento fallito: {e}")

    # Alpha Vantage risponde 200 anche quando non ha il dato: l'errore sta nel
    # corpo, come 'Note' (limite di 25 richieste al giorno) o 'Information'.
    avviso = str(payload.get("Note") or payload.get("Information") or "")
    # Lo sbarramento al secondo è momentaneo, quello al giorno no: ritenta una
    # volta sola, altrimenti un prospetto che esiste resterebbe "non
    # disponibile" per un quarto d'ora per colpa della cache negativa.
    if avviso and "per second" in avviso.lower():
        time.sleep(_AV_PAUSA * 2)
        try:
            raw, payload = _scarica()
            avviso = str(payload.get("Note") or payload.get("Information") or "")
        except Exception as e:
            return _fallito(f"scaricamento fallito: {e}")
    if avviso:
        return _fallito(avviso)
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
    """Le quattro righe della cascata del conto economico, più i margini.

    Fatturato → utile lordo → reddito operativo → utile netto: sono lo stesso
    numero di partenza al netto di tre strati di costo diversi, ed è il
    confronto fra loro a dire dove finiscono i soldi. Il reddito operativo è
    quello che misura il mestiere dell'azienda: sotto di lui restano solo
    interessi e imposte, che dipendono da come è finanziata e da dove ha sede,
    non da come lavora.

    `modo='ttm'` somma i 4 trimestri scorrevoli (trailing twelve months),
    `modo='annuale'` usa gli esercizi come li pubblica l'azienda.
    """
    vuoto = {"date": [], "fatturato": [], "utile": [], "lordo": [],
             "operativo": [], "margine": [], "margine_lordo": [],
             "margine_op": [], "valuta": "USD"}
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
    # Il reddito operativo è il risultato della gestione caratteristica: ricavi
    # meno costo del venduto meno le spese di struttura (ricerca, vendite,
    # amministrazione, ammortamenti). Dove Alpha Vantage non lo pubblica si
    # ricostruisce come utile lordo − costi operativi, che è la sua definizione.
    def _op(r):
        v = _av_num(r.get("operatingIncome"))
        if v is not None:
            return v
        lo = (_av_num(r.get("totalRevenue")) - _av_num(r.get("costOfRevenue"))
              if (_av_num(r.get("totalRevenue")) is not None
                  and _av_num(r.get("costOfRevenue")) is not None)
              else _av_num(r.get("grossProfit")))
        oc = _av_num(r.get("operatingExpenses"))
        return (lo - oc) if (lo is not None and oc is not None) else None

    operativi = [_op(r) for r in reports]
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

        date, ricavi, utili, lordi, operativi = (
            date[3:], _ttm(ricavi)[3:], _ttm(utili)[3:], _ttm(lordi)[3:],
            _ttm(operativi)[3:])

    def _marg(serie):
        return [(x / r * 100) if (x is not None and r) else None
                for x, r in zip(serie, ricavi)]

    return {"date": date, "fatturato": ricavi, "utile": utili, "lordo": lordi,
            "operativo": operativi, "margine": _marg(utili),
            "margine_lordo": _marg(lordi), "margine_op": _marg(operativi),
            "valuta": valuta}


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


# ── Valori consigliati per gli slider ────────────────────────────────────────
# Ogni slider parte da un numero ricavato dai conti del titolo, non da un
# default buono per tutti: con parametri arbitrari la stessa schermata dice
# tutto e il contrario di tutto: bastano due punti di WACC. Le formule stanno
# qui, il tab "🎯 Parametri" le mostra con i numeri con cui sono state
# calcolate, così un consiglio si può rifiutare sapendo cosa si rifiuta.
_ERP          = 5.0      # premio per il rischio azionario di un mercato maturo (%)
_RF_RISERVA   = 4.25     # usati solo se FRED non risponde
_AAA_RISERVA  = 5.50
_G_TERM_DEF   = 2.5      # crescita perpetua: inflazione + crescita reale di lungo periodo
_GRAHAM_G_MAX = 15.0     # oltre, la formula di Graham regala multipli irreali

_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
_FRED_KEY = os.environ.get("FRED_API_KEY", "65061ed1fa4c47d53b1d644e1cd858d3")
_FRED_TTL = 12 * 3600
_FRED_MEM = {}           # serie → (valore, data, quando)

# Multipli mediani di settore: rete di sicurezza per i titoli la cui storia non
# basta (meno di due esercizi con utile o EBITDA positivi). La mediana del
# titolo stesso, quando c'è, è un ancoraggio migliore di una media di settore.
_PE_SETTORE = {
    "Technology": 26.0, "Communication Services": 18.0, "Consumer Cyclical": 20.0,
    "Consumer Defensive": 20.0, "Healthcare": 20.0, "Financial Services": 13.0,
    "Industrials": 20.0, "Energy": 12.0, "Basic Materials": 15.0,
    "Utilities": 17.0, "Real Estate": 20.0,
}
_EV_SETTORE = {
    "Technology": 17.0, "Communication Services": 9.0, "Consumer Cyclical": 12.0,
    "Consumer Defensive": 13.0, "Healthcare": 13.0, "Financial Services": 10.0,
    "Industrials": 13.0, "Energy": 6.0, "Basic Materials": 8.0,
    "Utilities": 10.0, "Real Estate": 17.0,
}
_PE_DEF, _EV_DEF = 18.0, 11.0

# Slider → (passo, minimo, massimo): il consiglio deve poter essere scritto
# nello slider, quindi nasce già sul suo passo e dentro i suoi estremi.
_SLIDER_LIMITI = {
    "val-wacc":       (0.5,  4.0, 20.0),
    "val-g1":         (0.5,  0.0, 60.0),
    "val-g2":         (0.5,  0.0, 45.0),
    "val-gterm":      (0.25, 0.0,  8.75),
    "val-fcf-margin": (0.5,  1.0, 50.0),
    "val-ke":         (0.5,  4.0, 20.0),
    "val-ddm-g":      (0.25, 0.0, 10.0),
    "val-graham-g":   (0.5,  0.0, 25.0),
    "val-bond-yield": (0.25, 1.0, 10.0),
    "val-pe-sector":  (1.0,  5.0, 60.0),
    "val-ev-ebitda":  (0.5,  3.0, 30.0),
}

# Etichetta e unità di misura di ogni consiglio, per il tab Parametri.
_SLIDER_ORDINE = ["val-wacc", "val-g1", "val-g2", "val-gterm", "val-fcf-margin",
                  "val-ke", "val-ddm-g", "val-graham-g", "val-bond-yield",
                  "val-pe-sector", "val-ev-ebitda"]

_SLIDER_ETICHETTE = [
    ("val-wacc",       "WACC — tasso di sconto",            "%",  "📉 DCF"),
    ("val-g1",         f"Crescita ricavi anni 1-{ANNI_FASE1}", "%", "📉 DCF"),
    ("val-g2",         f"Crescita ricavi anni {ANNI_FASE1+1}-{ANNI_DCF}", "%", "📉 DCF"),
    ("val-gterm",      "Tasso finale (perpetuità)",         "%",  "📉 DCF"),
    ("val-fcf-margin", "Margine FCF unlevered / ricavi",    "%",  "📉 DCF"),
    ("val-ke",         "Ke — costo del capitale proprio",   "%",  "💰 DDM"),
    ("val-ddm-g",      "Crescita del dividendo",            "%",  "💰 DDM"),
    ("val-graham-g",   "Crescita EPS attesa",               "%",  "📐 Graham"),
    ("val-bond-yield", "Rendimento AAA bond",               "%",  "📐 Graham"),
    ("val-pe-sector",  "P/E di confronto",                  "x",  "📈 Multipli"),
    ("val-ev-ebitda",  "EV/EBITDA di confronto",            "x",  "📈 Multipli"),
]


def _ssl_ctx():
    """Contesto SSL con i certificati di `certifi` quando ci sono.

    Sul server i certificati di sistema bastano; su macOS il Python del
    framework non ne ha e senza questo la chiamata a FRED fallirebbe solo in
    locale, facendo sembrare rotto un consiglio che in produzione funziona.
    """
    try:
        import ssl, certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _num(x):
    """float, oppure None per NaN e valori non numerici."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _mediana(valori):
    v = sorted(x for x in valori if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _anni_fra(col0, col1, riserva):
    """Anni fra due colonne di bilancio; `riserva` se le date non si leggono."""
    try:
        n = abs(int(col1.year) - int(col0.year))
        return n if n > 0 else riserva
    except Exception:
        return riserva


def _cagr(v0, v1, anni):
    """Crescita annua composta. None quando i segni non la rendono definita."""
    if not v0 or not v1 or v0 <= 0 or v1 <= 0 or anni <= 0:
        return None
    return (v1 / v0) ** (1.0 / anni) - 1.0


def _riga_bilancio(df, *chiavi):
    """Riga di un prospetto yfinance, per nome esatto (maiuscole indifferenti).

    Prima chiave trovata, in ordine: i nomi delle voci cambiano da titolo a
    titolo (`Interest Expense` / `Interest Expense Non Operating`, `EBITDA` /
    `Normalized EBITDA`) e un `in` sulla sottostringa pescherebbe la voce
    sbagliata.
    """
    if df is None or getattr(df, "empty", True):
        return None
    mappa = {str(i).strip().lower(): i for i in df.index}
    for k in chiavi:
        i = mappa.get(k.lower())
        if i is not None:
            return df.loc[i]
    return None


def _cella(riga, col):
    """Valore di una colonna (esercizio) di una riga di bilancio."""
    if riga is None:
        return None
    try:
        return _num(riga.get(col))
    except Exception:
        return None


def _su_slider(slider, valore):
    """Porta un valore sul passo dello slider e dentro i suoi estremi.

    Torna anche il valore grezzo quando il taglio l'ha spostato davvero: un
    consiglio limitato dal massimo dello slider deve dirlo, altrimenti si legge
    come un dato del titolo (NVDA cresce dell'83%, lo slider si ferma a 60).
    """
    if valore is None:
        return None, None
    passo, mn, mx = _SLIDER_LIMITI[slider]
    v = min(mx, max(mn, valore))
    v = round(round(v / passo) * passo, 4)
    return v, (valore if abs(valore - v) > passo / 2 else None)


def _fred_ultimo(serie):
    """Ultima osservazione valida di una serie FRED, in cache per 12 ore.

    Senza cache ogni caricamento di un titolo sarebbe una chiamata di rete in
    più su un dato che si muove una volta al giorno.
    """
    ora = time.time()
    c = _FRED_MEM.get(serie)
    if c and ora - c[2] < _FRED_TTL:
        return c[0], c[1]
    url = (f"{_FRED_URL}?series_id={serie}&api_key={_FRED_KEY}"
           f"&file_type=json&sort_order=desc&limit=24")
    ctx = _ssl_ctx()
    req = (urllib.request.urlopen(url, timeout=8, context=ctx) if ctx
           else urllib.request.urlopen(url, timeout=8))
    with req as r:
        oss = json.loads(r.read().decode()).get("observations", [])
    for o in oss:
        v = _num(o.get("value"))
        if v is not None:
            _FRED_MEM[serie] = (v, o.get("date", ""), ora)
            return v, o.get("date", "")
    return None, None


def _tassi_mercato():
    """Risk-free (Treasury 10 anni) e rendimento AAA correnti, da FRED.

    Sono i due soli input dei modelli che non stanno nei conti del titolo — la
    base del CAPM e la Y della formula di Graham — e sono anche i due che si
    muovono di più: lasciarli a un default rende finto tutto il resto.
    """
    rf = aaa = None
    fonti = []
    try:
        rf, d = _fred_ultimo("DGS10")
        if rf is not None:
            fonti.append(f"risk-free {rf:.2f}% (FRED DGS10 {d})")
    except Exception:
        pass
    try:
        aaa, d = _fred_ultimo("AAA")
        if aaa is not None:
            fonti.append(f"AAA {aaa:.2f}% (FRED {d})")
    except Exception:
        pass
    if rf is None:
        rf = _RF_RISERVA
        fonti.append(f"risk-free {rf:.2f}% (valore di riserva: FRED non raggiungibile)")
    if aaa is None:
        aaa = _AAA_RISERVA
        fonti.append(f"AAA {aaa:.2f}% (valore di riserva: FRED non raggiungibile)")
    return rf, aaa, " · ".join(fonti)


def stima_crescita_flussi(righe):
    """Crescita del flusso di cassa stimata partendo dal reddito netto.

    `righe` = [(data, utile_netto, fcf)], dal più vecchio al più recente.

    Perché passare dal reddito netto invece di misurare il FCF e basta: il
    flusso di cassa è molto più ballerino dell'utile, perché il capex è
    grumoso — un anno di fabbriche nuove lo taglia a metà senza che il
    mestiere sia cambiato. L'utile netto, che gli investimenti li spalma in
    ammortamenti, è la parte stabile; il resto è il **tasso di conversione**
    FCF/utile netto, che si guarda a parte.

    I due pezzi si ricompongono esattamente, non per approssimazione:

        FCF = utile × conversione   ⟹   (1+g_fcf) = (1+g_utile) × (1+g_conv)

    Da qui la stima prospettica: si tiene la crescita dell'utile e si assume
    che la conversione smetta di scivolare (g_conv = 0). Dire quanto è
    scivolata finora resta compito di chi guarda, e infatti si stampa.

    Le medie sono due perché rispondono a domande diverse: la media
    aritmetica delle variazioni è "quanto è cambiato in un anno tipico" ed è
    sempre **più alta** della crescita davvero realizzata (da 100 a 50 a 100
    fa −50% e +100%, media +25%, crescita vera zero). Per far crescere i
    flussi nel DCF serve il CAGR, che parte e arriva dove è arrivata l'azienda.
    """
    righe = [(d_, u_, f_) for d_, u_, f_ in righe
             if u_ is not None and f_ is not None]
    if len(righe) < 2:
        return None

    anni = []
    for i, (data, utile, fcf) in enumerate(righe):
        prec_u = righe[i - 1][1] if i else None
        prec_f = righe[i - 1][2] if i else None
        anni.append({
            "data": data,
            "utile": utile,
            "fcf": fcf,
            # Da una base negativa la variazione percentuale non significa
            # niente: resta un buco, non un numero che sembra un'informazione.
            "var_utile": ((utile / prec_u - 1) * 100
                          if (prec_u is not None and prec_u > 0) else None),
            "var_fcf": ((fcf / prec_f - 1) * 100
                        if (prec_f is not None and prec_f > 0) else None),
            "conversione": (fcf / utile * 100) if utile > 0 else None,
        })

    var_u = [a["var_utile"] for a in anni if a["var_utile"] is not None]
    var_f = [a["var_fcf"] for a in anni if a["var_fcf"] is not None]
    conv  = [a["conversione"] for a in anni if a["conversione"] is not None]

    def _pct(x):
        return None if x is None else x * 100

    # Il composto vuole due estremi positivi. Con esercizi in perdita in testa
    # alla serie (AT&T 2022, −6,9 mld) non si butta via tutto: si parte dal
    # primo anno in utile e si dice su quanti anni si è misurato. Una mediana
    # delle variazioni al posto del composto sarebbe peggio del silenzio — su
    # AT&T dava +34,6% l'anno per un'azienda che cresce a una cifra.
    i0 = next((i for i, (_, u_, _) in enumerate(righe) if u_ > 0), None)
    if i0 is not None and i0 >= len(righe) - 1:
        i0 = None                                  # solo l'ultimo anno è in utile
    n_anni = (len(righe) - 1 - i0) if i0 is not None else 0
    parziale = bool(i0)                            # i0 > 0: serie accorciata

    cagr_u = cagr_f = None
    if i0 is not None:
        cagr_u = _pct(_cagr(righe[i0][1], righe[-1][1], n_anni))
        cagr_f = _pct(_cagr(righe[i0][2], righe[-1][2], n_anni))
    # La conversione è il ponte fra i due: cresce come il rapporto fra i due
    # CAGR, e per costruzione (1+g_u)(1+g_c) = (1+g_f).
    cagr_c = None
    if cagr_u is not None and cagr_f is not None:
        cagr_c = ((1 + cagr_f / 100) / (1 + cagr_u / 100) - 1) * 100

    # Su banche e assicurazioni il free cash flow non vuole dire niente:
    # prestiti e depositi passano dal rendiconto come se fossero investimenti,
    # e il rapporto con l'utile salta da +2281% a −483% (Intesa Sanpaolo).
    # Quando il segno cambia o l'escursione è enorme il ponte va dichiarato rotto.
    conv_rotta = bool(conv) and (min(conv) <= 0 or max(conv) > 4 * max(min(conv), 1e-9))

    stima, come, avviso = None, "", ""
    if cagr_u is not None:
        stima = cagr_u
        come = (f"crescita composta del reddito netto su {n_anni} "
                f"{'anno' if n_anni == 1 else 'anni'}"
                + (" (la serie parte dal primo esercizio in utile)"
                   if parziale else "")
                + ", con il tasso di conversione in cassa tenuto fermo dov'è oggi")
        if conv_rotta:
            avviso = ("il rapporto fra flusso di cassa e utile è troppo "
                      "ballerino perché questa stima si trasferisca ai flussi: "
                      "succede sulle banche e sulle società che stanno "
                      "cambiando pelle. Qui vale come crescita degli utili, "
                      "non come crescita della cassa")
        elif cagr_c is not None and abs(cagr_c) >= 3.0:
            verso = "scivolata" if cagr_c < 0 else "salita"
            avviso = (
                f"la conversione FCF/utile è {verso} del {cagr_c:+.1f}% "
                f"l'anno: è questo, non l'utile, a spiegare perché il flusso "
                f"di cassa è cresciuto del {cagr_f:+.1f}% invece del "
                f"{cagr_u:+.1f}%. Se il movimento continua la crescita dei "
                f"flussi resta più "
                f"{'bassa' if cagr_c < 0 else 'alta'} di questa stima")
    else:
        come = ("il reddito netto è in perdita nell'ultimo esercizio: non c'è "
                "una crescita da comporre")
        avviso = "imposta la crescita a mano, è un'ipotesi tua, non un dato"

    return {
        "anni": anni,
        "n_var": len(var_u),
        "media_utile": (sum(var_u) / len(var_u)) if var_u else None,
        "mediana_utile": _mediana(var_u),
        "cagr_utile": cagr_u,
        "media_fcf": (sum(var_f) / len(var_f)) if var_f else None,
        "mediana_fcf": _mediana(var_f),
        "cagr_fcf": cagr_f,
        "cagr_conv": cagr_c,
        "conv_mediana": _mediana(conv),
        "conv_rotta": conv_rotta,
        "n_anni": n_anni,
        "parziale": parziale,
        "conv_min": min(conv) if conv else None,
        "conv_max": max(conv) if conv else None,
        "stima": stima,
        "come": come,
        "avviso": avviso,
    }


def parametri_consigliati(info, fin, cf, bs, storia=None, dividendi=None):
    """Valore di partenza di ogni slider, ricavato dai conti del titolo.

    Torna `{id_slider: {"v": valore, "come": spiegazione, "avviso": ...}}` più
    la chiave `_det` con i passaggi intermedi (beta, aliquota, costo del debito,
    pesi del WACC) che il tab Parametri stampa in testa.

    Nessuna eccezione esce da qui: un consiglio mancante è una riga "N/D" nel
    tab, non un titolo che non si carica.
    """
    out, det = {}, {}

    def _g(k):
        return _num(info.get(k))

    def _put(slider, valore, come, avviso=""):
        v, grezzo = _su_slider(slider, valore)
        if grezzo is not None:
            _, mn, mx = _SLIDER_LIMITI[slider]
            taglio = (f"calcolato {grezzo:.1f}, riportato dentro la corsa dello "
                      f"slider ({mn:g}–{mx:g})")
            avviso = f"{avviso} · {taglio}" if avviso else taglio
        out[slider] = {"v": v, "come": come, "avviso": avviso}
        return v

    # ── tassi di mercato e struttura del capitale ────────────────────────────
    rf, aaa, fonte_tassi = _tassi_mercato()
    beta = _g("beta") or 1.0
    mc   = _g("marketCap") or 0.0
    debt = _g("totalDebt") or 0.0
    det["tassi"] = fonte_tassi
    det["beta"], det["mc"], det["debito"] = beta, mc, debt

    # Aliquota effettiva: mediana degli esercizi in utile. Un solo anno con una
    # posta straordinaria darebbe un'aliquota che non descrive l'azienda.
    r_tax = _riga_bilancio(fin, "Tax Provision")
    r_pre = _riga_bilancio(fin, "Pretax Income")
    aliquote = []
    if r_pre is not None:
        for col in r_pre.index:
            t_, p_ = _cella(r_tax, col), _cella(r_pre, col)
            if t_ is not None and p_ and p_ > 0 and 0 <= t_ / p_ <= 0.6:
                aliquote.append(t_ / p_)
    aliq = _mediana(aliquote)
    if aliq is None:
        aliq, det["aliquota_fonte"] = 0.25, "nessun esercizio in utile: 25% convenzionale"
    else:
        aliq = min(0.35, max(0.05, aliq))
        det["aliquota_fonte"] = (f"mediana di {len(aliquote)} eserciz"
                                 f"{'io' if len(aliquote) == 1 else 'i'} in utile")
    det["aliquota"] = aliq

    # Costo del debito: quanto paga davvero, non quanto pagherebbe in teoria.
    r_int = _riga_bilancio(fin, "Interest Expense", "Interest Expense Non Operating")
    kd = None
    if debt > 0 and r_int is not None:
        for col in r_int.index:
            i_ = _cella(r_int, col)
            if i_:
                kd = abs(i_) / debt * 100
                break
    if kd is None:
        kd = aaa + 0.5
        det["kd_fonte"] = f"AAA {aaa:.2f}% + 0,50 (oneri finanziari non esposti)"
    elif not 0.5 <= kd <= 15.0:
        kd = aaa + 0.5
        det["kd_fonte"] = (f"AAA {aaa:.2f}% + 0,50 (oneri ÷ debito fuori scala, "
                           f"dato di bilancio non confrontabile)")
    else:
        det["kd_fonte"] = "oneri finanziari dell'ultimo esercizio ÷ debito totale"
    # Gli oneri di bilancio sono la cedola media del debito già emesso, spesso
    # acceso quando i tassi erano altri: come costo del capitale conta quanto
    # costerebbe rifinanziarsi oggi, e nessuno si rifinanzia sotto il rendimento
    # delle emittenti migliori.
    if kd < aaa:
        det["kd_fonte"] = (f"{det['kd_fonte']}, {kd:.2f}%, portato al rendimento "
                           f"AAA corrente: il debito in bilancio è più vecchio "
                           f"dei tassi di oggi")
        kd = aaa
    det["kd"] = kd

    ke_capm = rf + beta * _ERP
    # L'azionista viene dopo i creditori in ogni scenario: pretendere dalle
    # azioni meno di quanto la stessa azienda paga sui suoi bond è una
    # contraddizione, e su un titolo a beta basso il CAPM ci arriva davvero
    # (ENI: 5,9% di CAPM contro un debito che costa di più).
    ke = max(ke_capm, kd + 2.0)
    det["ke_capm"], det["ke"] = ke_capm, ke
    if ke > ke_capm:
        det["ke_nota"] = (f"CAPM {ke_capm:.2f}% alzato a costo del debito + 2 punti: "
                          f"il capitale proprio è subordinato, non può costare "
                          f"meno del debito della stessa azienda")
    if mc > 0 and debt > 0:
        we, wd = mc / (mc + debt), debt / (mc + debt)
        wacc = we * ke + wd * kd * (1 - aliq)
    else:
        we, wd = 1.0, 0.0
        wacc = ke
    det["we"], det["wd"], det["wacc"] = we, wd, wacc

    _put("val-wacc", wacc,
         f"CAPM sul capitale proprio e costo effettivo del debito, pesati a valori "
         f"di mercato: Ke = {rf:.2f}% + {beta:.2f} × {_ERP:.1f}% = {ke_capm:.2f}%"
         + (f" → {ke:.2f}% (minimo: costo del debito + 2 punti)" if ke > ke_capm else "")
         + f"; Kd = {kd:.2f}% × (1 − {aliq*100:.0f}%) = {kd*(1-aliq):.2f}%; "
           f"pesi E {we*100:.0f}% / D {wd*100:.0f}% → WACC {wacc:.2f}%")
    _put("val-ke", ke,
         f"CAPM: risk-free {rf:.2f}% + beta {beta:.2f} × premio per il rischio "
         f"{_ERP:.1f}% = {ke_capm:.2f}%"
         + (f", alzato a {ke:.2f}% (costo del debito + 2 punti)" if ke > ke_capm else "")
         + ". È il tasso del DDM, che sconta un flusso già al netto degli "
           "interessi e quindi non usa il WACC",
         det.get("ke_nota", ""))

    # ── crescita dei ricavi ──────────────────────────────────────────────────
    r_rev = _riga_bilancio(fin, "Total Revenue", "Operating Revenue")
    ricavi = []
    if r_rev is not None:
        for col in r_rev.index:            # colonne dalla più recente
            v = _cella(r_rev, col)
            if v and v > 0:
                ricavi.append((col, v))
    cagr_rev = (_cagr(ricavi[-1][1], ricavi[0][1],
                      _anni_fra(ricavi[-1][0], ricavi[0][0], len(ricavi) - 1))
                if len(ricavi) >= 2 else None)
    yoy_bil  = (ricavi[0][1] / ricavi[1][1] - 1) if len(ricavi) >= 2 else None
    yoy_info = _g("revenueGrowth")
    voci = []
    if cagr_rev is not None: voci.append(f"CAGR {len(ricavi)-1} anni {cagr_rev*100:+.1f}%")
    if yoy_bil  is not None: voci.append(f"ultimo esercizio {yoy_bil*100:+.1f}%")
    if yoy_info is not None: voci.append(f"ultimo trimestre {yoy_info*100:+.1f}%")
    g1_raw = _mediana([x * 100 for x in (cagr_rev, yoy_bil, yoy_info) if x is not None])
    avviso_g1 = ""
    if g1_raw is not None and g1_raw < 0:
        avviso_g1 = ("i ricavi stanno calando: decidi tu se è un passaggio "
                     "temporaneo — allora alza la fase 1 — o la nuova normalità")
    _put("val-g1", 8.0 if g1_raw is None else g1_raw,
         ("mediana delle misure di crescita dei ricavi disponibili ("
          + " · ".join(voci) + ")" if voci else
          "storico dei ricavi troppo corto per misurare una crescita (serve più "
          "di un esercizio): 8% convenzionale, da correggere a mano"),
         avviso_g1)

    gterm_raw = min(_G_TERM_DEF, max(0.0, wacc - 0.25))
    _put("val-gterm", gterm_raw,
         f"{_G_TERM_DEF:.1f}% (inflazione più crescita reale di lungo periodo): "
         f"nessuna azienda cresce più dell'economia per sempre"
         + ("" if gterm_raw >= _G_TERM_DEF else
            f" — qui abbassato a {gterm_raw:.2f}% perché deve restare sotto il "
            f"WACC {wacc:.2f}%, altrimenti Gordon non converge"))

    g1_eff = out["val-g1"]["v"] if out["val-g1"]["v"] is not None else 8.0
    g2_raw = (g1_eff + gterm_raw) / 2
    _put("val-g2", g2_raw,
         f"punto di mezzo fra la crescita di fase 1 ({g1_eff:.1f}%) e il tasso "
         f"finale ({gterm_raw:.2f}%): la crescita non si spegne di colpo, sfuma")

    # ── margine FCF unlevered ────────────────────────────────────────────────
    # Il DCF sconta al WACC e sottrae il debito netto, quindi il flusso deve
    # essere quello che spetta a tutti i finanziatori: il "Free Cash Flow" del
    # rendiconto è già al netto degli interessi pagati, e gli interessi netti
    # d'imposta vanno rimessi dentro. Senza questo passaggio il debito verrebbe
    # contato due volte, nel flusso e nel ponte finale.
    r_fcf   = _riga_bilancio(cf, "Free Cash Flow")
    r_ocf   = _riga_bilancio(cf, "Operating Cash Flow",
                             "Cash Flow From Continuing Operating Activities")
    r_capex = _riga_bilancio(cf, "Capital Expenditure", "Purchase Of PPE")
    margini, dettaglio_marg = [], []
    for col, rev_v in ricavi:
        f_ = _cella(r_fcf, col)
        if f_ is None:
            a, b = _cella(r_ocf, col), _cella(r_capex, col)
            f_ = (a - abs(b)) if (a is not None and b is not None) else None
        if f_ is None:
            continue
        i_ = _cella(r_int, col) or 0.0
        m = (f_ + abs(i_) * (1 - aliq)) / rev_v * 100
        margini.append(m)
        try:
            dettaglio_marg.append(f"{col.year}: {m:.1f}%")
        except Exception:
            dettaglio_marg.append(f"{m:.1f}%")
    marg = _mediana(margini)
    det["margini_fcf"] = dettaglio_marg
    # Un'azienda che brucia cassa non ha un margine da consigliare: portarlo
    # all'1% del minimo dello slider inventerebbe un flusso positivo e il DCF
    # stamperebbe un prezzo obiettivo per una società che non ne ha uno.
    if marg is not None and marg <= 0:
        out["val-fcf-margin"] = {
            "v": None,
            "come": (f"il flusso di cassa unlevered è negativo negli esercizi "
                     f"disponibili ({' · '.join(dettaglio_marg)}): non c'è un "
                     f"margine da consigliare"),
            "avviso": ("il DCF non si applica a un'azienda che brucia cassa. "
                       "Se vuoi comunque usarlo, imposta a mano il margine che "
                       "ti aspetti a regime — è un'ipotesi tua, non un dato"),
        }
        marg = None
    else:
        _put("val-fcf-margin", marg,
             (f"mediana del margine FCF unlevered degli esercizi disponibili "
              f"({' · '.join(dettaglio_marg)}). Unlevered = FCF di rendiconto + "
              f"oneri finanziari × (1 − {aliq*100:.0f}%): è il flusso che spetta "
              f"a tutti i finanziatori, coerente con lo sconto al WACC"
              if marg is not None else
              "rendiconto finanziario non disponibile: imposta il margine a mano"))

    # ── multipli: la mediana del titolo stesso, non una media di settore ─────
    r_eps = _riga_bilancio(fin, "Diluted EPS", "Basic EPS")
    r_sh  = _riga_bilancio(fin, "Diluted Average Shares", "Basic Average Shares")
    r_eb  = _riga_bilancio(fin, "EBITDA", "Normalized EBITDA")
    r_nd  = _riga_bilancio(bs, "Net Debt")
    chiusure = None
    try:
        if storia is not None and not storia.empty:
            chiusure = storia["Close"]
            if getattr(chiusure.index, "tz", None) is not None:
                chiusure.index = chiusure.index.tz_localize(None)
    except Exception:
        chiusure = None
    pe_st, ev_st = [], []
    if chiusure is not None:
        for col, _rev in ricavi:
            try:
                p = _num(chiusure.asof(col))     # ultima chiusura entro la data di bilancio
            except Exception:
                p = None
            if not p:
                continue
            e_ = _cella(r_eps, col)
            if e_ and e_ > 0:
                pe_st.append(p / e_)
            s_, b_ = _cella(r_sh, col), _cella(r_eb, col)
            n_ = _cella(r_nd, col) or 0.0
            if s_ and s_ > 0 and b_ and b_ > 0:
                ev_st.append((p * s_ + n_) / b_)
    settore = (info.get("sector") or "").strip()
    pe_med, ev_med = _mediana(pe_st), _mediana(ev_st)
    if pe_med is not None and len(pe_st) >= 2:
        _put("val-pe-sector", pe_med,
             f"mediana del P/E del titolo negli ultimi {len(pe_st)} esercizi "
             f"({' · '.join(f'{x:.1f}x' for x in pe_st)}): il multiplo che il "
             f"mercato gli riconosce di solito, più significativo di una media "
             f"di settore")
    else:
        _put("val-pe-sector", _PE_SETTORE.get(settore, _PE_DEF),
             f"mediana di settore ({settore or 'non classificato'}): la storia "
             f"del titolo non basta (servono due esercizi con utile positivo)")
    if ev_med is not None and len(ev_st) >= 2:
        _put("val-ev-ebitda", ev_med,
             f"mediana dell'EV/EBITDA del titolo negli ultimi {len(ev_st)} "
             f"esercizi ({' · '.join(f'{x:.1f}x' for x in ev_st)}), con il debito "
             f"netto di ciascun esercizio")
    else:
        _put("val-ev-ebitda", _EV_SETTORE.get(settore, _EV_DEF),
             f"mediana di settore ({settore or 'non classificato'}): la storia "
             f"del titolo non basta (servono due esercizi con EBITDA positivo)")

    # ── dividendo ────────────────────────────────────────────────────────────
    div_g, anni_div = None, 0
    try:
        if dividendi is not None and len(dividendi) > 0:
            per_anno = dividendi.groupby(dividendi.index.year).sum()
            # l'anno in corso è incompleto e falserebbe il CAGR: si toglie solo
            # se è davvero l'anno corrente (un titolo che ha smesso di pagare
            # dividendi nel 2019 ha il 2019 chiuso, non da scartare)
            if len(per_anno) >= 2 and int(per_anno.index[-1]) == time.localtime().tm_year:
                per_anno = per_anno.iloc[:-1]
            if len(per_anno) >= 3:
                anni_div = min(5, len(per_anno) - 1)
                div_g = _cagr(float(per_anno.iloc[-1 - anni_div]),
                              float(per_anno.iloc[-1]), anni_div)
    except Exception:
        div_g = None
    tetto_ddm = max(0.0, min(6.0, ke - 0.5))
    if div_g is None:
        _put("val-ddm-g", min(2.0, tetto_ddm),
             "storico dei dividendi troppo corto (servono tre anni chiusi): "
             "valore prudenziale")
    else:
        avviso = ""
        if div_g * 100 > tetto_ddm:
            avviso = (f"crescita storica {div_g*100:+.1f}%, tenuta sotto "
                      f"{tetto_ddm:.2f}%: Gordon esige g < Ke e un dividendo non "
                      f"cresce a quel ritmo per sempre")
        elif div_g < 0:
            avviso = (f"il dividendo è calato del {abs(div_g)*100:.1f}% l'anno: "
                      f"il consiglio è 0, non una crescita negativa")
        _put("val-ddm-g", max(0.0, min(div_g * 100, tetto_ddm)),
             f"CAGR del dividendo per azione sugli ultimi {anni_div} anni chiusi",
             avviso)

    # ── crescita degli utili per Graham ──────────────────────────────────────
    # gli esercizi senza EPS si saltano, ma la distanza va contata sulle date:
    # con un anno mancante in mezzo, "numero di dati − 1" gonfierebbe il CAGR
    eps_serie = [(c, _cella(r_eps, c)) for c, _ in ricavi]
    eps_serie = [(c, v) for c, v in eps_serie if v]
    cagr_eps = None
    if len(eps_serie) >= 2:
        cagr_eps = _cagr(eps_serie[-1][1], eps_serie[0][1],
                         _anni_fra(eps_serie[-1][0], eps_serie[0][0], len(eps_serie) - 1))
    g_eps_info = _g("earningsGrowth")
    voci_eps = []
    if cagr_eps is not None:   voci_eps.append(f"CAGR EPS {cagr_eps*100:+.1f}%")
    if g_eps_info is not None: voci_eps.append(f"ultimo trimestre {g_eps_info*100:+.1f}%")
    g_eps = _mediana([x * 100 for x in (cagr_eps, g_eps_info) if x is not None])
    avviso_eps = ""
    if g_eps is not None and g_eps > _GRAHAM_G_MAX:
        avviso_eps = (f"crescita storica {g_eps:.1f}%, tenuta a {_GRAHAM_G_MAX:.0f}%: "
                      f"Graham intendeva la crescita sostenibile dei prossimi 7-10 "
                      f"anni, oltre questa soglia la formula stampa multipli irreali")
    _put("val-graham-g", 5.0 if g_eps is None else min(max(g_eps, 0.0), _GRAHAM_G_MAX),
         ("mediana fra " + " e ".join(voci_eps) if voci_eps
          else "nessuno storico di utili: 5% prudenziale"), avviso_eps)

    _put("val-bond-yield", aaa,
         "rendimento corrente delle obbligazioni societarie AAA (Moody's, serie "
         "FRED «AAA»): è la Y della formula di Graham, il rendimento "
         "dell'alternativa senza rischio azionario")

    out["_det"] = det
    return out


def _consigli_titolo(t, info):
    """Scarica il resto dei prospetti e calcola i consigli. Non solleva mai."""
    def _p(nome):
        try:
            v = getattr(t, nome)
            return v if (v is not None and not v.empty) else None
        except Exception:
            return None

    try:
        storia = None
        try:
            storia = t.history(period="7y", interval="1mo")
        except Exception:
            storia = None
        dividendi = None
        try:
            dividendi = t.dividends
        except Exception:
            dividendi = None
        return parametri_consigliati(info, _p("financials"), _p("cashflow"),
                                     _p("balance_sheet"), storia, dividendi)
    except Exception:
        import traceback
        traceback.print_exc()
        return {}


def _val_dcf(revenue, fcf_margin, wacc, g1, g2, gterm, shares, net_debt=0.0):
    """DCF a 2 fasi: anni 1-ANNI_FASE1 a g1, poi fino ad ANNI_DCF a g2,
    infine il tasso finale in perpetuità (Gordon).

    Torna `(prezzo, flussi, pv_tv, gordon_ok, ev)`.

    Il flusso scontato è **unlevered** (spetta a tutti i finanziatori) e il
    tasso di sconto è il WACC (costo medio di tutto il capitale): la somma
    dei valori attuali è quindi il valore dell'**impresa**, non quello degli
    azionisti. Il ponte finale è obbligatorio — enterprise value meno debito
    netto, poi diviso per le azioni — perché chi compra l'azione eredita
    anche i debiti. Dividere l'enterprise value per le azioni regalava
    all'azionista tutto il capitale dei creditori: su un titolo indebitato
    come AT&T (debito netto ≈ market cap) il fair value usciva quasi
    raddoppiato, su una cassaforte netta come Alphabet usciva sottostimato.

    Il quarto valore dice se la perpetuità è applicabile. Gordon converge
    solo per un tasso finale **sotto** il costo del capitale: da lì in su la
    serie diverge e il valore terminale non è zero, è indefinito. Chi chiama
    deve dirlo, non spacciare per valutazione la somma dei soli anni
    espliciti — sarebbe un prezzo bassissimo che sembra un giudizio sul
    titolo mentre è solo il modello che ha smesso di valere.
    """
    if revenue <= 0 or shares <= 0:
        return None, [], 0.0, True, None
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
    gordon_ok = gterm < wacc
    tv = fcf_t * (1 + gterm) / (wacc - gterm) if gordon_ok else 0.0
    pv_tv = tv / (1 + wacc) ** ANNI_DCF
    pv_sum += pv_tv
    # pv_sum = valore d'impresa; l'azionista prende quel che avanza dopo i
    # creditori. Con cassa netta il debito netto è negativo e si somma.
    equity = pv_sum - net_debt
    fair_price = equity / shares
    return fair_price, fcf_rows, pv_tv, gordon_ok, pv_sum


def _dcf_implicito(prezzo, revenue, fcf_margin, wacc, g1, g2, gterm, shares,
                   net_debt):
    """Cosa deve credere chi compra al prezzo di mercato.

    Il DCF diretto risponde «quanto vale»; questo risponde «quale tasso di
    sconto e quale crescita rendono giusto il prezzo di oggi». È il numero che
    serve davvero per posizionare gli slider: dice se un WACC del 9% è una
    scelta o una distrazione, perché lo mette accanto a quello che il mercato
    sta effettivamente usando su questo titolo.

    Torna `(wacc_implicito, g1_implicita)` in percentuale, `None` dove
    l'equazione non ha soluzione dentro limiti sensati.
    """
    if not prezzo or prezzo <= 0 or revenue <= 0 or shares <= 0:
        return None, None

    def _prezzo(w=None, gg=None):
        fv, _, _, ok, _ = _val_dcf(revenue, fcf_margin,
                                   wacc if w is None else w,
                                   g1 if gg is None else gg,
                                   g2 if gg is None else gg, gterm, shares,
                                   net_debt)
        return fv if ok else None

    # Il valore scende al salire del tasso: bisezione fra poco sopra il tasso
    # finale (dove Gordon esplode) e un tasso che nessuno userebbe.
    w_imp = None
    lo, hi = gterm + 0.0025, 0.60
    p_lo, p_hi = _prezzo(w=lo), _prezzo(w=hi)
    if p_lo is not None and p_hi is not None and p_hi <= prezzo <= p_lo:
        for _ in range(60):
            mid = (lo + hi) / 2
            if (_prezzo(w=mid) or 0) > prezzo:
                lo = mid
            else:
                hi = mid
        w_imp = (lo + hi) / 2 * 100

    # Stessa cosa sulla crescita, tenendo fermo il tasso di sconto: qui il
    # valore sale con la crescita, quindi il verso della bisezione si inverte.
    g_imp = None
    lo, hi = -0.50, 1.00
    p_lo, p_hi = _prezzo(gg=lo), _prezzo(gg=hi)
    if p_lo is not None and p_hi is not None and p_lo <= prezzo <= p_hi:
        for _ in range(60):
            mid = (lo + hi) / 2
            if (_prezzo(gg=mid) or 0) < prezzo:
                lo = mid
            else:
                hi = mid
        g_imp = (lo + hi) / 2 * 100
    return w_imp, g_imp


def layout():
    """Tab valutazione titolo azionario — 6 modelli + heatmap sensitività."""

    def _inp(id_, placeholder, value="", width="100%", type_="text"):
        return dcc.Input(id=id_, type=type_, placeholder=placeholder, value=value,
                         debounce=True,
                         persistence=True, persistence_type="session",
                         style={"width": width, "padding": "5px 8px",
                                "border": "1px solid #ccc", "border-radius": "4px",
                                "font-size": "12px"})

    def _lbl(text):
        return html.Label(text, style={"font-size": "10px", "color": "#555",
                                       "margin-top": "8px", "display": "block"})

    def _sl(id_, mn, mx, step, val, label):
        # Gli slider devono ricordarsi come li ha lasciati l'utente, esattamente
        # come i tab: se la pagina si ricarica e i tab tornano dov'erano ma gli
        # slider ripartono dai valori di partenza, la stessa schermata mostra
        # una valutazione diversa senza che nulla lo segnali.
        return html.Div([
            html.Label(label, style={"font-size": "10px", "color": "#555"}),
            dcc.Slider(id=id_, min=mn, max=mx, step=step, value=val,
                       tooltip={"placement": "bottom", "always_visible": True},
                       persistence=True, persistence_type="session",
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

        html.Button("🎯 Applica valori consigliati", id="btn-val-consigli",
                    n_clicks=0,
                    style={"width": "100%", "margin-top": "6px",
                           "background": "white", "color": "#1a3a5c",
                           "border": "1px solid #1a3a5c", "border-radius": "6px",
                           "padding": "6px", "font-size": "11px",
                           "cursor": "pointer"}),
        html.Div("Li applica già ▶ al caricamento. Il tab 🎯 Parametri mostra "
                 "da dove viene ogni numero.",
                 style={"font-size": "9px", "color": "#888", "margin-top": "4px",
                        "line-height": "1.45"}),

        html.Hr(style={"margin": "10px 0"}),

        _grp("📉 DCF — flussi di cassa"),
        _sl("val-wacc",    4.0, 20.0, 0.5,  9.0, "WACC — tasso di sconto (%)"),
        _sl("val-g1", 0.0, 60.0, 0.5, 12.0,
            f"Crescita anni 1-{ANNI_FASE1} (%)"),
        _sl("val-g2", 0.0, 45.0, 0.5, 6.0,
            f"Crescita anni {ANNI_FASE1 + 1}-{ANNI_DCF} (%)"),
        # Il massimo qui sotto è solo quello di partenza: `_val_limita_gterm` lo
        # riporta sempre appena sotto il WACC, perché oltre quella soglia
        # Gordon non converge e il DCF non sarebbe calcolabile.
        _sl("val-gterm",   0.0, 8.75, 0.25, 2.5,
            f"Tasso finale — da anno {ANNI_DCF + 1} in poi, per sempre (%)"),
        html.Div("Crescita perpetua: il massimo segue il WACC (resta un quarto "
                 "di punto sotto), perché una crescita perpetua pari o "
                 "superiore al tasso di sconto fa divergere Gordon. Alza il "
                 "WACC se ti serve più corsa. In pratica 2-3% (inflazione + "
                 "crescita reale): per una crescita alta ma temporanea usa le "
                 "due fasi qui sopra.",
                 style={"font-size": "9px", "color": "#888",
                        "line-height": "1.45", "margin": "-4px 0 8px"}),
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
            persistence=True, persistence_type="session",
            labelStyle={"display": "block", "font-size": "11px",
                        "color": "#333", "margin-bottom": "4px"}),
        html.Div("Vale per i grafici e le tabelle del tab Bilanci. "
                 "Per la variazione anno su anno scegli «Annuali».",
                 style={"font-size": "9px", "color": "#888",
                        "line-height": "1.45", "margin-top": "2px"}),

    ], style={"width": "270px", "min-width": "270px", "padding": "14px",
              "background": "#fafafa", "border-right": "1px solid #ddd",
              "overflow-y": "auto", "height": "calc(100vh - 250px)",
              "min-height": "520px"})

    results = html.Div([
        # `persistence` sulla sessione del browser, come lo store qui sotto: se
        # la pagina si ricarica il tab aperto resta quello, invece di riportare
        # l'utente sul Riepilogo perdendo il punto in cui era.
        dcc.Tabs(id="val-result-tabs", value="val-tab-summary",
                 persistence=True, persistence_type="session",
                 children=[
                     dcc.Tab(label="📊 Riepilogo",       value="val-tab-summary"),
                     dcc.Tab(label="🎯 Parametri",        value="val-tab-parametri"),
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

    # ── il tasso finale non può arrivare al WACC ─────────────────────────────
    @app.callback(
        Output("val-gterm", "max"),
        Output("val-gterm", "value"),
        Input("val-wacc",   "value"),
        State("val-gterm",  "value"),
    )
    def _val_limita_gterm(wacc, gterm):
        """Tiene il tasso finale sotto il costo del capitale.

        Gordon converge solo per una crescita perpetua **minore** del WACC:
        da lì in su la serie diverge e il valore terminale non esiste. Invece di
        lasciare allo slider una corsa che finisce su "non calcolabile", il suo
        massimo insegue il WACC: ogni posizione produce un numero e il legame
        fra i due parametri si vede muovendoli. Chi vuole un tasso finale alto
        alza prima il tasso di sconto, che è esattamente il vincolo economico.

        Ripara anche la sessione: il valore che il browser si ricorda viene
        riportato dentro il limite alla prima apertura, senza che la pagina
        nasca con il DCF a N/D.
        """
        w = 9.0 if wacc is None else float(wacc)
        tetto = max(0.25, round(w - 0.25, 2))
        v = 2.5 if gterm is None else float(gterm)
        # `no_update` quando il valore è già buono: questo callback scatta anche
        # subito dopo che i valori consigliati hanno scritto WACC e tasso
        # finale insieme, e riscriverlo lo riporterebbe a quello di prima.
        return tetto, (min(v, tetto) if v > tetto else no_update)

    # ── scarica i fondamentali e posiziona gli slider sui dati del titolo ─────
    @app.callback(
        Output("store-valuation",  "data"),
        Output("val-fetch-status", "children"),
        Output("val-fcf-nota",     "children"),
        Input("btn-run-valuation", "n_clicks"),
        State("val-ticker",        "value"),
        prevent_initial_call=True,
    )
    def run_valuation(n_clicks, ticker):
        import json, traceback
        import yfinance as yf

        if not ticker:
            return no_update, "⚠ Inserisci un ticker.", no_update

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
            fin = None                 # serve anche più sotto, per i ricavi d'esercizio
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
            fcf_bilancio, fcf_esercizio, rev_esercizio = 0, "", 0
            try:
                cf = t.cashflow
                if cf is not None and not cf.empty:
                    k_fcf = [k for k in cf.index if "free cash flow" in k.lower()]
                    if k_fcf:
                        serie = cf.loc[k_fcf[0]].dropna()
                        if len(serie) > 0:
                            fcf_bilancio  = float(serie.iloc[0])
                            col_fcf       = serie.index[0]
                            fcf_esercizio = str(col_fcf.date())
                            # Il margine va rapportato ai ricavi dello **stesso**
                            # esercizio: dividere il FCF di un anno chiuso per i
                            # ricavi TTM mescola due periodi diversi.
                            k_rev = ([k for k in fin.index
                                      if k.lower() == "total revenue"]
                                     if fin is not None else [])
                            if k_rev and col_fcf in fin.columns:
                                v_rev = fin.loc[k_rev[0], col_fcf]
                                if v_rev == v_rev and v_rev:
                                    rev_esercizio = float(v_rev)
            except Exception:
                pass

            # Reddito netto e flusso di cassa presi dallo **stesso** prospetto:
            # il rendiconto porta entrambi (parte dall'utile e ci arriva), così
            # le due serie hanno per forza gli stessi esercizi e il rapporto fra
            # loro è confrontabile riga per riga.
            flussi_storici = []
            try:
                cf = t.cashflow
                if cf is not None and not cf.empty:
                    def _riga_cf(*chiavi):
                        for k in chiavi:
                            for idx in cf.index:
                                if str(idx).strip().lower() == k.lower():
                                    return cf.loc[idx]
                        return None
                    r_ni = _riga_cf("Net Income From Continuing Operations",
                                    "Net Income", "Net Income Continuous Operations")
                    r_fc = _riga_cf("Free Cash Flow")
                    r_oc = _riga_cf("Operating Cash Flow")
                    r_cx = _riga_cf("Capital Expenditure")
                    for col in sorted(cf.columns):          # dal più vecchio
                        u_ = _num(r_ni[col]) if r_ni is not None else None
                        f_ = _num(r_fc[col]) if r_fc is not None else None
                        if f_ is None and r_oc is not None and r_cx is not None:
                            a_, b_ = _num(r_oc[col]), _num(r_cx[col])
                            f_ = (a_ - abs(b_)) if (a_ is not None
                                                    and b_ is not None) else None
                        if u_ is None or f_ is None:
                            continue
                        flussi_storici.append((str(col.date()), u_, f_))
            except Exception:
                flussi_storici = []

            crescita = stima_crescita_flussi(flussi_storici)

            # Il "dato reale" con cui si confronta lo slider dev'essere il
            # rendiconto: `info["freeCashflow"]` è un TTM che yfinance sbaglia
            # spesso e di molto (MSFT: 16.5 mld dichiarati contro i 67 del
            # rendiconto, 5% invece del 20%), e un riferimento falso è peggio di
            # nessun riferimento — ci si tara sopra lo slider.
            if fcf_bilancio and rev_esercizio > 0:
                fcf_margin_actual = fcf_bilancio / rev_esercizio
                fcf_margin_fonte  = f"rendiconto {fcf_esercizio}"
            elif fcf_bilancio and revenue > 0:
                fcf_margin_actual = fcf_bilancio / revenue
                fcf_margin_fonte  = f"rendiconto {fcf_esercizio} su ricavi TTM"
            elif revenue > 0 and fcf_yf:
                fcf_margin_actual = fcf_yf / revenue
                fcf_margin_fonte  = "yfinance TTM (dato spesso inaffidabile)"
            else:
                fcf_margin_actual, fcf_margin_fonte = 0, "non disponibile"

            d = {
                "ticker": ticker, "name": name, "sector": sector,
                "industry": industry, "currency": currency,
                "price": price, "market_cap": market_cap, "shares": shares,
                "eps_ttm": eps_ttm, "eps_fwd": eps_fwd,
                "revenue": revenue, "ebitda": ebitda, "fcf_yf": fcf_yf,
                "fcf_margin_actual": fcf_margin_actual,
                "fcf_bilancio": fcf_bilancio, "fcf_esercizio": fcf_esercizio,
                "rev_esercizio": rev_esercizio, "fcf_margin_fonte": fcf_margin_fonte,
                "crescita": crescita,
                "total_debt": total_debt, "cash": cash, "net_debt": net_debt,
                "dividend": dividend, "beta": beta,
                "pe_trailing": pe_trailing, "pe_forward": pe_forward,
                "book_val": book_val, "revenue_growth": revenue_growth,
                "gross_margins": gross_margins, "gross_profits": gross_profits,
                "ebitda_margins": ebitda_margins, "operating_margins": operating_margins,
                "ps_trailing": ps_trailing, "ev": ev, "rd_expense": rd_expense,
            }

            # Gli slider si posizionano sui dati del titolo — tutti, non due —
            # e da lì restano tuoi: nessun modello usa un valore diverso da
            # quello che vedi. Li scrive `_val_applica_consigli` leggendo questa
            # chiave, così caricamento e pulsante 🎯 fanno la stessa cosa.
            d["sugg"] = _consigli_titolo(t, info)

            nota = []
            if fcf_bilancio:
                nota.append(f"rendiconto {fcf_esercizio}: {fcf_bilancio/1e9:,.1f} mld "
                            f"({fcf_margin_actual*100:.1f}% dei ricavi)")
            if fcf_yf and revenue > 0:
                # Si mostra lo stesso, ma detto per quello che è: serve a capire
                # da dove viene un numero diverso se lo si è visto altrove.
                nota.append(f"yfinance TTM (inaffidabile): {fcf_yf/1e9:,.1f} mld "
                            f"({fcf_yf/revenue*100:.1f}%)")
            if not nota:
                nota.append("nessun FCF disponibile: imposta il margine a mano")

            status = f"✅ {name} ({ticker}) — {sector} | {currency} | prezzo: {price:.2f}"
            return json.dumps(d), status, " · ".join(nota)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"=== VALUATION ERROR ===\n{tb}")
            return no_update, f"❌ {e}", no_update

    # ── i valori consigliati entrano negli slider ────────────────────────────
    # Un solo posto che scrive gli slider, due modi di scatenarlo: il
    # caricamento di un titolo e il pulsante 🎯. Prima erano due slider scritti
    # dentro `run_valuation` e nove lasciati ai default di fabbrica, uguali per
    # una utility e per una società che raddoppia i ricavi ogni anno.
    @app.callback(
        [Output(s, "value", allow_duplicate=True) for s in _SLIDER_ORDINE],
        Input("store-valuation",  "data"),
        Input("btn-val-consigli", "n_clicks"),
        prevent_initial_call=True,
    )
    def _val_applica_consigli(stored, n_clicks):
        import json
        fermi = [no_update] * len(_SLIDER_ORDINE)
        if not stored:
            return fermi
        try:
            d = json.loads(stored) if isinstance(stored, str) else stored
            sugg = (d or {}).get("sugg") or {}
        except Exception:
            return fermi
        if not sugg:
            return fermi
        # Un consiglio mancante lascia lo slider dov'è: meglio il valore di
        # prima che un default che finge di essere un dato del titolo.
        return [(sugg.get(s) or {}).get("v", None) if (sugg.get(s) or {}).get("v") is not None
                else no_update for s in _SLIDER_ORDINE]

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

        # Senza questa rete di protezione un errore di calcolo lascia il
        # pannello con il contenuto del tab precedente e il click sembra non
        # aver fatto nulla: meglio dire cosa è andato storto.
        try:
            return _val_build_content(
                d, active_tab,
                _n(wacc, 9.0) / 100, _n(g1, 12.0) / 100,
                _n(g2, 6.0) / 100, _n(gterm, 2.5) / 100,
                _n(fcf_margin, 15.0) / 100, _n(pe_sector, 22.0),
                _n(ev_ebitda_mult, 12.0), _n(ke, 10.0) / 100,
                _n(ddm_g, 2.5) / 100, _n(graham_g, 8.0),
                _n(bond_yield, 4.5) / 100, bilanci_modo or "ttm")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return html.Div([
                html.Div("Questo tab non si è potuto costruire.",
                         style={"font-weight": "700", "margin-bottom": "6px"}),
                html.Div(f"{type(e).__name__}: {e}",
                         style={"font-family": "monospace", "font-size": "11px"}),
                html.Div("Gli altri tab restano disponibili; ricarica il titolo "
                         "con ▶ se il problema resta.",
                         style={"margin-top": "8px", "color": "#888"}),
            ], style={"padding": "30px", "color": "#8a1f11", "font-size": "13px",
                      "background": "#fdf2f0", "border": "1px solid #f0c8c0",
                      "border-radius": "6px", "margin": "20px"})


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

        # FCF margin: il rapporto già calcolato sul rendiconto (vedi
        # `run_valuation`), non il TTM di yfinance — su MSFT erano 5% contro 20%.
        fcf_margin_act = d.get("fcf_margin_actual")
        if not fcf_margin_act:
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
             "la sostenibilità della crescita. >15% = eccellente; >0% = autofinanziante. "
             f"Fonte: {d.get('fcf_margin_fonte', 'n/d')}."),

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
        dcf_price, fcf_rows, pv_tv, gordon_ok, dcf_ev = _val_dcf(
            revenue, fcf_m, wacc, g1, g2, gterm, shares, net_debt)
        # Sopra il WACC il DCF non è calcolabile: fuori dal confronto fra
        # modelli, altrimenti trascinerebbe giù la media come se fosse un
        # giudizio sul titolo.
        if not gordon_ok:
            dcf_price = None
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
                f"finale {gterm*100:.2f}% · − debito netto "
                f"{_val_fmt_num(net_debt)} ÷ {_val_fmt_num(shares, 0)} azioni"
                + ("" if gordon_ok else
                   f" — non calcolabile: il tasso finale è ≥ WACC "
                   f"{wacc*100:.1f}%, la perpetuità di Gordon non converge")),
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
                ("FCF (yfinance TTM, inaffidabile)",
                 _val_fmt_num(d.get("fcf_yf"), suffix=f" {currency}")),
                ("FCF (rendiconto)",   _val_fmt_num(fcf_bilancio, suffix=f" {currency}")
                                       if fcf_bilancio else "N/D"),
                ("Margine FCF reale",
                 f"{fcf_margin_actual*100:.1f}% ({d.get('fcf_margin_fonte', 'n/d')})"
                 if fcf_margin_actual else "N/D"),
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

            # ── Crescita del flusso di cassa, stimata dal reddito netto ───────
            # Sta nel Riepilogo perché è il numero da confrontare con lo slider
            # della fase 1: nel DCF il margine resta fermo, quindi far crescere
            # i ricavi di g1 vuol dire far crescere il flusso di g1. Qui si
            # vede a che ritmo è cresciuto davvero.
            cr = d.get("crescita") or None

            def _p(x, dec=1):
                return "—" if x is None else f"{x:+.{dec}f}%"

            def _blocco_crescita():
                if not cr:
                    return html.Div(
                        "Rendiconto finanziario non disponibile: la crescita "
                        "dei flussi non si può misurare, lo slider della fase 1 "
                        "resta un'ipotesi tua.",
                        style={"font-size": "11px", "color": "#888",
                               "padding": "10px 12px", "background": "#fafafa",
                               "border": "1px solid #e5e5e5",
                               "border-radius": "6px", "margin": "14px 0 0"})

                st = cr.get("stima")
                col = ("#2ca02c" if (st or 0) > 0 else
                       "#d62728" if st is not None else "#888")
                righe_tbl = [html.Tr([
                    html.Td(a["data"], style={**td_style, "color": "#555"}),
                    html.Td(_val_fmt_num(a["utile"], 1),
                            style={**td_style, "text-align": "right",
                                   "font-variant-numeric": "tabular-nums",
                                   **({"color": "#b0413e"} if a["utile"] < 0 else {})}),
                    html.Td(_p(a["var_utile"]),
                            style={**td_style, "text-align": "right",
                                   "font-variant-numeric": "tabular-nums",
                                   "color": ("#888" if a["var_utile"] is None else
                                             "#2e7d32" if a["var_utile"] >= 0
                                             else "#b0413e")}),
                    html.Td(_val_fmt_num(a["fcf"], 1),
                            style={**td_style, "text-align": "right",
                                   "font-variant-numeric": "tabular-nums",
                                   **({"color": "#b0413e"} if a["fcf"] < 0 else {})}),
                    html.Td(_p(a["var_fcf"]),
                            style={**td_style, "text-align": "right",
                                   "font-variant-numeric": "tabular-nums",
                                   "color": ("#888" if a["var_fcf"] is None else
                                             "#2e7d32" if a["var_fcf"] >= 0
                                             else "#b0413e")}),
                    html.Td("—" if a["conversione"] is None
                            else f"{a['conversione']:.0f}%",
                            style={**td_style, "text-align": "right",
                                   "font-variant-numeric": "tabular-nums",
                                   "color": "#555"}),
                ]) for a in cr["anni"]]

                # Le due medie rispondono a domande diverse e la differenza non
                # è un dettaglio: quella aritmetica è sempre la più alta, ed è
                # quella che fa sembrare più ricca un'azienda ballerina.
                sintesi = [
                    ("Media aritmetica delle variazioni",
                     _p(cr["media_utile"]), _p(cr["media_fcf"]),
                     "quanto è cambiato in un anno tipico"),
                    ("Mediana delle variazioni",
                     _p(cr["mediana_utile"]), _p(cr["mediana_fcf"]),
                     "l'anno di mezzo, insensibile all'anno anomalo"),
                    (f"Crescita composta ({cr['n_anni']} "
                     f"{'anno' if cr['n_anni'] == 1 else 'anni'})",
                     _p(cr["cagr_utile"]), _p(cr["cagr_fcf"]),
                     "dove è partita e dove è arrivata: è questa che si usa "
                     "per far crescere i flussi"),
                ]

                blocchi = [
                    html.Div([
                        html.Span("Stima della crescita annua del flusso di cassa: ",
                                  style={"font-size": "12px", "color": "#555"}),
                        html.Span(_p(st), style={"font-size": "20px",
                                                 "font-weight": "bold", "color": col}),
                    ], style={"margin": "0 0 4px"}),
                    html.Div(cr["come"], style={"font-size": "11px", "color": "#666",
                                                "line-height": "1.6",
                                                "margin": "0 0 10px"}),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th(c, style=th_style) for c in
                            ("Esercizio", "Reddito netto", "Var.",
                             "Free cash flow", "Var.", "FCF/utile")])),
                        html.Tbody(righe_tbl),
                    ], style=tbl_style),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th(c, style=th_style) for c in
                            ("", "Reddito netto", "Flusso di cassa", "")])),
                        html.Tbody([
                            html.Tr([
                                html.Td(et, style={**td_style, "color": "#555"}),
                                html.Td(u_, style={**td_style, "font-weight": "bold",
                                                   "text-align": "right",
                                                   "font-variant-numeric": "tabular-nums"}),
                                html.Td(f_, style={**td_style, "font-weight": "bold",
                                                   "text-align": "right",
                                                   "font-variant-numeric": "tabular-nums"}),
                                html.Td(nota, style={**td_style, "color": "#888",
                                                     "font-size": "10px",
                                                     "line-height": "1.5"}),
                            ]) for et, u_, f_, nota in sintesi
                        ]),
                    ], style={**tbl_style, "margin-top": "10px"}),
                ]

                # L'identità che tiene insieme i due CAGR: non è
                # un'approssimazione, è come sono definiti.
                if cr["cagr_conv"] is not None and not cr["conv_rotta"]:
                    blocchi.append(html.Div([
                        html.Div("Perché i due numeri non coincidono",
                                 style={"font-size": "11px", "font-weight": "bold",
                                        "color": "#1a3a5c", "margin-bottom": "4px"}),
                        html.Div([
                            html.Div(
                                "Il flusso di cassa è l'utile moltiplicato per "
                                "la quota di utile che diventa cassa, quindi le "
                                "due crescite si moltiplicano — non è "
                                "un'approssimazione, è come sono definite:",
                                style={"margin-bottom": "5px"}),
                            html.Div(
                                f"{1 + cr['cagr_utile']/100:.3f} × "
                                f"{1 + cr['cagr_conv']/100:.3f} = "
                                f"{1 + cr['cagr_fcf']/100:.3f}",
                                style={"font-family": "monospace",
                                       "font-size": "12px", "color": "#1a3a5c",
                                       "font-weight": "bold",
                                       "margin": "0 0 5px 6px"}),
                            html.Div(
                                f"utile {_p(cr['cagr_utile'])} l'anno, "
                                f"conversione {_p(cr['cagr_conv'])} l'anno → "
                                f"flusso di cassa {_p(cr['cagr_fcf'])} l'anno."
                                + (f" La quota che diventa cassa è "
                                   f"{'scesa' if cr['anni'][-1]['conversione'] < cr['anni'][0]['conversione'] else 'salita'}"
                                   f" da {cr['anni'][0]['conversione']:.0f}% a "
                                   f"{cr['anni'][-1]['conversione']:.0f}% "
                                   f"dell'utile."
                                   if (cr["anni"][0]["conversione"] is not None
                                       and cr["anni"][-1]["conversione"] is not None)
                                   else "")),
                        ], style={"font-size": "11px", "color": "#555",
                                  "line-height": "1.65"}),
                    ], style={"background": "#f7f9fc", "border": "1px solid #dbe4f0",
                              "border-radius": "6px", "padding": "9px 11px",
                              "margin-top": "10px"}))

                if cr["avviso"]:
                    blocchi.append(html.Div(
                        "⚠ " + cr["avviso"],
                        style={"font-size": "11px", "color": "#8a5a00",
                               "background": "#fff8e6", "border": "1px solid #f0dca8",
                               "border-radius": "6px", "padding": "9px 11px",
                               "line-height": "1.65", "margin-top": "10px"}))

                # Il collegamento allo slider: senza questo il numero resta
                # una curiosità invece che una decisione.
                if st is not None:
                    scarto = st - g1 * 100
                    blocchi.append(html.Div([
                        html.Span("Nel DCF il margine FCF resta fermo, quindi la "
                                  "crescita dei ricavi è anche quella dei flussi. "
                                  "Slider «Crescita fase 1»: ",
                                  style={"color": "#555"}),
                        html.Span(f"{g1*100:.1f}%",
                                  style={"font-weight": "bold", "color": "#1a3a5c"}),
                        html.Span(f" — {abs(scarto):.1f} punti "
                                  f"{'sotto' if scarto > 0 else 'sopra'} questa stima."
                                  if abs(scarto) >= 0.5 else " — in linea con questa stima.",
                                  style={"color": "#555"}),
                    ], style={"font-size": "11px", "line-height": "1.65",
                              "margin-top": "10px", "padding": "9px 11px",
                              "background": "#fafafa", "border": "1px solid #e5e5e5",
                              "border-radius": "6px"}))
                return html.Div(blocchi)

            blocco_crescita = _blocco_crescita()

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
                        html.H4("Crescita dei flussi, misurata dal reddito netto",
                                style={"font-size": "13px", "margin": "18px 0 8px",
                                       "color": "#1a3a5c"}),
                        blocco_crescita,
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

            # Unità scelta una volta sola sul valore più grande dei quattro
            # aggregati: i colossi in miliardi, tutti gli altri in milioni.
            # Con il /1e9 fisso una società da 300 mln di ricavi disegnava una
            # riga schiacciata sullo zero.
            _tutti = [abs(x) for k in ("fatturato", "lordo", "operativo", "utile")
                      for x in s[k] if x is not None]
            _div, _um = ((1e9, "mld") if max(_tutti, default=0) >= 1e10
                         else (1e6, "mln"))

            def _mld(v):
                return None if v is None else v / _div

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

            def _var_testo(y):
                """L'ultima variazione su un anno prima, per la riga di sintesi."""
                v = _var_anno(y)
                ult = next((x for x in reversed(v) if x is not None), None)
                return "—" if ult is None else f"{ult:+.1f}%"

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

            def _grafico_margini():
                """I tre margini sullo stesso asse: è il confronto fra loro a
                dire dove se ne va il fatturato. La distanza fra lordo e
                operativo è la struttura (ricerca, vendite, amministrazione),
                quella fra operativo e netto sono interessi e imposte."""
                fig = go.Figure()
                for chiave, nome, colore in (
                        ("margine_lordo", "Margine lordo",     "#9467bd"),
                        ("margine_op",    "Margine operativo", "#ff7f0e"),
                        ("margine",       "Margine netto",     "#2ca02c")):
                    fig.add_trace(go.Scatter(
                        x=s["date"], y=s[chiave], mode="lines+markers",
                        line=dict(color=colore, width=2), marker=dict(size=4),
                        name=nome, hovertemplate="%{y:.1f}%<extra></extra>"))
                fig.add_hline(y=0, line_color="#999", line_width=1)
                fig.update_layout(
                    title=dict(text="I tre margini a confronto", font=dict(size=11)),
                    yaxis=dict(title="% dei ricavi", ticksuffix="%"),
                    margin=dict(t=40, b=30, l=55, r=18),
                    paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)))
                return dcc.Graph(figure=fig, style={"height": "290px"},
                                 config={"displayModeBar": False})

            # Ordine della cascata: ogni riga è la precedente meno uno strato di
            # costo, e ognuna porta accanto la propria variazione su un anno
            # prima — è lì che si vede se un margine si allarga perché cresce il
            # numeratore o perché si è fermato il denominatore.
            grafici = [
                _grafico([_mld(v) for v in s["fatturato"]],
                         "Fatturato", "#1f77b4", f"{_um} {val}",
                         y2=_var_anno(s["fatturato"]),
                         y2_titolo="Variazione su un anno prima", y2_unita="var. %"),
                _grafico([_mld(v) for v in s["lordo"]],
                         "Utile lordo (ricavi − costo del venduto)", "#9467bd",
                         f"{_um} {val}", y2=_var_anno(s["lordo"]),
                         y2_titolo="Variazione su un anno prima", y2_unita="var. %"),
                _grafico([_mld(v) for v in s["operativo"]],
                         "Reddito operativo", "#ff7f0e", f"{_um} {val}",
                         y2=_var_anno(s["operativo"]),
                         y2_titolo="Variazione su un anno prima", y2_unita="var. %"),
                _grafico([_mld(v) for v in s["utile"]],
                         "Utile netto", "#2ca02c", f"{_um} {val}",
                         y2=_var_anno(s["utile"]),
                         y2_titolo="Variazione su un anno prima", y2_unita="var. %"),
                _grafico_margini(),
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
                (f"Fatturato ({_um} {val})", [_mld(v) for v in s["fatturato"]], 2),
                (f"Utile lordo ({_um} {val})", [_mld(v) for v in s["lordo"]], 2),
                (f"Reddito operativo ({_um} {val})",
                 [_mld(v) for v in s["operativo"]], 2),
                (f"Utile netto ({_um} {val})", [_mld(v) for v in s["utile"]], 2),
                ("Margine lordo (%)", s["margine_lordo"], 2),
                ("Margine operativo (%)", s["margine_op"], 2),
                ("Margine netto (%)", s["margine"], 2),
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

            # ── Reddito operativo periodo per periodo ─────────────────────────
            # Il grafico dice la forma, questa dice i numeri: quanto ha fatto la
            # gestione caratteristica, di quanto è cambiata rispetto a un anno
            # prima e quanta parte del fatturato è rimasta a valle dei costi.
            op_var  = _var_anno(s["operativo"])
            ric_var = _var_anno(s["fatturato"])
            n_op    = min(10, len(s["date"]))
            op_tbl = html.Table([
                html.Thead(html.Tr([
                    html.Th(c, style=th_style) for c in
                    ("Periodo", f"Reddito operativo ({_um} {val})",
                     "Var. su un anno prima", "Margine operativo",
                     "Var. fatturato")])),
                html.Tbody([
                    html.Tr([
                        html.Td(s["date"][i], style=td_style),
                        html.Td("—" if s["operativo"][i] is None
                                else f"{s['operativo'][i]/_div:,.2f}",
                                style={**td_style, "text-align": "right",
                                       "font-weight": "bold",
                                       "font-variant-numeric": "tabular-nums",
                                       **({"color": "#b0413e"}
                                          if (s["operativo"][i] or 0) < 0 else {})}),
                        html.Td("—" if op_var[i] is None else f"{op_var[i]:+.1f}%",
                                style={**td_style, "text-align": "right",
                                       "font-variant-numeric": "tabular-nums",
                                       "color": ("#888" if op_var[i] is None else
                                                 "#2e7d32" if op_var[i] >= 0
                                                 else "#b0413e")}),
                        html.Td("—" if s["margine_op"][i] is None
                                else f"{s['margine_op'][i]:.1f}%",
                                style={**td_style, "text-align": "right",
                                       "font-variant-numeric": "tabular-nums",
                                       "color": "#555"}),
                        html.Td("—" if ric_var[i] is None else f"{ric_var[i]:+.1f}%",
                                style={**td_style, "text-align": "right",
                                       "font-variant-numeric": "tabular-nums",
                                       "color": "#888"}),
                    ]) for i in range(len(s["date"]) - n_op, len(s["date"]))
                ]),
            ], style=tbl_style)

            # Il confronto fra le due ultime colonne è il punto: il reddito
            # operativo che corre più del fatturato vuol dire margini che si
            # allargano, il contrario vuol dire costi che scappano.
            op_nota = html.P(
                f"Ultimo periodo: reddito operativo {_var_testo(s['operativo'])} "
                f"su un anno prima, fatturato {_var_testo(s['fatturato'])}. "
                "Quando il reddito operativo cresce più del fatturato l'azienda "
                "sta guadagnando margine (i costi crescono meno dei ricavi); "
                "quando cresce meno lo sta perdendo, anche se i ricavi salgono.",
                style={"font-size": "11px", "color": "#666", "line-height": "1.6",
                       "margin": "8px 0 0"})

            # Cosa sono le quattro righe della cascata, in italiano e una volta
            # sola: senza questo i grafici sono quattro barre che si assomigliano.
            legenda_ce = html.Div([
                html.Div("Come si legge la cascata",
                         style={"font-size": "11px", "font-weight": "bold",
                                "color": "#1a3a5c", "margin-bottom": "6px"}),
                html.Table([html.Tbody([
                    html.Tr([
                        html.Td(t_, style={**td_style, "font-weight": "bold",
                                           "white-space": "nowrap",
                                           "color": c_, "width": "150px"}),
                        html.Td(x_, style={**td_style, "color": "#555",
                                           "line-height": "1.6"}),
                    ]) for t_, c_, x_ in (
                        ("Fatturato", "#1f77b4",
                         "quanto ha venduto, prima di qualsiasi costo."),
                        ("− costo del venduto", "#888",
                         "quello che è servito a produrre proprio ciò che è "
                         "stato venduto: materie prime, manodopera diretta, "
                         "server e banda per un servizio online. Non ci sono "
                         "dentro stipendi di struttura, ricerca o pubblicità."),
                        ("= Utile lordo", "#9467bd",
                         "quanto resta su ogni euro venduto per pagare tutto il "
                         "resto. È la misura di quanto è scalabile il "
                         "business: un software ha un margine lordo altissimo "
                         "perché la copia in più non costa quasi nulla, un "
                         "supermercato bassissimo perché la merce va ricomprata "
                         "ogni volta. Da solo non dice se l'azienda guadagna."),
                        ("− costi operativi", "#888",
                         "la struttura: ricerca e sviluppo, vendite e "
                         "marketing, amministrazione, ammortamenti."),
                        ("= Reddito operativo", "#ff7f0e",
                         "il risultato del mestiere dell'azienda, prima di "
                         "interessi e imposte. È il più comparabile fra due "
                         "società, perché non dipende da quanto debito hanno "
                         "né da dove pagano le tasse."),
                        ("− interessi e imposte", "#888",
                         "il costo del debito e il fisco."),
                        ("= Utile netto", "#2ca02c",
                         "quello che resta agli azionisti ed entra nell'EPS. "
                         "È il più esposto a poste straordinarie: una "
                         "svalutazione o una plusvalenza una tantum lo muovono "
                         "senza che il mestiere sia cambiato."),
                    )
                ])], style={**tbl_style, "margin": "0"}),
            ], style={"background": "#f7f9fc", "border": "1px solid #dbe4f0",
                      "border-radius": "6px", "padding": "10px 12px",
                      "margin": "10px 0"})

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
                html.Div(f"Valori in {_um} {val} · {len(s['date'])} {periodo} · "
                         f"la linea rossa a destra è la variazione rispetto allo "
                         f"stesso periodo dell'anno prima",
                         style={"font-size": "11px", "color": "#666",
                                "margin": "2px 0 0"}),
                legenda_ce,
                html.Div([html.Div(g, style={"flex": "1 1 46%", "min-width": "320px"})
                          for g in grafici],
                         style={"display": "flex", "flex-wrap": "wrap", "gap": "10px",
                                "margin": "10px 0"}),

                html.Hr(),
                html.H5(f"Reddito operativo — ultimi {n_op} {periodo}",
                        style={"font-size": "12px", "margin": "10px 0 8px",
                               "color": "#1a3a5c"}),
                op_tbl,
                op_nota,

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

        # ── TAB PARAMETRI ─────────────────────────────────────────────────────
        # Da dove viene ogni slider. Serve a poter dire di no al consiglio
        # sapendo cosa si sta rifiutando: un WACC è un'opinione finché non si
        # vede il beta, l'aliquota e i pesi che l'hanno prodotto.
        elif active_tab == "val-tab-parametri":
            sugg = d.get("sugg") or {}
            if not sugg:
                return html.Div(
                    "I valori consigliati si calcolano quando carichi il titolo: "
                    "premi ▶ Carica & Valuta.",
                    style={"padding": "40px", "color": "#888",
                           "text-align": "center", "font-size": "13px"})
            det = sugg.get("_det") or {}

            correnti = {
                "val-wacc": wacc * 100, "val-g1": g1 * 100, "val-g2": g2 * 100,
                "val-gterm": gterm * 100, "val-fcf-margin": fcf_m * 100,
                "val-ke": ke * 100, "val-ddm-g": ddm_g * 100,
                "val-graham-g": graham_g_pct, "val-bond-yield": bond_yield * 100,
                "val-pe-sector": pe_sector, "val-ev-ebitda": ev_mult,
            }

            def _fmt_par(val, unita, passo):
                if val is None:
                    return "N/D"
                dec = 2 if passo < 0.5 else (1 if passo < 1 else 0)
                return f"{val:.{dec}f}{unita}"

            righe_par = []
            gruppo_prec = None
            for slider, etichetta, unita, gruppo in _SLIDER_ETICHETTE:
                voce   = sugg.get(slider) or {}
                cons   = voce.get("v")
                attuale = correnti.get(slider)
                passo  = _SLIDER_LIMITI[slider][0]
                diverso = (cons is not None and attuale is not None
                           and abs(cons - attuale) > passo / 2)
                if gruppo != gruppo_prec:
                    righe_par.append(html.Tr([
                        html.Td(gruppo, colSpan=4,
                                style={**td_style, "background": "#eaf4fb",
                                       "font-weight": "bold", "color": "#1a5276",
                                       "font-size": "11px"}),
                    ]))
                    gruppo_prec = gruppo
                righe_par.append(html.Tr([
                    html.Td(etichetta, style={**td_style, "width": "22%"}),
                    html.Td(_fmt_par(cons, unita, passo),
                            style={**td_style, "font-weight": "bold",
                                   "text-align": "right", "width": "11%",
                                   "color": "#1a3a5c"}),
                    html.Td(_fmt_par(attuale, unita, passo),
                            style={**td_style, "text-align": "right", "width": "11%",
                                   "font-weight": "bold" if diverso else "normal",
                                   "color": "#e67e22" if diverso else "#555"}),
                    html.Td([
                        html.Div(voce.get("come", "") or "—",
                                 style={"line-height": "1.5"}),
                    ] + ([html.Div("⚠ " + voce["avviso"],
                                   style={"color": "#8a6d3b", "margin-top": "3px",
                                          "line-height": "1.5"})]
                         if voce.get("avviso") else []),
                        style={**td_style, "font-size": "10px", "color": "#666"}),
                ]))

            def _riga_det(k, v_):
                return html.Tr([
                    html.Td(k, style={**td_style, "color": "#555", "width": "45%"}),
                    html.Td(v_, style={**td_style, "font-weight": "bold"}),
                ])

            def _blocco_implicito():
                """Il DCF al contrario: cosa deve credere chi compra oggi.

                È il confronto che dice se un parametro è una scelta o una
                distrazione — il consiglio da solo non basta, perché il DCF
                risponde sempre qualcosa anche a un'ipotesi assurda.
                """
                w_imp, g_imp = _dcf_implicito(price, revenue, fcf_m, wacc, g1, g2,
                                              gterm, shares, net_debt)
                if w_imp is None and g_imp is None:
                    return html.Div()
                w_cons = (sugg.get("val-wacc") or {}).get("v")
                voci = []
                if w_imp is not None:
                    conf = ""
                    if w_cons:
                        diff = w_imp - w_cons
                        conf = (f" — {abs(diff):.1f} punti "
                                f"{'sopra' if diff > 0 else 'sotto'} il WACC "
                                f"consigliato ({w_cons:.1f}%)")
                    voci.append(html.Li([
                        html.B(f"Tasso di sconto implicito: {w_imp:.1f}%"), conf,
                        html.Div("Il rendimento annuo che il prezzo di oggi "
                                 "promette se i flussi vanno come dicono gli "
                                 "altri slider. Più alto del WACC che il titolo "
                                 "merita = il mercato chiede un premio, cioè "
                                 "prezza un rischio che le tue ipotesi non "
                                 "contengono.",
                                 style={"color": "#777", "margin": "2px 0 6px"}),
                    ]))
                if g_imp is not None:
                    voci.append(html.Li([
                        html.B(f"Crescita implicita dei flussi: {g_imp:+.1f}% "
                               f"l'anno per {ANNI_DCF} anni"),
                        html.Div("La crescita che giustifica il prezzo tenendo "
                                 "fermo il tuo tasso di sconto. Confrontala con "
                                 "la crescita storica dei ricavi qui sotto: se "
                                 "il mercato ne chiede il doppio, il titolo è "
                                 "caro anche quando il DCF dice di no.",
                                 style={"color": "#777", "margin": "2px 0 0"}),
                    ]))
                return html.Div([
                    html.H5(f"Cosa sconta il prezzo di mercato "
                            f"({price:.2f} {currency})",
                            style={"font-size": "12px", "margin": "0 0 6px",
                                   "color": "#1a3a5c"}),
                    html.Ul(voci, style={"margin": "0", "padding-left": "18px",
                                          "font-size": "11px", "line-height": "1.5"}),
                ], style={"background": "#f0f4fa", "border": "1px solid #d6e0ee",
                          "border-radius": "6px", "padding": "10px 12px",
                          "margin-bottom": "14px", "max-width": "900px"})

            det_rows = [
                _riga_det("Beta (yfinance)", f"{det.get('beta', 1.0):.2f}"),
                _riga_det("Premio per il rischio azionario", f"{_ERP:.1f}% (mercato maturo)"),
                _riga_det("Costo del capitale proprio (Ke)",
                          f"{det.get('ke', 0):.2f}%"),
                _riga_det("Costo del debito (Kd)",
                          f"{det.get('kd', 0):.2f}% — {det.get('kd_fonte', '')}"),
                _riga_det("Aliquota fiscale effettiva",
                          f"{det.get('aliquota', 0)*100:.1f}% — {det.get('aliquota_fonte', '')}"),
                _riga_det("Pesi a valori di mercato",
                          f"capitale proprio {det.get('we', 1)*100:.0f}% "
                          f"({_val_fmt_num(det.get('mc'))}) · debito "
                          f"{det.get('wd', 0)*100:.0f}% ({_val_fmt_num(det.get('debito'))})"),
                _riga_det("WACC risultante", f"{det.get('wacc', 0):.2f}%"),
            ]

            return html.Div([
                html.H4(f"Valori consigliati — {name}",
                        style={"font-size": "14px", "margin": "0 0 6px",
                               "color": "#1a3a5c", "border-bottom": "2px solid #1a3a5c",
                               "padding-bottom": "6px"}),
                html.P([
                    "Ogni slider parte da un numero preso dai conti di questo titolo, "
                    "non da un default uguale per tutti. Sono ", html.B("punti di "
                    "partenza documentati"), ", non prezzi obiettivi: la colonna "
                    "«come si ottiene» dice con quali dati è stato calcolato, così "
                    "un consiglio si può cambiare sapendo cosa si sta cambiando. "
                    "Il pulsante ", html.B("🎯 Applica valori consigliati"),
                    " nella barra laterale li riscrive tutti."],
                    style={"font-size": "11px", "color": "#666",
                           "line-height": "1.6", "max-width": "900px"}),
                html.Div(f"Dati di mercato: {det.get('tassi', 'n.d.')}",
                         style={"font-size": "10px", "color": "#888",
                                "margin-bottom": "10px"}),

                _blocco_implicito(),

                # Il DCF unlevered su una banca non vuol dire niente: il debito
                # è la sua materia prima, non una fonte di finanziamento, e
                # "debito netto" e "FCF" perdono significato. Meglio dirlo qui
                # che lasciar leggere come giudizio un numero senza senso.
            ] + ([html.Div([
                    html.B("Banche e assicurazioni: "),
                    "per questo settore il DCF sui flussi unlevered non è "
                    "applicabile — la raccolta è la materia prima dell'attività, "
                    "non un modo di finanziarla, e sia il debito netto sia il "
                    "flusso di cassa libero perdono significato. Restano validi "
                    "il DDM (il modello nato per le banche), il P/E e il "
                    "confronto con il patrimonio netto.",
                 ], style={"font-size": "11px", "color": "#8a6d3b",
                           "background": "#fdf7e6", "border": "1px solid #e6d9a8",
                           "border-radius": "6px", "padding": "10px 12px",
                           "margin-bottom": "14px", "max-width": "900px",
                           "line-height": "1.6"})]
                 if d.get("sector") == "Financial Services" else []) + [

                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Parametro", style=th_style),
                        html.Th("Consigliato", style={**th_style, "text-align": "right"}),
                        html.Th("Nello slider", style={**th_style, "text-align": "right"}),
                        html.Th("Come si ottiene", style=th_style),
                    ])),
                    html.Tbody(righe_par),
                ], style={**tbl_style, "margin-bottom": "16px"}),

                html.H5("Come nasce il costo del capitale",
                        style={"font-size": "12px", "margin": "0 0 8px",
                               "color": "#1a3a5c"}),
                html.Table([html.Tbody(det_rows)],
                           style={**tbl_style, "max-width": "620px"}),
                html.Div([
                    html.B("WACC = "),
                    "peso del capitale proprio × Ke + peso del debito × Kd × (1 − aliquota). ",
                    "Il Ke viene dal CAPM (risk-free + beta × premio per il rischio); il Kd "
                    "è quanto il titolo paga davvero sul suo debito, non un tasso teorico, "
                    "ed è deducibile, per questo entra al netto d'imposta.",
                ], style={"font-size": "10px", "color": "#888", "line-height": "1.6",
                          "margin-top": "8px", "max-width": "620px"}),

                html.Div([
                    html.B("Perché due tassi diversi. "),
                    "Il DCF sconta al WACC un flusso che spetta a tutti i finanziatori "
                    "e poi toglie il debito netto; il DDM sconta al Ke un dividendo che "
                    "è già quello che resta all'azionista, e quindi non toglie niente. "
                    "Usare lo stesso tasso per entrambi è l'errore che fa sembrare i due "
                    "modelli d'accordo quando non lo sono.",
                ], style={"font-size": "10px", "color": "#666", "line-height": "1.6",
                          "margin-top": "12px", "max-width": "900px",
                          "background": "#f8f9fa", "padding": "10px 12px",
                          "border-radius": "4px", "border-left": "3px solid #1a3a5c"}),
            ], style={"padding": "14px 16px 30px"})

        # ── TAB DCF ───────────────────────────────────────────────────────────
        elif active_tab == "val-tab-dcf":
            if not gordon_ok:
                return html.Div([
                    html.Div(f"Tasso finale {gterm*100:.2f}% ≥ WACC {wacc*100:.1f}%: "
                             "il DCF non è calcolabile.",
                             style={"font-weight": "700", "font-size": "14px",
                                    "margin-bottom": "10px"}),
                    html.Div([
                        "Il tasso finale è la crescita che il flusso di cassa "
                        "mantiene ", html.B("per sempre"), ", e la formula di "
                        "Gordon somma una serie infinita: converge solo se quella "
                        "crescita resta sotto il costo del capitale. Da lì in su "
                        "ogni flusso futuro cresce più in fretta di quanto lo si "
                        "sconti, la somma diverge e il valore terminale non "
                        "esiste — non è zero, è indefinito."],
                        style={"line-height": "1.6", "margin-bottom": "10px"}),
                    html.Div([
                        "Un'azienda non può crescere più dell'economia in eterno: "
                        "in pratica questo tasso sta fra ", html.B("2% e 3%"),
                        " (inflazione più crescita reale di lungo periodo). "
                        "Per una crescita alta ma temporanea usa gli slider "
                        f"delle fasi 1 e 2, che valgono {ANNI_DCF} anni e non "
                        "l'eternità."],
                        style={"line-height": "1.6", "margin-bottom": "10px"}),
                    html.Div(f"Abbassa il tasso finale sotto {wacc*100:.1f}% "
                             "per tornare a vedere la valutazione.",
                             style={"color": "#555"}),
                ], style={"padding": "26px 30px", "color": "#8a6d3b",
                          "background": "#fdf7e6", "border": "1px solid #e6d9a8",
                          "border-radius": "6px", "margin": "20px",
                          "font-size": "13px", "max-width": "760px"})
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

                # Dal valore d'impresa al valore per azione: il passaggio che
                # decide il risultato su ogni titolo indebitato, quindi si vede.
                html.Div([
                    html.H5("Dal valore d'impresa al prezzo per azione",
                            style={"font-size": "12px", "margin": "0 0 8px"}),
                    html.Table([
                        html.Tbody([
                            html.Tr([
                                html.Td("Valore d'impresa (somma dei valori attuali)",
                                        style={**td_style, "color": "#555"}),
                                html.Td(_val_fmt_num(dcf_ev, suffix=f" {currency}"),
                                        style={**td_style, "font-weight": "bold",
                                               "text-align": "right"}),
                            ]),
                            html.Tr([
                                html.Td(("− Debito netto" if net_debt >= 0
                                         else "+ Cassa netta (debito netto negativo)"),
                                        style={**td_style, "color": "#555"}),
                                html.Td(_val_fmt_num(-net_debt, suffix=f" {currency}"),
                                        style={**td_style, "font-weight": "bold",
                                               "text-align": "right",
                                               "color": "#d62728" if net_debt > 0
                                                        else "#2ca02c"}),
                            ]),
                            html.Tr([
                                html.Td("= Valore per gli azionisti",
                                        style={**td_style, "color": "#555"}),
                                html.Td(_val_fmt_num((dcf_ev or 0) - net_debt,
                                                     suffix=f" {currency}"),
                                        style={**td_style, "font-weight": "bold",
                                               "text-align": "right"}),
                            ]),
                            html.Tr([
                                html.Td("÷ Azioni in circolazione",
                                        style={**td_style, "color": "#555"}),
                                html.Td(_val_fmt_num(shares, 0),
                                        style={**td_style, "text-align": "right"}),
                            ]),
                            html.Tr([
                                html.Td("= Fair value per azione",
                                        style={**td_style, "color": "#555"}),
                                html.Td(f"{dcf_price:.2f} {currency}",
                                        style={**td_style, "font-weight": "bold",
                                               "text-align": "right", "color": vcol}),
                            ], style={"background": "#f3f7fc"}),
                        ])
                    ], style={**tbl_style, "max-width": "520px"}),
                    html.Div("Il flusso scontato spetta a tutti i finanziatori e il "
                             "WACC è il costo di tutto il capitale: la somma dei "
                             "valori attuali è il valore dell'impresa. Chi compra "
                             "l'azione eredita anche i debiti, quindi il debito netto "
                             "va tolto prima di dividere per le azioni — e il margine "
                             "FCF dello slider va inteso unlevered (prima degli "
                             "interessi), altrimenti il debito peserebbe due volte.",
                             style={"font-size": "10px", "color": "#888",
                                    "line-height": "1.5", "margin-top": "6px",
                                    "max-width": "520px"}),
                ] + ([html.Div(
                        f"Il debito netto ({_val_fmt_num(net_debt)} {currency}) supera "
                        f"il valore d'impresa stimato: con queste ipotesi il modello "
                        f"dice che agli azionisti non resta nulla. Non è un prezzo "
                        f"obiettivo, è un avviso sulla struttura finanziaria.",
                        style={"font-size": "11px", "color": "#8a6d3b",
                               "background": "#fdf7e6", "border": "1px solid #e6d9a8",
                               "border-radius": "4px", "padding": "8px 10px",
                               "margin-top": "8px", "max-width": "520px"})]
                     if ((dcf_ev or 0) - net_debt) <= 0 else []),
                   style={"margin": "12px 0"}),

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
                                    ("Debito netto",    _val_fmt_num(net_debt, suffix=f" {currency}")),
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
                    fv, _, _, ok, _ = _val_dcf(revenue, fcf_m, wc, g1, g2, gt,
                                               shares, net_debt)
                    # La cella sulla diagonale g ≥ WACC resta vuota: lì Gordon
                    # non vale e un numero qualunque si leggerebbe come crollo.
                    row.append(round(fv, 2) if (ok and fv) else None)
                z_vals.append(row)

            # Calcola % rispetto al prezzo corrente
            z_pct = [[None if v is None else
                      ((v - price) / price * 100 if price > 0 else 0)
                      for v in row] for row in z_vals]

            fig_heat = go.Figure(go.Heatmap(
                z=z_pct,
                x=[f"{g*100:.0f}%" for g in g_range],
                y=[f"{w*100:.0f}%" for w in wacc_range],
                colorscale="RdYlGn",
                zmid=0,
                text=[["n.d." if v is None else f"{v:+.0f}%" for v in row]
                      for row in z_pct],
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
                text=[["n.d." if v is None else f"{v:.1f}" for v in row]
                      for row in z_vals],
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
