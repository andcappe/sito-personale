"""
Pianificazione finanziaria familiare — modulo condiviso (sezione Clienti).

Risponde alla domanda "quando e a quali condizioni questo nucleo puo' smettere
di lavorare", con simulazione Monte Carlo e scenari avversi (premorienza,
invalidita', uscita graduale dal lavoro), e ne scrive la relazione.

Il nucleo puo' essere di una sola persona
-----------------------------------------
La composizione e' una scelta esplicita (`pl-composizione`), non si deduce dai
campi lasciati vuoti: un campo vuoto torna None e ripiegherebbe sul default,
cioe' un coniuge di 44 anni che nessuno ha inserito. Tolto il coniguge o il
figlio, spariscono i loro campi, spariscono gli scenari che li riguardano
(premorienza di lei, reversibilita', "dopo di noi") e la relazione lo dichiara.

Perche' la simulazione e' in euro NOMINALI e non in termini reali
---------------------------------------------------------------
Il modo rapido di fare questi conti e' lavorare in euro di oggi con un
rendimento reale (nominale meno inflazione) e spese costanti. Qui non si puo':
l'imposta italiana del 26% colpisce la plusvalenza **nominale**, cioe' anche la
parte di guadagno che e' solo inflazione. Su orizzonti di 70 anni la differenza
non e' un dettaglio — un modello in termini reali sottostima sistematicamente il
prelievo fiscale. Quindi si simula in nominale (patrimonio, costo di carico,
imposte, bollo) e si deflaziona solo alla fine, per mostrare tutto in euro di
oggi.

Regole del modello
------------------
* Un anno per passo, fino all'eta' `orizzonte` del piu' giovane degli adulti
  presenti (l'ipotesi del cliente: il patrimonio deve sopravvivere ai genitori,
  quindi 120 anni).
* Il risparmio NON e' un parametro libero: e' cio' che avanza, reddito meno
  spesa. Altrimenti i tre numeri (reddito, spesa, risparmio) potrebbero
  contraddirsi. Il risparmio implicito e' mostrato in pagina come verifica.
* Flussi positivi: comprano quote, quindi alzano anche il costo di carico.
* Flussi negativi: si vende il lordo necessario a ottenere il netto che serve.
  Con plusvalenza latente g = (valore - carico) / valore, per incassare `n` netti
  bisogna vendere n / (1 - g x 26%). Il carico scende in proporzione (costo medio:
  l'Italia userebbe il LIFO, che su un PAC ventennale darebbe un carico leggermente
  diverso — e' l'approssimazione standard di questi studi, dichiarata).
* Rendimenti lognormali calibrati perche' media e deviazione standard aritmetiche
  siano quelle impostate: cosi' il patrimonio non puo' diventare negativo e le code
  hanno la forma giusta.
* Bollo dossier 0,2% annuo sul valore di fine anno.
* Successione: per i titoli ereditati il costo fiscale si rivaluta al valore di
  successione (art. 68 TUIR), e coniuge e figlio hanno 1 milione di franchigia a
  testa. Quindi l'eredita' e' confrontata al LORDO: non c'e' imposta da scontare.
* Seme fisso: ricalcolando con gli stessi parametri esce lo stesso numero. Senza,
  la probabilita' ballerebbe di mezzo punto a ogni modifica e non si capirebbe
  piu' quale variazione dipende da cosa.

Profili di investimento e cono di Ibbotson
------------------------------------------
Rendimento e volatilita' del piano non sono numeri inventati: escono dai prezzi
veri di un portafoglio. I cinque profili standard (Cauto, Prudente, Moderato,
Attivo, Dinamico) sono combinazioni fisse di azionario globale e governativo
euro — sostituibili, ma nelle proporzioni del profilo; la
scelta autonoma accetta fino a cinque ticker con i loro pesi. Sulla serie del
portafoglio si misurano media, volatilita', VaR e VaR condizionale al 5%, e da
li' parte il cono di Ibbotson — i percentili del valore futuro in forma chiusa,
calibrati sulla STESSA lognormale del motore Monte Carlo, cosi' il cono e il
grafico del piano non possono raccontare due storie diverse. Il cono proietta il
solo patrimonio di oggi, senza versamenti ne' prelievi: quelli stanno nel
grafico del piano, e tenerli separati e' l'unico modo di vedere quanto rischio
c'e' nel portafoglio in quanto tale.

Uso da un'app Dash:
    import pianificazione
    ...  pianificazione.layout()              # contenuto del tab
    pianificazione.register_callbacks(app)    # callback (una volta sola)
"""
import json
import math
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State, ALL, ctx, no_update

BLU    = '#1a3a6b'
VERDE  = '#1e7d4f'
ROSSO  = '#b3261e'
GIALLO = '#b8860b'
SEME   = 42

NOMI_DEFAULT = {'lui': 'Lui', 'lei': 'Lei', 'figlio': 'Figlio'}


# ─────────────────────────────────────────────────────────────────────────────
# Parametri
# Un'unica lista: da qui nascono i campi della pagina, gli Input del callback e
# il dizionario che arriva al motore. Aggiungere un parametro = aggiungere una
# riga (l'ordine dei valori nel callback segue questa lista).
#   (id, etichetta, default, passo, suffisso, gruppo, nota, chi)
# `etichetta` e `nota` possono contenere {lui} {lei} {figlio}: li sostituisce il
# nome scelto dall'utente. `chi` dice a chi appartiene il campo ('' = al nucleo):
# se quella persona non c'e', il campo sparisce dalla pagina.
# I default sono un foglio bianco, non un caso di esempio: eta' 18 per gli
# adulti e 0 per il figlio, redditi e spesa a zero (quindi risparmio zero).
# Si riempiono con i dati del cliente; un default plausibile ma inventato
# rischia di restare li' e di finire in relazione come se fosse un dato vero.
# ─────────────────────────────────────────────────────────────────────────────
CAMPI = [
    ('eta_lui',      'Età {lui}',                  18,    1,   'anni', 'Famiglia', '', ''),
    ('eta_lei',      'Età {lei}',                  18,    1,   'anni', 'Famiglia', '', 'lei'),
    ('eta_figlio',   'Età {figlio}',                0,    1,   'anni', 'Famiglia', '', 'figlio'),
    ('orizzonte',    'Il piano deve reggere fino a', 120, 1, 'anni', 'Famiglia',
     'Ipotesi del cliente: il patrimonio deve sopravvivere a tutti gli adulti del nucleo.', ''),

    ('patrimonio',   'Patrimonio finanziario',   538000, 1000, '€', 'Patrimonio', '', ''),
    ('carico',       'Costo di carico',          352000, 1000, '€', 'Patrimonio',
     'Serve a calcolare l’imposta del 26% sui disinvestimenti futuri.', ''),

    ('reddito_lui',  'Reddito netto {lui}',           0,  500, '€/anno', 'Redditi e spese', '', ''),
    ('reddito_lei',  'Reddito netto {lei}',           0,  500, '€/anno', 'Redditi e spese', '', 'lei'),
    ('spesa',        'Spesa del nucleo',              0,  500, '€/anno', 'Redditi e spese', '', ''),
    ('spesa_ridotta','Spesa dopo l’uscita dal lavoro', 0, 500, '€/anno', 'Redditi e spese',
     'Si applica dall’anno in cui ha smesso di lavorare l’ultimo degli adulti. '
     'Metterla pari alla spesa del nucleo per non ipotizzare alcun taglio.', ''),

    ('stop_lui',     '{lui} smette di lavorare a',     67,   1, 'anni', 'Uscita dal lavoro', '', ''),
    ('stop_lei',     '{lei} smette di lavorare a',     67,   1, 'anni', 'Uscita dal lavoro', '', 'lei'),
    ('pt_eta_lui',   '{lui} passa a orario ridotto a', 67,   1, 'anni', 'Uscita dal lavoro', '', ''),
    ('pt_rid_lui',   'Riduzione orario {lui}',          0,   5, '%',    'Uscita dal lavoro',
     'Lasciando 0% la riduzione non ha effetto: è così che si modella l’uscita graduale.', ''),
    ('pt_eta_lei',   '{lei} passa a orario ridotto a', 67,   1, 'anni', 'Uscita dal lavoro', '', 'lei'),
    ('pt_rid_lei',   'Riduzione orario {lei}',          0,   5, '%',    'Uscita dal lavoro', '', 'lei'),

    ('pens_lui',     'Pensione netta {lui}',      22000,  500, '€/anno', 'Previdenza',
     'DA SOSTITUIRE con la stima dell’estratto conto INPS: è un segnaposto.', ''),
    ('pens_lei',     'Pensione netta {lei}',      18000,  500, '€/anno', 'Previdenza',
     'DA SOSTITUIRE con la stima dell’estratto conto INPS: è un segnaposto.', 'lei'),
    ('eta_pens_lui', 'Decorrenza pensione {lui}',    67,   1, 'anni', 'Previdenza', '', ''),
    ('eta_pens_lei', 'Decorrenza pensione {lei}',    67,   1, 'anni', 'Previdenza', '', 'lei'),
    ('reversibilita','Aliquota di reversibilità',    60,   5, '%',    'Previdenza',
     'Coniuge con figlio a carico: 60% (80% con un figlio minore, da verificare caso per caso).',
     'lei'),

    ('rendimento',   'Rendimento nominale atteso',  7.0, 0.25, '%/anno', 'Mercato',
     'Lo riempie il profilo di investimento, se la spunta «usa nel piano» è attiva. '
     'Al netto dei costi dello strumento.', ''),
    ('volatilita',   'Volatilità',                 16.0, 0.5,  '%/anno', 'Mercato',
     'Come sopra: la misura il profilo di investimento sui prezzi veri.', ''),
    ('rendimento_pens', 'Rendimento in decumulo',   4.0, 0.25, '%/anno', 'Mercato',
     'Dall’anno in cui ha smesso di lavorare l’ultimo degli adulti il capitale si '
     'sposta su un portafoglio più prudente: chi vive di prelievi non può correre il '
     'rischio dell’accumulo. Metterlo pari al rendimento atteso per non ipotizzare '
     'alcun cambio di portafoglio.', ''),
    ('volatilita_pens', 'Volatilità in decumulo',   7.0, 0.5,  '%/anno', 'Mercato',
     'Il cambio di fase scatta nello stesso anno in cui parte la spesa ridotta.', ''),
    ('inflazione',   'Inflazione',                  2.0, 0.25, '%/anno', 'Mercato', '', ''),

    ('tassa',        'Imposta su plusvalenze',     26.0, 0.5,  '%', 'Fisco', '', ''),
    ('bollo',        'Imposta di bollo dossier',   0.20, 0.05, '%/anno', 'Fisco', '', ''),

    ('eredita',      'Eredità da lasciare',      100000, 5000, '€ di oggi', 'Obiettivi', '', ''),
    ('soglia',       'Il piano è valido se riesce nel', 97, 1, '% dei casi', 'Obiettivi',
     'Il cliente ha indicato una tolleranza al fallimento del 3%.', ''),
    ('simulazioni',  'Numero di simulazioni',      3000, 500, '',  'Obiettivi', '', ''),

    ('anno_evento',  'L’evento avverso accade fra',    1,   1, 'anni', 'Scenari avversi',
     'Impostato a 1 è l’ipotesi peggiore: colpisce prima che il capitale sia cresciuto.', ''),
    ('fatt_superstite', 'Spesa dopo la perdita del coniuge', 75, 5, '% della spesa',
     'Scenari avversi', '', 'lei'),
    ('fatt_figlio',  'Spesa di {figlio} rimasto solo', 50,   5, '% della spesa',
     'Scenari avversi', '', 'figlio'),
    ('eta_indip_figlio', '{figlio} è autonomo a',      25,   1, 'anni', 'Scenari avversi', '', 'figlio'),
    ('pens_inv',     'Pensione di invalidità',     10000,  500, '€/anno', 'Scenari avversi', '', ''),
    ('costo_disab',  'Maggior costo per invalidità di un adulto', 12000, 1000, '€/anno',
     'Scenari avversi', '', ''),
    ('costo_disab_figlio', 'Maggior costo per invalidità di {figlio}', 15000, 1000, '€/anno',
     'Scenari avversi', '', 'figlio'),
    ('orizzonte_figlio', 'In quello scenario il capitale deve reggere fino ai suoi',
     90, 1, 'anni', 'Scenari avversi', '', 'figlio'),
    ('eta_decesso',  'In quello scenario gli adulti mancano a',  95, 1, 'anni', 'Scenari avversi',
     'Serve a far emergere il problema del “dopo di noi”: da lì il capitale sostiene '
     'il solo {figlio}.', 'figlio'),
]

# L'ordine in cui si compila: prima chi e' e con che cosa parte, poi come e'
# investito e che cosa gli trattiene il fisco — i tre blocchi che decidono il
# rendimento del patrimonio — e solo dopo redditi, lavoro, pensioni e obiettivi.
# Il profilo di rischio si incastra prima di «Mercato» perche' ne riempie i campi.
_ORDINE_GRUPPI = ['Famiglia', 'Patrimonio', 'Mercato', 'Fisco', 'Redditi e spese',
                  'Uscita dal lavoro', 'Previdenza', 'Obiettivi', 'Scenari avversi']


# ─────────────────────────────────────────────────────────────────────────────
# Profili di investimento e dati di mercato
#
# I cinque profili standard sono due soli strumenti in proporzioni diverse:
# azionario globale e governativo euro a breve. Non e' una semplificazione
# grafica ma la struttura del rischio — la quota azionaria e' cio' che
# distingue un profilo dall'altro. I due strumenti sono pero' solo i
# rappresentanti di quelle due classi: con la spunta «modifica» si sostituiscono
# con altri due ticker, e le percentuali dei cinque profili restano quelle.
# Con la scelta autonoma il portafoglio si costruisce invece da un massimo di
# cinque ticker, e valgono solo quelli effettivamente scritti.
#
# Rendimento, volatilita', VaR e VaR condizionale NON sono parametri scritti a
# mano: si misurano sui prezzi veri degli strumenti scelti, gia' riportati in
# euro da data_core. Da qui la cautela d'obbligo nel leggerli: sono il
# realizzato di una finestra, non una previsione.
# ─────────────────────────────────────────────────────────────────────────────
AZIONARIO = 'ISAC.L'         # iShares MSCI ACWI, azionario globale (quotato in USD)
OBBLIG    = 'CSBGE3.MI'      # iShares Euro Government Bond 3-5y, governativo euro

NOME_STRUMENTO = {
    AZIONARIO: 'azionario globale (MSCI ACWI)',
    OBBLIG:    'governativo euro 3-5 anni',
}

# In ordine di quota azionaria crescente: e' l'ordine in cui vanno letti, ed e'
# anche quello in cui compaiono nel pannello.
PROFILI = [
    ('cauto',    'Cauto',           [(AZIONARIO, 20), (OBBLIG, 80)]),
    ('prudente', 'Prudente',        [(AZIONARIO, 40), (OBBLIG, 60)]),
    ('moderato', 'Moderato',        [(AZIONARIO, 60), (OBBLIG, 40)]),
    ('attivo',   'Attivo',          [(AZIONARIO, 80), (OBBLIG, 20)]),
    ('dinamico', 'Dinamico',        [(AZIONARIO, 100)]),
    ('autonomo', 'Scelta autonoma', []),
]
PESI_PROFILO = {k: v for k, _, v in PROFILI}
NOME_PROFILO = {k: n for k, n, _ in PROFILI}
N_AUTONOMO   = 5                    # righe ticker/percentuale della scelta autonoma

GIORNI_ANNO = 252.0
_ALFA = 0.05                        # coda del VaR: il 5% dei casi peggiori
# Quantili della normale standard: servono al cono senza tirarsi dietro scipy.
_Z = {5: -1.6448536269514722, 25: -0.6744897501960817, 50: 0.0,
      75: 0.6744897501960817, 95: 1.6448536269514722}

_MERC_DIR     = Path(__file__).resolve().parent / 'sessions' / '_mercato'
_MERC_TTL     = 12 * 3600           # i prezzi si riscaricano due volte al giorno
_MERC_ERR_TTL = 900                 # un ticker sbagliato resta "non trovato" 15 minuti
_MERC_MEM     = {}
_MERC_ERR     = {}
_MERC_LOCK    = threading.Lock()


def _file_cache(ticker):
    return _MERC_DIR / (re.sub(r'[^A-Za-z0-9._-]', '_', ticker) + '.json')


def serie_eur(ticker):
    """Prezzi giornalieri di un ticker in EUR, in cache.  → (Series | None, nota).

    La cache non e' un'ottimizzazione, e' una necessita': il pannello si
    ridisegna a ogni modifica dei parametri del piano e senza di essa ogni tasto
    premuto sarebbe uno scaricamento da Yahoo. Sta sotto `sessions/`, quindi
    finisce anche su R2 e sopravvive al riavvio del server (il nome con
    l'underscore non viene scambiato per un utente dai job notturni, che cercano
    un `current.json`). Anche i fallimenti restano in memoria un quarto d'ora,
    altrimenti un ticker inesistente costerebbe un tentativo di rete per ogni
    carattere digitato."""
    tk = (ticker or '').strip().upper()
    if not tk:
        return None, ''
    ora = time.time()
    with _MERC_LOCK:
        voce = _MERC_MEM.get(tk)
        if voce and ora - voce[0] < _MERC_TTL:
            return voce[1], voce[2]
        err = _MERC_ERR.get(tk)
        if err and ora - err[1] < _MERC_ERR_TTL:
            return None, err[0]

    f = _file_cache(tk)
    if f.exists() and ora - f.stat().st_mtime < _MERC_TTL:
        try:
            d = json.loads(f.read_text())
            px = pd.Series(d['p'], index=pd.to_datetime(d['d']), name=tk, dtype=float)
            nota = f"prezzi al {px.index[-1]:%d/%m/%Y}"
            with _MERC_LOCK:
                _MERC_MEM[tk] = (f.stat().st_mtime, px, nota)
            return px, nota
        except Exception:
            pass

    try:
        import data_core
        px, _valuta, errore = data_core.download_series_eur(tk, 'EUR')
    except Exception as e:
        px, errore = None, f'scaricamento fallito: {e}'
    if px is None or len(px) < 120:
        msg = errore or f"«{tk}»: Yahoo non ha dati sufficienti (il ticker è corretto?)"
        with _MERC_LOCK:
            _MERC_ERR[tk] = (msg, time.time())
        return None, msg

    px = px.astype(float).dropna()
    try:
        _MERC_DIR.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix('.tmp')
        tmp.write_text(json.dumps({'d': [x.strftime('%Y-%m-%d') for x in px.index],
                                   'p': [round(float(v), 6) for v in px.values]}))
        tmp.replace(f)
        import data_core
        data_core.cloud_push(f)
    except Exception:
        pass
    nota = f"prezzi al {px.index[-1]:%d/%m/%Y}"
    with _MERC_LOCK:
        _MERC_MEM[tk] = (time.time(), px, nota)
    return px, nota


def composizione(profilo, tickers, pesi, tk_az=None, tk_ob=None):
    """Le righe (ticker, peso) del portafoglio scelto. Nei profili standard le
    percentuali sono fisse, ma i due strumenti che le riempiono si possono
    sostituire: `tk_az` prende il posto dell'azionario e `tk_ob` quello
    dell'obbligazionario; vuoti o assenti, restano quelli di serie. Con la
    scelta autonoma valgono SOLO le righe compilate.
    In tutti i casi un ticker ripetuto somma i suoi pesi invece di entrare due
    volte: due colonne uguali conterebbero doppio nelle misure."""
    if profilo != 'autonomo':
        sost = {AZIONARIO: (tk_az or '').strip().upper() or AZIONARIO,
                OBBLIG:    (tk_ob or '').strip().upper() or OBBLIG}
        coppie = [(sost.get(t, t), w) for t, w in PESI_PROFILO.get(profilo, [])]
    else:
        coppie = list(zip(tickers or [], pesi or []))
    somma = {}
    ordine = []
    for t, w in coppie:
        t = (t or '').strip().upper()
        if not t:
            continue
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if w <= 0:
            continue
        if t not in somma:
            ordine.append(t)
            somma[t] = 0.0
        somma[t] += w
    return [(t, somma[t]) for t in ordine]


def _lognormale(mu, sd):
    """Parametri (mu_log, sd_log) della lognormale annua con media e deviazione
    standard ARITMETICHE pari a mu e sd. E' la stessa calibrazione di
    `_rendimenti`: cono e Monte Carlo del piano non possono contraddirsi."""
    var_log = math.log1p((sd ** 2) / ((1 + mu) ** 2))
    return math.log1p(mu) - var_log / 2, math.sqrt(var_log)


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def var_modello(mu, sd, t=1.0):
    """VaR e VaR condizionale al 5% della lognormale, in forma chiusa.
    Il condizionale e' la media dei casi che stanno sotto il VaR: per una
    lognormale E[X | X ≤ q] = e^(m+s²/2) · Φ(z − s) / 0,05."""
    m, s = _lognormale(mu, sd)
    rad = math.sqrt(t)
    q = math.exp(m * t + _Z[5] * s * rad)
    c = math.exp(m * t + s * s * t / 2) * _phi(_Z[5] - s * rad) / _ALFA
    return q - 1.0, c - 1.0


def statistiche(righe, anni=10):
    """Misura il portafoglio sui prezzi veri: rendimento medio, volatilita',
    VaR e VaR condizionale a un anno.  → dict serializzabile, pronto per lo Store."""
    out = {'ok': False, 'avvisi': [], 'anni': int(anni), 'dettaglio': []}
    if not righe:
        out['avvisi'].append('Nessuno strumento indicato: scrivi almeno un ticker '
                             'con la sua percentuale.')
        return out

    serie, buone, avvisi = {}, [], []
    for tk, w in righe:
        px, nota = serie_eur(tk)
        if px is None:
            avvisi.append(nota or f'«{tk}» non disponibile')
            continue
        serie[tk] = px
        buone.append((tk, w))
    out['avvisi'] = avvisi
    if not buone:
        return out

    prezzi = pd.concat([serie[t] for t, _ in buone], axis=1,
                       keys=[t for t, _ in buone]).dropna()
    if len(prezzi) < 120:
        out['avvisi'] = avvisi + ['Gli strumenti scelti hanno meno di sei mesi di '
                                  'storia in comune: non basta per misurare il rischio.']
        return out
    taglio = prezzi.index[-1] - pd.DateOffset(years=int(anni))
    finestra = prezzi[prezzi.index >= taglio]
    if len(finestra) >= 120:
        prezzi = finestra
    else:
        out['avvisi'] = avvisi + [f'Storia più corta dei {int(anni)} anni richiesti: '
                                  'le misure usano tutto lo storico disponibile.']

    # Stesso limite di data_core: un tick corrotto non deve diventare volatilita'.
    r = prezzi.pct_change().dropna().clip(-0.5, 0.5)
    w = np.array([x for _, x in buone], dtype=float)
    totale = float(w.sum())
    if totale <= 0:
        return out
    out['normalizzato'] = abs(totale - 100.0) > 0.51
    out['totale_scritto'] = totale
    w = w / totale

    # Portafoglio a pesi costanti: la media pesata dei rendimenti giornalieri.
    rp = r.values @ w
    mu = float(np.mean(rp) * GIORNI_ANNO)
    sd = float(np.std(rp, ddof=1) * math.sqrt(GIORNI_ANNO))
    crescita = float(np.prod(1.0 + rp))
    cagr = float(crescita ** (GIORNI_ANNO / len(rp)) - 1.0)

    # VaR e VaR condizionale a un anno dalla distribuzione EMPIRICA delle
    # finestre mobili di 252 giorni. Sono sovrapposte, quindi meno indipendenti
    # di quanto il loro numero suggerisca, ma e' l'unico modo di vedere le code
    # vere — e non quelle di una normale — su dieci anni di dati.
    cum = np.concatenate(([1.0], np.cumprod(1.0 + rp)))
    if len(cum) > 252 + 60:
        annui = cum[252:] / cum[:-252] - 1.0
        var = float(np.percentile(annui, _ALFA * 100))
        coda = annui[annui <= var]
        out['var'] = var
        out['cvar'] = float(coda.mean()) if len(coda) else var
        out['finestre'] = int(len(annui))
        out['peggio'] = float(annui.min())
    else:
        out['var'] = out['cvar'] = None
        out['finestre'] = 0
        out['peggio'] = None

    vm, cm = var_modello(mu, sd)
    for i, (tk, _) in enumerate(buone):
        ri = r.values[:, i]
        out['dettaglio'].append({
            'ticker': tk, 'nome': NOME_STRUMENTO.get(tk, ''),
            'peso': float(w[i] * 100),
            'mu': float(np.mean(ri) * GIORNI_ANNO),
            'sd': float(np.std(ri, ddof=1) * math.sqrt(GIORNI_ANNO)),
        })
    out.update({
        'ok': True, 'mu': mu, 'sd': sd, 'cagr': cagr,
        'var_modello': vm, 'cvar_modello': cm,
        'sd_pesata': float(sum(d['peso'] / 100 * d['sd'] for d in out['dettaglio'])),
        'da': f"{prezzi.index[0]:%m/%Y}", 'a': f"{prezzi.index[-1]:%d/%m/%Y}",
        'giorni': int(len(r)),
    })
    return out


def cono(v0, mu, sd, anni):
    """Cono di Ibbotson: i percentili del valore del portafoglio anno per anno,
    partendo dal valore di oggi. Nessuna simulazione — a orizzonte t il valore e'
    lognormale in forma chiusa e i percentili sono esatti.
    La riga `cvar` e' il VaR condizionale: non il 5° percentile, ma la MEDIA di
    tutti i casi che stanno sotto di esso, cioe' quanto resta quando le cose
    vanno male sul serio.  → (anni, {percentile: valori})"""
    m, s = _lognormale(mu, sd)
    t = np.arange(int(max(1, anni)) + 1, dtype=float)
    rad = np.sqrt(t)
    q = {p: v0 * np.exp(m * t + z * s * rad) for p, z in _Z.items()}
    atteso = v0 * np.exp(m * t + (s ** 2) * t / 2)
    q['cvar'] = atteso * np.array([_phi(_Z[5] - s * x) for x in rad]) / _ALFA
    return t, q


def nomi_da(n_lui, n_lei, n_figlio):
    """Nomi ripuliti, con i default quando il campo e' vuoto."""
    grezzi = {'lui': n_lui, 'lei': n_lei, 'figlio': n_figlio}
    return {k: (str(v).strip() if v and str(v).strip() else NOMI_DEFAULT[k])
            for k, v in grezzi.items()}


def testo(modello, nomi):
    """Sostituisce {lui} {lei} {figlio} in un'etichetta."""
    return modello.format(**nomi) if modello else modello


def scenari_applicabili(comp, nomi):
    """Gli scenari che hanno senso per questa composizione.
    Senza coniuge non esistono premorienza di lei ne' reversibilita'; senza figlio
    non esiste il "dopo di noi". Quando l'adulto e' uno solo, la sua premorienza
    E' gia' il caso in cui il nucleo non c'e' piu'."""
    ha_lei, ha_figlio = 'lei' in comp, 'figlio' in comp
    s = [('base', 'Base — si lavora fino alle età indicate')]
    if ha_lei:
        s += [('morte_lui', f'Premorienza di {nomi["lui"]}'),
              ('morte_lei', f'Premorienza di {nomi["lei"]}'),
              ('morte_entrambi', 'Premorienza di entrambi')]
    else:
        s += [('morte_lui', f'Premorienza di {nomi["lui"]}')]
    s += [('invalidita_lui', f'Invalidità di {nomi["lui"]}')]
    if ha_lei:
        s += [('invalidita_lei', f'Invalidità di {nomi["lei"]}'),
              ('invalidita_entrambi', 'Invalidità di entrambi')]
    if ha_figlio:
        s += [('invalidita_figlio', f'Invalidità di {nomi["figlio"]}')]
    return s


def etichetta_scenario(scenario, comp, nomi):
    for v, lbl in scenari_applicabili(comp, nomi):
        if v == scenario:
            return lbl
    return scenario


# ─────────────────────────────────────────────────────────────────────────────
# Flussi annui (in euro di oggi): reddito e spesa anno per anno
# ─────────────────────────────────────────────────────────────────────────────
def _reddito_persona(eta, p, chi, morto_da=None, invalido_da=None):
    """Reddito annuo di una persona per ogni anno, in euro di oggi.
    `eta` e' il vettore delle sue eta'; `morto_da`/`invalido_da` sono indici di anno
    (None = non accade). Chi muore smette di percepire: la reversibilita' la somma
    il chiamante al coniuge superstite."""
    reddito  = np.where(eta < p[f'stop_{chi}'], p[f'reddito_{chi}'], 0.0)
    # Orario ridotto: vale solo finche' si lavora.
    ridotto  = (eta >= p[f'pt_eta_{chi}']) & (eta < p[f'stop_{chi}'])
    reddito  = np.where(ridotto, reddito * (1 - p[f'pt_rid_{chi}'] / 100), reddito)
    pensione = np.where(eta >= p[f'eta_pens_{chi}'], p[f'pens_{chi}'], 0.0)

    anni = np.arange(len(eta))
    if invalido_da is not None:
        # Il lavoro finisce, la pensione di vecchiaia matura lo stesso e nel
        # frattempo interviene quella di invalidita'.
        colpito  = anni >= invalido_da
        reddito  = np.where(colpito, 0.0, reddito)
        pensione = np.where(colpito, pensione + p['pens_inv'], pensione)
    if morto_da is not None:
        morto    = anni >= morto_da
        reddito  = np.where(morto, 0.0, reddito)
        pensione = np.where(morto, 0.0, pensione)
    return reddito + pensione


def _reversibilita(eta, p, chi, morto_da):
    """Quota di reversibilita' che il superstite incassa. Decorre dalla morte, ma
    non prima dell'eta' in cui il defunto sarebbe andato in pensione."""
    anni = np.arange(len(eta))
    attiva = (anni >= morto_da) & (eta >= p[f'eta_pens_{chi}'])
    return np.where(attiva, p[f'pens_{chi}'] * p['reversibilita'] / 100, 0.0)


def profilo(p, scenario, comp):
    """Costruisce i flussi annui reali dello scenario per questa composizione.
    → dict: anni, reddito[], spesa[], note[]"""
    ha_lei, ha_figlio = 'lei' in comp, 'figlio' in comp
    ev = int(max(0, p['anno_evento']))
    note = []

    morto_lui = ev if scenario in ('morte_lui', 'morte_entrambi') else None
    morto_lei = ev if (ha_lei and scenario in ('morte_lei', 'morte_entrambi')) else None
    inv_lui   = ev if scenario in ('invalidita_lui', 'invalidita_entrambi') else None
    inv_lei   = ev if (ha_lei and scenario in ('invalidita_lei', 'invalidita_entrambi')) else None
    # Con un adulto solo, la sua premorienza e' gia' la fine del nucleo.
    tutti_morti = (morto_lui is not None) and (morto_lei is not None or not ha_lei)

    # ── Orizzonte ───────────────────────────────────────────────────────────
    T = int(p['orizzonte'] - p['eta_lui'])
    if ha_lei:
        T = int(max(T, p['orizzonte'] - p['eta_lei']))
    if scenario == 'invalidita_figlio':
        T = int(max(T, p['orizzonte_figlio'] - p['eta_figlio']))
    if tutti_morti:
        # Da li' il capitale serve al figlio fino alla sua autonomia; se figli non
        # ce ne sono, si misura subito quanto resta agli eredi.
        T = int(max(ev, p['eta_indip_figlio'] - p['eta_figlio'])) if ha_figlio else max(ev, 1)
    T = max(T, 1)

    anni    = np.arange(T)
    eta_lui = p['eta_lui'] + anni
    eta_lei = p['eta_lei'] + anni

    # ── Redditi ─────────────────────────────────────────────────────────────
    reddito = _reddito_persona(eta_lui, p, 'lui', morto_lui, inv_lui)
    if ha_lei:
        reddito = reddito + _reddito_persona(eta_lei, p, 'lei', morto_lei, inv_lei)
        if morto_lui is not None and morto_lei is None:
            reddito = reddito + _reversibilita(eta_lui, p, 'lui', ev)
            note.append(f"{p['_nomi']['lei']} incassa il {p['reversibilita']:.0f}% della "
                        f"pensione di {p['_nomi']['lui']} dalla decorrenza "
                        f"({p['eta_pens_lui']:.0f} anni).")
        if morto_lei is not None and morto_lui is None:
            reddito = reddito + _reversibilita(eta_lei, p, 'lei', ev)
            note.append(f"{p['_nomi']['lui']} incassa il {p['reversibilita']:.0f}% della "
                        f"pensione di {p['_nomi']['lei']} dalla decorrenza "
                        f"({p['eta_pens_lei']:.0f} anni).")

    # Orario ridotto dichiarato ma finestra vuota: la riduzione non puo' agire e
    # senza avviso sembrerebbe applicata. Capita col default, che fa partire il
    # part time alla stessa eta' in cui si smette di lavorare.
    for k in (['lui', 'lei'] if ha_lei else ['lui']):
        if p[f'pt_rid_{k}'] > 0 and p[f'pt_eta_{k}'] >= p[f'stop_{k}']:
            note.append(f"La riduzione d’orario di {p['_nomi'][k]} "
                        f"({p[f'pt_rid_{k}']:.0f}%) non ha effetto: l’orario ridotto "
                        f"partirebbe a {p[f'pt_eta_{k}']:.0f} anni, quando ha già "
                        f"smesso di lavorare ({p[f'stop_{k}']:.0f} anni).")

    # ── Spesa ───────────────────────────────────────────────────────────────
    # Si riduce quando ha smesso di lavorare l'ultimo degli adulti.
    uscite = [p['stop_lui'] - p['eta_lui']]
    if ha_lei:
        uscite.append(p['stop_lei'] - p['eta_lei'])
    spesa = np.where(anni >= max(uscite), p['spesa_ridotta'], p['spesa']).astype(float)

    if tutti_morti:
        reddito = np.where(anni >= ev, 0.0, reddito)
        if ha_figlio:
            spesa = np.where(anni >= ev, p['spesa'] * p['fatt_figlio'] / 100, spesa)
            note.append(f"Dopo l’evento resta la sola spesa di {p['_nomi']['figlio']} "
                        f"({p['fatt_figlio']:.0f}% della spesa del nucleo) fino ai "
                        f"{p['eta_indip_figlio']:.0f} anni; lì si misura quanto resta.")
        else:
            spesa = np.where(anni >= ev, 0.0, spesa)
            note.append('Nessun superstite a carico: si misura quanto resta agli eredi '
                        'al momento dell’evento.')
    elif scenario in ('morte_lui', 'morte_lei'):
        spesa = np.where(anni >= ev, spesa * p['fatt_superstite'] / 100, spesa)
        note.append(f"La spesa scende al {p['fatt_superstite']:.0f}% dall’anno dell’evento.")

    if scenario in ('invalidita_lui', 'invalidita_lei'):
        spesa = np.where(anni >= ev, spesa + p['costo_disab'], spesa)
    if scenario == 'invalidita_entrambi':
        spesa = np.where(anni >= ev, spesa + 2 * p['costo_disab'], spesa)
    if scenario == 'invalidita_figlio':
        spesa = spesa + p['costo_disab_figlio']
        # "Dopo di noi": quando gli adulti mancano cadono redditi e pensioni e
        # resta il solo figlio, che il capitale deve mantenere da solo.
        orfano = eta_lui >= p['eta_decesso']
        if ha_lei:
            orfano = orfano & (eta_lei >= p['eta_decesso'])
        reddito = np.where(orfano, 0.0, reddito)
        spesa   = np.where(orfano, p['spesa'] * p['fatt_figlio'] / 100
                           + p['costo_disab_figlio'], spesa)
        note.append(f"Dai {p['eta_decesso']:.0f} anni degli adulti il capitale sostiene "
                    f"{p['_nomi']['figlio']} da solo, fino ai suoi "
                    f"{p['orizzonte_figlio']:.0f}.")

    # L'anno in cui il portafoglio passa alla fase di decumulo e' lo stesso in cui
    # parte la spesa ridotta: una data sola, cosi' le due cose non possono
    # disallinearsi, e se l'utente sposta l'uscita dal lavoro si spostano insieme.
    anno_pens = int(np.clip(int(max(uscite)), 0, T))

    return {'anni': T, 'reddito': reddito, 'spesa': spesa, 'note': note,
            'anno_pens': anno_pens}


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
def _rendimenti(p, n, T, seme, anno_pens=None):
    """Matrice (T, n) di rendimenti nominali lognormali, calibrati perche' media e
    deviazione standard ARITMETICHE siano quelle impostate. Lognormale e non
    normale: un rendimento normale puo' scendere sotto -100% e il patrimonio
    diventerebbe negativo.

    Due fasi, non una: fino ad `anno_pens` valgono rendimento e volatilita'
    dell'accumulo, da li' in poi quelli del decumulo. Chi vive di prelievi sposta
    il capitale su un portafoglio piu' prudente, e un tasso solo per mezzo secolo
    non descrive nessuna delle due fasi. La taratura e' la stessa di prima, fatta
    anno per anno invece che una volta sola."""
    mu = np.full(T, p['rendimento'] / 100, dtype=float)
    sd = np.full(T, p['volatilita'] / 100, dtype=float)
    if anno_pens is not None:
        k = int(np.clip(anno_pens, 0, T))
        mu[k:] = p['rendimento_pens'] / 100
        sd[k:] = p['volatilita_pens'] / 100
    var_log = np.log1p((sd ** 2) / ((1 + mu) ** 2))
    sd_log  = np.sqrt(var_log)
    mu_log  = np.log1p(mu) - var_log / 2
    rng = np.random.default_rng(seme)
    return np.exp(rng.normal(mu_log[:, None], sd_log[:, None], size=(T, n))) - 1.0


def simula(p, prof, n=None, seme=SEME):
    """Esegue la simulazione. → dict con traccia (euro di oggi), successo, ecc."""
    T = prof['anni']
    n = int(n or p['simulazioni'])
    infl  = p['inflazione'] / 100
    tau   = p['tassa'] / 100
    bollo = p['bollo'] / 100
    R = _rendimenti(p, n, T, seme, prof.get('anno_pens'))

    W = np.full(n, float(p['patrimonio']))
    B = np.full(n, float(p['carico']))
    fallito = np.zeros(n, dtype=bool)
    traccia = np.zeros((T + 1, n))
    traccia[0] = W
    anno_ko = np.full(n, -1)

    for t in range(T):
        fatt   = (1 + infl) ** (t + 1)
        flusso = float(prof['reddito'][t] - prof['spesa'][t]) * fatt

        if flusso >= 0:
            # Si compra: sale il valore e sale anche il costo di carico.
            W = W + flusso
            B = B + flusso
        else:
            netto_serve = -flusso
            attivo = W > 0
            g = np.zeros(n)
            np.divide(W - B, W, out=g, where=attivo)
            g = np.clip(g, 0.0, 1.0)          # in perdita non si paga imposta
            lordo = netto_serve / (1 - g * tau)
            ko = lordo > W + 1e-6
            anno_ko = np.where(ko & ~fallito, t, anno_ko)
            fallito |= ko
            venduto = np.minimum(lordo, np.maximum(W, 0.0))
            quota_b = np.zeros(n)
            np.divide(venduto * B, W, out=quota_b, where=attivo)
            W = np.maximum(W - venduto, 0.0)
            B = np.maximum(B - quota_b, 0.0)

        W = W * (1 + R[t])
        W = W * (1 - bollo)
        W = np.where(fallito, 0.0, np.maximum(W, 0.0))
        traccia[t + 1] = W

    deflat = (1 + infl) ** np.arange(T + 1)
    traccia_reale = traccia / deflat[:, None]
    finale_reale  = traccia_reale[-1]
    successo = (~fallito) & (finale_reale >= p['eredita'])
    return {
        'T': T,
        'traccia': traccia_reale,
        'successo': successo,
        'prob': float(successo.mean() * 100),
        'fallito': fallito,
        'anno_ko': anno_ko,
        'finale': finale_reale,
    }


def scansione_uscita(p, comp, n=800, passo=1):
    """Per ogni possibile data di uscita dal lavoro calcola la probabilita' di
    successo. Con il coniuge: smettono entrambi / solo lui / solo lei. Da soli,
    una curva sola."""
    ha_lei = 'lei' in comp
    massimo = int(p['eta_pens_lui'] - p['eta_lui'])
    if ha_lei:
        massimo = int(max(massimo, p['eta_pens_lei'] - p['eta_lei']))
    massimo = max(massimo, 1)
    offsets = list(range(0, massimo + 1, max(1, int(passo))))
    quali = ['entrambi', 'lui', 'lei'] if ha_lei else ['solo']
    curve = {k: [] for k in quali}
    for off in offsets:
        for chi in quali:
            q = dict(p)
            if chi in ('entrambi', 'lui', 'solo'):
                q['stop_lui'] = p['eta_lui'] + off
            if chi in ('entrambi', 'lei'):
                q['stop_lei'] = p['eta_lei'] + off
            curve[chi].append(simula(q, profilo(q, 'base', comp), n=n)['prob'])
    return offsets, {k: np.array(v) for k, v in curve.items()}


def prima_uscita(offsets, probabilita, soglia):
    """Primo anno in cui la probabilita' raggiunge la soglia (None se mai)."""
    ok = np.where(probabilita >= soglia)[0]
    return offsets[ok[0]] if len(ok) else None


def limite(p, scenario, comp, soglia, applica, lo, hi, meglio_alto, n=700, giri=11):
    """Bisezione: fin dove puo' spostarsi un parametro prima che il piano scenda
    sotto la soglia. `applica(q, x)` scrive il valore x nei parametri.
    meglio_alto=True  → la probabilita' cresce con x (rendimento, patrimonio):
                        si cerca il valore MINIMO ancora accettabile.
    meglio_alto=False → la probabilita' cala con x (spesa): si cerca il MASSIMO.
    Ritorna None se il piano non regge la soglia nemmeno nel caso migliore."""
    def prob(x):
        q = dict(p)
        applica(q, x)
        return simula(q, profilo(q, scenario, comp), n=n)['prob']

    if meglio_alto:
        if prob(hi) < soglia:
            return None
        if prob(lo) >= soglia:
            return lo
        for _ in range(giri):
            m = (lo + hi) / 2
            if prob(m) >= soglia:
                hi = m
            else:
                lo = m
        return hi
    if prob(lo) < soglia:
        return None
    if prob(hi) >= soglia:
        return hi
    for _ in range(giri):
        m = (lo + hi) / 2
        if prob(m) >= soglia:
            lo = m
        else:
            hi = m
    return lo


def margini(p, scenario, comp, soglia):
    """I tre margini che contano: quanto puo' deludere il mercato, quanto puo'
    crescere la spesa, quanto patrimonio serve come minimo."""
    def _rend(q, x):
        # Se il mercato delude, delude in tutte e due le fasi: il rendimento del
        # decumulo scende degli stessi punti, altrimenti il margine misurerebbe
        # solo la meta' del piano.
        q['rendimento'] = x
        q['rendimento_pens'] = max(0.0, p['rendimento_pens'] - (p['rendimento'] - x))

    def _spesa(q, x):
        # La spesa ridotta segue in proporzione: il tenore di vita e' uno solo.
        k = x / p['spesa'] if p['spesa'] else 1
        q['spesa'], q['spesa_ridotta'] = x, p['spesa_ridotta'] * k

    def _patr(q, x):
        # Il costo di carico segue in proporzione: la plusvalenza latente resta quella.
        k = x / p['patrimonio'] if p['patrimonio'] else 1
        q['patrimonio'], q['carico'] = x, p['carico'] * k

    return {
        'rendimento': limite(p, scenario, comp, soglia, _rend, 0.0, p['rendimento'], True),
        'spesa':      limite(p, scenario, comp, soglia, _spesa, p['spesa'], p['spesa'] * 4, False),
        'patrimonio': limite(p, scenario, comp, soglia, _patr, 0.0, p['patrimonio'], True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formattazione
# ─────────────────────────────────────────────────────────────────────────────
def _eur(v):
    if v is None or not np.isfinite(v):
        return '—'
    return f"{v:,.0f}".replace(',', '.') + ' €'


def _card(titolo, valore, nota='', colore=BLU):
    return html.Div([
        html.Div(titolo, style={'fontSize': '10px', 'color': '#666', 'fontWeight': '700',
                                'textTransform': 'uppercase', 'letterSpacing': '0.04em'}),
        html.Div(valore, style={'fontSize': '24px', 'fontWeight': '700', 'color': colore,
                                'lineHeight': '1.2', 'margin': '4px 0 2px'}),
        html.Div(nota, style={'fontSize': '11px', 'color': '#666'}),
    ], style={'flex': '1 1 170px', 'minWidth': '160px', 'padding': '12px 14px',
              'background': '#f8fafd', 'border': '1px solid #e8edf5', 'borderRadius': '10px'})


def _titolo(t, sottotitolo='', margine='18px'):
    return html.Div([
        html.Div(t, style={'fontSize': '13px', 'fontWeight': '700', 'color': BLU}),
        html.Div(sottotitolo, style={'fontSize': '11px', 'color': '#666'}) if sottotitolo else None,
    ], style={'marginTop': margine, 'marginBottom': '5px'})


def descrizione_nucleo(p, comp, nomi):
    """La frase che apre la relazione. E' il punto in cui il programma dichiara
    di chi sta parlando: se il nucleo e' di una persona sola, lo dice."""
    ha_lei, ha_figlio = 'lei' in comp, 'figlio' in comp
    fig = (f"{nomi['figlio']}, {p['eta_figlio']:.0f} anni" if ha_figlio else None)
    if ha_lei:
        base = (f"Nucleo di due adulti: {nomi['lui']}, {p['eta_lui']:.0f} anni, e "
                f"{nomi['lei']}, {p['eta_lei']:.0f} anni")
    else:
        base = (f"Nucleo formato dal solo {nomi['lui']}, {p['eta_lui']:.0f} anni")
    return base + (f", con un figlio a carico ({fig})." if ha_figlio
                   else ", senza figli a carico.")


# ─────────────────────────────────────────────────────────────────────────────
# Grafici
# ─────────────────────────────────────────────────────────────────────────────
def _fig_ventaglio(p, ris, nomi):
    """Ventaglio dei percentili del patrimonio, in euro di oggi."""
    tr = ris['traccia']
    x = np.arange(tr.shape[0]) + p['eta_lui']
    pc = {q: np.percentile(tr, q, axis=1) for q in (5, 25, 50, 75, 95)}
    fig = go.Figure()
    for lo, hi, col, nome in ((5, 95, 'rgba(26,58,107,0.10)', '5°–95° percentile'),
                              (25, 75, 'rgba(26,58,107,0.22)', '25°–75° percentile')):
        fig.add_trace(go.Scatter(x=x, y=pc[hi], line={'width': 0}, showlegend=False,
                                 hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x, y=pc[lo], fill='tonexty', fillcolor=col,
                                 line={'width': 0}, name=nome, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=pc[50], line={'color': BLU, 'width': 2.5},
                             name='Mediana',
                             hovertemplate='Età %{x}<br>%{y:,.0f} €<extra></extra>'))
    fig.add_hline(y=p['eredita'], line={'color': VERDE, 'width': 1.5, 'dash': 'dot'},
                  annotation_text=f"Eredità obiettivo {_eur(p['eredita'])}",
                  annotation_font={'size': 10, 'color': VERDE})
    fig.update_layout(
        margin={'l': 60, 'r': 20, 't': 30, 'b': 40}, height=360,
        plot_bgcolor='white', paper_bgcolor='white',
        font={'family': 'Inter, sans-serif', 'size': 11},
        legend={'orientation': 'h', 'y': 1.12, 'x': 0, 'font': {'size': 10}},
        xaxis={'title': f"Età di {nomi['lui']}", 'gridcolor': '#eef2f7'},
        yaxis={'title': 'Patrimonio (€ di oggi)', 'gridcolor': '#eef2f7', 'tickformat': ',.0f'},
        hovermode='x unified',
    )
    return fig


def _fig_uscita(p, offsets, curve, soglia, nomi):
    """Probabilita' di successo in funzione di quando si smette di lavorare."""
    fig = go.Figure()
    stile = {'entrambi': (BLU, 'Smettono entrambi'),
             'solo':     (BLU, f"{nomi['lui']} smette di lavorare"),
             'lui':      ('#5b8cc7', f"Smette solo {nomi['lui']}"),
             'lei':      ('#c77e5b', f"Smette solo {nomi['lei']}")}
    for chi, valori in curve.items():
        col, nome = stile[chi]
        fig.add_trace(go.Scatter(
            x=[p['eta_lui'] + o for o in offsets], y=valori, name=nome,
            line={'color': col, 'width': 2.2},
            hovertemplate=nome + '<br>Età %{x}: %{y:.1f}%<extra></extra>'))
    fig.add_hline(y=soglia, line={'color': ROSSO, 'width': 1.5, 'dash': 'dash'},
                  annotation_text=f'soglia {soglia:.0f}%',
                  annotation_font={'size': 10, 'color': ROSSO})
    fig.update_layout(
        margin={'l': 55, 'r': 20, 't': 30, 'b': 40}, height=300,
        plot_bgcolor='white', paper_bgcolor='white',
        font={'family': 'Inter, sans-serif', 'size': 11},
        legend={'orientation': 'h', 'y': 1.15, 'x': 0, 'font': {'size': 10}},
        xaxis={'title': f"Età di {nomi['lui']} quando si smette di lavorare",
               'gridcolor': '#eef2f7'},
        yaxis={'title': 'Piani riusciti', 'ticksuffix': '%', 'range': [0, 101],
               'gridcolor': '#eef2f7'},
        hovermode='x unified',
    )
    return fig


def _tabella_scenari(righe, soglia):
    intestazioni = ['Scenario', 'Piani riusciti', 'Patrimonio mediano finale',
                    '5° percentile finale', 'Esito']
    th = {'padding': '7px 10px', 'fontSize': '10px', 'textTransform': 'uppercase',
          'letterSpacing': '0.04em', 'color': '#666', 'textAlign': 'left',
          'borderBottom': '2px solid #e8edf5', 'whiteSpace': 'nowrap'}
    corpo = []
    for nome, r in righe:
        ok = r['prob'] >= soglia
        td = {'padding': '7px 10px', 'fontSize': '12px', 'borderBottom': '1px solid #f0f3f8'}
        corpo.append(html.Tr([
            html.Td(nome, style={**td, 'fontWeight': '600'}),
            html.Td(f"{r['prob']:.1f}%", style={**td, 'fontWeight': '700',
                                                'color': VERDE if ok else ROSSO}),
            html.Td(_eur(np.percentile(r['finale'], 50)), style=td),
            html.Td(_eur(np.percentile(r['finale'], 5)), style=td),
            html.Td('tiene' if ok else 'non tiene',
                    style={**td, 'color': VERDE if ok else ROSSO, 'fontWeight': '600'}),
        ]))
    return html.Div(
        html.Table([html.Thead(html.Tr([html.Th(h, style=th) for h in intestazioni])),
                    html.Tbody(corpo)],
                   style={'width': '100%', 'minWidth': '560px', 'borderCollapse': 'collapse',
                          'fontFamily': 'Inter, sans-serif'}),
        style={'overflowX': 'auto'})


# ─────────────────────────────────────────────────────────────────────────────
# Relazione
# ─────────────────────────────────────────────────────────────────────────────
def _par(*pezzi):
    """Paragrafo: le stringhe sono testo, le tuple (testo, True) vanno in neretto."""
    figli = []
    for pz in pezzi:
        if isinstance(pz, tuple):
            figli.append(html.Span(pz[0], style={'fontWeight': '700', 'color': '#222'}))
        else:
            figli.append(html.Span(pz))
    return html.P(figli, style={'fontSize': '12.5px', 'lineHeight': '1.7', 'color': '#444',
                                'margin': '0 0 9px'})


def _righe_portafoglio(mercato):
    """Il paragrafo della relazione che descrive il portafoglio: che cosa c'e'
    dentro, quanto ha reso e quanto ha perso quando ha perso. Se le misure non
    sono disponibili la relazione tace, invece di inventare numeri."""
    m = mercato or {}
    if not m.get('ok'):
        return []
    nome = NOME_PROFILO.get(m.get('profilo'), 'non definito')
    strumenti = ', '.join(f"{d['peso']:.0f}% {d['ticker']}" for d in m['dettaglio'])
    frase = (f"Il patrimonio è investito secondo il profilo «{nome}» ({strumenti}). "
             f"Sui prezzi da {m['da']} al {m['a']} quel portafoglio ha reso in media "
             + f"{m['mu']:+.1%}".replace('.', ',') + ' l’anno, con una volatilità del '
             + f"{m['sd']:.1%}".replace('.', ',') + '.')
    if m.get('var') is not None:
        frase += (' Nella finestra misurata il 5% degli anni peggiori ha perso almeno il '
                  + f"{abs(m['var']):.1%}".replace('.', ',')
                  + ', e in quei casi la perdita media è stata del '
                  + f"{abs(m['cvar']):.1%}".replace('.', ',')
                  + ': è il rischio che il piano deve poter attraversare senza '
                    'costringere a vendere nel momento sbagliato.')
    if not m.get('applicato'):
        frase += (' Il piano però non usa queste misure: rendimento e volatilità '
                  'sono quelli impostati a mano nel gruppo Mercato.')
    righe = [_par(frase)]
    # Il portafoglio del decumulo e' un secondo portafoglio, non un'ipotesi sul
    # primo: se e' stato scelto un profilo se ne dicono i numeri misurati, se il
    # tasso e' scritto a mano il paragrafo tace e parla solo _frase_fasi.
    d = m.get('decumulo')
    if d:
        righe.append(_par(
            f"Dall’inizio del decumulo il portafoglio passa al profilo «{d['nome']}», "
            'misurato sugli stessi prezzi: rendimento medio '
            + f"{d['mu']:+.1%}".replace('.', ',') + " l’anno e volatilità "
            + f"{d['sd']:.1%}".replace('.', ',')
            + '. È il portafoglio con cui si convive quando il capitale non cresce '
              'più di versamenti ma viene venduto per vivere.'))
    return righe


def _frase_fasi(p, comp, nomi):
    """Le due fasi del piano dette a parole: quando finisce l'accumulo, che cosa
    cambia nel portafoglio e che cosa cambia nella spesa. Il lettore deve poter
    ritrovare nel testo i numeri che ha scritto nei campi, non fidarsi."""
    quali = ['lui', 'lei'] if 'lei' in comp else ['lui']
    off = max(int(p[f'stop_{k}'] - p[f'eta_{k}']) for k in quali)
    ultimo = max(quali, key=lambda k: p[f'stop_{k}'] - p[f'eta_{k}'])
    if off <= 0:
        quando = 'già da oggi'
    else:
        quando = (f"fra {off} anni, quando {nomi[ultimo]} arriva a "
                  f"{p[f'stop_{ultimo}']:.0f} anni")
    num = lambda v, d=2: f"{v:.{d}f}".replace('.', ',')

    testo = (f"Il piano ha due fasi. Nell’accumulo il capitale è investito al "
             f"{num(p['rendimento'])}% con volatilità {num(p['volatilita'], 1)}% e "
             f"cresce anche del risparmio; il decumulo comincia {quando}")
    if (p['rendimento_pens'] != p['rendimento']
            or p['volatilita_pens'] != p['volatilita']):
        testo += (f", e da lì il portafoglio passa al {num(p['rendimento_pens'])}% "
                  f"con volatilità {num(p['volatilita_pens'], 1)}%: chi vive di "
                  'prelievi non può permettersi il rischio dell’accumulo. ')
    else:
        testo += (', con lo stesso portafoglio: non è ipotizzato alcuno spostamento '
                  'su strumenti più prudenti. ')
    if p['spesa_ridotta'] != p['spesa']:
        testo += ('Da quell’anno la spesa scende a ' + _eur(p['spesa_ridotta'])
                  + ' l’anno e il capitale viene venduto per coprire la differenza '
                    'fra spesa e pensioni, pagando l’imposta sulla plusvalenza a '
                    'ogni prelievo.')
    else:
        testo += ('Da quell’anno la spesa resta invariata e il capitale viene venduto '
                  'per coprire la differenza fra spesa e pensioni, pagando l’imposta '
                  'sulla plusvalenza a ogni prelievo.')
    return testo


def relazione(p, comp, nomi, scenario, ris, righe, marg, soglia, offsets, curve,
              mercato=None):
    """La relazione in prosa: chi è il nucleo, da dove parte, che cosa dicono i
    numeri, che cosa può andare storto, che cosa fare. È scritta dai risultati
    appena calcolati, quindi si aggiorna da sé a ogni modifica dei parametri."""
    ha_lei, ha_figlio = 'lei' in comp, 'figlio' in comp
    adulti = [nomi['lui']] + ([nomi['lei']] if ha_lei else [])
    redditi = p['reddito_lui'] + (p['reddito_lei'] if ha_lei else 0)
    risparmio = (redditi - p['spesa']) / 12
    plus = p['patrimonio'] - p['carico']
    plus_netta = plus * (1 - p['tassa'] / 100)

    chiave = 'entrambi' if 'entrambi' in curve else 'solo'
    primo = prima_uscita(offsets, curve[chiave], soglia)
    deboli = [n for n, r in righe if r['prob'] < soglia]
    ok = ris['prob'] >= soglia
    med = np.percentile(ris['finale'], 50)
    p05 = np.percentile(ris['finale'], 5)

    sezioni = []

    # ── 1. Chi è il nucleo ──────────────────────────────────────────────────
    lavoro = ' e '.join(f"{nomi[k]} fino a {p[f'stop_{k}']:.0f} anni"
                        for k in (['lui', 'lei'] if ha_lei else ['lui']))
    sezioni.append(('Il nucleo familiare', [
        _par(descrizione_nucleo(p, comp, nomi)),
        _par('Nelle ipotesi di base lavora ', lavoro,
             ', con la pensione che decorre a ',
             ' e '.join(f"{p[f'eta_pens_{k}']:.0f} anni per {nomi[k]}"
                        for k in (['lui', 'lei'] if ha_lei else ['lui'])),
             f". L’analisi copre {ris['T']:.0f} anni: il capitale deve reggere fino ai "
             f"{p['orizzonte']:.0f} anni "
             + ('del più giovane dei due.' if ha_lei else f"di {nomi['lui']}.")),
    ] + ([] if ha_lei else [
        _par(('Attenzione: ', True), 'il calcolo è impostato su una sola persona. '
             'Non esistono quindi né reversibilità né scenari di premorienza del '
             'coniuge, e la spesa non si riduce per la perdita di un componente.')
    ])))

    # ── 2. Da dove si parte ─────────────────────────────────────────────────
    sezioni.append(('La situazione di partenza', [
        _par('Il patrimonio finanziario è di ', (_eur(p['patrimonio']), True),
             ', a fronte di un costo di carico di ', (_eur(p['carico']), True),
             f": la plusvalenza latente è di {_eur(plus)}, che al netto dell’imposta "
             f"del {p['tassa']:.0f}% vale {_eur(plus_netta)}. È il motivo per cui i "
             'disinvestimenti futuri costano più di quanto si incassa: nel modello '
             'ogni prelievo è calcolato al lordo dell’imposta.'),
        _par(f"I redditi netti ammontano a {_eur(redditi)} l’anno e la spesa del nucleo a "
             f"{_eur(p['spesa'])}: ne risulta un risparmio di ", (_eur(risparmio) + ' al mese', True),
             '. Dopo l’uscita dal lavoro la spesa è ipotizzata a '
             f"{_eur(p['spesa_ridotta'])} l’anno."
             if p['spesa_ridotta'] != p['spesa'] else
             '. La spesa è ipotizzata invariata anche dopo l’uscita dal lavoro.'),
        _par(_frase_fasi(p, comp, nomi)),
    ] + _righe_portafoglio(mercato)))

    # ── 3. Il risultato ─────────────────────────────────────────────────────
    if primo is None:
        frase_uscita = ('Con questi parametri non esiste un’età di uscita dal lavoro che '
                        f"raggiunga il {soglia:.0f}%: o si lavora più a lungo delle "
                        'ipotesi, o si riduce la spesa, o servono più risparmi.')
    elif primo == 0:
        frase_uscita = ('La soglia è già raggiunta oggi: '
                        + ('entrambi potrebbero smettere di lavorare da subito.'
                           if ha_lei else f"{nomi['lui']} potrebbe smettere da subito."))
    else:
        eta_fin = [f"{nomi[k]} ne avrà {p[f'eta_{k}'] + primo:.0f}"
                   for k in (['lui', 'lei'] if ha_lei else ['lui'])]
        frase_uscita = (f"La soglia viene raggiunta fra {primo} anni: " + ' e '.join(eta_fin)
                        + '. Prima di allora la probabilità resta sotto il livello '
                          'che il cliente ha indicato come accettabile.')
    sezioni.append(('Il risultato', [
        _par(f"Nello scenario «{etichetta_scenario(scenario, comp, nomi)}» il piano riesce nel ",
             (f"{ris['prob']:.1f}% delle simulazioni", True),
             f", contro una soglia richiesta del {soglia:.0f}%: ",
             ('è quindi sostenibile.' if ok else 'non è quindi sostenibile così com’è.')),
        _par(f"A fine orizzonte il patrimonio mediano residuo è di {_eur(med)} in euro di "
             f"oggi, e nel 5% dei casi peggiori si ferma a {_eur(p05)}. L’obiettivo di "
             f"lasciare {_eur(p['eredita'])} a valori odierni è incluso nella definizione "
             'di successo: un piano che si esaurisce, o che arriva in fondo sotto quella '
             'cifra, conta come fallito.'),
        _par(frase_uscita),
    ]))

    # ── 4. I rischi ─────────────────────────────────────────────────────────
    if deboli:
        rischi = [_par('Gli scenari che non reggono la soglia sono ', (', '.join(deboli), True),
                       '. Sono la misura del capitale umano ancora scoperto: '
                       'finché il patrimonio non basta da solo, quegli eventi vanno '
                       'trasferiti a un assicuratore invece che tenuti in proprio.')]
    else:
        rischi = [_par('Tutti gli scenari simulati restano sopra la soglia richiesta: '
                       'anche negli eventi avversi il piano non si interrompe.')]
    rischi.append(_par(
        'Il rischio che questi numeri non catturano è la ', ('sequenza dei rendimenti', True),
        ': il modello estrae ogni anno in modo indipendente, mentre i mercati reali '
        'incatenano le annate negative. Una crisi nei primi anni dopo l’uscita dal lavoro '
        'pesa molto più di una crisi tardiva della stessa entità.'))
    if not ha_figlio:
        rischi.append(_par('Non essendoci figli a carico, l’obiettivo di eredità vincola '
                           'solo il patrimonio finale e non genera un fabbisogno di '
                           'mantenimento nel frattempo.'))
    sezioni.append(('I rischi principali', rischi))

    # ── 5. Che fare ─────────────────────────────────────────────────────────
    azioni = []
    if primo is not None and primo > 0:
        azioni.append(f"Fissare la data di uscita dal lavoro non prima di {primo} anni da "
                      f"oggi, e riverificarla ogni anno con i valori aggiornati.")
    if deboli:
        azioni.append('Quantificare la copertura assicurativa necessaria per gli scenari '
                      'scoperti (' + ', '.join(deboli) + '), che è il passo successivo '
                      'naturale di questo studio.')
    azioni.append('Sostituire le pensioni stimate con quelle dell’estratto conto INPS: '
                  'sono il parametro che sposta di più la data di uscita.')
    if marg['spesa'] is not None and p['spesa']:
        azioni.append(f"Tenere la spesa del nucleo entro {_eur(marg['spesa'])} l’anno a "
                      'valori odierni: oltre quella cifra il piano scende sotto la soglia.')
    if ha_figlio:
        azioni.append(f"Valutare gli strumenti di tutela di {nomi['figlio']} "
                      '(trust, vincolo di destinazione, polizza): lo scenario «dopo di noi» '
                      'è quello con l’orizzonte più lungo e meno rimediabile.')
    azioni.append('Rivedere il piano a ogni cambiamento di reddito, di composizione '
                  'familiare o di regime fiscale, e comunque una volta l’anno.')
    sezioni.append(('Conclusioni e indicazioni operative', [
        html.Ol([html.Li(a, style={'fontSize': '12.5px', 'lineHeight': '1.7',
                                   'color': '#444', 'marginBottom': '5px'})
                 for a in azioni], style={'paddingLeft': '20px', 'margin': '0'}),
    ]))

    return html.Details([
        html.Summary([
            html.Span('📄 Relazione', style={'fontWeight': '700', 'color': BLU,
                                             'fontSize': '13px'}),
            html.Span('  — leggi l’analisi in forma discorsiva',
                      style={'fontSize': '11px', 'color': '#777'}),
        ], style={'cursor': 'pointer', 'padding': '2px 0'}),
        html.Div([html.Div([
            html.Div(t, style={'fontSize': '12px', 'fontWeight': '700', 'color': BLU,
                               'textTransform': 'uppercase', 'letterSpacing': '0.04em',
                               'margin': '14px 0 6px'}),
            *corpo,
        ]) for t, corpo in sezioni], style={'maxWidth': '760px'}),
    ], open=True, style={'padding': '16px 20px', 'background': 'white',
                         'border': '1px solid #e8edf5', 'borderRadius': '10px',
                         'marginBottom': '18px'})


# ─────────────────────────────────────────────────────────────────────────────
# Margini di sicurezza
# ─────────────────────────────────────────────────────────────────────────────
def _pannello_margini(p, marg, soglia, scenario, comp, nomi):
    """Di quanto la realta' puo' scostarsi dalle ipotesi prima che il piano vada
    rivisto. E' la parte che il cliente ha chiesto come "margini di sicurezza" e
    "eventi che richiedono un aggiornamento": sono numeri calcolati, non massime."""
    voci, campanelli = [], []

    r = marg['rendimento']
    if r is None:
        voci.append(('Rendimento', 'nessun margine',
                     f'il piano non raggiunge il {soglia:.0f}% nemmeno con le ipotesi attuali'))
    elif r <= 0.01:
        voci.append(('Rendimento medio', 'irrilevante',
                     'il piano regge anche con rendimento nullo: lo sostengono '
                     'redditi e pensioni, non il mercato'))
    else:
        voci.append(('Rendimento medio', f'{r:.2f}% annuo',
                     f'il minimo che regge la soglia — oggi se ne ipotizza '
                     f'{p["rendimento"]:.2f}%, cuscinetto {p["rendimento"] - r:.2f} punti'
                     + (' (anche la fase di decumulo scende degli stessi punti)'
                        if p['rendimento_pens'] != p['rendimento'] else '')))
        campanelli.append(f'il rendimento medio realizzato scende stabilmente sotto '
                          f'il {r:.1f}% annuo nominale (circa {r - p["inflazione"]:.1f}% reale)')

    sp = marg['spesa']
    if sp is None:
        voci.append(('Spesa del nucleo', 'nessun margine',
                     'già al livello attuale la soglia non è raggiunta'))
    else:
        voci.append(('Spesa sostenibile', _eur(sp) + '/anno',
                     f'contro i {_eur(p["spesa"])} attuali: '
                     f'+{(sp / p["spesa"] - 1) * 100:.0f}% di margine'
                     if p['spesa'] else ''))
        campanelli.append(f'la spesa del nucleo supera stabilmente {_eur(sp)} l’anno '
                          f'(a valori di oggi)')

    pa = marg['patrimonio']
    if pa is None:
        voci.append(('Patrimonio minimo', 'non sufficiente',
                     'nemmeno l’intero patrimonio attuale regge la soglia'))
    elif pa <= 1000:
        voci.append(('Patrimonio minimo necessario', 'nessuno',
                     'con queste date di uscita il piano si regge sui soli redditi '
                     'e pensioni: il patrimonio è tutto margine'))
    else:
        voci.append(('Patrimonio minimo necessario', _eur(pa),
                     f'contro {_eur(p["patrimonio"])} attuali: si può perdere fino al '
                     f'{(1 - pa / p["patrimonio"]) * 100:.0f}% e restare nel piano'
                     if p['patrimonio'] else ''))
        campanelli.append(f'il patrimonio scende sotto {_eur(pa)} e non recupera')

    campanelli += [
        'cambiano le date di decorrenza o gli importi delle pensioni rispetto '
        'all’estratto conto INPS usato qui',
        'cambia l’aliquota sulle rendite finanziarie o il regime di successione',
        'cambia la composizione del nucleo (convivenza, matrimonio, figli, separazione): '
        'da lì il piano va rifatto, non aggiustato',
        'arriva uno degli eventi simulati (premorienza, invalidità)',
        'cambia la composizione del portafoglio: rendimento e volatilità impostati '
        'valgono per un azionario globale',
    ]

    riga = {'display': 'flex', 'justifyContent': 'space-between', 'gap': '12px',
            'padding': '7px 0', 'borderBottom': '1px solid #f0f3f8', 'fontSize': '12px'}
    return html.Div([
        _titolo('Margini di sicurezza',
                f'Di quanto le cose possono andare peggio del previsto prima che il piano '
                f'scenda sotto il {soglia:.0f}%. Scenario '
                f'«{etichetta_scenario(scenario, comp, nomi).split(" — ")[0]}», '
                f'un parametro per volta, gli altri fermi.', margine='0'),
        html.Div([html.Div([
            html.Div([html.Div(t, style={'fontWeight': '600', 'color': '#333'}),
                      html.Div(n, style={'fontSize': '11px', 'color': '#777'})]),
            html.Div(v, style={'fontWeight': '700', 'color': BLU, 'whiteSpace': 'nowrap'}),
        ], style=riga) for t, v, n in voci]),
        _titolo('Quando il piano va rivisto', margine='16px'),
        html.Ul([html.Li(c, style={'fontSize': '11.5px', 'color': '#555', 'marginBottom': '3px',
                                   'lineHeight': '1.5'}) for c in campanelli],
                style={'paddingLeft': '18px', 'margin': '0'}),
    ], style={'padding': '14px 16px', 'background': '#fbfcfe',
              'border': '1px solid #e8edf5', 'borderRadius': '10px', 'marginTop': '16px'})


def _fig_cono(v0, t, q, nome):
    """Il cono: fasce dei percentili, mediana e riga del VaR condizionale.
    Su orizzonti lunghi la forbice fra il 5° e il 95° percentile arriva a valere
    centinaia di volte: su scala lineare la mediana finirebbe schiacciata sullo
    zero e il grafico non direbbe piu' nulla. Oltre le cento volte si passa alla
    scala logaritmica, dove una distanza uguale e' una PERCENTUALE uguale."""
    fig = go.Figure()
    for lo, hi, col, etichetta in ((5, 95, 'rgba(26,58,107,0.10)', '5°–95° percentile'),
                                   (25, 75, 'rgba(26,58,107,0.22)', '25°–75° percentile')):
        fig.add_trace(go.Scatter(x=t, y=q[hi], line={'width': 0}, showlegend=False,
                                 hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=t, y=q[lo], fill='tonexty', fillcolor=col,
                                 line={'width': 0}, name=etichetta, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=t, y=q[50], line={'color': BLU, 'width': 2.5},
                             name='Mediana',
                             hovertemplate='Fra %{x:.0f} anni<br>%{y:,.0f} €<extra></extra>'))
    fig.add_trace(go.Scatter(x=t, y=q['cvar'],
                             line={'color': ROSSO, 'width': 1.8, 'dash': 'dash'},
                             name='VaR condizionale (media del 5% peggiore)',
                             hovertemplate='Fra %{x:.0f} anni<br>%{y:,.0f} €<extra></extra>'))
    fig.add_hline(y=v0, line={'color': '#9aa5b5', 'width': 1, 'dash': 'dot'},
                  annotation_text=f'valore di oggi {_eur(v0)}',
                  annotation_position='top left',
                  annotation_font={'size': 10, 'color': '#777'})
    log = float(q[95][-1]) > 50 * max(float(v0), 1.0)
    fig.update_layout(
        margin={'l': 62, 'r': 20, 't': 30, 'b': 40}, height=340,
        plot_bgcolor='white', paper_bgcolor='white',
        font={'family': 'Inter, sans-serif', 'size': 11},
        legend={'orientation': 'h', 'y': 1.14, 'x': 0, 'font': {'size': 10}},
        title={'text': '', 'font': {'size': 11}},
        xaxis={'title': 'Anni da oggi', 'gridcolor': '#eef2f7'},
        yaxis=dict({'title': f'Valore del portafoglio — profilo {nome} (€ correnti)'
                             + (' — scala logaritmica' if log else ''),
                    'gridcolor': '#eef2f7'},
                   **({'type': 'log', 'tickformat': '.2s'} if log
                      else {'tickformat': ',.0f'})),
        hovermode='x unified',
    )
    return fig


def _riquadro(testo_html, colore, sfondo, bordo):
    return html.Div(testo_html, style={
        'fontSize': '11.5px', 'padding': '8px 12px', 'borderRadius': '8px',
        'background': sfondo, 'border': f'1px solid {bordo}', 'color': colore,
        'lineHeight': '1.55', 'marginBottom': '10px'})


def _tabella_cono(t, q, anni, infl):
    """I numeri del cono agli orizzonti che interessano, perché un grafico si
    guarda ma non si cita in una relazione."""
    tappe = sorted({x for x in (5, 10, 20, 30, anni) if 0 < x <= anni})
    intestazioni = ['Fra', 'Sfavorevole (5°)', 'Mediana', 'Favorevole (95°)',
                    'Media del 5% peggiore', 'Mediana in € di oggi']
    th = {'padding': '7px 10px', 'fontSize': '10px', 'textTransform': 'uppercase',
          'letterSpacing': '0.04em', 'color': '#666', 'textAlign': 'left',
          'borderBottom': '2px solid #e8edf5', 'whiteSpace': 'nowrap'}
    td = {'padding': '7px 10px', 'fontSize': '12px', 'borderBottom': '1px solid #f0f3f8',
          'whiteSpace': 'nowrap'}
    corpo = []
    for x in tappe:
        i = int(x)
        oggi = q[50][i] / ((1 + infl) ** i)
        corpo.append(html.Tr([
            html.Td(f'{i} anni', style={**td, 'fontWeight': '600'}),
            html.Td(_eur(q[5][i]), style={**td, 'color': ROSSO}),
            html.Td(_eur(q[50][i]), style={**td, 'fontWeight': '700'}),
            html.Td(_eur(q[95][i]), style={**td, 'color': VERDE}),
            html.Td(_eur(q['cvar'][i]), style={**td, 'color': ROSSO}),
            html.Td(_eur(oggi), style={**td, 'color': '#666'}),
        ]))
    return html.Div(
        html.Table([html.Thead(html.Tr([html.Th(h, style=th) for h in intestazioni])),
                    html.Tbody(corpo)],
                   style={'width': '100%', 'minWidth': '620px', 'borderCollapse': 'collapse',
                          'fontFamily': 'Inter, sans-serif'}),
        style={'overflowX': 'auto', 'marginTop': '6px'})


def _tabella_strumenti(dati):
    intestazioni = ['Strumento', 'Peso', 'Rendimento medio', 'Volatilità']
    th = {'padding': '6px 10px', 'fontSize': '10px', 'textTransform': 'uppercase',
          'letterSpacing': '0.04em', 'color': '#666', 'textAlign': 'left',
          'borderBottom': '2px solid #e8edf5', 'whiteSpace': 'nowrap'}
    td = {'padding': '6px 10px', 'fontSize': '12px', 'borderBottom': '1px solid #f0f3f8'}
    corpo = []
    for d in dati['dettaglio']:
        corpo.append(html.Tr([
            html.Td([html.Span(d['ticker'], style={'fontWeight': '700'}),
                     html.Span(f"  {d['nome']}" if d['nome'] else '',
                               style={'color': '#888', 'fontSize': '11px'})], style=td),
            html.Td(f"{d['peso']:.0f}%", style=td),
            html.Td(f"{d['mu']:+.1%}".replace('.', ','), style=td),
            html.Td(f"{d['sd']:.1%}".replace('.', ','), style=td),
        ]))
    corpo.append(html.Tr([
        html.Td('Portafoglio', style={**td, 'fontWeight': '700'}),
        html.Td('100%', style={**td, 'fontWeight': '700'}),
        html.Td(f"{dati['mu']:+.1%}".replace('.', ','), style={**td, 'fontWeight': '700'}),
        html.Td(f"{dati['sd']:.1%}".replace('.', ','), style={**td, 'fontWeight': '700'}),
    ]))
    return html.Div(
        html.Table([html.Thead(html.Tr([html.Th(h, style=th) for h in intestazioni])),
                    html.Tbody(corpo)],
                   style={'width': '100%', 'minWidth': '480px', 'borderCollapse': 'collapse',
                          'fontFamily': 'Inter, sans-serif'}),
        style={'overflowX': 'auto', 'marginTop': '6px'})


def _pct(v):
    return '—' if v is None else f'{v:+.1%}'.replace('.', ',')


def _blocco_proiezione(p, dati):
    """Proiezione del patrimonio, subito sotto il riepilogo: da quanto si parte
    oggi e dove puo' arrivare il portafoglio scelto, solo per effetto del
    mercato. I versamenti e i prelievi sono un'altra cosa e stanno nel grafico
    del piano, piu' sotto: tenerli separati e' l'unico modo di vedere quanto
    rischio c'e' nel portafoglio in quanto tale."""
    dati = dati or {}
    nome = NOME_PROFILO.get(dati.get('profilo'), 'non definito')
    avvisi = dati.get('avvisi') or []

    if not dati.get('ok'):
        return html.Div([
            _titolo(f'Proiezione del portafoglio — profilo {nome}',
                    'Il portafoglio non è ancora misurabile.'),
            _riquadro([html.Div('• ' + a) for a in avvisi] or
                      [html.Div('Scegli un profilo oppure indica i ticker.')],
                      ROSSO, '#fdf3f2', '#f3ddda'),
        ])

    # La proiezione dura quanto il piano: se il piano deve reggere fino a 120
    # anni, il portafoglio va guardato per tutti quegli anni, non per trenta.
    # Il tetto a 120 non morde mai su eta' vere: e' solo una rete se l'utente
    # scrive un orizzonte assurdo.
    anni = int(max(1, min(120, p['orizzonte'] - p['eta_lui'])))
    v0   = float(p['patrimonio'])
    infl = p['inflazione'] / 100
    mu, sd = dati['mu'], dati['sd']
    t, q = cono(v0, mu, sd, anni)

    sotto = (f"Dal patrimonio di oggi ({_eur(v0)}), solo per effetto del mercato: "
             f"nessun versamento e nessun prelievo. Misure su {dati['giorni']} giorni "
             f"di borsa, da {dati['da']} al {dati['a']}.")
    if dati.get('standard_modificato'):
        sotto += ' Gli strumenti del profilo sono stati sostituiti a mano.'
    pezzi = [_titolo(f'Proiezione del portafoglio — profilo {nome}', sotto)]
    if avvisi:
        pezzi.append(_riquadro([html.Div('• ' + a) for a in avvisi], ROSSO,
                               '#fdf3f2', '#f3ddda'))
    if dati.get('normalizzato'):
        pezzi.append(_riquadro(
            f"Le percentuali indicate sommano a {dati['totale_scritto']:.0f}%: "
            'sono state riproporzionate a 100 mantenendo i rapporti fra loro.',
            '#7a5c00', '#fdf8ec', '#f0e2bd'))
    if dati.get('applicato'):
        pezzi.append(_riquadro([
            html.Span('Il piano sta usando questi numeri: ', style={'fontWeight': '700'}),
            html.Span(f"rendimento {mu:+.1%} e volatilità {sd:.1%}".replace('.', ','),
                      style={'fontWeight': '700'}),
            html.Span('. Sono il realizzato della finestra scelta, non una previsione: '
                      'su un orizzonte di decenni si usa di norma un’ipotesi più '
                      'prudente. Per deciderla a mano togli la spunta «usa nel piano» '
                      'e scrivi i valori nel gruppo Mercato.'),
        ], '#7a5c00', '#fdf8ec', '#f0e2bd'))

    pezzi.append(html.Div([
        _card('Rendimento medio', f'{mu:+.1%}'.replace('.', ','),
              f"composto {dati['cagr']:+.1%}".replace('.', ',') + ' l’anno',
              VERDE if mu > 0 else ROSSO),
        _card('Volatilità', f'{sd:.1%}'.replace('.', ','), 'annua, misurata sui prezzi'),
        _card('VaR 95% a 1 anno', _pct(dati.get('var_modello')),
              'perdita superata da un anno su venti'
              + (f" — realizzato {_pct(dati['var'])}" if dati.get('var') is not None
                 else ''), ROSSO),
        _card('VaR condizionale', _pct(dati.get('cvar_modello')),
              'perdita media quando si sta sotto il VaR'
              + (f" — realizzato {_pct(dati['cvar'])}" if dati.get('cvar') is not None
                 else ''), ROSSO),
    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'margin': '8px 0 4px'}))

    pezzi.append(dcc.Graph(figure=_fig_cono(v0, t, q, nome),
                           config={'displayModeBar': False}))

    # ── Le stesse cose, dette a parole ──────────────────────────────────────
    # Due VaR con lo stesso nome: quello del modello disegna il bordo basso del
    # cono, quello realizzato dice come e' andata davvero. Dirli entrambi, e
    # dire quale dei due e' piu' severo, e' l'unico modo di non farli sembrare
    # un errore di calcolo.
    if dati.get('var') is not None:
        severo = dati['var_modello'] < dati['var']
        frase_var = (
            'Il bordo basso del cono è disegnato dal VaR del modello: a un anno vale '
            + _pct(dati['var_modello']) + ', e il condizionale — la media proprio di '
            'quel 5% — ' + _pct(dati['cvar_modello']) +
            '. Nella finestra di dati misurata il 5% degli anni mobili peggiori ha '
            'fatto peggio di ' + _pct(dati['var']) + ', con una perdita media di '
            + _pct(dati['cvar']) + ' e un minimo di ' + _pct(dati['peggio']) + '. ' +
            ('Il modello è dunque più severo di quanto sia davvero successo: dieci anni '
             'di borsa non contengono tutte le code possibili, ed è giusto che il piano '
             'si difenda da quelle che non si sono viste.' if severo else
             'Il modello è dunque più mite di quanto sia davvero successo: la finestra '
             'contiene una crisi che la sola volatilità non basta a descrivere, e vale '
             'la pena alzare a mano la volatilità del piano.'))
    else:
        frase_var = ('Lo storico disponibile non basta per misurare il VaR su finestre '
                     'annue: restano validi rendimento e volatilità.')
    paragrafi = [
        _par('La linea centrale è la mediana, le due fasce contengono rispettivamente il '
             '50% e il 90% dei percorsi possibili, la riga rossa tratteggiata è il ',
             ('VaR condizionale', True),
             ': non il 5° percentile, ma la media di quanto resterebbe nel 5% dei '
             'percorsi peggiori. È la domanda giusta da porsi — non «quanto posso '
             'perdere al massimo» ma «quanto perdo quando perdo».'),
        _par('Sulla finestra misurata il portafoglio ha reso in media ',
             (f'{mu:+.1%}'.replace('.', ',') + ' l’anno', True),
             f" ({dati['cagr']:+.1%} composto)".replace('.', ',') +
             f", con una volatilità del {sd:.1%}".replace('.', ',') + '.'),
        _par(frase_var),
    ]
    if len(dati['dettaglio']) > 1:
        paragrafi.append(_par(
            f"La volatilità del portafoglio è del {sd:.1%}".replace('.', ',') +
            f", contro il {dati['sd_pesata']:.1%}".replace('.', ',') +
            ' che si otterrebbe sommando quelle dei singoli strumenti in proporzione '
            'al peso: la differenza è l’effetto della diversificazione.'))
    # Attenzione a non passare _eur() dentro un .replace('.', ','): i punti li usa
    # come separatore delle migliaia.
    infl_txt = f"{p['inflazione']:.1f}".replace('.', ',')
    paragrafi.append(_par(
        f'Fra {anni} anni la mediana vale ', (_eur(q[50][anni]), True),
        ' in euro correnti, cioè ' + _eur(q[50][anni] / ((1 + infl) ** anni)) +
        f' in euro di oggi con un’inflazione del {infl_txt}%. '
        'Nello scenario sfavorevole (5° percentile) si fermerebbe a ' +
        _eur(q[5][anni]) + '.'))
    paragrafi.append(_par(
        ('Come leggere questi numeri: ', True),
        'sono il comportamento realizzato dagli strumenti scelti nella finestra '
        'indicata, proiettato in avanti come se il futuro somigliasse a quel '
        'passato. Non è una previsione, e una finestra diversa dà numeri diversi — '
        'sui titoli di Stato, per esempio, gli ultimi dieci anni comprendono il '
        'rialzo dei tassi del 2022 e raccontano un rendimento molto più basso di '
        'quello che oggi si può ottenere a scadenza.'))
    # Su orizzonti da vita intera il numero finale diventa enorme per pura
    # matematica dell'interesse composto: va detto, o si legge come una promessa.
    if anni >= 40:
        paragrafi.append(_par(
            ('Attenzione all’orizzonte: ', True),
            f'la proiezione copre {anni} anni perché tanto dura il piano, ma ',
            (f"{mu:+.1%}".replace('.', ',') + ' l’anno per ' + str(anni) + ' anni', True),
            ' moltiplica il capitale per ' +
            f"{q[50][anni] / v0:,.0f}".replace(',', '.') +
            ' volte: è la matematica dell’interesse composto, non una previsione '
            'credibile su un secolo. Il numero a trent’anni è già il limite oltre '
            'il quale conviene ragionare per ordini di grandezza, e la parte finale '
            'del grafico va letta come un riferimento alto, non come un obiettivo.'))
    pezzi.extend(paragrafi)

    pezzi.append(_titolo('Il portafoglio agli orizzonti principali', margine='12px'))
    pezzi.append(_tabella_cono(t, q, anni, infl))
    pezzi.append(_titolo('Composizione e misure dei singoli strumenti', margine='14px'))
    pezzi.append(_tabella_strumenti(dati))
    return html.Div(pezzi, style={'marginTop': '6px'})


# ─────────────────────────────────────────────────────────────────────────────
# Pagina
# ─────────────────────────────────────────────────────────────────────────────
def _campo(idc, etichetta, default, passo, suffisso, nota):
    figli = [
        html.Label(etichetta, id={'type': 'pl-etichetta', 'index': idc},
                   style={'fontSize': '11px', 'color': '#333',
                          'display': 'block', 'marginBottom': '2px'}),
        html.Div([
            dcc.Input(id=f'pl-{idc}', type='number', value=default, step=passo,
                      debounce=True,
                      style={'width': '104px', 'padding': '4px 7px', 'fontSize': '12px',
                             'border': '1px solid #ccd9ee', 'borderRadius': '5px',
                             'fontFamily': 'Inter, sans-serif'}),
            html.Span(suffisso, style={'fontSize': '10px', 'color': '#888',
                                       'marginLeft': '6px'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
    ]
    if nota:
        figli.append(html.Div(nota, id={'type': 'pl-nota', 'index': idc},
                              style={'fontSize': '10px', 'color': '#999', 'marginTop': '2px',
                                     'maxWidth': '230px'}))
    return html.Div(figli, id={'type': 'pl-campo', 'index': idc},
                    style={'marginBottom': '9px'})


def _campo_nome(idc, etichetta, default):
    return html.Div([
        html.Label(etichetta, style={'fontSize': '10px', 'color': '#666',
                                     'display': 'block', 'marginBottom': '2px'}),
        dcc.Input(id=f'pl-{idc}', type='text', value=default, debounce=True,
                  placeholder=default,
                  style={'width': '100px', 'padding': '4px 7px', 'fontSize': '12px',
                         'border': '1px solid #ccd9ee', 'borderRadius': '5px',
                         'fontFamily': 'Inter, sans-serif'}),
    ], id=f'pl-box-{idc}', style={'display': 'block'})


def _pannello_nucleo():
    """Chi compone il nucleo e come si chiama. Sta in cima perché decide tutto il
    resto: quali campi compaiono, quali scenari esistono, come è scritta la
    relazione."""
    return html.Div([
        html.Div('Chi compone il nucleo', style={'fontSize': '11px', 'fontWeight': '700',
                                                 'color': BLU, 'marginBottom': '5px'}),
        dcc.Checklist(
            id='pl-composizione', value=['lei', 'figlio'],
            options=[{'label': ' Coniuge o convivente', 'value': 'lei'},
                     {'label': ' Figlio a carico', 'value': 'figlio'}],
            labelStyle={'display': 'block', 'fontSize': '11.5px', 'color': '#333',
                        'marginBottom': '2px', 'cursor': 'pointer'},
            inputStyle={'marginRight': '4px'}),
        html.Div('Togliendo la spunta spariscono i campi e gli scenari che riguardano '
                 'quella persona, e la relazione lo dichiara.',
                 style={'fontSize': '10px', 'color': '#999', 'margin': '4px 0 10px',
                        'lineHeight': '1.45'}),
        html.Div([
            _campo_nome('nome_lui', 'Nome', NOMI_DEFAULT['lui']),
            _campo_nome('nome_lei', 'Coniuge', NOMI_DEFAULT['lei']),
            _campo_nome('nome_figlio', 'Figlio', NOMI_DEFAULT['figlio']),
        ], style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap'}),
    ], style={'padding': '12px 14px', 'background': '#f0f4fa', 'borderRadius': '9px',
              'marginBottom': '10px'})


def _etichetta_profilo(nome, mix):
    """L'etichetta dice subito quanta parte e' azionaria: e' l'unica cosa che
    davvero distingue un profilo dall'altro."""
    if not mix:
        return ' Scelta autonoma'
    quota = sum(w for t, w in mix if t == AZIONARIO)
    return f' {nome} — {quota:.0f}% azioni'


def _nota_standard(az, ob):
    """La riga sotto ai profili: dice quali due strumenti li stanno riempiendo.
    Se sono stati sostituiti il nome descrittivo non c'e', e allora si dichiara
    al posto di chi sta — non deve restare il dubbio su che cosa sia in uso."""
    def descr(tk, base):
        nome = NOME_STRUMENTO.get(tk, '')
        return f'{tk} ({nome})' if nome else f'{tk} (al posto di {base})'
    return f'Profili standard: {descr(az, AZIONARIO)} e {descr(ob, OBBLIG)}.'


def _pannello_profilo():
    """Il profilo di investimento. Sta sopra il gruppo Mercato perché è quello
    che ne riempie i campi: rendimento e volatilità non si scrivono a mano, si
    misurano sui prezzi degli strumenti scelti."""
    st_tk = {'width': '92px', 'padding': '3px 6px', 'fontSize': '11px',
             'border': '1px solid #ccd9ee', 'borderRadius': '5px',
             'fontFamily': 'Inter, sans-serif'}
    st_pk  = dict(st_tk, width='52px')
    st_std = dict(st_tk, width='116px')
    nota_st = {'fontSize': '10px', 'color': '#999', 'margin': '4px 0 6px',
               'lineHeight': '1.45'}

    def riga_std(etichetta, ident, base):
        return html.Div([
            html.Span(etichetta, style={'fontSize': '10.5px', 'color': '#555',
                                        'width': '86px'}),
            dcc.Input(id=ident, type='text', value=base, debounce=True,
                      placeholder=base, style=st_std),
        ], style={'display': 'flex', 'gap': '4px', 'alignItems': 'center',
                  'marginBottom': '4px'})

    righe = [html.Div([
        dcc.Input(id={'type': 'pl-tk', 'index': i}, type='text', value='',
                  debounce=True, placeholder='ticker', style=st_tk),
        dcc.Input(id={'type': 'pl-pk', 'index': i}, type='number', value=None,
                  min=0, step=5, debounce=True, placeholder='%', style=st_pk),
        html.Span('%', style={'fontSize': '10px', 'color': '#888'}),
    ], style={'display': 'flex', 'gap': '4px', 'alignItems': 'center',
              'marginBottom': '4px'}) for i in range(N_AUTONOMO)]

    return html.Details([
        html.Summary('Profilo di rischio',
                     style={'fontSize': '12px', 'fontWeight': '700', 'color': BLU,
                            'cursor': 'pointer', 'padding': '6px 0'}),
        html.Div([
            dcc.RadioItems(
                id='pl-profilo', value='dinamico',
                options=[{'label': _etichetta_profilo(n, mix), 'value': k}
                         for k, n, mix in PROFILI],
                labelStyle={'display': 'block', 'fontSize': '11.5px', 'color': '#333',
                            'marginBottom': '2px', 'cursor': 'pointer'},
                inputStyle={'marginRight': '4px'}),
            html.Div([
                html.Div(_nota_standard(AZIONARIO, OBBLIG), id='pl-nota-standard',
                         style=nota_st),
                dcc.Checklist(
                    id='pl-modifica-std', value=[],
                    options=[{'label': ' Modifica gli ETF dei profili standard',
                              'value': 'si'}],
                    labelStyle={'display': 'block', 'fontSize': '11px', 'color': '#333',
                                'cursor': 'pointer'},
                    inputStyle={'marginRight': '4px'}),
                html.Div([
                    riga_std('Azionario', 'pl-tk-az', AZIONARIO),
                    riga_std('Obbligazionario', 'pl-tk-ob', OBBLIG),
                    html.Div('Le percentuali dei cinque profili non cambiano: cambia '
                             'solo lo strumento che le riempie. Un campo lasciato vuoto '
                             'torna al ticker di partenza.', style=nota_st),
                ], id='pl-righe-standard', style={'display': 'none'}),
            ], id='pl-blocco-standard', style={'margin': '4px 0 6px'}),

            html.Div(righe + [
                html.Div('Entrano nel portafoglio solo i ticker scritti. Se le '
                         'percentuali non fanno 100 vengono riproporzionate.',
                         style={'fontSize': '10px', 'color': '#999', 'marginTop': '3px',
                                'lineHeight': '1.45'}),
            ], id='pl-righe-autonomo', style={'display': 'none'}),

            html.Div([
                html.Label('Storia usata per le misure',
                           style={'fontSize': '11px', 'color': '#333', 'display': 'block',
                                  'marginBottom': '2px'}),
                dcc.Dropdown(id='pl-storia', value=10, clearable=False,
                             options=[{'label': f'{a} anni', 'value': a}
                                      for a in (3, 5, 10)],
                             style={'width': '110px', 'fontSize': '11px'}),
            ], style={'margin': '8px 0 6px'}),

            dcc.Checklist(
                id='pl-usa-mercato', value=['si'],
                options=[{'label': ' Usa rendimento e volatilità del profilo nel piano',
                          'value': 'si'}],
                labelStyle={'display': 'block', 'fontSize': '11px', 'color': '#333',
                            'cursor': 'pointer'},
                inputStyle={'marginRight': '4px'}),
            html.Div('Togliendo la spunta il piano usa i valori scritti a mano nel '
                     'gruppo Mercato, e il profilo resta solo una proiezione.',
                     style={'fontSize': '10px', 'color': '#999', 'marginTop': '3px',
                            'lineHeight': '1.45'}),

            # Il portafoglio del decumulo si sceglie come quello dell'accumulo: o
            # un profilo, e allora i suoi numeri si MISURANO sui prezzi veri, o un
            # tasso scritto a mano. Sono due portafogli diversi, non due ipotesi
            # sullo stesso: chi vive di prelievi cambia davvero strumenti.
            html.Div([
                html.Label('Portafoglio in fase di decumulo',
                           style={'fontSize': '11px', 'color': '#333', 'display': 'block',
                                  'marginBottom': '2px', 'fontWeight': '600'}),
                dcc.Dropdown(id='pl-profilo-dec', value='cauto', clearable=False,
                             options=[{'label': _etichetta_profilo(n, mix).strip(),
                                       'value': k} for k, n, mix in PROFILI if mix]
                                     + [{'label': 'Tasso a mano', 'value': 'tasso'}],
                             style={'width': '200px', 'fontSize': '11px'}),
                html.Div(id='pl-nota-dec', style=nota_st),
            ], style={'margin': '10px 0 2px', 'borderTop': '1px solid #eef2f7',
                      'paddingTop': '9px'}),
        ], style={'paddingLeft': '4px', 'paddingBottom': '8px'}),
    ], open=True, style={'borderBottom': '1px solid #eef2f7'})


def _pannello_parametri():
    gruppi = [_pannello_nucleo()]
    for g in _ORDINE_GRUPPI:
        if g == 'Mercato':
            gruppi.append(_pannello_profilo())
        campi = [c for c in CAMPI if c[5] == g]
        gruppi.append(html.Details([
            html.Summary(g, style={'fontSize': '12px', 'fontWeight': '700', 'color': BLU,
                                   'cursor': 'pointer', 'padding': '6px 0'}),
            html.Div([_campo(c[0], testo(c[1], NOMI_DEFAULT), c[2], c[3], c[4],
                             testo(c[6], NOMI_DEFAULT)) for c in campi],
                     style={'paddingLeft': '4px'}),
        ], open=(g in ('Famiglia', 'Patrimonio', 'Redditi e spese', 'Uscita dal lavoro')),
            style={'borderBottom': '1px solid #eef2f7'}))
    return html.Div(gruppi, style={
        'flex': '0 0 272px', 'maxHeight': '80vh', 'overflowY': 'auto',
        'paddingRight': '12px', 'borderRight': '1px solid #e8edf5'})


def _metodo():
    voci = [
        ('Simulazione', 'Monte Carlo su rendimenti lognormali indipendenti, calibrati '
                        'perché media e volatilità aritmetiche siano quelle impostate. '
                        'Seme fisso: stessi parametri, stesso risultato.'),
        ('Profilo di investimento', 'Rendimento, volatilità, VaR e VaR condizionale non '
                                    'sono ipotesi scritte a mano: si misurano sui prezzi '
                                    'veri degli strumenti del profilo, riportati in euro. '
                                    'Restano il realizzato di una finestra, non una '
                                    'previsione. I due strumenti dei profili standard sono '
                                    'sostituibili con la spunta «modifica»: cambia lo '
                                    'strumento, non le percentuali del profilo.'),
        ('VaR e VaR condizionale', 'Al 5%, su finestre mobili di dodici mesi. Il VaR è la '
                                   'perdita che solo un anno su venti supera; il '
                                   'condizionale è la perdita media proprio di quegli anni. '
                                   'Le finestre si sovrappongono — meno indipendenti di '
                                   'quante sembrino — ma mostrano le code vere invece di '
                                   'quelle di una normale.'),
        ('Cono di Ibbotson', 'I percentili del valore futuro in forma chiusa, dalla stessa '
                             'lognormale del Monte Carlo: cono e grafico del piano non '
                             'possono contraddirsi. Proietta il solo patrimonio di oggi, '
                             'senza versamenti né prelievi.'),
        ('Valuta del calcolo', 'Tutto in euro nominali anno per anno, poi deflazionato: '
                               'l’imposta del 26% colpisce la plusvalenza nominale, quindi '
                               'un modello in termini reali la sottostimerebbe.'),
        ('Composizione del nucleo', 'È una scelta esplicita, non si deduce dai campi '
                                    'lasciati vuoti: un campo vuoto ripiegherebbe sul '
                                    'valore di default, cioè su un componente che nessuno '
                                    'ha inserito.'),
        ('Risparmio', 'Non è un dato di ingresso: è reddito meno spesa. Così i tre numeri '
                      'non possono contraddirsi e smettere di lavorare azzera il risparmio '
                      'da sé.'),
        ('Disinvestimenti', 'Per incassare un netto si vende il lordo che, tolta l’imposta '
                            'sulla plusvalenza latente, lascia quel netto. Costo di carico '
                            'medio (la norma italiana userebbe il LIFO: differenza modesta '
                            'su un PAC pluriennale, ma è un’approssimazione).'),
        ('Successione', 'Per i titoli ereditati il costo fiscale si rivaluta al valore di '
                        'successione (art. 68 TUIR) e la franchigia è di 1 milione per '
                        'coniuge e figlio: l’eredità è quindi confrontata al lordo.'),
        ('Costi ricorrenti', 'Imposta di bollo dello 0,2% annuo sul dossier. I costi dello '
                             'strumento vanno già sottratti al rendimento atteso.'),
        ('Successo di un piano', 'Il patrimonio non si esaurisce mai e a fine orizzonte '
                                 'resta almeno l’eredità obiettivo, in euro di oggi.'),
        ('Cosa NON è modellato', 'Sequenze di rendimenti autocorrelate, imposte di successione '
                                 'oltre le franchigie, polizze di protezione del capitale umano '
                                 '(da inserire quando il cliente le comunicherà), TFR e fondi '
                                 'pensione integrativi, variazioni normative.'),
    ]
    return html.Details([
        html.Summary([
            html.Span('🧮 Metodologia e ipotesi', style={'fontWeight': '700', 'color': BLU,
                                                         'fontSize': '13px'}),
            html.Span('  — come sono fatti i conti', style={'fontSize': '11px', 'color': '#777'}),
        ], style={'cursor': 'pointer', 'padding': '2px 0'}),
        html.Div([html.Div([
            html.Span(t + ' — ', style={'fontWeight': '700', 'color': '#333'}),
            html.Span(d, style={'color': '#555'}),
        ], style={'fontSize': '11.5px', 'lineHeight': '1.6', 'marginBottom': '5px'})
            for t, d in voci], style={'marginTop': '10px', 'maxWidth': '760px'}),
    ], style={'marginTop': '22px', 'padding': '16px 18px', 'background': '#f8fafd',
              'border': '1px solid #e8edf5', 'borderRadius': '10px'})


def layout():
    return html.Div([
        dcc.Store(id='pl-mercato'),
        html.H2('Indipendenza finanziaria — studio di pianificazione',
                style={'color': BLU, 'fontSize': '20px', 'margin': '0 0 4px'}),
        html.Div('Quando e a quali condizioni il nucleo può smettere di lavorare, '
                 'tenuto conto di fisco, previdenza, eredità da lasciare e degli scenari '
                 'avversi. Ogni parametro a sinistra ricalcola tutto.',
                 style={'color': '#666', 'fontSize': '12px', 'marginBottom': '14px'}),
        html.Div([
            _pannello_parametri(),
            html.Div([
                html.Div([
                    html.Label('Scenario:', style={'fontSize': '11px', 'fontWeight': '700',
                                                   'color': BLU, 'marginRight': '8px'}),
                    dcc.Dropdown(id='pl-scenario', value='base', clearable=False,
                                 options=[{'label': l, 'value': v} for v, l
                                          in scenari_applicabili(['lei', 'figlio'],
                                                                 NOMI_DEFAULT)],
                                 style={'width': '330px', 'fontSize': '12px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
                dcc.Loading(type='dot', color=BLU, children=html.Div(id='pl-risultati')),
            ], style={'flex': '1 1 auto', 'paddingLeft': '18px', 'minWidth': '0'}),
        ], style={'display': 'flex', 'alignItems': 'flex-start', 'gap': '4px'}),
        _metodo(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────────────────
def register_callbacks(app):

    # ── I campi e le etichette seguono la composizione del nucleo ───────────
    @app.callback(
        Output({'type': 'pl-campo', 'index': ALL}, 'style'),
        Output({'type': 'pl-etichetta', 'index': ALL}, 'children'),
        Output({'type': 'pl-nota', 'index': ALL}, 'children'),
        Output('pl-box-nome_lei', 'style'),
        Output('pl-box-nome_figlio', 'style'),
        Input('pl-composizione', 'value'),
        Input('pl-nome_lui', 'value'),
        Input('pl-nome_lei', 'value'),
        Input('pl-nome_figlio', 'value'),
    )
    def _campi_visibili(comp, n_lui, n_lei, n_figlio):
        comp = comp or []
        nomi = nomi_da(n_lui, n_lei, n_figlio)
        per_id = {c[0]: c for c in CAMPI}

        def visibile(idc):
            chi = per_id[idc][7]
            return (not chi) or (chi in comp)

        # L'ordine dei match pattern non e' quello di CAMPI: lo si legge da
        # ctx.outputs_list, altrimenti si assegnano stili all'elemento sbagliato.
        stili = [{'marginBottom': '9px'} if visibile(o['id']['index']) else {'display': 'none'}
                 for o in ctx.outputs_list[0]]
        etich = [testo(per_id[o['id']['index']][1], nomi) for o in ctx.outputs_list[1]]
        note  = [testo(per_id[o['id']['index']][6], nomi) for o in ctx.outputs_list[2]]
        mostra = {'display': 'block'}
        nascondi = {'display': 'none'}
        return (stili, etich, note,
                mostra if 'lei' in comp else nascondi,
                mostra if 'figlio' in comp else nascondi)

    # ── Gli scenari disponibili dipendono da chi c'è ────────────────────────
    @app.callback(
        Output('pl-scenario', 'options'),
        Output('pl-scenario', 'value'),
        Input('pl-composizione', 'value'),
        Input('pl-nome_lui', 'value'),
        Input('pl-nome_lei', 'value'),
        Input('pl-nome_figlio', 'value'),
        State('pl-scenario', 'value'),
    )
    def _opzioni_scenario(comp, n_lui, n_lei, n_figlio, scelto):
        comp = comp or []
        scen = scenari_applicabili(comp, nomi_da(n_lui, n_lei, n_figlio))
        valori = [v for v, _ in scen]
        return ([{'label': l, 'value': v} for v, l in scen],
                scelto if scelto in valori else 'base')

    # ── Il portafoglio: misure sui prezzi veri ──────────────────────────────
    @app.callback(
        Output('pl-mercato', 'data'),
        Output('pl-righe-autonomo', 'style'),
        Output('pl-blocco-standard', 'style'),
        Output('pl-righe-standard', 'style'),
        Output('pl-nota-standard', 'children'),
        Output('pl-tk-az', 'value'),
        Output('pl-tk-ob', 'value'),
        Output('pl-rendimento', 'value'),
        Output('pl-volatilita', 'value'),
        Output('pl-rendimento_pens', 'value'),
        Output('pl-volatilita_pens', 'value'),
        Output('pl-nota-dec', 'children'),
        Input('pl-profilo', 'value'),
        Input('pl-storia', 'value'),
        Input('pl-usa-mercato', 'value'),
        Input('pl-modifica-std', 'value'),
        Input('pl-tk-az', 'value'),
        Input('pl-tk-ob', 'value'),
        Input('pl-profilo-dec', 'value'),
        Input({'type': 'pl-tk', 'index': ALL}, 'value'),
        Input({'type': 'pl-pk', 'index': ALL}, 'value'),
    )
    def _mercato(profilo, storia, usa, modifica, tk_az, tk_ob, dec, tickers, pesi):
        standard = profilo != 'autonomo'
        libero   = bool(modifica)
        az = (tk_az or '').strip().upper() or AZIONARIO
        ob = (tk_ob or '').strip().upper() or OBBLIG
        if not libero:
            az, ob = AZIONARIO, OBBLIG

        dati = statistiche(composizione(profilo, tickers, pesi, az, ob), storia or 10)
        dati['profilo'] = profilo
        dati['standard_modificato'] = standard and (az, ob) != (AZIONARIO, OBBLIG)
        applica = bool(usa) and bool(dati.get('ok'))
        dati['applicato'] = applica

        st_auto = {'display': 'block' if not standard else 'none'}
        st_bloc = {'margin': '4px 0 6px', 'display': 'block' if standard else 'none'}
        st_righe = {'display': 'block' if libero else 'none'}
        # Tolta la spunta i due campi tornano a vista ai ticker di serie: se
        # restassero scritti quelli sostituiti mostrerebbero un portafoglio che
        # il calcolo non sta piu' usando. Con la spunta attiva non si toccano.
        v_az = no_update if libero or (tk_az or '').strip().upper() == AZIONARIO else AZIONARIO
        v_ob = no_update if libero or (tk_ob or '').strip().upper() == OBBLIG else OBBLIG

        # I due campi del gruppo Mercato si riempiono da soli e restano
        # modificabili: la modifica a mano dura fino al prossimo cambio di
        # profilo. Senza spunta non si toccano (no_update), altrimenti ogni
        # ricalcolo cancellerebbe il valore scritto dall'utente.
        rend = round(dati['mu'] * 100, 2) if applica else no_update
        vol  = round(dati['sd'] * 100, 2) if applica else no_update

        # Il portafoglio del decumulo: stessi due strumenti (anche se sostituiti),
        # altre percentuali. La misura passa dalla stessa cache dei prezzi, quindi
        # non costa un secondo download.
        dec = dec or 'cauto'
        nome_dec = NOME_PROFILO.get(dec, '')
        rend_dec = vol_dec = no_update
        dati['decumulo'] = None
        if dec == 'tasso':
            nota_dec = ('Valgono i valori scritti a mano in «Rendimento in decumulo» '
                        'e «Volatilità in decumulo», qui sotto nel gruppo Mercato.')
        elif not applica:
            nota_dec = (f'Profilo {nome_dec}: si misura quando la spunta «usa nel piano» '
                        'è attiva. Ora valgono i due valori scritti a mano.')
        else:
            d2 = statistiche(composizione(dec, [], [], az, ob), storia or 10)
            if d2.get('ok'):
                rend_dec = round(d2['mu'] * 100, 2)
                vol_dec  = round(d2['sd'] * 100, 2)
                dati['decumulo'] = {'profilo': dec, 'nome': nome_dec,
                                    'mu': d2['mu'], 'sd': d2['sd']}
                nota_dec = ('Profilo {}: rendimento {} e volatilità {}, misurati sugli '
                            'stessi prezzi dell’accumulo.').format(
                                nome_dec,
                                f'{rend_dec:.2f}%'.replace('.', ','),
                                f'{vol_dec:.2f}%'.replace('.', ','))
            else:
                nota_dec = (f'Profilo {nome_dec} non misurabile ora: restano i due valori '
                            'scritti a mano.')

        return (dati, st_auto, st_bloc, st_righe, _nota_standard(az, ob),
                v_az, v_ob, rend, vol, rend_dec, vol_dec, nota_dec)

    # ── Il calcolo ──────────────────────────────────────────────────────────
    @app.callback(
        Output('pl-risultati', 'children'),
        Input('pl-scenario', 'value'),
        Input('pl-composizione', 'value'),
        Input('pl-nome_lui', 'value'),
        Input('pl-nome_lei', 'value'),
        Input('pl-nome_figlio', 'value'),
        Input('pl-mercato', 'data'),
        *[Input(f'pl-{c[0]}', 'value') for c in CAMPI],
    )
    def _calcola(scenario, comp, n_lui, n_lei, n_figlio, mercato, *valori):
        comp = comp or []
        nomi = nomi_da(n_lui, n_lei, n_figlio)
        # Un campo svuotato dall'utente torna None: si riprende il default, altrimenti
        # il calcolo esploderebbe mentre si sta ancora scrivendo. NON e' cosi' che si
        # toglie una persona dal nucleo: per quello c'e' la spunta.
        p = {c[0]: (c[2] if v is None else float(v)) for c, v in zip(CAMPI, valori)}
        p['simulazioni'] = int(max(200, min(20000, p['simulazioni'])))
        p['_nomi'] = nomi
        soglia = p['soglia']

        scen = scenari_applicabili(comp, nomi)
        if scenario not in [v for v, _ in scen]:
            scenario = 'base'      # la spunta e' cambiata prima che il dropdown si aggiornasse

        prof = profilo(p, scenario, comp)
        ris  = simula(p, prof)
        offsets, curve = scansione_uscita(p, comp)
        marg = margini(p, scenario, comp, soglia)

        ha_lei = 'lei' in comp
        redditi = p['reddito_lui'] + (p['reddito_lei'] if ha_lei else 0)
        risparmio = (redditi - p['spesa']) / 12
        chiave = 'entrambi' if ha_lei else 'solo'
        primo = prima_uscita(offsets, curve[chiave], soglia)

        ok = ris['prob'] >= soglia
        med = np.percentile(ris['finale'], 50)
        p05 = np.percentile(ris['finale'], 5)

        # L'eta' di TUTTI gli adulti, non solo di lui: la domanda di chi legge e'
        # «io a che eta' smetto», e con due persone le eta' sono due.
        adulti_k = ['lui', 'lei'] if ha_lei else ['lui']
        if primo is None:
            testo_fire, nota_fire = 'mai', f'sotto la soglia del {soglia:.0f}% a ogni età'
        else:
            quando = 'già da oggi' if not primo else f'fra {primo} anni'
            testo_fire = ' e '.join(f"{p[f'eta_{k}'] + primo:.0f}" for k in adulti_k) + ' anni'
            nota_fire = (f"{' e '.join(nomi[k] for k in adulti_k)} — {quando}"
                         if ha_lei else quando)

        # Tutti gli scenari applicabili a confronto, con gli stessi parametri.
        righe = [(lbl.split(' — ')[0],
                  simula(p, profilo(p, v, comp), n=max(1000, p['simulazioni'] // 2)))
                 for v, lbl in scen]
        deboli = [n for n, r in righe if r['prob'] < soglia]

        titolo_fire = ('Possono smettere di lavorare a' if ha_lei
                       else f"{nomi['lui']} può smettere a")

        # L'ordine di lettura: in cima i quattro numeri che rispondono alla domanda
        # («regge? a che eta' si smette? quanto resta alla fine?»), subito sotto la
        # vita intera del capitale — accumulo e poi decumulo fino all'eta' del
        # piano — e solo dopo il portafoglio da solo e la relazione in prosa.
        return html.Div([
            _titolo('Riepilogo del piano',
                    'I quattro numeri che rispondono alla domanda: regge o no.',
                    margine='0'),
            html.Div([
                _card('Piani riusciti', f"{ris['prob']:.1f}%",
                      f"soglia richiesta {soglia:.0f}%", VERDE if ok else ROSSO),
                _card(titolo_fire, testo_fire, nota_fire,
                      BLU if primo is not None else ROSSO),
                _card('Patrimonio a fine piano', _eur(med),
                      f"mediana a {p['orizzonte']:.0f} anni, in € di oggi · "
                      f"5° percentile {_eur(p05)}"),
                _card('Risparmio implicito', _eur(risparmio) + '/mese',
                      'reddito netto meno spesa',
                      BLU if risparmio >= 0 else ROSSO),
            ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap',
                      'marginBottom': '14px'}),

            _titolo(f'Crescita e decumulo del capitale — '
                    f"{etichetta_scenario(scenario, comp, nomi)}",
                    f"Dal patrimonio di oggi fino ai {p['orizzonte']:.0f} anni, con "
                    'versamenti, spese, pensioni e imposte. In euro di oggi.'
                    + (f" · Accumulo al {p['rendimento']:.2f}% per "
                       f"{prof['anno_pens']} anni, poi decumulo al "
                       f"{p['rendimento_pens']:.2f}%".replace('.', ',')
                       if p['rendimento_pens'] != p['rendimento'] else '')
                    + (' · ' + ' · '.join(prof['note']) if prof['note'] else '')),
            dcc.Graph(figure=_fig_ventaglio(p, ris, nomi),
                      config={'displayModeBar': False}),

            _blocco_proiezione(p, mercato),

            html.Div([
                html.Span('⚠ ', style={'color': GIALLO, 'fontWeight': '700'}),
                html.Span('Le pensioni impostate ('
                          + ', '.join(f"{_eur(p[f'pens_{k}'])} {nomi[k]}"
                                      for k in (['lui', 'lei'] if ha_lei else ['lui']))
                          + ', al netto) sono segnaposto: vanno sostituite con la stima '
                          'dell’estratto conto INPS. Sono il parametro che sposta di '
                          'più la data di uscita dal lavoro.',
                          style={'color': '#7a5c00'}),
            ], style={'fontSize': '11.5px', 'padding': '8px 12px', 'borderRadius': '8px',
                      'background': '#fdf8ec', 'border': '1px solid #f0e2bd',
                      'marginBottom': '12px', 'lineHeight': '1.5'}),

            relazione(p, comp, nomi, scenario, ris, righe, marg, soglia, offsets, curve,
                      mercato),

            _titolo('Quando si può smettere di lavorare',
                    'Probabilità che il piano regga, al variare della data di uscita. '
                    'Scenario base, parametri correnti.', margine='0'),
            dcc.Graph(figure=_fig_uscita(p, offsets, curve, soglia, nomi),
                      config={'displayModeBar': False}),

            _titolo('Tutti gli scenari a confronto'),
            _tabella_scenari(righe, soglia),
            html.Div(
                ('Scenari che non reggono la soglia: ' + ', '.join(deboli) +
                 '. Sono questi a dire di quanta copertura assicurativa c’è bisogno.')
                if deboli else
                'Tutti gli scenari reggono la soglia richiesta.',
                style={'fontSize': '11px', 'marginTop': '8px', 'padding': '9px 12px',
                       'borderRadius': '8px',
                       'background': '#fdf3f2' if deboli else '#f1f8f4',
                       'color': ROSSO if deboli else VERDE,
                       'border': f"1px solid {'#f3ddda' if deboli else '#d9ece1'}"}),

            _pannello_margini(p, marg, soglia, scenario, comp, nomi),
        ])
