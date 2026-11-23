"""
ARIMA+GARCH per utente — un file per utente, condiviso da Frontiera e dal job notturno.

File: sessions/<utente>/arima_garch.json (prefisso sincronizzato su R2)
    {"finestra": 250, "aggiornato": "23/11/2026 03:00",
     "asset": {"<nome>": {"mu": <giornaliero>, "vol": <giornaliera>,
                          "ticker": "...", "ultimo": "2026-11-21", "calcolato": "..."}}}

- Ogni asset è stimato da solo sugli ultimi FINESTRA rendimenti della sua serie
  (tutto current.json, non le date scelte in Frontiera): μ = media incondizionale
  dell'ARIMA(p,0,q) col miglior AIC, σ = GARCH(1,1) a un passo sui residui.
- Il job notturno (portafoglio/app.py, dopo l'aggiornamento dei dati utente) rifà
  tutti gli asset. Frontiera calcola soltanto quelli che mancano (asset nuovi o
  con ticker cambiato) e li aggiunge al file.
- La covarianza non si salva: si ricostruisce a ogni uso (mu_cov) con le
  correlazioni degli ultimi FINESTRA giorni, così vale per qualunque gruppo di asset.
"""
import threading
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import data_core

FINESTRA = 250            # giorni di rendimenti per asset (il «default»)
_NOME_FILE = 'arima_garch.json'
_LOCK = threading.Lock()  # lettura-fusione-scrittura del file, fra thread dello stesso processo


def percorso(username):
    return data_core.ROOT / 'sessions' / username / _NOME_FILE


def leggi(username):
    """Contenuto del file ({} se manca o è illeggibile)."""
    import json
    try:
        with open(percorso(username)) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def stima_asset(s, finestra=FINESTRA):
    """(mu, vol) giornalieri di una serie di rendimenti; None se non ci sono dati."""
    s = pd.Series(s).dropna().tail(finestra)
    if len(s) < 2:
        return None
    media = float(s.mean())
    if len(s) < 30:
        return media, float(s.std())
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return media, float(s.std())

    migliore, aic, residui = None, np.inf, (s - media).values
    for p in range(3):
        for q in range(3):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m = ARIMA(s.values, order=(p, 0, q)).fit()
                if m.aic < aic:
                    migliore, aic, residui = m, m.aic, np.asarray(m.resid)
            except Exception:
                pass

    # μ incondizionale = const / (1 − Σφ). Con Σφ vicino a 1 è mal identificata
    # (si torna alla media) e oltre 5 volte la media storica è un artefatto.
    mu = media
    if migliore is not None:
        try:
            nomi = list(getattr(migliore.model, 'param_names', []) or [])
            par = dict(zip(nomi, np.asarray(migliore.params)))
            den = 1.0 - sum(v for k, v in par.items() if k.startswith('ar.'))
            if 'const' in par and abs(den) > 0.15:
                cand = float(par['const'] / den)
                mu = media if (media != 0 and abs(cand) > 5 * abs(media) + 1e-5) else cand
            else:
                mu = float(np.mean(migliore.fittedvalues))
        except Exception:
            mu = media

    vol = float(np.std(residui))
    if len(residui) > 20:
        try:
            from arch import arch_model
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                g = arch_model(residui * 100, vol='Garch', p=1, q=1, rescale=False).fit(disp='off')
                vol = float(np.sqrt(g.forecast(horizon=1).variance.values[-1, 0])) / 100
        except Exception:
            pass
    if not np.isfinite(mu):
        mu = media
    if not np.isfinite(vol) or vol <= 0:
        vol = float(s.std())
    return mu, vol


def mancanti(dati, assets, tickers=None):
    """Asset senza stima nel file, o stimati quando avevano un altro ticker."""
    voci = (dati or {}).get('asset', {})
    out = []
    for a in assets:
        v = voci.get(a)
        if not v:
            out.append(a)
        elif tickers and tickers.get(a) and v.get('ticker') and v['ticker'] != tickers[a]:
            out.append(a)
    return out


def aggiorna(username, returns_df, assets, tickers=None, avanzamento=None):
    """Stima gli `assets` sui rendimenti completi `returns_df` e li scrive nel file,
    senza toccare gli altri asset già presenti. `avanzamento(fatti, totale)` per la
    barra. Restituisce il contenuto aggiornato del file."""
    assets = [a for a in assets if a in returns_df.columns]
    nuove = {}
    ora = datetime.now().strftime('%d/%m/%Y %H:%M')
    for i, a in enumerate(assets):
        s = returns_df[a].dropna()
        st = stima_asset(s)
        voce = {'mu': None, 'vol': None, 'calcolato': ora,
                'ultimo': str(s.index[-1])[:10] if len(s) else ''}
        if st is not None:
            voce['mu'], voce['vol'] = st
        if tickers and tickers.get(a):
            voce['ticker'] = tickers[a]
        nuove[a] = voce
        if avanzamento:
            avanzamento(i + 1, len(assets))
    with _LOCK:
        dati = leggi(username)          # riletto: un altro thread può aver scritto nel frattempo
        dati.setdefault('asset', {}).update(nuove)
        dati['finestra'] = FINESTRA
        dati['aggiornato'] = ora
        data_core.write_json_atomic(percorso(username), dati)
    return dati


def ripulisci(username, assets_presenti):
    """Toglie dal file gli asset che non stanno più in current.json."""
    with _LOCK:
        dati = leggi(username)
        voci = dati.get('asset', {})
        via = [a for a in voci if a not in set(assets_presenti)]
        if not via:
            return
        for a in via:
            voci.pop(a, None)
        data_core.write_json_atomic(percorso(username), dati)


def mu_vol(dati, asset):
    """(mu, vol) giornalieri di un asset dal file, o None."""
    v = (dati or {}).get('asset', {}).get(asset) or {}
    if v.get('mu') is None or v.get('vol') is None:
        return None
    return float(v['mu']), float(v['vol'])


def mu_cov(dati, returns_all, assets):
    """(mu Series, cov DataFrame) giornalieri per `assets`: σ dal file, correlazioni
    sugli ultimi FINESTRA giorni di returns_all (a coppie). Un asset senza stima nel
    file prende media e deviazione degli ultimi FINESTRA rendimenti."""
    mu, vol = {}, {}
    for a in assets:
        mv = mu_vol(dati, a)
        if mv is None:
            s = returns_all[a].dropna().tail(FINESTRA) if a in returns_all.columns else pd.Series(dtype=float)
            mv = (float(s.mean()), float(s.std())) if len(s) > 1 else (0.0, 0.0)
        mu[a], vol[a] = mv
    cols = [a for a in assets if a in returns_all.columns]
    corr = returns_all[cols].tail(FINESTRA).corr().reindex(index=assets, columns=assets)
    corr = np.nan_to_num(corr.values, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    v = np.array([vol[a] for a in assets])
    cov = np.diag(v) @ corr @ np.diag(v) + 1e-8 * np.eye(len(assets))
    return pd.Series(mu), pd.DataFrame(cov, index=assets, columns=assets)
