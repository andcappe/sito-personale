"""
Strategie Opzioni — App standalone
Pricing B-S + greche 1°/2°/3° livello, GEX/VEX/DEX, IV Surface,
Skew, Term Structure, scanner automatico di pattern.
Dati: yfinance (catena reale) + Black-Scholes europeo locale.
"""
import json
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from settings.browser_css import BROWSER_RESET_CSS
from navbar import make_navbar

_NU = no_update

# ─── App ─────────────────────────────────────────────────────────────────────
_EXT = [
    'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700'
    '&family=Inter:wght@400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=_EXT,
           requests_pathname_prefix='/opzioni/',
           routes_pathname_prefix='/opzioni/')

# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 1 — BLACK-SCHOLES + GRECHE
# ══════════════════════════════════════════════════════════════════════════════

def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if opt_type == 'call' else (K - S))
    try:
        d1, d2 = _d1d2(S, K, T, r, sigma)
        if opt_type == 'call':
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    except Exception:
        return 0.0

def bs_greeks(S, K, T, r, sigma, opt_type):
    """Greche di primo livello: Δ Γ Θ V ρ."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return dict(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    try:
        d1, d2 = _d1d2(S, K, T, r, sigma)
        nd1 = norm.pdf(d1)
        gamma = nd1 / (S * sigma * np.sqrt(T))
        vega  = S * nd1 * np.sqrt(T) / 100.0
        if opt_type == 'call':
            delta = norm.cdf(d1)
            theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
            rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
        else:
            delta = norm.cdf(d1) - 1.0
            theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
            rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
        return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
    except Exception:
        return dict(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

def bs_higher_greeks(S, K, T, r, sigma, opt_type):
    """Greche di 2°/3° livello via differenze finite: Vanna, Charm, Vomma, Color."""
    out = dict(vanna=0.0, charm=0.0, vomma=0.0, color=0.0)
    if T <= 1/365 or sigma <= 0:
        return out
    try:
        ds   = max(sigma * 0.01, 0.001)
        dt   = 1.0 / 365.0
        g0   = bs_greeks(S, K, T, r, sigma, opt_type)
        g_su = bs_greeks(S, K, T, r, sigma + ds, opt_type)
        g_sd = bs_greeks(S, K, T, r, sigma - ds, opt_type)
        g_t  = bs_greeks(S, K, T - dt, r, sigma, opt_type) if T > dt else g0

        # Vanna = ∂Δ/∂σ
        out['vanna'] = (g_su['delta'] - g_sd['delta']) / (2 * ds)
        # Charm = change in delta for 1 day passing (daily)
        out['charm'] = g_t['delta'] - g0['delta']
        # Vomma = ∂Vega/∂σ  (× 100 to express per 1% vol move)
        out['vomma'] = (g_su['vega'] - g_sd['vega']) / (2 * ds) * 0.01
        # Color = change in gamma for 1 day passing (daily)
        out['color'] = g_t['gamma'] - g0['gamma']
    except Exception:
        pass
    return out

def aggregate_all_greeks(legs, spot, T, r):
    """Greche aggregate (1° e 2°/3° livello) su tutte le gambe."""
    first  = dict(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    higher = dict(vanna=0.0, charm=0.0, vomma=0.0, color=0.0)
    for leg in legs:
        K, opt_type = float(leg.get('strike', spot)), leg.get('type', 'call')
        factor = int(leg.get('dir', 1)) * int(leg.get('qty', 1))
        iv = max(0.01, float(leg.get('iv', 0.25)))
        for k, v in bs_greeks(spot, K, T, r, iv, opt_type).items():
            first[k]  += factor * v
        for k, v in bs_higher_greeks(spot, K, T, r, iv, opt_type).items():
            higher[k] += factor * v
    return first, higher

def days_to_expiry_years(expiry_str):
    try:
        exp  = datetime.strptime(expiry_str, '%Y-%m-%d')
        days = (exp - datetime.now()).days
        return max(0.0, days / 365.0)
    except Exception:
        return 0.0

# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 2 — GEX / VEX / DEX
# ══════════════════════════════════════════════════════════════════════════════

def compute_exposure(calls_df, puts_df, spot, T, r):
    """
    GEX = Σ (OI_call × Γ_call - OI_put × Γ_put) × S² × 100
    VEX = Σ (OI_call × Vanna_call - OI_put × Vanna_put) × S × 100
    DEX = Σ (OI_call × Δ_call - OI_put × |Δ_put|) × 100
    Ritorna dict con per-strike e aggregati.
    """
    if T <= 0:
        return None

    rows = []
    for _, row in (calls_df.iterrows() if calls_df is not None else []):
        iv = float(row.get('impliedVolatility', 0) or 0)
        oi = int(row.get('openInterest', 0) or 0)
        K  = float(row['strike'])
        if iv <= 0 or oi <= 0:
            continue
        g = bs_greeks(spot, K, T, r, iv, 'call')
        h = bs_higher_greeks(spot, K, T, r, iv, 'call')
        rows.append({'strike': K, 'side': 'call', 'oi': oi,
                     'delta': g['delta'], 'gamma': g['gamma'],
                     'vanna': h['vanna'], 'iv': iv})

    for _, row in (puts_df.iterrows() if puts_df is not None else []):
        iv = float(row.get('impliedVolatility', 0) or 0)
        oi = int(row.get('openInterest', 0) or 0)
        K  = float(row['strike'])
        if iv <= 0 or oi <= 0:
            continue
        g = bs_greeks(spot, K, T, r, iv, 'put')
        h = bs_higher_greeks(spot, K, T, r, iv, 'put')
        rows.append({'strike': K, 'side': 'put', 'oi': oi,
                     'delta': g['delta'], 'gamma': g['gamma'],
                     'vanna': h['vanna'], 'iv': iv})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    strikes = sorted(df['strike'].unique())
    gex_per_strike, vex_per_strike, dex_per_strike = [], [], []

    for K in strikes:
        c = df[(df['strike'] == K) & (df['side'] == 'call')]
        p = df[(df['strike'] == K) & (df['side'] == 'put')]
        c_oi  = c['oi'].sum() if not c.empty else 0
        p_oi  = p['oi'].sum() if not p.empty else 0
        c_gam = c['gamma'].mean() if not c.empty else 0
        p_gam = p['gamma'].mean() if not p.empty else 0
        c_van = c['vanna'].mean() if not c.empty else 0
        p_van = p['vanna'].mean() if not p.empty else 0
        c_dlt = c['delta'].mean() if not c.empty else 0
        p_dlt = p['delta'].mean() if not p.empty else 0  # negative

        gex = (c_oi * c_gam - p_oi * p_gam) * spot ** 2 * 100
        vex = (c_oi * c_van - p_oi * p_van) * spot * 100
        dex = (c_oi * c_dlt + p_oi * p_dlt) * 100  # put delta already negative

        gex_per_strike.append({'strike': K, 'gex': gex})
        vex_per_strike.append({'strike': K, 'vex': vex})
        dex_per_strike.append({'strike': K, 'dex': dex})

    total_gex = sum(x['gex'] for x in gex_per_strike)
    total_vex = sum(x['vex'] for x in vex_per_strike)
    total_dex = sum(x['dex'] for x in dex_per_strike)

    return {
        'gex': gex_per_strike,
        'vex': vex_per_strike,
        'dex': dex_per_strike,
        'total_gex': total_gex,
        'total_vex': total_vex,
        'total_dex': total_dex,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 3 — IV SURFACE / SKEW / TERM STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_iv_surface(chains_dict, spot):
    """
    chains_dict: {expiry_str: {'calls': records, 'puts': records}}
    Ritorna {'z': matrix, 'strikes': list, 'expiries': list, 'dte': list}
    """
    all_strikes = set()
    for exp_data in chains_dict.values():
        for rec in exp_data.get('calls', []):
            all_strikes.add(rec['strike'])
        for rec in exp_data.get('puts', []):
            all_strikes.add(rec['strike'])

    strikes = sorted(all_strikes)
    expiries = sorted(chains_dict.keys())
    dte_list = []
    for exp in expiries:
        try:
            d = (datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days
            dte_list.append(max(0, d))
        except Exception:
            dte_list.append(0)

    # Near ATM strikes only (±30%)
    lo, hi = spot * 0.70, spot * 1.30
    strikes = [s for s in strikes if lo <= s <= hi]
    if not strikes:
        return None

    z = np.full((len(expiries), len(strikes)), np.nan)
    for i, exp in enumerate(expiries):
        exp_data = chains_dict[exp]
        iv_map = {}
        for rec in exp_data.get('calls', []):
            iv = rec.get('impliedVolatility', 0) or 0
            if iv > 0:
                iv_map[rec['strike']] = iv * 100
        for rec in exp_data.get('puts', []):
            iv = rec.get('impliedVolatility', 0) or 0
            if iv > 0 and rec['strike'] not in iv_map:
                iv_map[rec['strike']] = iv * 100
        for j, K in enumerate(strikes):
            if K in iv_map:
                z[i, j] = iv_map[K]

    return {
        'z': z.tolist(),
        'strikes': strikes,
        'expiries': expiries,
        'dte': dte_list,
    }

def compute_skew(calls_df, puts_df, spot, T, r):
    """IV Skew: IV put OTM vs IV call OTM per ogni strike — e 25Δ risk reversal."""
    result = {'strikes': [], 'call_iv': [], 'put_iv': [], 'skew': [], 'rr_25': None}
    if calls_df is None or puts_df is None:
        return result

    skew_rows = []
    # merge on strike
    for _, row in calls_df.iterrows():
        K  = float(row['strike'])
        iv = float(row.get('impliedVolatility', 0) or 0)
        if iv > 0:
            skew_rows.append({'strike': K, 'call_iv': iv * 100, 'put_iv': None})
    iv_map_put = {float(r['strike']): float(r.get('impliedVolatility', 0) or 0) * 100
                  for _, r in puts_df.iterrows() if (r.get('impliedVolatility') or 0) > 0}
    for row in skew_rows:
        row['put_iv'] = iv_map_put.get(row['strike'])

    skew_rows = [r for r in skew_rows if r['put_iv'] is not None]
    skew_rows.sort(key=lambda x: x['strike'])

    result['strikes'] = [r['strike'] for r in skew_rows]
    result['call_iv']  = [r['call_iv'] for r in skew_rows]
    result['put_iv']   = [r['put_iv'] for r in skew_rows]
    result['skew']     = [round(r['put_iv'] - r['call_iv'], 2) for r in skew_rows]

    # 25Δ Risk Reversal: find strikes nearest to delta ≈ 0.25 put / 0.25 call
    if T > 0:
        target_d = 0.25
        best_call = best_put = None
        best_cd = best_pd = 999
        for _, row in calls_df.iterrows():
            K  = float(row['strike'])
            iv = float(row.get('impliedVolatility', 0) or 0)
            if iv <= 0:
                continue
            d = abs(bs_greeks(spot, K, T, r, iv, 'call')['delta'] - target_d)
            if d < best_cd:
                best_cd  = d
                best_call = iv * 100
        for _, row in puts_df.iterrows():
            K  = float(row['strike'])
            iv = float(row.get('impliedVolatility', 0) or 0)
            if iv <= 0:
                continue
            d = abs(abs(bs_greeks(spot, K, T, r, iv, 'put')['delta']) - target_d)
            if d < best_pd:
                best_pd = d
                best_put = iv * 100
        if best_call and best_put:
            result['rr_25'] = round(best_put - best_call, 2)

    return result

def compute_term_structure(chains_dict, spot):
    """ATM IV per scadenza → curva Term Structure."""
    rows = []
    for exp, exp_data in sorted(chains_dict.items()):
        try:
            dte = max(0, (datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days)
        except Exception:
            dte = 0
        # ATM call IV
        best_iv = None
        best_dist = 999
        for rec in exp_data.get('calls', []):
            iv = rec.get('impliedVolatility', 0) or 0
            if iv <= 0:
                continue
            dist = abs(rec['strike'] - spot)
            if dist < best_dist:
                best_dist = dist
                best_iv   = iv * 100
        if best_iv:
            rows.append({'dte': dte, 'iv': best_iv, 'exp': exp})
    return rows

# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 4 — STRATEGY PRESETS + PAYOFF
# ══════════════════════════════════════════════════════════════════════════════

STRATEGY_PRESETS = {
    'custom':           [],
    'long_call':        [{'type':'call','dir':1,'sidx':0,'qty':1}],
    'long_put':         [{'type':'put', 'dir':1,'sidx':0,'qty':1}],
    'short_call':       [{'type':'call','dir':-1,'sidx':0,'qty':1}],
    'short_put':        [{'type':'put', 'dir':-1,'sidx':0,'qty':1}],
    'bull_call_spread': [{'type':'call','dir':1,'sidx':0,'qty':1},
                         {'type':'call','dir':-1,'sidx':2,'qty':1}],
    'bear_put_spread':  [{'type':'put', 'dir':1,'sidx':0,'qty':1},
                         {'type':'put', 'dir':-1,'sidx':-2,'qty':1}],
    'straddle':         [{'type':'call','dir':1,'sidx':0,'qty':1},
                         {'type':'put', 'dir':1,'sidx':0,'qty':1}],
    'strangle':         [{'type':'call','dir':1,'sidx':2,'qty':1},
                         {'type':'put', 'dir':1,'sidx':-2,'qty':1}],
    'butterfly':        [{'type':'call','dir':1,'sidx':-2,'qty':1},
                         {'type':'call','dir':-1,'sidx':0,'qty':2},
                         {'type':'call','dir':1,'sidx':2,'qty':1}],
    'iron_condor':      [{'type':'put', 'dir':1,'sidx':-4,'qty':1},
                         {'type':'put', 'dir':-1,'sidx':-2,'qty':1},
                         {'type':'call','dir':-1,'sidx':2,'qty':1},
                         {'type':'call','dir':1,'sidx':4,'qty':1}],
}
STRATEGY_LABELS = {
    'custom':           'Custom',
    'long_call':        'Long Call',
    'long_put':         'Long Put',
    'short_call':       'Short Call',
    'short_put':        'Short Put',
    'bull_call_spread': 'Bull Call Spread',
    'bear_put_spread':  'Bear Put Spread',
    'straddle':         'Straddle (long)',
    'strangle':         'Strangle (long)',
    'butterfly':        'Butterfly (call)',
    'iron_condor':      'Iron Condor',
}

def resolve_preset(preset_key, calls_df, puts_df, spot):
    template = STRATEGY_PRESETS.get(preset_key, [])
    if not template:
        return []

    def get_strikes_atm(df):
        if df is None or df.empty:
            return [], 0
        ss  = sorted(df['strike'].tolist())
        idx = int(np.argmin(np.abs(np.array(ss) - spot)))
        return ss, idx

    cs, ca = get_strikes_atm(calls_df)
    ps, pa = get_strikes_atm(puts_df)

    legs = []
    for t in template:
        ot      = t['type']
        strikes = cs if ot == 'call' else ps
        atm     = ca if ot == 'call' else pa
        df      = calls_df if ot == 'call' else puts_df
        idx     = max(0, min(len(strikes) - 1, atm + t['sidx'])) if strikes else 0
        strike  = float(strikes[idx]) if strikes else round(spot)

        premium, iv_val = 0.0, 0.25
        if df is not None and not df.empty:
            row = df[df['strike'] == strike]
            if not row.empty:
                bid = float(row['bid'].values[0] or 0)
                ask = float(row['ask'].values[0] or 0)
                mid = (bid + ask) / 2 if (bid + ask) > 0 else float(row['lastPrice'].values[0] or 0)
                premium = round(float(mid), 4) if mid > 0 else 0.0
                iv_raw  = row['impliedVolatility'].values[0]
                if iv_raw and not np.isnan(float(iv_raw)) and float(iv_raw) > 0:
                    iv_val = float(iv_raw)

        legs.append({'type': ot, 'dir': t['dir'], 'strike': strike,
                     'qty': t['qty'], 'premium': premium, 'iv': round(iv_val, 4)})
    return legs

def compute_payoff(legs, spot, T, r):
    """P&L a scadenza e attuale su range ±35% dello spot."""
    if not legs or spot <= 0:
        return None
    S_range     = np.linspace(spot * 0.65, spot * 1.35, 250)
    pnl_expiry  = np.zeros(250)
    pnl_current = np.zeros(250)
    net_cost    = 0.0

    for leg in legs:
        K        = float(leg.get('strike', spot))
        ot       = leg.get('type', 'call')
        factor   = int(leg.get('dir', 1)) * int(leg.get('qty', 1))
        premium  = float(leg.get('premium', 0))
        iv       = max(0.01, float(leg.get('iv', 0.25)))
        net_cost += factor * premium * 100

        intr = np.maximum(0, (S_range - K) if ot == 'call' else (K - S_range))
        pnl_expiry += factor * (intr - premium) * 100
        bs_vals     = np.array([bs_price(S, K, T, r, iv, ot) for S in S_range])
        pnl_current += factor * (bs_vals - premium) * 100

    # Break-even (zero crossing a scadenza)
    bes = []
    sign_chg = np.where(np.diff(np.sign(pnl_expiry)))[0]
    for i in sign_chg:
        denom = pnl_expiry[i + 1] - pnl_expiry[i]
        if denom != 0:
            be = S_range[i] - pnl_expiry[i] * (S_range[i + 1] - S_range[i]) / denom
            bes.append(round(float(be), 2))

    return {
        'S_range':     S_range.tolist(),
        'pnl_expiry':  pnl_expiry.tolist(),
        'pnl_current': pnl_current.tolist(),
        'net_cost':    round(net_cost, 2),
        'breakevens':  bes,
    }

def compute_heatmap(legs, spot, T, r):
    """P&L per (Δprice%, ΔIV%) — griglia 21×21."""
    if not legs or T <= 0:
        return None
    pc  = np.linspace(-25, 25, 21)
    ivc = np.linspace(-50, 50, 21)
    Z   = np.zeros((21, 21))
    for j, p in enumerate(pc):
        S_new = spot * (1 + p / 100)
        for i, iv_chg in enumerate(ivc):
            pnl = 0.0
            for leg in legs:
                K        = float(leg.get('strike', spot))
                ot       = leg.get('type', 'call')
                factor   = int(leg.get('dir', 1)) * int(leg.get('qty', 1))
                premium  = float(leg.get('premium', 0))
                iv       = max(0.01, float(leg.get('iv', 0.25)))
                iv_new   = max(0.001, iv * (1 + iv_chg / 100))
                pnl     += factor * (bs_price(S_new, K, T, r, iv_new, ot) - premium) * 100
            Z[i, j] = pnl
    return {'z': Z.tolist(), 'x': pc.tolist(), 'y': ivc.tolist()}

def compute_rolling_iv_greeks(hist_df, legs, expiry_str, r, window):
    """Rolling HV + greche aggregate della strategia su storia prezzi."""
    if hist_df is None or hist_df.empty or not legs:
        return None
    closes  = hist_df['Close']
    returns = closes.pct_change().dropna()
    hv      = returns.rolling(window).std() * np.sqrt(252)
    hv      = hv.dropna()

    exp_dt = None
    if expiry_str:
        try:
            exp_dt = datetime.strptime(expiry_str, '%Y-%m-%d')
        except Exception:
            pass

    dates, deltas, gammas, thetas, vegas = [], [], [], [], []
    for date in hv.index:
        S    = float(closes.loc[date])
        vol  = float(hv.loc[date])
        if vol <= 0 or np.isnan(vol):
            continue
        T_hist = max(0.0, (exp_dt - datetime(date.year, date.month, date.day)).days / 365.0) \
                 if exp_dt else 0.25
        agg = dict(delta=0.0, gamma=0.0, theta=0.0, vega=0.0)
        for leg in legs:
            K   = float(leg.get('strike', S))
            ot  = leg.get('type', 'call')
            fct = int(leg.get('dir', 1)) * int(leg.get('qty', 1))
            iv_use = (vol + max(0.01, float(leg.get('iv', 0.25)))) / 2
            g = bs_greeks(S, K, T_hist, r, iv_use, ot)
            for k in agg:
                agg[k] += fct * g[k]
        dates.append(date.strftime('%Y-%m-%d'))
        deltas.append(round(agg['delta'], 4))
        gammas.append(round(agg['gamma'], 6))
        thetas.append(round(agg['theta'], 4))
        vegas.append(round(agg['vega'],  4))

    price_series = [{'x': d.strftime('%Y-%m-%d'), 'y': float(closes.loc[d])}
                    for d in closes.dropna().index]
    hv_series    = [{'x': d.strftime('%Y-%m-%d'), 'y': float(hv.loc[d])}
                    for d in hv.index]

    return {
        'dates':  dates,
        'price':  price_series,
        'hv':     hv_series,
        'delta':  deltas,
        'gamma':  gammas,
        'theta':  thetas,
        'vega':   vegas,
    }

def compute_scanner_signals(exposure, skew_data, term_data):
    """
    Scanner automatico di pattern basato su:
    - GEX regime (positivo → mean rev, negativo → breakout)
    - IV Skew (deviazione da media storica)
    - Term Structure (contango vs backwardation)
    """
    signals = []
    regime  = 'neutral'

    if exposure:
        gex = exposure['total_gex']
        vex = exposure['total_vex']
        regime = 'positive' if gex > 0 else 'negative'

        if gex > 0:
            signals.append({
                'type': 'GEX+', 'level': 'info',
                'msg': f'GEX Positivo ({gex/1e6:.1f}M): i Dealer coprono al rialzo — '
                       'mercato pinned, preferire Mean Reversion.',
                'strategy': ['Iron Condor', 'Short Strangle', 'Calendar Spread'],
            })
        else:
            signals.append({
                'type': 'GEX-', 'level': 'warning',
                'msg': f'GEX Negativo ({gex/1e6:.1f}M): i Dealer amplificano i movimenti — '
                       'preferire Breakout / Long Vol.',
                'strategy': ['Long Straddle', 'Long Strangle', 'Long Call/Put'],
            })

        if abs(vex) > abs(gex) * 0.1:
            dir_vex = 'rialzo' if vex > 0 else 'ribasso'
            signals.append({
                'type': 'VEX', 'level': 'info',
                'msg': f'VEX {dir_vex} elevato: un calo della IV innescherà '
                       f'{"acquisti" if vex > 0 else "vendite"} forzati dai Market Maker (Vanna Rally/Trap).',
                'strategy': [],
            })

    if skew_data and skew_data.get('rr_25') is not None:
        rr = skew_data['rr_25']
        if rr > 5:
            signals.append({
                'type': 'SKEW', 'level': 'warning',
                'msg': f'25Δ Risk Reversal = +{rr:.1f}% → Put premium elevata: '
                       'paura di ribasso strutturale. Skew Put/Call deviato.',
                'strategy': ['Bull Put Spread', 'Short Put (su supporto)'],
            })
        elif rr < 1:
            signals.append({
                'type': 'SKEW', 'level': 'info',
                'msg': f'25Δ Risk Reversal = {rr:.1f}% → Skew piatto/Call premium: '
                       'mercato complacente o directional bull.',
                'strategy': ['Bear Call Spread', 'Long Put'],
            })

    if term_data and len(term_data) >= 2:
        # Check backwardation: IV breve > IV lunga
        sorted_td = sorted(term_data, key=lambda x: x['dte'])
        short_iv  = sorted_td[0]['iv']
        long_iv   = sorted_td[-1]['iv']
        if short_iv > long_iv * 1.05:
            signals.append({
                'type': 'TERM', 'level': 'warning',
                'msg': f'Term Structure in BACKWARDATION (IV breve {short_iv:.1f}% > '
                       f'lunga {long_iv:.1f}%): panico in atto. '
                       'Aspettare normalizzazione per vendere volatilità.',
                'strategy': ['Calendar Spread', 'Short Straddle (scad. lunga)'],
            })
        else:
            signals.append({
                'type': 'TERM', 'level': 'ok',
                'msg': f'Term Structure in CONTANGO (IV breve {short_iv:.1f}% < '
                       f'lunga {long_iv:.1f}%): condizioni normali.',
                'strategy': [],
            })

    return {'signals': signals, 'regime': regime}

# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 5 — LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

_CARD = {
    'background': '#ffffff', 'border': '1px solid #e0e6ef', 'borderRadius': '8px',
    'padding': '12px', 'marginBottom': '12px',
    'boxShadow': '0 1px 4px rgba(26,58,107,0.06)',
}

def _lbl(txt, color='#1a3a6b'):
    return html.Span(txt, style={
        'fontSize': '9px', 'fontWeight': '700', 'color': color,
        'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        'fontFamily': 'Inter, sans-serif',
    })

_TAB_STYLE = {
    'fontFamily': 'Inter, sans-serif', 'fontSize': '11px',
    'padding': '8px 14px',
}
_TAB_SEL = {
    **_TAB_STYLE, 'fontWeight': '700',
    'borderTop': '3px solid #1a3a6b', 'color': '#1a3a6b',
}

app.index_string = (
    '<!DOCTYPE html><html><head>{%metas%}'
    '<title>Strategie Opzioni — Andrea Cappelletti</title>'
    '{%favicon%}{%css%}<style>'
    + BROWSER_RESET_CSS +
    '.greek-chip{display:inline-flex;flex-direction:column;align-items:center;'
    'padding:6px 10px;border-radius:6px;min-width:68px;margin:3px}'
    '.signal-card{border-radius:6px;padding:8px 12px;margin-bottom:6px;'
    'border-left:4px solid #ccc}'
    '</style></head><body>{%app_entry%}'
    '<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'
)

# ─────────────────────────────────────────────────────────────────────────────
# FASE 1 — Cruscotto Macro / Sentiment (VIX & intermarket, EOD giornaliero)
# ─────────────────────────────────────────────────────────────────────────────
_SENT_TICKERS = {
    "^VIX":     "VIX — Volatilità S&P 500",
    "^VIX9D":   "VIX 9 giorni",
    "^VIX3M":   "VIX 3 mesi",
    "^VVIX":    "VVIX — Volatilità del VIX",
    "^SKEW":    "SKEW Index",
    "^VXN":     "VXN — Volatilità Nasdaq 100",
    "^RVX":     "RVX — Volatilità Russell 2000",
    "^TNX":     "Treasury 10Y (rendimento)",
    "DX-Y.NYB": "Dollar Index (DXY)",
    "SPY":      "S&P 500 (SPY)",
}


# Serie DERIVATE: non si scaricano, si calcolano dai ticker qui sopra. Sono
# selezionabili come indice principale e come indice di confronto.
_SENT_DERIVED = {
    "VIX9D-VIX": "Differenziale  VIX 9g − VIX 30g",
    "VIX9D/VIX": "Rapporto  VIX 9g / VIX 30g",
}


def _sent_label(key):
    """Etichetta di un ticker scaricato o di una serie derivata."""
    return _SENT_TICKERS.get(key) or _SENT_DERIVED.get(key) or key


def _add_derived(px):
    """Aggiunge a px le serie derivate della struttura a termine breve."""
    if "^VIX9D" in px.columns and "^VIX" in px.columns:
        px = px.copy()
        px["VIX9D-VIX"] = px["^VIX9D"] - px["^VIX"]
        px["VIX9D/VIX"] = px["^VIX9D"] / px["^VIX"].replace(0, np.nan)
    return px


def _attraversa_zero(s):
    """True se la serie cambia segno o passa vicinissima allo zero: in quel caso
    rapporti e rebase % su di essa esplodono e non vanno calcolati."""
    v = np.asarray(s.dropna().values, dtype=float)
    if v.size == 0:
        return True
    return bool(np.nanmin(v) <= 0 <= np.nanmax(v))


def download_sentiment():
    """2 anni EOD dei ticker macro/vol. (Il Put/Call CBOE via CSV pubblico non è
    più disponibile: si aggancia qui su px['Equity_PC'] se colleghi una sorgente.)"""
    tks = list(_SENT_TICKERS.keys())
    d = yf.download(tks, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        return d["Close"].copy()
    return d[["Close"]].rename(columns={"Close": tks[0]})


def _pct_change_n(s, n):
    s = s.dropna()
    if len(s) <= n:
        return None
    return (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100


def _sent_signal(key, val, pct):
    g = ("#e6f4ea", "#137333"); y = ("#fff3e0", "#e65100"); r = ("#fce8e6", "#c5221f"); nne = ("#f5f5f5", "#555")
    if key == "^VIX":
        if val >= 25: return "Premi molto gonfiati → vendi volatilità (CSP / Iron Condor)", g
        if val >= 20: return "Premi cari → favorevole ai venditori di opzioni", g
        if val < 14:  return "Premi compressi → compra protezione / strategie a debito", r
        return "Regime neutro (14–20)", nne
    if key == "^SKEW":
        if val >= 145: return "Coda sinistra estrema: istituzionali comprano Put", r
        if val >= 135: return "Skew elevato: rischio asimmetrico al ribasso", y
        return "Skew nella norma", nne
    if key == "^VVIX":
        if pct is not None and pct >= 80: return "Vol della vol alta → riduci la size (semaforo giallo)", y
        return "Stabile", nne
    if key == "^TNX":
        return "Filtro macro: se rompe i massimi relativi, riduci esposizione Nasdaq", nne
    return "", nne


def _build_sentiment_table(px):
    idx = px.dropna(how="all").index
    if len(idx) == 0:
        return html.Div("Nessun dato disponibile.", style={"padding": "12px", "fontSize": "12px"})

    th = {"padding": "8px 10px", "fontSize": "11px", "fontWeight": "700", "color": "#fff",
          "background": "#1a3a5c", "border": "1px solid #ddd", "position": "sticky", "top": "0"}
    heads = ["Indicatore", "Ultimo", "1g Δ%", "5g Δ%", "20g Δ%", "Perc. 2a", "Segnale operativo"]
    rows = [html.Tr([html.Th(h, style={**th, "textAlign": "left" if i in (0, 6) else "right"})
                     for i, h in enumerate(heads)])]

    def _chg(v):
        if v is None:
            return html.Td("—", style={"padding": "7px 10px", "textAlign": "right", "fontSize": "11px",
                                        "color": "#aaa", "border": "1px solid #eee"})
        c = ("#e6f4ea", "#137333") if v >= 0 else ("#fce8e6", "#c5221f")
        return html.Td(f"{v:+.1f}%", style={"padding": "7px 10px", "textAlign": "right", "fontSize": "11px",
                                            "background": c[0], "color": c[1], "border": "1px solid #eee"})

    def _row(name, val, c1, c5, c20, pct, sig, sigcol, fmt="{:.2f}"):
        return html.Tr([
            html.Td(name, style={"padding": "7px 10px", "fontSize": "11px", "fontWeight": "600",
                                 "border": "1px solid #eee", "whiteSpace": "nowrap",
                                 "position": "sticky", "left": "0", "background": "#fff"}),
            html.Td(fmt.format(val) if val is not None else "—",
                    style={"padding": "7px 10px", "textAlign": "right", "fontSize": "11px",
                           "fontWeight": "700", "border": "1px solid #eee"}),
            _chg(c1), _chg(c5), _chg(c20),
            html.Td(f"{pct:.0f}%" if pct is not None else "—",
                    style={"padding": "7px 10px", "textAlign": "right", "fontSize": "11px",
                           "border": "1px solid #eee",
                           "background": "#fff3e0" if (pct is not None and pct >= 70) else "#fff"}),
            html.Td(sig, style={"padding": "7px 10px", "fontSize": "10.5px", "border": "1px solid #eee",
                                "background": sigcol[0], "color": sigcol[1]}),
        ])

    # Term structure VIX/VIX3M (backwardation = venditori favoriti)
    if "^VIX" in px.columns and "^VIX3M" in px.columns:
        v, v3 = px["^VIX"].dropna(), px["^VIX3M"].dropna()
        com = v.index.intersection(v3.index)
        if len(com) > 5:
            ratio = float((v.reindex(com) / v3.reindex(com)).dropna().iloc[-1])
            if ratio > 1:
                sig, col = "BACKWARDATION → paura sul presente: momento migliore per vendere opzioni", ("#e6f4ea", "#137333")
            else:
                sig, col = "Contango (normale): struttura a termine regolare", ("#f5f5f5", "#555")
            rows.append(_row("Term Structure  VIX / VIX3M", ratio, None, None, None, None, sig, col, "{:.3f}"))

    for key, label in _SENT_TICKERS.items():
        if key not in px.columns:
            continue
        s = px[key].dropna()
        if s.empty:
            continue
        val = float(s.iloc[-1])
        pct = float((s < val).mean() * 100)
        sig, col = _sent_signal(key, val, pct)
        rows.append(_row(label, val, _pct_change_n(s, 1), _pct_change_n(s, 5), _pct_change_n(s, 20), pct, sig, col))

    return html.Table(rows, style={"borderCollapse": "collapse", "width": "100%",
                                   "fontFamily": "Inter, Arial, sans-serif"})


# Soglie di regime del VIX. Sono livelli ASSOLUTI: si disegnano solo sugli indici
# in scala VIX (vol. implicita annualizzata dell'S&P, stessa unità su ogni scadenza:
# 9 giorni, 1 mese = il VIX stesso, 3 mesi) e solo su assi in scala reale, mai sui
# grafici rebasati a 0%.
_VIX_LEVELS = [
    (15.0,  "#137333", "Calma — premi compressi"),
    (22.5,  "#e65100", "Tensione — premi cari"),
    (40.0,  "#c5221f", "Panico — stress di mercato"),
]
_VIX_SCALE = {"^VIX9D", "^VIX", "^VIX3M"}
# Tutti gli indici di volatilità implicita. Su questi la lettura degli oscillatori
# è ROVESCIATA rispetto all'azionario: valore alto = premi gonfiati = zona del
# VENDITORE di opzioni (verde), valore basso = premi compressi (rosso).
_VOL_TICKERS = {"^VIX", "^VVIX", "^SKEW", "^VIX9D", "^VIX3M", "^VXN", "^RVX"}


def _add_vix_levels(fig, row=None, col=None, brief=False, secondary_y=None):
    """Linee orizzontali 15 / 22,5 / 40 sul grafico del VIX."""
    for y, color, desc in _VIX_LEVELS:
        txt = f"{y:.1f}".replace(".0", "").replace(".", ",")
        fig.add_hline(
            y=y, line_color=color, line_dash="dash", line_width=1,
            annotation_text=txt if brief else f"{txt} · {desc}",
            annotation_position="top left",
            annotation_font=dict(size=8 if brief else 9, color=color),
            row=row, col=col, secondary_y=secondary_y,
        )


def _centra_soglie(fig, ratios, diffs, soglia_text, row=None, col=None):
    """Centra l'asse sinistro su 1 (rapporto) e il destro su 0 (differenziale), così
    le due soglie cadono alla STESSA altezza e le curve si confrontano a occhio: il
    rapporto sta sopra 1 esattamente quando il differenziale sta sopra 0, quindi con
    assi automatici le due curve taglierebbero la loro soglia in punti diversi e la
    lettura sarebbe ingannevole. Una riga sola marca il confine per entrambe."""
    r = np.concatenate([np.asarray(x, dtype=float) for x in ratios])
    d = np.concatenate([np.asarray(x, dtype=float) for x in diffs])
    mr = float(np.nanmax(np.abs(r - 1.0))) * 1.12 or 0.1
    md = float(np.nanmax(np.abs(d))) * 1.12 or 1.0
    fig.update_yaxes(range=[1 - mr, 1 + mr], title_text="Rapporto (sx)",
                     title_font=dict(size=9), row=row, col=col, secondary_y=False)
    fig.update_yaxes(range=[-md, md], title_text="Differenziale (dx)",
                     title_font=dict(size=9), showgrid=False,
                     row=row, col=col, secondary_y=True)
    fig.add_hline(y=1.0, line_color="#c5221f", line_dash="dash", line_width=1.2,
                  annotation_text=soglia_text, annotation_position="top left",
                  annotation_font=dict(size=9, color="#c5221f"),
                  row=row, col=col, secondary_y=False)


def _add_ratio_diff(fig, px, ticker, compare, row=None, col=None):
    """Per ogni serie di confronto: rapporto principale/confronto (asse sx, tinta unita)
    e differenziale principale − confronto (asse dx, tratteggiato, stesso colore).
    Ritorna False se non c'è nulla da disegnare."""
    label = _sent_label(ticker)
    ratios, diffs = [], []
    for j, tk in enumerate(compare):
        sub = px[[ticker, tk]].dropna()
        if sub.empty:
            continue
        c = _CMP_PAL[j % len(_CMP_PAL)]
        lbl = _sent_label(tk)
        diff = sub[ticker] - sub[tk]
        fig.add_trace(go.Scatter(x=diff.index, y=diff.values,
            name=f"Differenziale  {label} − {lbl}",
            line=dict(color=c, width=1.2, dash='dot'),
            hovertemplate="Differenziale: %{y:+.2f}<br>%{x|%d/%m/%y}<extra></extra>"),
            row=row, col=col, secondary_y=True)
        diffs.append(diff.dropna().values)
        # Il rapporto si calcola solo se il denominatore non passa per lo zero:
        # su una serie come il differenziale VIX 9g−30g esploderebbe a ±infinito.
        if _attraversa_zero(sub[tk]):
            continue
        ratio = sub[ticker] / sub[tk]
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values,
            name=f"Rapporto  {label} / {lbl}", line=dict(color=c, width=1.8),
            hovertemplate="Rapporto: %{y:.3f}<br>%{x|%d/%m/%y}<extra></extra>"),
            row=row, col=col, secondary_y=False)
        ratios.append(ratio.dropna().values)
    if not diffs:
        return False
    if not ratios:                     # nessun rapporto calcolabile: solo differenziali
        ratios = [np.array([1.0])]
    _centra_soglie(fig, ratios, diffs,
                   "1,00 / 0 — sopra: il principale sta più in alto del confronto",
                   row=row, col=col)
    return True


def _vix_term_chart(px, short="^VIX9D", long_="^VIX"):
    """Struttura a termine breve: rapporto e differenziale fra VIX 9 giorni e VIX 30
    giorni, sovrapposti su due assi Y (il rapporto sta intorno a 1, il differenziale
    intorno a 0: senza due assi uno dei due sarebbe una riga piatta).
    Sopra la soglia è BACKWARDATION: la paura è sul presente, i premi a breve sono
    gonfiati rispetto a quelli a un mese."""
    if short not in px.columns or long_ not in px.columns:
        return go.Figure()
    sub = px[[short, long_]].dropna()
    if sub.empty:
        return go.Figure()
    ratio = sub[short] / sub[long_]
    diff  = sub[short] - sub[long_]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, name="Rapporto  VIX 9g / VIX 30g",
        line=dict(color="#1a3a5c", width=1.8),
        hovertemplate="Rapporto: %{y:.3f}<br>%{x|%d/%m/%y}<extra></extra>"),
        secondary_y=False)
    fig.add_trace(go.Scatter(x=diff.index, y=diff.values, name="Differenziale  VIX 9g − VIX 30g",
        line=dict(color="#e65100", width=1.4),
        hovertemplate="Differenziale: %{y:+.2f}<br>%{x|%d/%m/%y}<extra></extra>"),
        secondary_y=True)

    _centra_soglie(fig, [ratio.values], [diff.values],
                   "1,00 / 0 — sopra: BACKWARDATION (premi a breve gonfiati)")

    ultimo_r, ultimo_d = float(ratio.iloc[-1]), float(diff.iloc[-1])
    regime = "BACKWARDATION" if ultimo_r > 1 else "Contango"
    fig.update_layout(
        title=dict(text=f"Struttura a termine breve — oggi: {regime} "
                        f"(rapporto {ultimo_r:.3f} · differenziale {ultimo_d:+.2f})",
                   font=dict(size=12, color="#1a3a5c"), x=0.01),
        height=300, hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        margin=dict(t=64, b=20, l=52, r=52),
        paper_bgcolor="white", plot_bgcolor="#f8f9fb",
        font=dict(family="Inter, sans-serif", size=10))
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    return fig


def _build_sentiment_charts(px):
    """Griglia di grafici (storico 2 anni) per ogni indice scaricato."""
    series = [(k, _SENT_TICKERS[k]) for k in _SENT_TICKERS
              if k in px.columns and not px[k].dropna().empty]
    if not series:
        return go.Figure()
    cols = 3
    rows_n = (len(series) + cols - 1) // cols
    fig = make_subplots(rows=rows_n, cols=cols,
                        subplot_titles=[lbl for _, lbl in series],
                        vertical_spacing=0.09, horizontal_spacing=0.06)
    for i, (key, lbl) in enumerate(series):
        r, c = i // cols + 1, i % cols + 1
        s = px[key].dropna()
        is_vol = key in _VOL_TICKERS
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color="#c5221f" if is_vol else "#1a3a5c", width=1.4),
            fill="tozeroy" if is_vol else None,
            fillcolor="rgba(197,34,31,0.06)",
            hovertemplate=lbl + ": %{y:.2f}<br>%{x|%d/%m/%y}<extra></extra>",
        ), row=r, col=c)
        if key in _VIX_SCALE:
            _add_vix_levels(fig, row=r, col=c, brief=True)
    fig.update_layout(height=rows_n * 200, showlegend=False,
                      margin=dict(t=34, b=18, l=44, r=12),
                      paper_bgcolor="white", plot_bgcolor="#f8f9fb",
                      font=dict(family="Inter, sans-serif", size=10))
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    for a in fig.layout.annotations:
        a.font.size = 11
        a.font.color = "#1a3a5c"
    return fig


# Tipi di indicatore selezionabili (facile aggiungerne di nuovi: qui + in _sent_detail_chart)
_IND_TYPES = {
    'rsi':    'RSI',
    'macd':   'MACD',
    'stoch':  'Stocastico',
    'ivrank': 'IV Rank',
    'ivpct':  'IV Percentile',
    'roc':    'Momentum ROC',
    'zscore': 'Z-Score',
    'perc':   'Percentile',
    'bb':     'Bollinger %B',
    'vol':    'Volatilità realizz.',
    'dist':   'Distanza da SMA %',
    'dd':     'Drawdown %',
}
_IND_DEFAULT_PERIOD = {'rsi': 14, 'macd': 12, 'stoch': 14, 'ivrank': 252, 'ivpct': 252,
                       'roc': 20, 'zscore': 252,
                       'perc': 252, 'bb': 20, 'vol': 20, 'dist': 50, 'dd': 252}

# Zone dell'IV Rank / IV Percentile lette dal punto di vista del VENDITORE di premio:
# in alto i premi sono cari (si vende), in basso sono compressi (si compra). È il
# rovescio della lettura azionaria "alto = ipercomprato = male" — su una misura di
# volatilità implicita quella lettura sarebbe fuorviante.
_IV_ZONE = [
    (75.0, '#137333', 'Premi cari → vendi premio'),
    (50.0, '#999999', 'Mediana'),
    (25.0, '#c5221f', 'Premi compressi → compra opzioni'),
]


def _iv_rank(s, n=252):
    """IV Rank: dove sta il valore di oggi nel RANGE min–max del periodo (0–100).
    Formula: (oggi − min) / (max − min) × 100."""
    lo, hi = s.rolling(n).min(), s.rolling(n).max()
    return (s - lo) / (hi - lo).replace(0, np.nan) * 100


def _iv_percentile(s, n=252):
    """IV Percentile: quante sedute del periodo hanno avuto un valore INFERIORE a
    quello di oggi, in % (0–100). Diverso dall'IV Rank: non guarda solo gli estremi,
    guarda tutta la distribuzione — un singolo picco storico gonfia il range e
    schiaccia l'IV Rank, mentre l'IV Percentile resta rappresentativo."""
    return s.rolling(n).apply(lambda w: (w < w[-1]).mean() * 100, raw=True)


def _add_iv_zones(fig, row, col):
    for y, color, desc in _IV_ZONE:
        fig.add_hline(y=y, line_color=color, line_dash='dot', line_width=1,
                      annotation_text=f'{y:.0f} · {desc}', annotation_position='top left',
                      annotation_font=dict(size=8, color=color), row=row, col=col)


def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(s, fast=12, slow=26, sig=9):
    macd = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return macd, macd.ewm(span=sig, adjust=False).mean()


def _stoch(s, n=14, d=3):
    lo, hi = s.rolling(n).min(), s.rolling(n).max()
    k = (s - lo) / (hi - lo).replace(0, np.nan) * 100
    return k, k.rolling(d).mean()


_CMP_PAL = ["#c5221f", "#137333", "#e65100", "#6a1b9a", "#00838f", "#5d4037"]


def _sent_detail_chart(px, ticker, slots, compare=None, scala='abs', smooth=1):
    """Grafico grande di un indice + un pannello per ogni slot (tipo, periodo).
    slots = lista di (tipo, periodo). compare = altri indici da sovrapporre.
    smooth = giorni della media mobile di smoothing sulle serie del pannello
    principale (1 = nessuno smoothing, serie grezze).
    scala:
      'abs'     = valori assoluti come scaricati (confronto sull'asse destro);
      'pct'     = variazione % dal 1° giorno comune, tutti sullo stesso asse;
      'rapdiff' = rapporto (asse sx) e differenziale (asse dx) fra il principale e
                  ogni confronto, con le soglie 1 e 0 allineate alla stessa altezza.
    'rapdiff' richiede almeno un indice di confronto: senza, ricade su 'abs'."""
    if ticker is None or ticker not in px.columns:
        return go.Figure()
    slots = [(t, int(p)) for t, p in (slots or []) if t in _IND_TYPES and p][:3]
    compare = [c for c in (compare or []) if c in px.columns and c != ticker]
    if scala == 'rapdiff' and not compare:
        scala = 'abs'          # senza confronto non c'è nessun rapporto da fare
        manca_confronto = True
    else:
        manca_confronto = False
    pct     = (scala == 'pct')
    rapdiff = (scala == 'rapdiff')
    # Smoothing: media mobile sulle SERIE degli indici del pannello principale.
    # Gli indicatori sotto restano sui dati grezzi (`s`): su una serie smussata
    # IV Rank, stocastico & co. sottostimerebbero proprio gli estremi che servono.
    try:
        k = int(smooth or 1)
    except (TypeError, ValueError):
        k = 1
    k = max(1, min(k, 250))
    pxs = px.rolling(k, min_periods=1).mean() if k > 1 else px
    s = px[ticker].dropna()                 # grezza → indicatori
    s_top = pxs[ticker].dropna()            # smussata → pannello principale
    label = _sent_label(ticker)
    suffisso = f"  ·  media mobile {k} gg" if k > 1 else ""
    nsub = len(slots)
    heights = ([0.55] + [0.45 / nsub] * nsub) if nsub else [1.0]
    # Doppio asse quando le due grandezze non condividono la scala: valori assoluti
    # con confronto (VIX ~18 vs SPY ~600) e rapporto+differenziale (~1 vs ~0).
    # In variazione % l'asse è uno solo: è tutto il punto del rebase.
    due_assi = (bool(compare) and not pct) or rapdiff
    if rapdiff:
        top_title = f"{label} — rapporto (sx) e differenziale (dx) vs confronto"
    elif pct:
        top_title = (f"{label} vs confronto — variazione % dal 1° giorno comune"
                     if compare else f"{label} — variazione % dal 1° giorno")
    else:
        top_title = (f"{label} (asse sx)  vs  confronto in valore assoluto (asse dx)"
                     if compare else label)
        if manca_confronto:
            top_title += "  ·  scegli un indice di confronto per rapporto e differenziale"
    top_title += suffisso
    titles = [top_title] + [f"{_IND_TYPES[t]} ({p})" for t, p in slots]
    specs = [[{"secondary_y": due_assi}]] + [[{"secondary_y": False}]] * nsub
    fig = make_subplots(rows=1 + nsub, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        row_heights=heights, subplot_titles=titles, specs=specs)

    if rapdiff:
        _add_ratio_diff(fig, pxs, ticker, compare, row=1, col=1)
    elif pct:
        # Rebase a 0% dal 1° giorno in cui TUTTE le serie mostrate hanno un dato.
        cols = [ticker] + compare
        sub = pxs[cols].dropna()
        if not sub.empty:
            # Una serie che attraversa lo zero (es. il differenziale VIX 9g−30g) non
            # è ribasabile in %: si dividerebbe per un valore vicino a zero e la curva
            # schizzerebbe a migliaia di punti percentuali. La si esclude e lo si dice.
            plottabili = [c for c in cols if not _attraversa_zero(sub[c])]
            escluse    = [c for c in cols if c not in plottabili]
            norm = (sub[plottabili] / sub[plottabili].iloc[0] - 1) * 100
            for j, tk in enumerate(plottabili):
                principale = (tk == ticker)
                fig.add_trace(go.Scatter(x=norm.index, y=norm[tk].values, name=_sent_label(tk),
                    line=dict(color='#1a3a5c' if principale
                              else _CMP_PAL[(j - 1) % len(_CMP_PAL)],
                              width=2.3 if principale else 1.5)), row=1, col=1)
            fig.add_hline(y=0, line_color='#999', line_dash='dot', line_width=1, row=1, col=1)
            fig.update_yaxes(title_text='Variazione %', title_font=dict(size=9), row=1, col=1)
            if escluse:
                fig.add_annotation(
                    text='Escluse dal rebase % (attraversano lo zero): '
                         + ', '.join(_sent_label(c) for c in escluse)
                         + ' — usa Valore assoluto',
                    xref='x domain', yref='y domain', x=0.01, y=0.02, showarrow=False,
                    font=dict(size=9, color='#c5221f'), align='left', row=1, col=1)
    else:
        # Valori assoluti, come scaricati (smussati se hai impostato una media).
        if k > 1:
            # la serie grezza resta sotto, in trasparenza: si vede cosa toglie la media
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name=f'{label} (grezzo)',
                line=dict(color='#9fb0c6', width=0.8)), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=s_top.index, y=s_top.values, mode='lines',
            name=label + (f' · media {k}gg' if k > 1 else ''),
            line=dict(color='#1a3a5c', width=1.7 if not compare else 2.3)),
            row=1, col=1, secondary_y=False)
        if compare:
            for j, tk in enumerate(compare):
                c = pxs[tk].dropna()
                fig.add_trace(go.Scatter(x=c.index, y=c.values,
                    name=_sent_label(tk),
                    line=dict(color=_CMP_PAL[j % len(_CMP_PAL)], width=1.5)),
                    row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text=label, title_font=dict(size=9),
                             row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text='Confronto (valore assoluto)', title_font=dict(size=9),
                             row=1, col=1, secondary_y=True)
        else:
            fig.add_trace(go.Scatter(x=s_top.index, y=s_top.rolling(50).mean(), name='SMA 50',
                line=dict(color='#f0a500', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=s_top.index, y=s_top.rolling(200).mean(), name='SMA 200',
                line=dict(color='#c5221f', width=1, dash='dot')), row=1, col=1)
        # Soglie di regime: solo sugli indici in scala VIX e solo qui, dove l'asse
        # è in scala reale (in variazione % i livelli assoluti non varrebbero).
        if ticker in _VIX_SCALE:
            _add_vix_levels(fig, row=1, col=1, secondary_y=False if due_assi else None)
    for i, (kind, period) in enumerate(slots):
        r = 2 + i
        if kind == 'rsi':
            v = _rsi(s, period)
            fig.add_trace(go.Scatter(x=v.index, y=v.values, line=dict(color='#6a1b9a', width=1.2),
                showlegend=False), row=r, col=1)
            fig.add_hline(y=70, line_color='#c5221f', line_dash='dot', line_width=1, row=r, col=1)
            fig.add_hline(y=30, line_color='#137333', line_dash='dot', line_width=1, row=r, col=1)
        elif kind == 'stoch':
            k, d_ = _stoch(s, period)
            fig.add_trace(go.Scatter(x=k.index, y=k.values, line=dict(color='#1565c0', width=1.2),
                name='%K', showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=d_.index, y=d_.values, line=dict(color='#e65100', width=1),
                name='%D', showlegend=False), row=r, col=1)
            # Colori delle bande per SIGNIFICATO, non per posizione: sul VIX & co. la
            # zona alta è "premi cari → vendi" (verde), non "ipercomprato" (rosso).
            if ticker in _VOL_TICKERS:
                hi_col, lo_col = '#137333', '#c5221f'      # 80 vendi vol, 20 compra
            else:
                hi_col, lo_col = '#c5221f', '#137333'      # 80 ipercomprato, 20 ipervenduto
            fig.add_hline(y=80, line_color=hi_col, line_dash='dot', line_width=1, row=r, col=1)
            fig.add_hline(y=20, line_color=lo_col, line_dash='dot', line_width=1, row=r, col=1)
        elif kind in ('ivrank', 'ivpct'):
            v = _iv_rank(s, period) if kind == 'ivrank' else _iv_percentile(s, period)
            fig.add_trace(go.Scatter(x=v.index, y=v.values,
                line=dict(color='#00695c' if kind == 'ivrank' else '#6a1b9a', width=1.3),
                fill='tozeroy',
                fillcolor='rgba(0,105,92,0.06)' if kind == 'ivrank' else 'rgba(106,27,154,0.06)',
                showlegend=False), row=r, col=1)
            _add_iv_zones(fig, r, 1)
            fig.update_yaxes(range=[0, 100], row=r, col=1)
        elif kind == 'macd':
            macd, sg = _macd(s, period, round(period * 26 / 12), max(2, round(period * 9 / 12)))
            fig.add_trace(go.Bar(x=(macd - sg).index, y=(macd - sg).values,
                marker_color='rgba(120,120,120,0.35)', showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=macd.index, y=macd.values, line=dict(color='#1a3a5c', width=1.2),
                showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=sg.index, y=sg.values, line=dict(color='#f0a500', width=1),
                showlegend=False), row=r, col=1)
        elif kind == 'roc':
            v = s.pct_change(period) * 100
            fig.add_trace(go.Scatter(x=v.index, y=v.values, line=dict(color='#00838f', width=1.2),
                showlegend=False), row=r, col=1)
            fig.add_hline(y=0, line_color='#999', line_dash='dot', line_width=1, row=r, col=1)
        elif kind == 'zscore':
            v = (s - s.rolling(period).mean()) / s.rolling(period).std()
            fig.add_trace(go.Scatter(x=v.index, y=v.values, line=dict(color='#e65100', width=1.2),
                showlegend=False), row=r, col=1)
            for yv, cc in ((2, '#c5221f'), (0, '#999'), (-2, '#137333')):
                fig.add_hline(y=yv, line_color=cc, line_dash='dot', line_width=1, row=r, col=1)
        elif kind == 'perc':
            v = s.rolling(period).apply(lambda w: (w < w[-1]).mean() * 100, raw=True)
            fig.add_trace(go.Scatter(x=v.index, y=v.values, line=dict(color='#2e7d32', width=1.2),
                showlegend=False), row=r, col=1)
        elif kind == 'bb':
            ma, sd = s.rolling(period).mean(), s.rolling(period).std()
            b = (s - (ma - 2 * sd)) / ((ma + 2 * sd) - (ma - 2 * sd)).replace(0, np.nan) * 100
            fig.add_trace(go.Scatter(x=b.index, y=b.values, line=dict(color='#00695c', width=1.2),
                showlegend=False), row=r, col=1)
            fig.add_hline(y=100, line_color='#c5221f', line_dash='dot', line_width=1, row=r, col=1)
            fig.add_hline(y=0, line_color='#137333', line_dash='dot', line_width=1, row=r, col=1)
        elif kind == 'vol':
            v = s.pct_change().rolling(period).std() * (252 ** 0.5) * 100
            fig.add_trace(go.Scatter(x=v.index, y=v.values, line=dict(color='#c5221f', width=1.2),
                fill='tozeroy', fillcolor='rgba(197,34,31,0.06)', showlegend=False), row=r, col=1)
        elif kind == 'dist':
            v = (s / s.rolling(period).mean() - 1) * 100
            fig.add_trace(go.Scatter(x=v.index, y=v.values, line=dict(color='#5d4037', width=1.2),
                showlegend=False), row=r, col=1)
            fig.add_hline(y=0, line_color='#999', line_dash='dot', line_width=1, row=r, col=1)
        elif kind == 'dd':
            v = (s / s.rolling(period, min_periods=1).max() - 1) * 100
            fig.add_trace(go.Scatter(x=v.index, y=v.values, line=dict(color='#b71c1c', width=1.2),
                fill='tozeroy', fillcolor='rgba(183,28,28,0.08)', showlegend=False), row=r, col=1)
    fig.update_layout(height=360 + nsub * 150, showlegend=True,
        legend=dict(orientation='h', y=1.03, x=0, font=dict(size=10)),
        margin=dict(t=44, b=20, l=48, r=15), paper_bgcolor='white', plot_bgcolor='#f8f9fb',
        font=dict(family='Inter, sans-serif', size=10), hovermode='x unified')
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    for a in fig.layout.annotations:
        a.font.size = 11
        a.font.color = '#1a3a5c'
    return fig


def _sent_slot(i, dtype='none', period=14):
    """Uno slot indicatore: dropdown tipo + input periodo."""
    return html.Div([
        dcc.Dropdown(id=f'opt-sent-t{i}',
            options=[{'label': '— nessuno', 'value': 'none'}]
                    + [{'label': v, 'value': k} for k, v in _IND_TYPES.items()],
            value=dtype, clearable=False,
            style={'width': '140px', 'fontSize': '11px'}),
        dcc.Input(id=f'opt-sent-p{i}', type='number', value=period, min=2, max=500, step=1,
            style={'width': '56px', 'fontSize': '11px', 'marginLeft': '6px', 'padding': '3px 5px',
                   'border': '1px solid #aaa', 'borderRadius': '3px'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '12px'})


app.layout = html.Div([
    make_navbar('Opzioni'),

    # ── Stores ────────────────────────────────────────────────────────────────
    dcc.Store(id='opt-stock-store'),
    dcc.Store(id='opt-chain-store'),
    dcc.Store(id='opt-surface-store'),
    dcc.Store(id='opt-hist-store'),
    dcc.Store(id='opt-legs-store', data=[]),
    dcc.Store(id='opt-rf-store',   data=3.5),
    dcc.Store(id='opt-expiry-store'),

    html.Div([

        # ── Barra ricerca ─────────────────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Input(id='opt-ticker-input', type='text', value='AAPL',
                          placeholder='Ticker US (AAPL, SPY…)',
                          style={
                              'width': '160px', 'fontSize': '13px', 'fontWeight': '700',
                              'border': '2px solid #1a3a6b', 'borderRadius': '5px',
                              'padding': '7px 10px', 'fontFamily': 'Inter,sans-serif',
                              'textTransform': 'uppercase',
                          }),
                html.Button(
                    [html.I(className='fa-solid fa-magnifying-glass',
                             style={'marginRight': '5px'}), 'Cerca'],
                    id='opt-search-btn', n_clicks=0,
                    style={
                        'background': '#1a3a6b', 'color': 'white', 'border': 'none',
                        'padding': '8px 18px', 'borderRadius': '5px', 'cursor': 'pointer',
                        'fontWeight': '700', 'fontSize': '12px',
                        'fontFamily': 'Inter,sans-serif',
                    }),
                dcc.Loading(type='dot',
                            children=html.Div(id='opt-stock-info',
                                              style={'display': 'flex', 'alignItems': 'center',
                                                     'gap': '10px', 'flexWrap': 'wrap'})),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px',
                      'flexWrap': 'wrap'}),
            html.Div([
                _lbl('Risk-Free %:'),
                dcc.Input(id='opt-rf-input', type='number', value=3.5,
                          min=0, max=20, step=0.1, debounce=True,
                          style={'width': '55px', 'fontSize': '11px',
                                 'border': '1px solid #aaa', 'borderRadius': '3px',
                                 'padding': '3px 5px', 'marginLeft': '4px'}),
            ], style={'display': 'flex', 'alignItems': 'center',
                      'marginLeft': 'auto', 'gap': '4px'}),
        ], style={
            'display': 'flex', 'alignItems': 'center', 'padding': '8px 0',
            'marginBottom': '12px', 'flexWrap': 'wrap', 'gap': '8px',
        }),

        # ── Corpo: builder sx | payoff dx ─────────────────────────────────────
        html.Div([

            # Pannello sinistro
            html.Div([

                html.Div([
                    html.Div([
                        _lbl('Strategia'),
                        dcc.Dropdown(
                            id='opt-strategy-select',
                            options=[{'label': v, 'value': k}
                                     for k, v in STRATEGY_LABELS.items()],
                            value='long_call', clearable=False,
                            style={'fontSize': '11px', 'marginTop': '3px'},
                        ),
                    ], style={'flex': '1', 'marginRight': '8px'}),
                    html.Div([
                        _lbl('Scadenza'),
                        dcc.Dropdown(id='opt-expiry-select', options=[], value=None,
                                     placeholder='Cerca ticker…',
                                     style={'fontSize': '11px', 'marginTop': '3px',
                                            'minWidth': '130px'}),
                    ], style={'flex': '1'}),
                ], style={**_CARD, 'display': 'flex', 'gap': '8px'}),

                html.Div([
                    html.Div([
                        _lbl('Gambe'),
                        html.Button('+ Gamba', id='opt-add-leg-btn', n_clicks=0, style={
                            'marginLeft': 'auto', 'background': '#1a3a6b', 'color': 'white',
                            'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer',
                            'fontSize': '9px', 'fontWeight': '700', 'padding': '3px 8px',
                        }),
                    ], style={'display': 'flex', 'alignItems': 'center',
                              'marginBottom': '6px'}),
                    html.Div([
                        html.Div(_lbl('Tipo'),    style={'width': '18%'}),
                        html.Div(_lbl('L/S'),     style={'width': '10%'}),
                        html.Div(_lbl('Strike'),  style={'width': '22%'}),
                        html.Div(_lbl('Qty'),     style={'width': '12%'}),
                        html.Div(_lbl('Prem.'),   style={'width': '18%'}),
                        html.Div(_lbl('IV%'),     style={'width': '13%'}),
                        html.Div('', style={'width': '7%'}),
                    ], style={
                        'display': 'flex', 'alignItems': 'center',
                        'background': '#eaf4fb', 'borderTop': '2px solid #2e6da4',
                        'borderBottom': '1px solid #aed6f1', 'padding': '3px 0',
                    }),
                    html.Div(id='opt-legs-table'),
                ], style=_CARD),

                # Greche 1° livello
                html.Div([
                    _lbl('Greche Strategia'),
                    html.Div(id='opt-greeks-display', style={'marginTop': '6px'}),
                ], style=_CARD),

                # Greche 2°/3° livello
                html.Div([
                    _lbl('Vanna · Charm · Vomma · Color', '#4a1a7c'),
                    html.Div(id='opt-higher-greeks-display', style={'marginTop': '6px'}),
                ], style=_CARD),

                # Riepilogo
                html.Div(id='opt-summary-card', style=_CARD),

            ], style={'width': '32%', 'paddingRight': '16px',
                      'overflowY': 'auto', 'maxHeight': '78vh'}),

            # Pannello destro — payoff
            html.Div([
                html.Div([
                    dcc.RadioItems(
                        id='opt-payoff-mode',
                        options=[
                            {'label': '📅 A scadenza', 'value': 'expiry'},
                            {'label': '🕐 Valore attuale B-S', 'value': 'current'},
                            {'label': '📊 Entrambi', 'value': 'both'},
                        ],
                        value='both', inline=True,
                        inputStyle={'marginRight': '4px'},
                        labelStyle={'marginRight': '14px', 'fontSize': '11px',
                                    'fontWeight': '600', 'cursor': 'pointer'},
                    ),
                ], style={
                    'padding': '6px 10px', 'background': '#f8fafd',
                    'border': '1px solid #e0e6ef', 'borderRadius': '6px',
                    'marginBottom': '8px',
                }),
                dcc.Loading(type='circle', children=[
                    dcc.Graph(id='opt-payoff-chart', style={'height': '360px'},
                              config={'displayModeBar': False}),
                ]),
            ], style={'width': '68%'}),

        ], style={'display': 'flex', 'marginBottom': '16px'}),

        # ── Tabs inferiori ─────────────────────────────────────────────────────
        dcc.Tabs(id='opt-tabs', value='sensitivity', children=[

            dcc.Tab(label='Sensitivity', value='sensitivity',
                    style=_TAB_STYLE, selected_style=_TAB_SEL,
                    children=html.Div([
                        html.Div([
                            dcc.Loading(type='circle', children=dcc.Graph(
                                id='opt-heatmap', style={'height': '340px'},
                                config={'displayModeBar': False})),
                        ], style={'width': '50%', 'paddingRight': '8px'}),
                        html.Div([
                            dcc.Loading(type='circle', children=dcc.Graph(
                                id='opt-greeks-chart', style={'height': '340px'},
                                config={'displayModeBar': False})),
                        ], style={'width': '50%'}),
                    ], style={'display': 'flex', 'paddingTop': '12px'})),

            dcc.Tab(label='GEX / VEX / DEX', value='gex',
                    style=_TAB_STYLE, selected_style=_TAB_SEL,
                    children=html.Div([
                        html.Div(id='opt-gex-summary', style={
                            'display': 'flex', 'gap': '12px', 'marginBottom': '10px',
                            'padding': '8px 0',
                        }),
                        dcc.Loading(type='circle', children=dcc.Graph(
                            id='opt-gex-chart', style={'height': '360px'},
                            config={'displayModeBar': False})),
                    ], style={'paddingTop': '12px'})),

            dcc.Tab(label='IV Skew & Superficie', value='surface',
                    style=_TAB_STYLE, selected_style=_TAB_SEL,
                    children=html.Div([
                        html.Div([
                            html.Button(
                                [html.I(className='fa-solid fa-layer-group',
                                         style={'marginRight': '6px'}),
                                 'Carica Superficie IV (tutte le scadenze)'],
                                id='opt-surface-btn', n_clicks=0, style={
                                    'background': '#4a1a7c', 'color': 'white',
                                    'border': 'none', 'padding': '7px 16px',
                                    'borderRadius': '5px', 'cursor': 'pointer',
                                    'fontWeight': '700', 'fontSize': '11px',
                                    'marginBottom': '12px',
                                }),
                            html.Div(id='opt-surface-status',
                                     style={'fontSize': '11px', 'color': '#666',
                                            'marginLeft': '10px'}),
                        ], style={'display': 'flex', 'alignItems': 'center',
                                  'paddingTop': '12px'}),
                        html.Div([
                            html.Div([
                                dcc.Loading(type='circle', children=dcc.Graph(
                                    id='opt-skew-chart', style={'height': '320px'},
                                    config={'displayModeBar': False})),
                            ], style={'width': '50%', 'paddingRight': '8px'}),
                            html.Div([
                                dcc.Loading(type='circle', children=dcc.Graph(
                                    id='opt-surface-chart', style={'height': '320px'},
                                    config={'displayModeBar': False})),
                            ], style={'width': '50%'}),
                        ], style={'display': 'flex'}),
                    ])),

            dcc.Tab(label='Storico IV & Greche', value='historical',
                    style=_TAB_STYLE, selected_style=_TAB_SEL,
                    children=html.Div([
                        html.Div([
                            _lbl('Finestra rolling (gg):'),
                            dcc.Input(id='opt-hist-window', type='number', value=30,
                                      min=5, max=252, step=1, debounce=True,
                                      style={'width': '55px', 'fontSize': '11px',
                                             'border': '1px solid #aaa', 'borderRadius': '3px',
                                             'padding': '3px 5px', 'marginLeft': '6px'}),
                        ], style={'display': 'flex', 'alignItems': 'center',
                                  'padding': '10px 0'}),
                        dcc.Loading(type='circle', children=dcc.Graph(
                            id='opt-hist-iv-chart', style={'height': '240px'},
                            config={'displayModeBar': False})),
                        dcc.Loading(type='circle', children=dcc.Graph(
                            id='opt-hist-greeks-chart', style={'height': '260px'},
                            config={'displayModeBar': False})),
                    ])),

            dcc.Tab(label='🔍 Scanner', value='scanner',
                    style=_TAB_STYLE, selected_style=_TAB_SEL,
                    children=html.Div([
                        html.Div(id='opt-scanner-output',
                                 style={'paddingTop': '12px'}),
                    ])),

            dcc.Tab(label='📊 Cruscotto Macro / VIX', value='sentiment',
                    style=_TAB_STYLE, selected_style=_TAB_SEL,
                    children=html.Div([
                        html.Div([
                            html.Button([html.I(className='fa-solid fa-rotate',
                                                 style={'marginRight': '6px'}),
                                         'Scarica / Aggiorna dati'],
                                id='opt-sent-btn', n_clicks=0, style={
                                    'background': '#1a3a5c', 'color': 'white', 'border': 'none',
                                    'padding': '7px 16px', 'borderRadius': '5px', 'cursor': 'pointer',
                                    'fontWeight': '700', 'fontSize': '11px'}),
                            html.Div(id='opt-sent-status',
                                     style={'fontSize': '11px', 'color': '#666', 'marginLeft': '12px'}),
                        ], style={'display': 'flex', 'alignItems': 'center', 'paddingTop': '12px'}),
                        dcc.Store(id='opt-sent-store'),
                        dcc.Loading(type='circle', children=[
                            dcc.Graph(id='opt-sent-charts', style={'marginTop': '10px'},
                                      config={'displayModeBar': False}),
                            dcc.Graph(id='opt-sent-term', style={'marginTop': '6px'},
                                      config={'displayModeBar': False}),
                            html.Div([
                                _lbl('Indice:'),
                                dcc.Dropdown(id='opt-sent-select', options=[], value=None,
                                             clearable=False,
                                             style={'width': '230px', 'fontSize': '11px'}),
                                dcc.Dropdown(id='opt-sent-compare', options=[], value=[], multi=True,
                                             placeholder='Confronta con…',
                                             style={'minWidth': '240px', 'fontSize': '11px'}),
                                _lbl('Scala:'),
                                dcc.RadioItems(id='opt-sent-scala',
                                    options=[{'label': ' Valore assoluto', 'value': 'abs'},
                                             {'label': ' Variazione %',    'value': 'pct'},
                                             {'label': ' Rapporto + differenziale',
                                              'value': 'rapdiff'}],
                                    value='abs', inline=True,
                                    inputStyle={'marginRight': '3px'},
                                    labelStyle={'marginRight': '10px', 'cursor': 'pointer'},
                                    style={'fontSize': '11px'}),
                                _lbl('Smoothing (gg):'),
                                dcc.Input(id='opt-sent-smooth', type='number', value=1,
                                          min=1, max=250, step=1, debounce=True,
                                          style={'width': '60px', 'fontSize': '11px',
                                                 'padding': '3px 5px', 'border': '1px solid #aaa',
                                                 'borderRadius': '3px'}),
                                _lbl('Indicatori (tipo · periodo):'),
                                _sent_slot(1, 'rsi', 14),
                                _sent_slot(2, 'macd', 12),
                                _sent_slot(3, 'none', 14),
                            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px',
                                      'flexWrap': 'wrap', 'marginTop': '18px',
                                      'borderTop': '1px solid #eee', 'paddingTop': '12px'}),
                            dcc.Graph(id='opt-sent-detail', config={'displayModeBar': False}),
                            html.Div(id='opt-sent-table',
                                     style={'overflowX': 'auto', 'marginTop': '14px'}),
                        ]),
                    ])),

        ]),

    ], style={
        'paddingTop': '112px', 'padding': '112px 5% 32px',
        'fontFamily': 'Inter, sans-serif',
    }),
])


# ── FASE 1: callback cruscotto macro / VIX ────────────────────────────────────
@app.callback(
    Output('opt-sent-charts', 'figure'),
    Output('opt-sent-term',   'figure'),
    Output('opt-sent-store',  'data'),
    Output('opt-sent-select', 'options'),
    Output('opt-sent-select', 'value'),
    Output('opt-sent-compare', 'options'),
    Output('opt-sent-table',  'children'),
    Output('opt-sent-status', 'children'),
    Input('opt-sent-btn', 'n_clicks'),
    State('opt-sent-select', 'value'),
    prevent_initial_call=True,
)
def _load_sentiment(_n, cur_sel):
    try:
        px = _add_derived(download_sentiment())
        fig = _build_sentiment_charts(px)
        term = _vix_term_chart(px)
        tbl = _build_sentiment_table(px)
        # Nei menu: prima gli indici scaricati, poi le serie derivate (differenziale
        # e rapporto della struttura a termine), utilizzabili anche come confronto.
        avail = [k for k in list(_SENT_TICKERS) + list(_SENT_DERIVED)
                 if k in px.columns and not px[k].dropna().empty]
        opts = [{'label': _sent_label(k), 'value': k} for k in avail]
        sel = cur_sel if cur_sel in avail else (avail[0] if avail else None)
        last = px.dropna(how='all').index[-1].strftime('%d/%m/%Y')
        return (fig, term, px.to_json(orient='split', date_format='iso'),
                opts, sel, opts, tbl, f'✓ Aggiornato — dati EOD al {last}')
    except Exception as e:
        return (go.Figure(), go.Figure(), None, [], None, [],
                html.Div(f'⚠ Errore nel download: {e}',
                         style={'color': '#c00', 'fontSize': '11px', 'padding': '8px'}), '')


@app.callback(
    Output('opt-sent-detail', 'figure'),
    Input('opt-sent-select', 'value'),
    Input('opt-sent-compare', 'value'),
    Input('opt-sent-scala', 'value'),
    Input('opt-sent-smooth', 'value'),
    Input('opt-sent-t1', 'value'), Input('opt-sent-p1', 'value'),
    Input('opt-sent-t2', 'value'), Input('opt-sent-p2', 'value'),
    Input('opt-sent-t3', 'value'), Input('opt-sent-p3', 'value'),
    State('opt-sent-store', 'data'),
    prevent_initial_call=True,
)
def _sent_detail(ticker, compare, scala, smooth, t1, p1, t2, p2, t3, p3, data):
    if not data or not ticker:
        raise PreventUpdate
    slots = [(t, p) for t, p in ((t1, p1), (t2, p2), (t3, p3)) if t and t != 'none' and p]
    import io as _io
    px = pd.read_json(_io.StringIO(data), orient='split')
    px.index = pd.to_datetime(px.index)
    return _sent_detail_chart(px, ticker, slots, compare, scala or 'abs', smooth or 1)

# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 6 — CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# ─── CB 1: Fetch ticker ───────────────────────────────────────────────────────
@app.callback(
    Output('opt-stock-store', 'data'),
    Output('opt-stock-info', 'children'),
    Output('opt-expiry-select', 'options'),
    Output('opt-hist-store', 'data'),
    Input('opt-search-btn', 'n_clicks'),
    State('opt-ticker-input', 'value'),
    prevent_initial_call=True,
)
def fetch_ticker(_, ticker_raw):
    ticker = (ticker_raw or '').strip().upper()
    if not ticker:
        raise PreventUpdate

    def chip(txt, bg='#e8f0fb', col='#1a3a6b'):
        return html.Span(txt, style={
            'background': bg, 'color': col, 'borderRadius': '4px',
            'padding': '3px 8px', 'fontSize': '11px', 'fontWeight': '600',
            'fontFamily': 'Inter,sans-serif',
        })

    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
        spot = (info.get('regularMarketPrice')
                or info.get('currentPrice')
                or info.get('previousClose')
                or 100.0)
        spot = float(spot)
        name = info.get('shortName') or info.get('longName') or ticker

        hist = t.history(period='1y')
        hist_json = hist[['Close', 'Volume']].reset_index().rename(
            columns={'Date': 'date', 'Close': 'Close', 'Volume': 'Volume'}
        ).to_json(orient='records', date_format='iso')

        expirations = t.options or []
        exp_options = [{'label': e, 'value': e} for e in expirations[:20]]

        prev_close = float(info.get('previousClose') or spot)
        chg_pct    = (spot / prev_close - 1) * 100 if prev_close else 0
        chg_col    = '#1b5e20' if chg_pct >= 0 else '#b71c1c'
        chg_bg     = '#e8f5e9' if chg_pct >= 0 else '#ffebee'

        chips = [
            chip(name),
            chip(f'${spot:.2f}', '#f0f4fa'),
            chip(f'{chg_pct:+.2f}%', chg_bg, chg_col),
        ]
        if info.get('fiftyTwoWeekLow') and info.get('fiftyTwoWeekHigh'):
            chips.append(chip(
                f"52w: {info['fiftyTwoWeekLow']:.1f}–{info['fiftyTwoWeekHigh']:.1f}",
                '#f8fafd', '#5a7099',
            ))
        if expirations:
            chips.append(chip(f'{len(expirations)} scadenze', '#fff3e0', '#e65100'))

        stock_data = {'ticker': ticker, 'spot': spot, 'name': name}
        return stock_data, chips, exp_options, hist_json

    except Exception as e:
        err = html.Span(f'Errore: {e}', style={'color': '#c0392b', 'fontSize': '11px'})
        return None, [err], [], None


# ─── CB 2: Fetch chain per scadenza ──────────────────────────────────────────
@app.callback(
    Output('opt-chain-store', 'data'),
    Output('opt-expiry-store', 'data'),
    Input('opt-expiry-select', 'value'),
    State('opt-stock-store', 'data'),
    prevent_initial_call=True,
)
def fetch_chain(expiry, stock_data):
    if not expiry or not stock_data:
        raise PreventUpdate
    try:
        ticker = stock_data['ticker']
        chain  = yf.Ticker(ticker).option_chain(expiry)
        calls  = chain.calls.to_dict(orient='records')
        puts   = chain.puts.to_dict(orient='records')
        return {'calls': calls, 'puts': puts}, expiry
    except Exception:
        raise PreventUpdate


# ─── CB 3: Costruisce gambe dal preset ───────────────────────────────────────
@app.callback(
    Output('opt-legs-store', 'data'),
    Input('opt-strategy-select', 'value'),
    Input('opt-chain-store', 'data'),
    State('opt-stock-store', 'data'),
    prevent_initial_call=True,
)
def build_legs(strategy, chain, stock_data):
    if not stock_data:
        raise PreventUpdate
    spot = float(stock_data.get('spot', 100))
    if not chain:
        # Crea gambe fittizie ATM senza dati di catena
        template = STRATEGY_PRESETS.get(strategy, [])
        legs = [{'type': t['type'], 'dir': t['dir'], 'strike': round(spot),
                 'qty': t['qty'], 'premium': 0.0, 'iv': 0.25}
                for t in template]
        return legs

    calls_df = pd.DataFrame(chain['calls'])
    puts_df  = pd.DataFrame(chain['puts'])
    return resolve_preset(strategy, calls_df, puts_df, spot)


# ─── CB 4: Render tabella gambe ───────────────────────────────────────────────
@app.callback(
    Output('opt-legs-table', 'children'),
    Input('opt-legs-store', 'data'),
    prevent_initial_call=False,
)
def render_legs(legs):
    if not legs:
        return html.Div('Nessuna gamba. Seleziona una strategia.',
                        style={'color': '#888', 'fontSize': '11px', 'padding': '8px'})

    rows = []
    for i, leg in enumerate(legs):
        ot   = leg.get('type', 'call')
        dir_ = int(leg.get('dir', 1))
        bg   = '#fff8f0' if ot == 'put' else '#f0f8ff'

        row = html.Div([
            html.Div(
                dcc.Dropdown(
                    id={'type': 'opt-leg-type', 'index': i},
                    options=[{'label': 'Call', 'value': 'call'},
                             {'label': 'Put',  'value': 'put'}],
                    value=ot, clearable=False,
                    style={'fontSize': '10px', 'width': '100%'},
                ), style={'width': '18%'}),
            html.Div(
                dcc.Dropdown(
                    id={'type': 'opt-leg-dir', 'index': i},
                    options=[{'label': 'L', 'value': 1},
                             {'label': 'S', 'value': -1}],
                    value=dir_, clearable=False,
                    style={'fontSize': '10px', 'width': '100%',
                           'color': '#1b5e20' if dir_ == 1 else '#b71c1c'},
                ), style={'width': '10%'}),
            html.Div(
                dcc.Input(id={'type': 'opt-leg-strike', 'index': i},
                          type='number', value=leg.get('strike', 100),
                          step=0.5, debounce=True,
                          style={'width': '95%', 'fontSize': '10px', 'height': '28px',
                                 'border': '1px solid #ccc', 'borderRadius': '3px',
                                 'padding': '2px 4px'}),
                style={'width': '22%'}),
            html.Div(
                dcc.Input(id={'type': 'opt-leg-qty', 'index': i},
                          type='number', value=leg.get('qty', 1),
                          min=1, max=100, step=1, debounce=True,
                          style={'width': '90%', 'fontSize': '10px', 'height': '28px',
                                 'border': '1px solid #ccc', 'borderRadius': '3px',
                                 'padding': '2px 4px'}),
                style={'width': '12%'}),
            html.Div(
                dcc.Input(id={'type': 'opt-leg-premium', 'index': i},
                          type='number', value=leg.get('premium', 0),
                          step=0.01, debounce=True,
                          style={'width': '95%', 'fontSize': '10px', 'height': '28px',
                                 'border': '1px solid #ccc', 'borderRadius': '3px',
                                 'padding': '2px 4px'}),
                style={'width': '18%'}),
            html.Div(
                dcc.Input(id={'type': 'opt-leg-iv', 'index': i},
                          type='number',
                          value=round(float(leg.get('iv', 0.25)) * 100, 1),
                          step=0.5, debounce=True,
                          style={'width': '90%', 'fontSize': '10px', 'height': '28px',
                                 'border': '1px solid #ccc', 'borderRadius': '3px',
                                 'padding': '2px 4px'}),
                style={'width': '13%'}),
            html.Div(
                html.Button('✕', id={'type': 'opt-leg-del', 'index': i},
                             n_clicks=0, style={
                                 'background': 'none', 'border': 'none', 'color': '#c0392b',
                                 'cursor': 'pointer', 'fontSize': '12px', 'padding': '0 4px',
                             }),
                style={'width': '7%', 'textAlign': 'center'}),
        ], style={
            'display': 'flex', 'alignItems': 'center',
            'padding': '3px 0', 'borderBottom': '1px dotted #eee',
            'background': bg,
        })
        rows.append(row)
    return rows


# ─── CB 5: Sync gambe da input tabella ───────────────────────────────────────
@app.callback(
    Output('opt-legs-store', 'data', allow_duplicate=True),
    Input({'type': 'opt-leg-type',    'index': ALL}, 'value'),
    Input({'type': 'opt-leg-dir',     'index': ALL}, 'value'),
    Input({'type': 'opt-leg-strike',  'index': ALL}, 'value'),
    Input({'type': 'opt-leg-qty',     'index': ALL}, 'value'),
    Input({'type': 'opt-leg-premium', 'index': ALL}, 'value'),
    Input({'type': 'opt-leg-iv',      'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def sync_legs(types, dirs, strikes, qtys, premiums, ivs):
    if not types:
        raise PreventUpdate
    legs = []
    for i in range(len(types)):
        legs.append({
            'type':    types[i]    or 'call',
            'dir':     int(dirs[i] or 1),
            'strike':  float(strikes[i] or 100),
            'qty':     int(qtys[i]   or 1),
            'premium': float(premiums[i] or 0),
            'iv':      max(0.001, float(ivs[i] or 25)) / 100.0,
        })
    return legs


# ─── CB 6: Aggiunge gamba ─────────────────────────────────────────────────────
@app.callback(
    Output('opt-legs-store', 'data', allow_duplicate=True),
    Input('opt-add-leg-btn', 'n_clicks'),
    State('opt-legs-store', 'data'),
    State('opt-stock-store', 'data'),
    prevent_initial_call=True,
)
def add_leg(_, legs, stock_data):
    legs  = list(legs or [])
    spot  = float((stock_data or {}).get('spot', 100))
    legs.append({'type': 'call', 'dir': 1, 'strike': round(spot),
                 'qty': 1, 'premium': 0.0, 'iv': 0.25})
    return legs


# ─── CB 7: Elimina gamba ─────────────────────────────────────────────────────
@app.callback(
    Output('opt-legs-store', 'data', allow_duplicate=True),
    Input({'type': 'opt-leg-del', 'index': ALL}, 'n_clicks'),
    State('opt-legs-store', 'data'),
    prevent_initial_call=True,
)
def delete_leg(n_clicks_list, legs):
    if not any(n_clicks_list):
        raise PreventUpdate
    ctx   = callback_context
    if not ctx.triggered or not ctx.triggered[0]['value']:
        raise PreventUpdate
    idx   = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
    legs  = list(legs or [])
    if 0 <= idx < len(legs):
        legs.pop(idx)
    return legs


# ─── CB 8: Relay RF ───────────────────────────────────────────────────────────
@app.callback(
    Output('opt-rf-store', 'data'),
    Input('opt-rf-input', 'value'),
    prevent_initial_call=True,
)
def relay_rf(v):
    return float(v or 3.5)


# ─── CB 9: Payoff + Greche (display) ─────────────────────────────────────────
@app.callback(
    Output('opt-payoff-chart', 'figure'),
    Output('opt-greeks-display', 'children'),
    Output('opt-higher-greeks-display', 'children'),
    Output('opt-summary-card', 'children'),
    Input('opt-legs-store', 'data'),
    Input('opt-payoff-mode', 'value'),
    State('opt-stock-store', 'data'),
    State('opt-expiry-store', 'data'),
    State('opt-rf-store', 'data'),
    prevent_initial_call=True,
)
def update_payoff_greeks(legs, mode, stock_data, expiry, rf):
    empty_fig = go.Figure().update_layout(
        paper_bgcolor='#f8fafd', plot_bgcolor='#f8fafd',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text='Seleziona ticker e scadenza',
                          showarrow=False, font=dict(size=13, color='#888'))],
        margin=dict(l=0, r=0, t=0, b=0),
    )

    if not legs or not stock_data:
        return empty_fig, _lbl('—'), _lbl('—'), html.Div()

    spot = float(stock_data.get('spot', 100))
    T    = days_to_expiry_years(expiry) if expiry else 0.25
    r    = float(rf or 3.5) / 100.0

    pdata = compute_payoff(legs, spot, T, r)
    if not pdata:
        return empty_fig, _lbl('—'), _lbl('—'), html.Div()

    # ── Payoff chart ──────────────────────────────────────────────────────────
    S_range     = pdata['S_range']
    pnl_exp     = pdata['pnl_expiry']
    pnl_cur     = pdata['pnl_current']
    bes         = pdata['breakevens']
    net_cost    = pdata['net_cost']
    max_profit  = max(pnl_exp)
    max_loss    = min(pnl_exp)

    fig = go.Figure()

    def add_zero():
        fig.add_hline(y=0, line_color='#888', line_width=1, line_dash='dash')
        fig.add_vline(x=spot, line_color='#1a3a6b', line_width=1.5,
                      annotation_text=f'Spot ${spot:.2f}',
                      annotation_position='top right',
                      annotation_font_size=10)

    if mode in ('expiry', 'both'):
        fig.add_trace(go.Scatter(
            x=S_range, y=pnl_exp, name='A scadenza',
            line=dict(color='#1a3a6b', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(26,58,107,0.08)',
        ))
    if mode in ('current', 'both'):
        fig.add_trace(go.Scatter(
            x=S_range, y=pnl_cur, name='Valore attuale B-S',
            line=dict(color='#e6194b', width=2, dash='dot'),
        ))

    add_zero()

    for be in bes:
        fig.add_vline(x=be, line_color='#e6194b', line_width=1,
                      line_dash='dot',
                      annotation_text=f'BE {be:.1f}',
                      annotation_position='bottom right',
                      annotation_font_size=9, annotation_font_color='#e6194b')

    fig.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#f8fafd',
        margin=dict(l=48, r=16, t=24, b=40),
        legend=dict(orientation='h', y=1.08, x=0, font=dict(size=10)),
        xaxis=dict(title='Prezzo Sottostante', gridcolor='#eeeeee',
                   tickformat='$.0f', title_font_size=10),
        yaxis=dict(title='P&L ($)', gridcolor='#eeeeee', tickprefix='$',
                   title_font_size=10),
        hovermode='x unified',
        font=dict(family='Inter, sans-serif', size=10),
    )

    # ── Greche 1° livello ────────────────────────────────────────────────────
    first, higher = aggregate_all_greeks(legs, spot, T, r)

    def greek_chip(name, val, fmt, bg, col):
        return html.Span([
            html.Span(name, style={
                'fontSize': '8px', 'fontWeight': '700', 'display': 'block',
                'color': col, 'letterSpacing': '0.05em',
            }),
            html.Span(fmt.format(val), style={
                'fontSize': '12px', 'fontWeight': '700', 'color': col,
            }),
        ], className='greek-chip', style={
            'background': bg, 'border': f'1px solid {col}22',
        })

    greeks_chips = html.Div([
        greek_chip('Δ Delta',  first['delta'], '{:+.3f}', '#e8f0fb', '#1a3a6b'),
        greek_chip('Γ Gamma',  first['gamma'], '{:.4f}',  '#e8f5e9', '#1b5e20'),
        greek_chip('Θ Theta',  first['theta'], '{:+.3f}', '#fff8e1', '#e65100'),
        greek_chip('ν Vega',   first['vega'],  '{:+.3f}', '#f3e5f5', '#6a1b9a'),
        greek_chip('ρ Rho',    first['rho'],   '{:+.3f}', '#fce4ec', '#880e4f'),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '4px'})

    higher_chips = html.Div([
        greek_chip('Vanna',  higher['vanna'], '{:+.4f}', '#ede7f6', '#4a148c'),
        greek_chip('Charm',  higher['charm'], '{:+.5f}', '#fce4ec', '#880e4f'),
        greek_chip('Vomma',  higher['vomma'], '{:+.4f}', '#e8eaf6', '#283593'),
        greek_chip('Color',  higher['color'], '{:+.6f}', '#e0f2f1', '#004d40'),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '4px'})

    # ── Summary card ─────────────────────────────────────────────────────────
    be_str = ' / '.join([f'${b:.2f}' for b in bes]) if bes else 'N/D'
    def _row(label, val, col='#222'):
        return html.Div([
            html.Span(label, style={'fontSize': '10px', 'color': '#666', 'width': '60%'}),
            html.Span(val,   style={'fontSize': '11px', 'fontWeight': '700',
                                    'color': col, 'textAlign': 'right'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between',
                  'padding': '3px 0', 'borderBottom': '1px dotted #eee'})

    nc_col = '#1b5e20' if net_cost < 0 else '#b71c1c'
    summary = html.Div([
        _lbl('Riepilogo Strategia'),
        _row('Costo netto (per 100)',  f'${net_cost:+.0f}',
             nc_col),
        _row('Max profitto (scad.)',   f'${max_profit:+.0f}' if np.isfinite(max_profit) else '∞',
             '#1b5e20'),
        _row('Max perdita (scad.)',    f'${max_loss:+.0f}' if np.isfinite(max_loss) else '−∞',
             '#b71c1c'),
        _row('Break-even (scad.)',    be_str),
        _row('T scadenza',            f'{T * 365:.0f} gg' if expiry else 'N/D'),
    ])

    return fig, greeks_chips, higher_chips, summary


# ─── CB 10: Sensitivity heatmap ──────────────────────────────────────────────
@app.callback(
    Output('opt-heatmap', 'figure'),
    Input('opt-legs-store', 'data'),
    State('opt-stock-store', 'data'),
    State('opt-expiry-store', 'data'),
    State('opt-rf-store', 'data'),
    prevent_initial_call=True,
)
def update_heatmap(legs, stock_data, expiry, rf):
    if not legs or not stock_data:
        raise PreventUpdate
    spot = float(stock_data.get('spot', 100))
    T    = days_to_expiry_years(expiry) if expiry else 0.25
    r    = float(rf or 3.5) / 100.0
    h    = compute_heatmap(legs, spot, T, r)
    if not h:
        raise PreventUpdate

    fig = go.Figure(go.Heatmap(
        z=h['z'], x=[f'{x:+.0f}%' for x in h['x']],
        y=[f'{y:+.0f}%' for y in h['y']],
        colorscale='RdYlGn', zmid=0,
        colorbar=dict(title='P&L $', title_side='right', tickprefix='$',
                      len=0.8, thickness=12),
        hovertemplate='ΔS: %{x}<br>ΔIV: %{y}<br>P&L: $%{z:.0f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Sensitivity: P&L vs ΔPrezzo × ΔIV', font_size=11, x=0.5),
        xaxis=dict(title='Variaz. Prezzo %', title_font_size=10),
        yaxis=dict(title='Variaz. IV %', title_font_size=10),
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        margin=dict(l=60, r=60, t=40, b=50),
        font=dict(family='Inter, sans-serif', size=10),
    )
    return fig


# ─── CB 11: Greche vs prezzo ──────────────────────────────────────────────────
@app.callback(
    Output('opt-greeks-chart', 'figure'),
    Input('opt-legs-store', 'data'),
    State('opt-stock-store', 'data'),
    State('opt-expiry-store', 'data'),
    State('opt-rf-store', 'data'),
    prevent_initial_call=True,
)
def update_greeks_chart(legs, stock_data, expiry, rf):
    if not legs or not stock_data:
        raise PreventUpdate
    spot = float(stock_data.get('spot', 100))
    T    = days_to_expiry_years(expiry) if expiry else 0.25
    r    = float(rf or 3.5) / 100.0
    S_range = np.linspace(spot * 0.70, spot * 1.30, 100)

    deltas, gammas, thetas, vegas = [], [], [], []
    for S in S_range:
        f, _ = aggregate_all_greeks(legs, S, T, r)
        deltas.append(f['delta'])
        gammas.append(f['gamma'])
        thetas.append(f['theta'])
        vegas.append(f['vega'])

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Delta', 'Gamma', 'Theta (daily)', 'Vega'],
                        shared_xaxes=True)

    def _add(row, col, name, y_vals, color):
        fig.add_trace(go.Scatter(
            x=S_range.tolist(), y=y_vals, name=name,
            line=dict(color=color, width=2),
            showlegend=False,
        ), row=row, col=col)
        fig.add_vline(x=spot, line_color='#888', line_width=1,
                      line_dash='dash', row=row, col=col)

    _add(1, 1, 'Delta', deltas, '#1a3a6b')
    _add(1, 2, 'Gamma', gammas, '#1b5e20')
    _add(2, 1, 'Theta', thetas, '#e65100')
    _add(2, 2, 'Vega',  vegas,  '#6a1b9a')

    fig.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#f8fafd',
        margin=dict(l=40, r=16, t=40, b=30),
        font=dict(family='Inter, sans-serif', size=9),
        height=340,
    )
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=10)
    return fig


# ─── CB 12: GEX / VEX / DEX ──────────────────────────────────────────────────
@app.callback(
    Output('opt-gex-chart', 'figure'),
    Output('opt-gex-summary', 'children'),
    Input('opt-chain-store', 'data'),
    State('opt-stock-store', 'data'),
    State('opt-expiry-store', 'data'),
    State('opt-rf-store', 'data'),
    prevent_initial_call=True,
)
def update_gex(chain, stock_data, expiry, rf):
    empty = go.Figure().update_layout(
        paper_bgcolor='#f8fafd', annotations=[
            dict(text='Seleziona ticker e scadenza per calcolare GEX/VEX/DEX',
                 showarrow=False, font=dict(size=12, color='#888'))],
        margin=dict(l=0, r=0, t=0, b=0))

    if not chain or not stock_data:
        return empty, []

    spot  = float(stock_data.get('spot', 100))
    T     = days_to_expiry_years(expiry) if expiry else 0.25
    r     = float(rf or 3.5) / 100.0
    cdf   = pd.DataFrame(chain['calls'])
    pdf   = pd.DataFrame(chain['puts'])
    exp   = compute_exposure(cdf, pdf, spot, T, r)

    if not exp:
        return empty, [html.Span('Dati insufficienti per GEX',
                                  style={'color': '#888', 'fontSize': '11px'})]

    strikes = [x['strike'] for x in exp['gex']]
    gex_v   = [x['gex']    for x in exp['gex']]
    vex_v   = [x['vex']    for x in exp['vex']]
    dex_v   = [x['dex']    for x in exp['dex']]

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=['GEX (Gamma Exposure)',
                                        'VEX (Vanna Exposure)',
                                        'DEX (Delta Exposure)'],
                        shared_xaxes=True, vertical_spacing=0.08)

    def bar(row, vals, name, color_pos, color_neg):
        colors = [color_pos if v >= 0 else color_neg for v in vals]
        fig.add_trace(go.Bar(x=strikes, y=vals, name=name,
                              marker_color=colors, showlegend=False),
                      row=row, col=1)
        fig.add_vline(x=spot, line_color='#1a3a6b', line_width=1.5,
                      line_dash='dash', row=row, col=1)

    bar(1, gex_v, 'GEX', '#1b5e20', '#b71c1c')
    bar(2, vex_v, 'VEX', '#1565c0', '#f57f17')
    bar(3, dex_v, 'DEX', '#4a148c', '#e65100')

    fig.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#f8fafd',
        margin=dict(l=50, r=16, t=40, b=40), height=360,
        font=dict(family='Inter, sans-serif', size=9),
    )

    # Summary chips
    def expo_chip(label, val, col_pos, col_neg):
        v_fmt = f'{val / 1e6:.1f}M' if abs(val) > 1e6 else f'{val / 1e3:.0f}K'
        col   = col_pos if val >= 0 else col_neg
        bg    = '#e8f5e9' if val >= 0 else '#ffebee'
        return html.Div([
            html.Span(label, style={'fontSize': '9px', 'fontWeight': '700',
                                     'color': col, 'display': 'block'}),
            html.Span(v_fmt, style={'fontSize': '16px', 'fontWeight': '700',
                                     'color': col}),
        ], style={
            'background': bg, 'border': f'1px solid {col}44',
            'borderRadius': '6px', 'padding': '8px 14px', 'textAlign': 'center',
        })

    chips = [
        expo_chip('GEX Total',  exp['total_gex'], '#1b5e20', '#b71c1c'),
        expo_chip('VEX Total',  exp['total_vex'], '#1565c0', '#f57f17'),
        expo_chip('DEX Total',  exp['total_dex'], '#4a148c', '#e65100'),
        html.Div([
            html.Span('Regime', style={'fontSize': '9px', 'fontWeight': '700',
                                        'display': 'block',
                                        'color': '#1b5e20' if exp['total_gex'] > 0 else '#b71c1c'}),
            html.Span('MEAN REV' if exp['total_gex'] > 0 else 'BREAKOUT',
                      style={'fontSize': '12px', 'fontWeight': '700',
                             'color': '#1b5e20' if exp['total_gex'] > 0 else '#b71c1c'}),
        ], style={
            'background': '#e8f5e9' if exp['total_gex'] > 0 else '#ffebee',
            'border': '1px solid #ccc', 'borderRadius': '6px',
            'padding': '8px 14px', 'textAlign': 'center',
        }),
    ]
    return fig, chips


# ─── CB 13: IV Surface + Skew ─────────────────────────────────────────────────
@app.callback(
    Output('opt-surface-store', 'data'),
    Output('opt-surface-status', 'children'),
    Input('opt-surface-btn', 'n_clicks'),
    State('opt-stock-store', 'data'),
    State('opt-rf-store', 'data'),
    prevent_initial_call=True,
)
def load_surface(_, stock_data, rf):
    if not stock_data:
        raise PreventUpdate
    ticker  = stock_data['ticker']
    t_obj   = yf.Ticker(ticker)
    exps    = t_obj.options[:8] if t_obj.options else []
    chains  = {}
    for exp in exps:
        try:
            ch = t_obj.option_chain(exp)
            chains[exp] = {
                'calls': ch.calls.to_dict(orient='records'),
                'puts':  ch.puts.to_dict(orient='records'),
            }
        except Exception:
            continue
    if not chains:
        return None, html.Span('Nessuna catena disponibile',
                                style={'color': '#c0392b', 'fontSize': '11px'})
    return chains, html.Span(
        f'{len(chains)} scadenze caricate',
        style={'color': '#1b5e20', 'fontSize': '11px', 'fontWeight': '600'})

@app.callback(
    Output('opt-skew-chart', 'figure'),
    Output('opt-surface-chart', 'figure'),
    Input('opt-surface-store', 'data'),
    Input('opt-chain-store', 'data'),
    State('opt-stock-store', 'data'),
    State('opt-expiry-store', 'data'),
    State('opt-rf-store', 'data'),
    prevent_initial_call=True,
)
def render_surface_skew(surface_data, chain, stock_data, expiry, rf):
    empty = go.Figure().update_layout(
        paper_bgcolor='#f8fafd', plot_bgcolor='#f8fafd',
        annotations=[dict(text='Carica la superficie',
                          showarrow=False, font=dict(size=12, color='#888'))],
        margin=dict(l=0, r=0, t=0, b=0))

    if not stock_data:
        return empty, empty

    spot = float(stock_data.get('spot', 100))
    r    = float(rf or 3.5) / 100.0
    T    = days_to_expiry_years(expiry) if expiry else 0.25

    # Skew dal chain corrente
    skew_fig = empty
    if chain:
        cdf = pd.DataFrame(chain['calls'])
        pdf = pd.DataFrame(chain['puts'])
        sk  = compute_skew(cdf, pdf, spot, T, r)
        if sk['strikes']:
            skew_fig = go.Figure()
            skew_fig.add_trace(go.Scatter(
                x=sk['strikes'], y=sk['call_iv'], name='Call IV %',
                line=dict(color='#1b5e20', width=2)))
            skew_fig.add_trace(go.Scatter(
                x=sk['strikes'], y=sk['put_iv'], name='Put IV %',
                line=dict(color='#b71c1c', width=2)))
            skew_fig.add_trace(go.Bar(
                x=sk['strikes'], y=sk['skew'], name='Skew (Put−Call)',
                marker_color='rgba(106,27,154,0.3)',
                yaxis='y2'))
            skew_fig.add_vline(x=spot, line_color='#1a3a6b',
                               line_dash='dash', line_width=1.5)
            rr_txt = f' · 25Δ RR: {sk["rr_25"]:+.1f}%' if sk.get('rr_25') is not None else ''
            skew_fig.update_layout(
                title=dict(text=f'IV Skew{rr_txt}', font_size=11, x=0.5),
                yaxis=dict(title='IV %', side='left'),
                yaxis2=dict(title='Skew %', side='right', overlaying='y',
                             showgrid=False),
                paper_bgcolor='#ffffff', plot_bgcolor='#f8fafd',
                legend=dict(orientation='h', y=1.08, font_size=9),
                margin=dict(l=50, r=50, t=40, b=40),
                font=dict(family='Inter, sans-serif', size=10),
            )

    # Superficie IV 3D
    surf_fig = empty
    if surface_data:
        surf = build_iv_surface(surface_data, spot)
        if surf and surf['z']:
            surf_fig = go.Figure(go.Surface(
                z=surf['z'],
                x=surf['strikes'],
                y=surf['dte'],
                colorscale='Viridis',
                colorbar=dict(title='IV %', title_side='right',
                               thickness=12, len=0.8),
                hovertemplate=(
                    'Strike: $%{x:.0f}<br>DTE: %{y} gg<br>IV: %{z:.1f}%<extra></extra>'),
            ))
            surf_fig.update_layout(
                title=dict(text='Volatility Surface', font_size=11, x=0.5),
                scene=dict(
                    xaxis_title='Strike', yaxis_title='DTE', zaxis_title='IV %',
                    xaxis=dict(tickprefix='$'),
                    zaxis=dict(ticksuffix='%'),
                ),
                paper_bgcolor='#ffffff',
                margin=dict(l=0, r=0, t=40, b=0),
                font=dict(family='Inter, sans-serif', size=9),
            )

    return skew_fig, surf_fig


# ─── CB 14: Storico IV e Greche rolling ───────────────────────────────────────
@app.callback(
    Output('opt-hist-iv-chart', 'figure'),
    Output('opt-hist-greeks-chart', 'figure'),
    Input('opt-hist-store', 'data'),
    Input('opt-legs-store', 'data'),
    Input('opt-hist-window', 'value'),
    State('opt-expiry-store', 'data'),
    State('opt-rf-store', 'data'),
    prevent_initial_call=True,
)
def update_historical(hist_json, legs, window, expiry, rf):
    empty = go.Figure().update_layout(
        paper_bgcolor='#f8fafd', plot_bgcolor='#f8fafd',
        annotations=[dict(text='Cerca un ticker per caricare lo storico',
                          showarrow=False, font=dict(size=11, color='#888'))],
        margin=dict(l=40, r=16, t=20, b=30))

    if not hist_json:
        return empty, empty

    try:
        hist_df = pd.read_json(hist_json, orient='records')
        hist_df = hist_df.rename(columns={'date': 'Date', 'Close': 'Close'})
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        hist_df = hist_df.set_index('Date').sort_index()
    except Exception:
        return empty, empty

    window = max(5, int(window or 30))
    r      = float(rf or 3.5) / 100.0

    rolling = compute_rolling_iv_greeks(hist_df, legs or [], expiry, r, window)
    if not rolling:
        return empty, empty

    # ── Chart 1: Prezzo + HV rolling ─────────────────────────────────────────
    iv_fig = make_subplots(specs=[[{'secondary_y': True}]])
    px_d   = [x['x'] for x in rolling['price']]
    px_v   = [x['y'] for x in rolling['price']]
    hv_d   = [x['x'] for x in rolling['hv']]
    hv_v   = [x['y'] for x in rolling['hv']]

    iv_fig.add_trace(go.Scatter(x=px_d, y=px_v, name='Prezzo',
                                line=dict(color='#1a3a6b', width=1.5)),
                     secondary_y=False)
    iv_fig.add_trace(go.Scatter(x=hv_d,
                                y=[v * 100 for v in hv_v],
                                name=f'HV {window}d %',
                                line=dict(color='#e6194b', width=1.5, dash='dot'),
                                fill='tozeroy',
                                fillcolor='rgba(230,25,75,0.06)'),
                     secondary_y=True)
    iv_fig.update_layout(
        title=dict(text=f'Prezzo e Volatilità Storica Rolling {window}d',
                   font_size=11, x=0.5),
        paper_bgcolor='#ffffff', plot_bgcolor='#f8fafd',
        legend=dict(orientation='h', y=1.1, font_size=9),
        margin=dict(l=50, r=50, t=40, b=30),
        font=dict(family='Inter, sans-serif', size=10),
    )
    iv_fig.update_yaxes(title_text='Prezzo', secondary_y=False)
    iv_fig.update_yaxes(title_text='HV %',   secondary_y=True,
                        ticksuffix='%', showgrid=False)

    # ── Chart 2: Greche rolling ───────────────────────────────────────────────
    if not legs or not rolling['delta']:
        g_fig = empty
    else:
        g_fig = make_subplots(rows=2, cols=2,
                              subplot_titles=['Delta rolling', 'Gamma rolling',
                                             'Theta rolling', 'Vega rolling'],
                              shared_xaxes=True, vertical_spacing=0.12)

        def add_line(row, col, name, vals, color):
            g_fig.add_trace(go.Scatter(
                x=rolling['dates'], y=vals, name=name,
                line=dict(color=color, width=1.5), showlegend=False,
            ), row=row, col=col)

        add_line(1, 1, 'Delta', rolling['delta'], '#1a3a6b')
        add_line(1, 2, 'Gamma', rolling['gamma'], '#1b5e20')
        add_line(2, 1, 'Theta', rolling['theta'], '#e65100')
        add_line(2, 2, 'Vega',  rolling['vega'],  '#6a1b9a')

        g_fig.update_layout(
            title=dict(text='Greche rolling della strategia (HV come σ)',
                       font_size=11, x=0.5),
            paper_bgcolor='#ffffff', plot_bgcolor='#f8fafd',
            margin=dict(l=40, r=16, t=40, b=30), height=260,
            font=dict(family='Inter, sans-serif', size=9),
        )

    return iv_fig, g_fig


# ─── CB 15: Scanner automatico ────────────────────────────────────────────────
@app.callback(
    Output('opt-scanner-output', 'children'),
    Input('opt-chain-store', 'data'),
    Input('opt-surface-store', 'data'),
    State('opt-stock-store', 'data'),
    State('opt-expiry-store', 'data'),
    State('opt-rf-store', 'data'),
    prevent_initial_call=True,
)
def update_scanner(chain, surface_data, stock_data, expiry, rf):
    if not stock_data:
        return html.Div('Cerca un ticker per attivare lo scanner.',
                        style={'color': '#888', 'fontSize': '12px', 'padding': '12px'})

    spot = float(stock_data.get('spot', 100))
    T    = days_to_expiry_years(expiry) if expiry else 0.25
    r    = float(rf or 3.5) / 100.0

    exposure   = None
    skew_data  = None
    term_data  = None

    if chain:
        cdf = pd.DataFrame(chain['calls'])
        pdf = pd.DataFrame(chain['puts'])
        exposure  = compute_exposure(cdf, pdf, spot, T, r)
        skew_data = compute_skew(cdf, pdf, spot, T, r)

    if surface_data:
        term_data = compute_term_structure(surface_data, spot)

    result = compute_scanner_signals(exposure, skew_data, term_data)

    if not result['signals']:
        return html.Div('Nessun segnale disponibile. Carica la superficie IV per l\'analisi completa.',
                        style={'color': '#888', 'fontSize': '12px'})

    level_cfg = {
        'ok':      ('#e8f5e9', '#1b5e20', '#2e7d32', 'fa-circle-check'),
        'info':    ('#e8f0fb', '#1a3a6b', '#1a3a6b', 'fa-circle-info'),
        'warning': ('#fff3e0', '#e65100', '#e65100', 'fa-triangle-exclamation'),
    }

    cards = []
    for sig in result['signals']:
        lvl  = sig.get('level', 'info')
        cfg  = level_cfg.get(lvl, level_cfg['info'])
        strs = sig.get('strategy', [])

        card = html.Div([
            html.Div([
                html.I(className=f'fa-solid {cfg[3]}',
                       style={'color': cfg[2], 'marginRight': '8px', 'fontSize': '14px'}),
                html.Span(sig['type'], style={
                    'fontSize': '10px', 'fontWeight': '700', 'color': cfg[2],
                    'textTransform': 'uppercase', 'letterSpacing': '0.08em',
                }),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
            html.P(sig['msg'], style={
                'fontSize': '12px', 'color': '#333', 'margin': '0 0 6px 0',
                'lineHeight': '1.5',
            }),
            html.Div([
                html.Span('Strategie suggerite: ', style={
                    'fontSize': '10px', 'color': '#666', 'fontWeight': '600',
                }),
                *[html.Span(s, style={
                    'fontSize': '10px', 'background': cfg[0],
                    'color': cfg[2], 'borderRadius': '3px',
                    'padding': '2px 6px', 'marginLeft': '4px',
                    'fontWeight': '700',
                }) for s in strs],
            ]) if strs else html.Div(),
        ], className='signal-card', style={
            'background': cfg[0], 'borderLeftColor': cfg[2],
        })
        cards.append(card)

    # Aggiunge la matrice 3D in testo
    matrix_note = html.Div([
        html.H4('Struttura della Matrice IV (attiva)',
                style={'fontSize': '11px', 'fontWeight': '700', 'color': '#1a3a6b',
                       'margin': '16px 0 4px 0'}),
        html.P(
            'Asse X: Strike Price · Asse Y: Scadenze (DTE) · Cella: IV% × OI × Gamma → GEX parziale. '
            'Usa "Carica Superficie IV" per visualizzare la superficie 3D nella tab IV Skew & Superficie.',
            style={'fontSize': '11px', 'color': '#555', 'lineHeight': '1.6'}
        ),
    ])
    cards.append(matrix_note)

    return html.Div(cards)
