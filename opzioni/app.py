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

        ]),

    ], style={
        'paddingTop': '112px', 'padding': '112px 5% 32px',
        'fontFamily': 'Inter, sans-serif',
    }),
])

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
