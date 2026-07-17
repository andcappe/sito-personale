"""
Dashboard Economica — FRED API
Analisi monetaria USA: M2, velocità, CPI, PIL reale
"""

import os
import io, base64, warnings, json, urllib.request, math
import statsmodels.api as sm
from scipy import stats as scipy_stats
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd
import pandas.tseries.offsets as offsets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback_context, Output, Input, State, ALL, no_update
from dash.exceptions import PreventUpdate

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

FRED_API_KEY = os.environ.get("FRED_API_KEY", "65061ed1fa4c47d53b1d644e1cd858d3")

DEFAULT_SERIES = {
    "M2SL":    ("M2 Money Supply",     "M"),
    "M2V":     ("M2 Velocity",         "Q"),
    "CPIAUCSL":("CPI All Items",        "M"),
    "CPILFESL":("CPI Core",             "M"),
    "GDPC1":   ("Real GDP",             "Q"),
    "FEDFUNDS":("Fed Funds Rate",       "M"),
    "UNRATE":  ("Unemployment Rate",    "M"),
    "T10Y2Y":  ("Yield Curve 10Y-2Y",  "M"),
}

# Equivalenti europei scaricati via FRED (ECB/Eurostat su FRED)
# Le colonne corrispondenti vengono rinominate con gli stessi label di DEFAULT_SERIES
# così sidebar, make_mvpq_chart e update_charts funzionano senza modifiche
EUR_MONETARY_FRED = {
    "ECBDFR":          ("Fed Funds Rate",      "M"),   # BCE: tasso deposito facility
    "LRHUTTTTEZM156S": ("Unemployment Rate",   "M"),   # Disoccupazione Area Euro (%)
    "IRLTLT01EZM156N": ("EUR 10Y Yield",       "M"),   # Rendimento BTP/Bund 10Y area euro
    "IRT3TM01EZM156N": ("EUR 3M Yield",        "M"),   # Tasso 3M area euro
}

COLORS = [
    "#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd",
    "#8c564b","#e377c2","#17becf","#bcbd22","#7f7f7f",
]

# =============================================================================
# FUNZIONI DATI
# =============================================================================

def fred_get(series_id: str, api_key: str, retries: int = 3, delay: float = 2.0) -> pd.Series | None:
    if not FRED_AVAILABLE:
        return None
    import time
    for attempt in range(retries):
        try:
            s = Fred(api_key=api_key).get_series(series_id)
            s.index = pd.to_datetime(s.index)
            return s.dropna()
        except Exception as e:
            msg = str(e)
            if "403" in msg or "Forbidden" in msg:
                print(f"  FRED [{series_id}]: 403 Forbidden — API key bloccata o rate limit")
                return None  # inutile riprovare
            if "mismatched tag" in msg or "429" in msg or "rate" in msg.lower():
                if attempt < retries - 1:
                    print(f"  FRED [{series_id}]: rate limit, attendo {delay}s (tentativo {attempt+1}/{retries})")
                    time.sleep(delay)
                    delay *= 2
                    continue
            print(f"  FRED [{series_id}]: {e}")
            return None
    return None


def to_monthly(s: pd.Series, freq: str) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    if freq == "M":
        s.index = s.index.to_period("M").to_timestamp()
        s = s[~s.index.duplicated(keep="last")]
    elif freq == "Q":
        s.index = s.index.to_period("Q").to_timestamp()
        s = s[~s.index.duplicated(keep="last")]
        full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
        return s.reindex(full).ffill()
    elif freq in ("W", "BW", "D"):
        return s.resample("MS").last()
    else:
        return s.resample("MS").last()
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(full).ffill()


def bce_get_m2() -> pd.Series | None:
    """Scarica M2 Area Euro direttamente dalla BCE via SDMX REST API.
    Serie: BSI / M.U2.Y.V.M20.X.1.U2.2300.Z01.E  (mensile, miliardi EUR)
    """
    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        "BSI/M.U2.Y.V.M20.X.1.U2.2300.Z01.E"
        "?format=csvdata&startPeriod=1997-01"
    )
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/csv"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        s = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        s.index = pd.to_datetime(df["TIME_PERIOD"])
        s = s.dropna().sort_index()
        # I valori BCE sono in milioni EUR → convertiamo in miliardi
        s = s / 1000.0
        print(f"  ✓ BCE M2 Area Euro: {len(s)} obs")
        return s
    except Exception as e:
        print(f"  ✗ BCE M2: {e}")
        return None


def bce_get_yields_df() -> pd.DataFrame:
    """Scarica la curva dei rendimenti AAA area euro dalla BCE SDMX REST API.
    Dataset YC — spot rates da 3M a 30Y, dati giornalieri."""
    frames = {}
    for code, (label, _mat) in BCE_YIELD_SERIES.items():
        url = (
            "https://data-api.ecb.europa.eu/service/data/"
            f"YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{code}"
            "?format=csvdata"
        )
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/csv"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8")
            df_raw = pd.read_csv(io.StringIO(raw))
            s = pd.to_numeric(df_raw["OBS_VALUE"], errors="coerce")
            s.index = pd.to_datetime(df_raw["TIME_PERIOD"])
            frames[label] = s.dropna().sort_index()
            print(f"  ✓ BCE YC {code} → '{label}': {len(frames[label])} obs")
        except Exception as e:
            print(f"  ✗ BCE YC {code}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def eurostat_get(dataset: str, params: dict, geo: str) -> pd.Series | None:
    """Scarica una serie temporale dall'API JSON di Eurostat."""
    base = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    all_params = {**params, "geo": geo}
    qstr = "&".join(f"{k}={v}" for k, v in all_params.items())
    url  = f"{base}/{dataset}?{qstr}&lang=en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EurostatDash/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"  Eurostat [{dataset}/{geo}]: {e}")
        return None
    try:
        ids    = raw["id"]
        sizes  = raw["size"]
        dims   = raw["dimension"]
        values = raw["value"]
        if "time" not in ids:
            return None
        t_idx     = ids.index("time")
        time_cats = list(dims["time"]["category"]["index"].keys())
        stride = 1
        for s in sizes[t_idx + 1:]:
            stride *= s
        result = {}
        for i, tcat in enumerate(time_cats):
            v = values.get(str(i * stride))
            if v is not None:
                result[tcat] = float(v)
        if not result:
            return None
        s = pd.Series(result).sort_index()
        sample = s.index[0]
        if "-Q" in sample:
            s.index = pd.PeriodIndex(s.index, freq="Q").to_timestamp()
        elif len(sample) == 7 and sample[4] == "M":
            s.index = pd.to_datetime(s.index, format="%YM%m")
        else:
            s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:
        print(f"  Eurostat parse [{dataset}/{geo}]: {e}")
        return None


def eurostat_hicp_extended(geo: str, indic: str = "TOTAL") -> pd.Series | None:
    """
    Scarica HICP indice da ei_cphi_m (flash estimate) — aggiornato fino al mese corrente.
    Usato per estendere prc_hicp_midx che si è fermato a fine 2025 (base 2025).
    indic: "TOTAL" = All Items, "TOT_X_NRG_FOOD" = Core (ex energia/alimentari).
    Restituisce indice su base 2025=100 (va riscalato su 2015=100 dal chiamante).
    """
    base = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    # Prima prova EA20, poi EA19
    for g in [geo, "EA20", "EA19"]:
        url = f"{base}/ei_cphi_m?indic={indic}&unit=HICP2025&geo={g}&lang=en"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EurostatDash/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = json.loads(r.read().decode("utf-8"))
            ids    = raw["id"]
            sizes  = raw["size"]
            dims   = raw["dimension"]
            values = raw["value"]
            t_idx  = ids.index("time")
            time_cats = list(dims["time"]["category"]["index"].keys())
            stride = 1
            for s in sizes[t_idx + 1:]:
                stride *= s
            result = {}
            for i, tcat in enumerate(time_cats):
                v = values.get(str(i * stride))
                if v is not None:
                    result[tcat] = float(v)
            if not result:
                continue
            s = pd.Series(result).sort_index()
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception:
            continue
    return None


def hicp_esteso(geo: str, coicop: str = "CP00", indic: str = "TOTAL",
                label: str = "HICP") -> pd.Series | None:
    """HICP da prc_hicp_midx (base 2015=100, definitivo) esteso con il flash ei_cphi_m
    per i mesi più recenti. coicop/indic identificano la stessa serie nei due dataset:
    "CP00"/"TOTAL" = All Items, "TOT_X_NRG_FOOD"/"TOT_X_NRG_FOOD" = Core.

    Serve perché prc_hicp_midx unit=I15 si è di fatto FERMATO a fine 2025 con il
    passaggio di Eurostat alla base 2025=100: da solo lascia l'inflazione EUR indietro
    di mesi. Il flash è su base 2025=100 e viene riscalato sul periodo di
    sovrapposizione, così la serie resta continua in base 2015=100. Stesso raccordo
    già usato nella sezione shock/Phillips (curva IS-PC): tenerli allineati."""
    hicp = eurostat_get("prc_hicp_midx", {"coicop": coicop, "unit": "I15"}, geo)
    if hicp is None and geo not in ("EA20", "EA19"):
        hicp = eurostat_get("prc_hicp_midx", {"coicop": coicop, "unit": "I15"}, "EA20")
    flash = eurostat_hicp_extended(geo, indic)
    if flash is not None and hicp is not None:
        overlap = hicp.index.intersection(flash.index)
        if len(overlap) >= 6:
            ratio = hicp.reindex(overlap).mean() / flash.reindex(overlap).mean()
            flash_scaled = flash * ratio
            new_idx = flash_scaled.index[flash_scaled.index > hicp.index.max()]
            if len(new_idx):
                hicp = pd.concat([hicp, flash_scaled.reindex(new_idx)]).sort_index()
                print(f"    ✓ {label} esteso con flash fino a "
                      f"{hicp.index.max().strftime('%Y-%m')}")
    elif flash is not None and hicp is None:
        hicp = flash
    return hicp


def build_eurostat_dataframe(geo: str) -> pd.DataFrame:
    """Scarica tutte le serie EUROSTAT_SERIES per il paese/area geo.
    Le tuple possono avere un 5° elemento opzionale (geo_override) per
    serie comuni all'intera area euro (es. Euribor) indipendenti dal paese."""
    frames = {}
    for key, entry in EUROSTAT_SERIES.items():
        dataset, params, label, freq = entry[0], entry[1], entry[2], entry[3]
        effective_geo = entry[4] if len(entry) > 4 else geo
        raw = eurostat_get(dataset, params, effective_geo)
        if raw is not None:
            frames[label] = to_monthly(raw, freq)
            print(f"  ✓ Eurostat {key}/{effective_geo}: {len(frames[label])} obs")
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def build_monetary_eur_df(geo: str, api_key: str) -> pd.DataFrame:
    """Costruisce il DataFrame monetario per l'Area Euro con gli stessi
    nomi colonna di DEFAULT_SERIES, in modo che sidebar e MV=PQ chart
    funzionino senza modifiche.

    Fonti:
    - M2 Area Euro      → BCE SDMX REST API BSI/M.U2.Y.V.M20.X.1.U2.2300.Z01.E (label: "M2 Money Supply")
    - HICP All Items    → Eurostat prc_hicp_midx CP00   (label: "CPI All Items")
    - HICP Core         → Eurostat prc_hicp_midx CORE   (label: "CPI Core")
    - PIL Reale         → Eurostat namq_10_gdp           (label: "Real GDP")
    - PIL Nominale      → Eurostat namq_10_gdp           (per calcolo velocità)
    - M2 Velocity       → PIL Nominale / M2              (label: "M2 Velocity")
    - BCE tasso dep.    → FRED ECBDFR                    (label: "Fed Funds Rate")
    - Disoccupazione    → FRED LRHUTTTTEZM156S           (label: "Unemployment Rate")
    - Curva rendimenti  → FRED IRLTLT01EZM156N − IRT3TM01EZM156N (label: "Yield Curve 10Y-2Y")
    """
    frames = {}

    # ── M2 Area Euro dalla BCE ────────────────────────────────────────────────
    print("  ▶ EUR Monetary — BCE M2...")
    m2 = bce_get_m2()
    if m2 is not None:
        frames["M2 Money Supply"] = to_monthly(m2, "M")

    # ── Serie FRED per l'Area Euro ────────────────────────────────────────────
    print(f"  ▶ EUR Monetary — FRED series ({len(EUR_MONETARY_FRED)})...")
    for sid, (label, freq) in EUR_MONETARY_FRED.items():
        s = fred_get(sid, api_key)
        if s is not None:
            frames[label] = to_monthly(s, freq)
            print(f"    ✓ {sid} → '{label}'  ({len(frames[label])} obs)")
        else:
            print(f"    ✗ {sid} non disponibile")

    # ── Yield Curve 10Y - 3M ─────────────────────────────────────────────────
    if "EUR 10Y Yield" in frames and "EUR 3M Yield" in frames:
        frames["Yield Curve 10Y-2Y"] = (
            frames.pop("EUR 10Y Yield") - frames.pop("EUR 3M Yield")
        )
        print("    ✓ Yield Curve EUR = 10Y − 3M")
    else:
        frames.pop("EUR 10Y Yield", None)
        frames.pop("EUR 3M Yield", None)

    # ── Serie Eurostat (HICP + PIL) ───────────────────────────────────────────
    print(f"  ▶ EUR Monetary — Eurostat [{geo}]...")
    hicp_all  = hicp_esteso(geo, "CP00", "TOTAL", "HICP All Items")   # + flash ei_cphi_m
    hicp_core = hicp_esteso(geo, "TOT_X_NRG_FOOD", "TOT_X_NRG_FOOD", "HICP Core")
    pil_r_raw = eurostat_get("namq_10_gdp",   {"na_item": "B1GQ", "unit": "CLV15_MEUR", "s_adj": "SCA"}, geo)
    pil_n_raw = eurostat_get("namq_10_gdp",   {"na_item": "B1GQ", "unit": "CP_MEUR",    "s_adj": "SCA"}, geo)

    if hicp_all is not None:
        frames["CPI All Items"] = to_monthly(hicp_all, "M")
        print(f"    ✓ HICP All Items ({len(frames['CPI All Items'])} obs)")
    if hicp_core is not None:
        frames["CPI Core"] = to_monthly(hicp_core, "M")
        print(f"    ✓ HICP Core ({len(frames['CPI Core'])} obs)")
    if pil_r_raw is not None:
        frames["Real GDP"] = to_monthly(pil_r_raw, "Q")
        print(f"    ✓ Real GDP EUR ({len(frames['Real GDP'])} obs)")

    # ── M2 Velocity = PIL Nominale / M2 ──────────────────────────────────────
    m2_s = frames.get("M2 Money Supply")
    if pil_n_raw is not None and m2_s is not None:
        pil_n = to_monthly(pil_n_raw, "Q")
        # allinea sull'indice comune
        idx = m2_s.dropna().index.intersection(pil_n.dropna().index)
        if len(idx) >= 8:
            # M2 in miliardi EUR, PIL nominale (mln EUR) → convertiamo in miliardi
            vel = pil_n.reindex(idx) / 1000.0 / m2_s.reindex(idx)
            vel.name = "M2 Velocity"
            frames["M2 Velocity"] = vel
            print(f"    ✓ M2 Velocity ({len(vel)} obs)")
    elif m2_s is None:
        print("    ✗ M2 non disponibile — Velocity non calcolata")

    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def yfinance_monthly(ticker: str, name: str) -> pd.Series | None:
    """Scarica serie mensile da Yahoo Finance (close di fine mese)."""
    try:
        import yfinance as yf
        raw = yf.download(ticker, period="30y", interval="1mo",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        s = raw["Close"].squeeze()
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
        s = s[~s.index.duplicated(keep="last")].dropna()
        s.name = name
        print(f"  ✓ yfinance [{ticker}]: {len(s)} obs")
        return s
    except Exception as e:
        print(f"  yfinance [{ticker}]: {e}")
        return None


def build_dataframe(series_dict: dict, api_key: str) -> pd.DataFrame:
    frames = {}
    for sid, (label, freq) in series_dict.items():
        raw = fred_get(sid, api_key)
        if raw is not None:
            frames[label] = to_monthly(raw, freq)
            print(f"  ✓ {sid}: {len(frames[label])} obs")
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def build_daily_dataframe(series_dict: dict, api_key: str) -> pd.DataFrame:
    frames = {}
    for sid, (label, freq) in series_dict.items():
        raw = fred_get(sid, api_key)
        if raw is None:
            continue
        raw = raw.dropna()
        if freq == "D":
            daily_idx = pd.date_range(raw.index.min(), raw.index.max(), freq="B")
            raw = raw.reindex(daily_idx).ffill()
        elif freq == "M":
            raw.index = raw.index.to_period("M").to_timestamp()
            raw = raw[~raw.index.duplicated(keep="last")]
            daily_idx = pd.date_range(raw.index.min(), raw.index.max(), freq="B")
            raw = raw.reindex(daily_idx).ffill()
        elif freq == "Q":
            raw.index = raw.index.to_period("Q").to_timestamp()
            raw = raw[~raw.index.duplicated(keep="last")]
            daily_idx = pd.date_range(raw.index.min(), raw.index.max(), freq="B")
            raw = raw.reindex(daily_idx).ffill()
        frames[label] = raw
        print(f"  ✓ {sid}: {len(raw)} obs giornalieri")
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def _has_internet(host="8.8.8.8", timeout=4):
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, 53))
        return True
    except Exception:
        return False


def parse_excel_upload(contents, filename) -> tuple[dict | None, str]:
    try:
        _, data = contents.split(",")
        xl = pd.read_excel(io.BytesIO(base64.b64decode(data)))
    except Exception as e:
        return None, f"Errore lettura: {e}"
    if xl.shape[1] < 2:
        return None, "Servono almeno 2 colonne: series_id | label"
    c = xl.columns.tolist()
    result = {}
    for _, row in xl.iterrows():
        sid   = str(row[c[0]]).strip().upper()
        label = str(row[c[1]]).strip()
        freq  = str(row[c[2]]).strip().upper() if len(c) >= 3 else "M"
        result[sid] = (label, freq)
    return result, "ok"


# =============================================================================
# TRASFORMAZIONI VISUALIZZAZIONE
# =============================================================================

def yoy(s: pd.Series) -> pd.Series:
    return ((s - s.shift(12)) / s.shift(12).abs()) * 100


def _cumprod_series(s: pd.Series) -> pd.Series:
    r = s.pct_change().fillna(0)
    return ((1 + r).cumprod() - 1) * 100


def transform_df(df: pd.DataFrame, mode: str,
                 start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    result = {}
    for col in df.columns:
        s_full = df[col].dropna()
        if s_full.empty:
            continue
        if mode == "abs":
            result[col] = s_full.loc[start:end]
        elif mode == "yoy":
            result[col] = yoy(s_full).loc[start:end]
        elif mode == "cumsum":
            s_slice = s_full.loc[start:end].dropna()
            if s_slice.empty:
                continue
            result[col] = _cumprod_series(s_slice)
    if not result:
        return pd.DataFrame()
    out = pd.DataFrame(result)
    out.index = pd.to_datetime(out.index)
    return out


# =============================================================================
# FIGURE HELPER
# =============================================================================

def empty_fig(msg="") -> go.Figure:
    fig = go.Figure()
    if msg:
        fig.add_annotation(text=msg, xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#bbb"), align="center")
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                      margin=dict(t=30, b=20, l=40, r=20))
    return fig


def make_line_chart(df: pd.DataFrame, title: str,
                    y_label: str, zero_line: bool = False) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        s = df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=col,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %Y}}: %{{y:.2f}}<extra></extra>",
        ))
    if zero_line:
        fig.add_hline(y=0, line_color="#666", line_dash="dot", line_width=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#333"), x=0.01),
        yaxis_title=y_label,
        hovermode="closest",      # ← era "x unified"
        autosize=True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=9),
                    bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=50, b=35, l=60, r=20),
        paper_bgcolor="white", plot_bgcolor="#f8f8f8",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
    return fig


def make_mvpq_chart(df: pd.DataFrame, mode: str,
                    start: pd.Timestamp, end: pd.Timestamp,
                    mvpq_show: list = None) -> go.Figure:
    col_m2 = next((c for c in df.columns if "M2 Money" in c or "M2 " in c), None)
    col_v  = next((c for c in df.columns if "Velocity" in c or "Velocit" in c), None)
    col_p  = next((c for c in df.columns if "CPI All" in c), None)
    col_q  = next((c for c in df.columns if "GDP" in c or "PIL" in c), None)

    missing = [n for n, c in [("M2",col_m2),("Velocity",col_v),
                                ("CPI",col_p),("GDP",col_q)] if c is None]
    if missing:
        return empty_fig(f"Mancano serie per MV=PQ: {', '.join(missing)}")

    common = (df[col_m2].dropna().index
              .intersection(df[col_v].dropna().index)
              .intersection(df[col_p].dropna().index)
              .intersection(df[col_q].dropna().index))
    if len(common) < 24:
        return empty_fig("Dati insufficienti per MV=PQ (< 24 mesi comuni)")

    m2 = df[col_m2].reindex(common)
    v  = df[col_v].reindex(common)
    p  = df[col_p].reindex(common)
    q  = df[col_q].reindex(common)

    mv_raw = (m2 * v)
    pq_raw = (p  * q / 100)

    mv_full = mv_raw / mv_raw.iloc[0] * 100
    pq_full = pq_raw / pq_raw.iloc[0] * 100
    mv_yoy_full = yoy(mv_full)
    pq_yoy_full = yoy(pq_full)

    mv_yoy = mv_yoy_full.loc[start:end].copy()
    pq_yoy = pq_yoy_full.loc[start:end].copy()

    if mv_yoy.empty or pq_yoy.empty:
        return empty_fig("Nessun dato nel range selezionato")

    gap = mv_yoy - pq_yoy

    mv_raw_sl = mv_raw.loc[start:end].dropna()
    pq_raw_sl = pq_raw.loc[start:end].dropna()

    def _cumprod_from_zero(s):
        r = s.pct_change().fillna(0)
        cp = (1 + r).cumprod() - 1
        return cp * 100

    mv = _cumprod_from_zero(mv_raw_sl)
    pq = _cumprod_from_zero(pq_raw_sl)

    show_abs = "abs" in (mvpq_show or [])
    show_yoy = "yoy" in (mvpq_show or [])
    show_cum = "cum" in (mvpq_show or [])

    if not (show_abs or show_yoy or show_cum):
        return empty_fig("Seleziona Livelli, YoY o CumProd per il grafico MV=PQ")

    # ── livelli indicizzati (base 100): le due curve M·V e P·Q che si inseguono ──
    mv_abs = mv_raw_sl / mv_raw_sl.iloc[0] * 100 if not mv_raw_sl.empty else mv_raw_sl
    pq_abs = pq_raw_sl / pq_raw_sl.iloc[0] * 100 if not pq_raw_sl.empty else pq_raw_sl

    # ── medie del periodo selezionato ─────────────────────────────────────────
    mv_yoy_mean = float(mv_yoy.dropna().mean())
    pq_yoy_mean = float(pq_yoy.dropna().mean())
    mv_cum_mean = float(mv.dropna().mean())
    pq_cum_mean = float(pq.dropna().mean())

    panels = [k for k, on in (("abs", show_abs), ("yoy", show_yoy), ("cum", show_cum)) if on]
    _titles = {
        "abs": "Livelli indicizzati (base 100) — M·V vs P·Q",
        "yoy": "Δ% YoY — M·V vs P·Q  |  Gap = eccesso monetario",
        "cum": "Crescita % cumulata (base 0 = inizio slider)",
    }
    _yax = {"abs": "Indice (base 100)", "yoy": "Δ% YoY", "cum": "Crescita % cum"}

    n = len(panels)
    if n == 1:
        fig = go.Figure()
    else:
        fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                            subplot_titles=[_titles[p] for p in panels])

    for _i, _p in enumerate(panels):
        rc = {} if n == 1 else dict(row=_i + 1, col=1)
        if _p == "abs":
            fig.add_trace(go.Scatter(x=mv_abs.index, y=mv_abs.values, name="M·V (livello)",
                line=dict(color="#1f77b4", width=2.5),
                hovertemplate="M·V: %{y:.1f}<extra></extra>"), **rc)
            fig.add_trace(go.Scatter(x=pq_abs.index, y=pq_abs.values, name="P·Q (livello)",
                line=dict(color="#d62728", width=2.5),
                hovertemplate="P·Q: %{y:.1f}<extra></extra>"), **rc)
        elif _p == "yoy":
            fig.add_trace(go.Scatter(x=mv_yoy.index, y=mv_yoy.values, name="M·V YoY",
                line=dict(color="#1f77b4", width=2.5),
                hovertemplate="M·V YoY: %{y:.2f}%<extra></extra>"), **rc)
            fig.add_trace(go.Scatter(x=pq_yoy.index, y=pq_yoy.values, name="P·Q YoY",
                line=dict(color="#d62728", width=2.5),
                hovertemplate="P·Q YoY: %{y:.2f}%<extra></extra>"), **rc)
            fig.add_trace(go.Bar(x=gap.clip(lower=0).index, y=gap.clip(lower=0).values,
                name="Gap+ (MV>PQ)", marker_color="rgba(44,160,44,0.45)"), **rc)
            fig.add_trace(go.Bar(x=gap.clip(upper=0).index, y=gap.clip(upper=0).values,
                name="Gap− (PQ>MV)", marker_color="rgba(214,39,40,0.35)"), **rc)
            fig.add_hline(y=0, line_color="#555", line_dash="dot", line_width=1, **rc)
            fig.add_hline(y=mv_yoy_mean, line_color="#1f77b4", line_dash="dashdot", line_width=1.2,
                          annotation_text=f"μ M·V={mv_yoy_mean:.2f}%", annotation_position="right",
                          annotation_font=dict(size=9, color="#1f77b4"), **rc)
            fig.add_hline(y=pq_yoy_mean, line_color="#d62728", line_dash="dashdot", line_width=1.2,
                          annotation_text=f"μ P·Q={pq_yoy_mean:.2f}%", annotation_position="right",
                          annotation_font=dict(size=9, color="#d62728"), **rc)
        else:  # cum
            fig.add_trace(go.Scatter(x=mv.index, y=mv.values, name="M·V CumProd",
                line=dict(color="#1f77b4", width=2.5),
                hovertemplate="M·V cum: %{y:.2f}%<extra></extra>"), **rc)
            fig.add_trace(go.Scatter(x=pq.index, y=pq.values, name="P·Q CumProd",
                line=dict(color="#d62728", width=2.5),
                hovertemplate="P·Q cum: %{y:.2f}%<extra></extra>"), **rc)
            fig.add_hline(y=0, line_color="#999", line_dash="dot", line_width=1, **rc)
            fig.add_hline(y=mv_cum_mean, line_color="#1f77b4", line_dash="dashdot", line_width=1.2,
                          annotation_text=f"μ M·V={mv_cum_mean:.2f}%", annotation_position="right",
                          annotation_font=dict(size=9, color="#1f77b4"), **rc)
            fig.add_hline(y=pq_cum_mean, line_color="#d62728", line_dash="dashdot", line_width=1.2,
                          annotation_text=f"μ P·Q={pq_cum_mean:.2f}%", annotation_position="right",
                          annotation_font=dict(size=9, color="#d62728"), **rc)
        if n == 1:
            fig.update_yaxes(title_text=_yax[_p])
        else:
            fig.update_yaxes(title_text=_yax[_p], title_font=dict(size=9), **rc)

    fig.update_layout(
        title=dict(text="MV = PQ  —  M·V vs P·Q", font=dict(size=11), x=0.01),
        hovermode="closest",
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=9),
                    bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=50, b=35, l=60, r=100),
        paper_bgcolor="white", plot_bgcolor="#f8f8f8",
        barmode="overlay",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
    return fig


def _extract_mvpq_components(df: pd.DataFrame, suffix: str):
    """Estrae e calcola M, V, P, Q da un df con colonne suffissate."""
    col_m2 = next((c for c in df.columns if ("M2 Money" in c or "M2 " in c) and c.endswith(suffix)), None)
    col_v  = next((c for c in df.columns if ("Velocity" in c or "Velocit" in c) and c.endswith(suffix)), None)
    col_p  = next((c for c in df.columns if "CPI All" in c and c.endswith(suffix)), None)
    col_q  = next((c for c in df.columns if ("GDP" in c or "PIL" in c) and c.endswith(suffix)), None)
    missing = [n for n, c in [("M2", col_m2), ("Velocity", col_v), ("CPI", col_p), ("GDP", col_q)] if c is None]
    if missing:
        return None, missing
    common = (df[col_m2].dropna().index
              .intersection(df[col_v].dropna().index)
              .intersection(df[col_p].dropna().index)
              .intersection(df[col_q].dropna().index))
    if len(common) < 24:
        return None, ["dati insufficienti"]
    m2 = df[col_m2].reindex(common)
    v  = df[col_v].reindex(common)
    p  = df[col_p].reindex(common)
    q  = df[col_q].reindex(common)
    mv_raw = m2 * v
    pq_raw = p * q / 100
    return {"mv_raw": mv_raw, "pq_raw": pq_raw, "common": common}, []


def make_mvpq_both_chart(df: pd.DataFrame,
                         start: pd.Timestamp, end: pd.Timestamp,
                         mvpq_show: list = None,
                         series_show: list = None) -> go.Figure:
    """MV=PQ confronto USA 🇺🇸 vs Europa 🇪🇺 — due grafici affiancati."""
    show_yoy = "yoy" in (mvpq_show or [])
    show_cum = "cum" in (mvpq_show or [])
    if not show_yoy and not show_cum:
        return empty_fig("Seleziona almeno YoY o CumProd per il grafico MV=PQ")

    visible = set(series_show or ["mv_usa", "pq_usa", "mv_eur", "pq_eur"])

    usa, miss_usa = _extract_mvpq_components(df, "🇺🇸")
    eur, miss_eur = _extract_mvpq_components(df, "🇪🇺")

    if usa is None and eur is None:
        return empty_fig(f"Mancano serie MV=PQ — USA: {miss_usa}  EUR: {miss_eur}")

    def _compute(comp):
        mv_raw = comp["mv_raw"]
        pq_raw = comp["pq_raw"]
        mv_full    = mv_raw / mv_raw.iloc[0] * 100
        pq_full    = pq_raw / pq_raw.iloc[0] * 100
        mv_yoy     = yoy(mv_full).loc[start:end].dropna()
        pq_yoy     = yoy(pq_full).loc[start:end].dropna()
        mv_raw_sl  = mv_raw.loc[start:end].dropna()
        pq_raw_sl  = pq_raw.loc[start:end].dropna()
        def _cum(s):
            return ((1 + s.pct_change().fillna(0)).cumprod() - 1) * 100
        return {"mv_yoy": mv_yoy, "pq_yoy": pq_yoy,
                "mv_cum": _cum(mv_raw_sl), "pq_cum": _cum(pq_raw_sl),
                "gap_yoy": mv_yoy - pq_yoy}

    usa_d = _compute(usa) if usa is not None else None
    eur_d = _compute(eur) if eur is not None else None

    n_rows = (1 if show_yoy else 0) + (1 if show_cum else 0)

    titles = []
    if show_yoy:
        titles.append("Δ% YoY  —  M·V vs P·Q  (USA 🇺🇸 e Europa 🇪🇺)")
    if show_cum:
        titles.append("Crescita % cumulata  (base 0 = inizio slider)")

    if n_rows == 2:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.10,
            row_heights=[0.50, 0.50],
            subplot_titles=titles,
        )
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=titles)

    yoy_row = 1
    cum_row = 2 if n_rows == 2 else 1

    COLORS = {
        "mv_usa": "#1f77b4",
        "pq_usa": "#d62728",
        "mv_eur": "#2ca02c",
        "pq_eur": "#ff7f0e",
    }

    series_cfg = [
        ("mv_usa", "mv_yoy", "mv_cum", "M·V 🇺🇸", usa_d),
        ("pq_usa", "pq_yoy", "pq_cum", "P·Q 🇺🇸", usa_d),
        ("mv_eur", "mv_yoy", "mv_cum", "M·V 🇪🇺", eur_d),
        ("pq_eur", "pq_yoy", "pq_cum", "P·Q 🇪🇺", eur_d),
    ]

    for key, yoy_k, cum_k, label, d in series_cfg:
        if key not in visible or d is None:
            continue
        color = COLORS[key]
        if show_yoy:
            fig.add_trace(go.Scatter(
                x=d[yoy_k].index, y=d[yoy_k].values,
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"{label} YoY: %{{y:.2f}}%<extra></extra>",
            ), row=yoy_row, col=1)
        if show_cum:
            fig.add_trace(go.Scatter(
                x=d[cum_k].index, y=d[cum_k].values,
                name=label + (" cum" if show_yoy else ""),
                line=dict(color=color, width=2.5),
                hovertemplate=f"{label} cum: %{{y:.2f}}%<extra></extra>",
                showlegend=not show_yoy,
            ), row=cum_row, col=1)

    if show_yoy:
        fig.add_hline(y=0, line_color="#555", line_dash="dot", line_width=1,
                      row=yoy_row, col=1)
        fig.update_yaxes(title_text="Δ% YoY", title_font=dict(size=9),
                         row=yoy_row, col=1)
    if show_cum:
        fig.add_hline(y=0, line_color="#999", line_dash="dot", line_width=1,
                      row=cum_row, col=1)
        fig.update_yaxes(title_text="Crescita % cum", title_font=dict(size=9),
                         row=cum_row, col=1)

    fig.update_layout(
        title=dict(
            text="MV = PQ  —  USA 🇺🇸 vs Europa 🇪🇺",
            font=dict(size=11), x=0.01),
        hovermode="x unified",
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=9),
                    bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=55, b=35, l=55, r=40),
        paper_bgcolor="white", plot_bgcolor="#f8f8f8",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
    return fig


# =============================================================================
# SERIE FRED
# =============================================================================

YIELD_SERIES = {
    "FEDFUNDS": ("Fed Funds",    "M"),
    "DGS3MO":   ("3M Treasury",  "D"),
    "DGS6MO":   ("6M Treasury",  "D"),
    "DGS1":     ("1Y Treasury",  "D"),
    "DGS2":     ("2Y Treasury",  "D"),
    "DGS3":     ("3Y Treasury",  "D"),
    "DGS5":     ("5Y Treasury",  "D"),
    "DGS7":     ("7Y Treasury",  "D"),
    "DGS10":    ("10Y Treasury", "D"),
    "DGS20":    ("20Y Treasury", "D"),
    "DGS30":    ("30Y Treasury", "D"),
}
YIELD_MATURITIES = [0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
YIELD_LABELS = [v[0] for v in YIELD_SERIES.values()]

# Curva dei rendimenti BCE (AAA govies area euro, spot rate)
BCE_YIELD_SERIES = {
    "SR_3M":  ("3M 🇪🇺",  0.25),
    "SR_6M":  ("6M 🇪🇺",  0.5),
    "SR_1Y":  ("1Y 🇪🇺",  1),
    "SR_2Y":  ("2Y 🇪🇺",  2),
    "SR_3Y":  ("3Y 🇪🇺",  3),
    "SR_5Y":  ("5Y 🇪🇺",  5),
    "SR_7Y":  ("7Y 🇪🇺",  7),
    "SR_10Y": ("10Y 🇪🇺", 10),
    "SR_15Y": ("15Y 🇪🇺", 15),
    "SR_20Y": ("20Y 🇪🇺", 20),
    "SR_25Y": ("25Y 🇪🇺", 25),
    "SR_30Y": ("30Y 🇪🇺", 30),
}
BCE_YIELD_LABELS      = [v[0] for v in BCE_YIELD_SERIES.values()]
BCE_YIELD_MATURITIES  = [v[1] for v in BCE_YIELD_SERIES.values()]

GDP_SERIES = {
    "GDP":    ("PIL Nominale (mld $)",        "Q"),
    "GDPC1":  ("PIL Reale (mld $)",           "Q"),
    "GDPDEF": ("Deflatore PIL",               "Q"),
    "PCEC":   ("Consumi Privati (mld $)",     "Q"),
    "GPDI":   ("Invest. Privati (mld $)",     "Q"),
    "GCE":    ("Spesa Pubblica (mld $)",      "Q"),
    "EXPGS":  ("Esportazioni (mld $)",        "Q"),
    "IMPGS":  ("Importazioni (mld $)",        "Q"),
    "COE":    ("Redditi da Lavoro (mld $)",   "Q"),
    "HOANBS": ("Ore Lavorate (indice)",       "Q"),
    "UNRATE": ("Tasso Disoccupazione (%)",    "M"),
}
GDP_LABELS = [v[0] for v in GDP_SERIES.values()]
NET_EXP_LABEL = "Esportazioni Nette (mld $)"

# Serie scaricate direttamente nel tab ADL quando si seleziona USA
ADL_USA_SERIES = {
    # PIL e componenti (trimestrali → mensile via ffill)
    "GDP":        ("PIL Nominale USA (mld $)",     "Q"),
    "GDPC1":      ("PIL Reale USA (mld $)",        "Q"),
    "GDPDEF":     ("Deflatore PIL USA",            "Q"),
    "PCEC":       ("Consumi Privati USA (mld $)",  "Q"),
    "GPDI":       ("Investimenti USA (mld $)",     "Q"),
    "GCE":        ("Spesa Pubblica USA (mld $)",   "Q"),
    "EXPGS":      ("Esportazioni USA (mld $)",     "Q"),
    "IMPGS":      ("Importazioni USA (mld $)",     "Q"),
    "COE":        ("Redditi Lavoro USA (mld $)",   "Q"),
    # Mensili
    "UNRATE":     ("Disoccupazione USA (%)",       "M"),
    "CPIAUCSL":   ("CPI USA (indice)",             "M"),
    "CPILFESL":   ("Core CPI USA (indice)",        "M"),
    "M2SL":       ("M2 USA (mld $)",              "M"),
    "FEDFUNDS":   ("Fed Funds Rate (%)",           "M"),
    # Tassi (giornalieri → mensile)
    "GS10":       ("Treasury 10y (%)",            "M"),
    "GS2":        ("Treasury 2y (%)",             "M"),
    # Energia / mercati (mensili o giornalieri → mensile)
    "MCOILBRENTEU": ("Brent Petrolio ($/barile)", "M"),
    "DCOILWTICO":   ("WTI Petrolio ($/barile)",   "D"),
    "DHHNGSP":      ("Gas Naturale ($/MMBtu)",    "D"),
    "SP500":        ("S&P 500",                   "D"),
    "VIXCLS":       ("VIX",                       "D"),
}
ADL_NET_EXP_LABEL = "Esp. Nette USA (mld $)"

# ── Eurostat ──────────────────────────────────────────────────────────────────
EUROSTAT_GEO = {
    "EA20": "Area Euro (20)",
    "DE":   "Germania",
    "FR":   "Francia",
    "IT":   "Italia",
    "ES":   "Spagna",
    "NL":   "Paesi Bassi",
    "BE":   "Belgio",
    "AT":   "Austria",
    "PT":   "Portogallo",
    "FI":   "Finlandia",
    "GR":   "Grecia",
    "IE":   "Irlanda",
}

# (dataset, filtri_dim, label, freq)
# Tutte le variabili sono scaricate in LIVELLI (valori originali) per permettere
# all'utente di applicare trasformazioni (log, ldiff, yoy) nell'ADL.
EUROSTAT_SERIES = {
    # ── Conti nazionali trimestrali (namq_10_gdp) ────────────────────────────
    "pil_nom":    ("namq_10_gdp",  {"na_item": "B1GQ", "unit": "CP_MEUR",      "s_adj": "SCA"}, "PIL Nominale (mln €)",         "Q"),
    "pil_reale":  ("namq_10_gdp",  {"na_item": "B1GQ", "unit": "CLV15_MEUR",   "s_adj": "SCA"}, "PIL Reale (mln €, 2015=100)",  "Q"),
    "deflatore":  ("namq_10_gdp",  {"na_item": "B1GQ", "unit": "PD15_EUR",     "s_adj": "SCA"}, "Deflatore PIL (2015=100)",     "Q"),
    "consumi":    ("namq_10_gdp",  {"na_item": "P31_S14_S15", "unit": "CP_MEUR",    "s_adj": "SCA"}, "Consumi Privati Nom. (mln €)", "Q"),
    "consumi_r":  ("namq_10_gdp",  {"na_item": "P31_S14_S15", "unit": "CLV15_MEUR", "s_adj": "SCA"}, "Consumi Privati Reali (mln €)", "Q"),
    "gov":        ("namq_10_gdp",  {"na_item": "P3_S13", "unit": "CP_MEUR",    "s_adj": "SCA"}, "Spesa Pubblica Nom. (mln €)",  "Q"),
    "gov_r":      ("namq_10_gdp",  {"na_item": "P3_S13", "unit": "CLV15_MEUR", "s_adj": "SCA"}, "Spesa Pubblica Reale (mln €)", "Q"),
    "invest":     ("namq_10_gdp",  {"na_item": "P51G",  "unit": "CP_MEUR",     "s_adj": "SCA"}, "Investimenti FBCF Nom. (mln €)","Q"),
    "invest_r":   ("namq_10_gdp",  {"na_item": "P51G",  "unit": "CLV15_MEUR",  "s_adj": "SCA"}, "Investimenti FBCF Reali (mln €)","Q"),
    "export":     ("namq_10_gdp",  {"na_item": "P6",    "unit": "CP_MEUR",     "s_adj": "SCA"}, "Esportazioni EUR Nom. (mln €)", "Q"),
    "export_r":   ("namq_10_gdp",  {"na_item": "P6",    "unit": "CLV15_MEUR",  "s_adj": "SCA"}, "Esportazioni EUR Reali (mln €)","Q"),
    "import_":    ("namq_10_gdp",  {"na_item": "P7",    "unit": "CP_MEUR",     "s_adj": "SCA"}, "Importazioni EUR Nom. (mln €)", "Q"),
    "import_r":   ("namq_10_gdp",  {"na_item": "P7",    "unit": "CLV15_MEUR",  "s_adj": "SCA"}, "Importazioni EUR Reali (mln €)","Q"),
    "redditi":    ("namq_10_gdp",  {"na_item": "D1",    "unit": "CP_MEUR",     "s_adj": "SCA"}, "Redditi da Lavoro (mln €)",    "Q"),
    # ── Mercato del lavoro mensile ────────────────────────────────────────────
    "disoc":      ("une_rt_m",     {"sex": "T", "age": "TOTAL", "s_adj": "SA", "unit": "PC_ACT"}, "Tasso Disoccupazione EUR (%)", "M"),
    # ── Prezzi: HICP scaricato come INDICE LIVELLO (2015=100) ─────────────────
    # usa ldiff o yoy nell'ADL per ottenere l'inflazione
    "hicp_idx":   ("prc_hicp_midx", {"coicop": "CP00",          "unit": "I15"}, "HICP Indice (2015=100)",              "M"),
    # coicop "TOT_X_NRG_FOOD" = tutto escluso energia e alimentari (core)
    "hicp_core":  ("prc_hicp_midx", {"coicop": "TOT_X_NRG_FOOD","unit": "I15"}, "HICP Core Indice ex Energia/Cibo (2015=100)", "M"),
    # ── Tassi d'interesse (mercato monetario mensile) ─────────────────────────
    # geo=EA (non il paese selezionato) — Euribor è comune a tutta l'area euro
    # 5° elemento opzionale = geo_override
    "rate_3m":    ("irt_st_m",      {"int_rt": "IRT_M3"},                        "Tasso 3M Euribor (%)",                "M", "EA"),
    "rate_12m":   ("irt_st_m",      {"int_rt": "IRT_M12"},                       "Tasso 12M Euribor (%)",               "M", "EA"),
}
NET_EXP_EUR_LABEL    = "Esportazioni Nette EUR Nom. (mln €)"
NET_EXP_EUR_R_LABEL  = "Esportazioni Nette EUR Reali (mln €)"

# ── Indici azionari per paese (Yahoo Finance) ────────────────────────────────
# (ticker_yahoo, nome_display)
EUROSTAT_EQUITY = {
    "EA20": ("^STOXX50E", "Euro Stoxx 50"),
    "DE":   ("^GDAXI",    "DAX (Germania)"),
    "FR":   ("^FCHI",     "CAC 40 (Francia)"),
    "IT":   ("FTSEMIB.MI","FTSE MIB (Italia)"),
    "ES":   ("^IBEX",     "IBEX 35 (Spagna)"),
    "NL":   ("^AEX",      "AEX (Paesi Bassi)"),
    "BE":   ("^BFX",      "BEL 20 (Belgio)"),
    "AT":   ("^ATX",      "ATX (Austria)"),
    "PT":   ("PSI20.LS",  "PSI 20 (Portogallo)"),
    "FI":   ("^OMXHPI",   "OMX Helsinki (Finlandia)"),
    "GR":   ("GD.AT",     "Athens General (Grecia)"),
    "IE":   ("^ISEQ",     "ISEQ (Irlanda)"),
}

SHOCK_SERIES = {
    "DCOILWTICO":  ("Petrolio WTI ($/barile)",    "D"),
    "DHHNGSP":     ("Gas Naturale ($/MMBtu)",      "D"),
    "SP500":       ("S&P 500",                     "D"),
    "PPIACO":      ("PPI Materie Prime",           "M"),
    "T10YIE":      ("Inflazione attesa 10Y (BE)",  "D"),
    "VIXCLS":      ("VIX (Volatilità)",            "D"),
    "WTISPLC":     ("Produzione petrolio USA (Mb/d)","M"),
    "DEXUSEU":     ("EUR/USD",                     "D"),
}


def _shock_tab_layout():
    from dash import html, dcc

    EVENTS = [
        ("2020-03-01", "COVID-19 lockdown", "#d62728"),
        ("2021-11-01", "Inflazione picco USA", "#ff7f0e"),
        ("2022-02-24", "Invasione Ucraina", "#9467bd"),
        ("2022-06-01", "Fed +75bps", "#2ca02c"),
        ("2023-07-01", "OPEC+ tagli", "#8c564b"),
        ("2024-01-01", "Recessione Europa", "#e377c2"),
    ]

    events_options = [
        {"label": f" {e[1]}", "value": e[0]}
        for e in EVENTS
    ]

    return html.Div([
        html.Div([
            html.H3("Analisi Shock Offerta & Politica Monetaria",
                    style={"margin": "0 16px 0 0", "font-size": "15px",
                           "white-space": "nowrap", "color": "#1a3a5c"}),
            html.Button("🔄  Ricarica dati shock", id="btn-reload-shock",
                        n_clicks=0,
                        style={"background": "#5d4037", "color": "white",
                               "border": "none", "padding": "7px 18px",
                               "border-radius": "5px", "cursor": "pointer",
                               "font-size": "12px", "font-weight": "bold",
                               "margin-right": "12px"}),
            html.Div(id="shock-status",
                     style={"font-size": "11px", "color": "#444",
                            "font-style": "italic", "margin-right": "24px"}),

            # ── Separatore ──────────────────────────────────────────────────
            html.Div(style={"width": "1px", "height": "28px",
                            "background": "#bbb", "margin-right": "16px"}),

            # ── Sezione Eurostat ─────────────────────────────────────────────
            html.Span("🇪🇺 Eurostat:", style={"font-size": "11px",
                                               "font-weight": "bold",
                                               "color": "#1a3a5c",
                                               "margin-right": "8px",
                                               "white-space": "nowrap"}),
            dcc.Dropdown(
                id="shock-eur-geo",
                options=[{"label": v, "value": k}
                         for k, v in EUROSTAT_GEO.items()],
                value="EA20",
                clearable=False,
                style={"font-size": "11px", "width": "155px",
                       "margin-right": "8px"},
            ),
            html.Button("📥  Carica Eurostat", id="btn-shock-eur-load",
                        n_clicks=0,
                        style={"background": "#1a5276", "color": "white",
                               "border": "none", "padding": "7px 16px",
                               "border-radius": "5px", "cursor": "pointer",
                               "font-size": "12px", "font-weight": "bold",
                               "margin-right": "12px"}),
            html.Div(id="shock-eur-status",
                     style={"font-size": "11px", "color": "#444",
                            "font-style": "italic"}),
        ], style={"display": "flex", "align-items": "center",
                  "padding": "8px 16px", "background": "#f0f4fa",
                  "border-bottom": "1px solid #dee2e6",
                  "flex-wrap": "wrap", "gap": "8px"}),

        dcc.Tabs(id="shock-subtabs", value="shock-monitor", children=[

            dcc.Tab(label="📉  Monitor Shock", value="shock-monitor", children=[
                html.Div([
                    html.Div([
                        html.B("Variabili", style={"font-size": "11px"}),
                        html.Hr(style={"margin": "6px 0"}),
                        *[
                            html.Div(
                                dcc.Checklist(
                                    id={"type": "shock-check", "index": lbl},
                                    options=[{"label": f" {lbl}", "value": lbl}],
                                    value=[lbl] if lbl in [
                                        "Petrolio WTI ($/barile)",
                                        "Gas Naturale ($/MMBtu)",
                                        "S&P 500",
                                    ] else [],
                                    style={"font-size": "10px"},
                                    inputStyle={"margin-right": "4px"},
                                ),
                                style={"margin-bottom": "4px"}
                            )
                            for lbl in [v[0] for v in SHOCK_SERIES.values()]
                        ],
                        # ── Variabili Eurostat (disponibili dopo caricamento) ──
                        html.Hr(style={"margin": "8px 0"}),
                        html.B("🇪🇺 Serie Eurostat", style={"font-size": "10px",
                                                              "color": "#1a5276"}),
                        html.Div(id="shock-eur-checklist-container",
                                 children=[html.Div("— carica dati Eurostat —",
                                                    style={"font-size": "9px",
                                                           "color": "#aaa",
                                                           "font-style": "italic",
                                                           "margin-top": "4px"})]),
                        html.Hr(style={"margin": "8px 0"}),
                        html.B("Marcatori eventi", style={"font-size": "10px"}),
                        html.Div(
                            dcc.Checklist(
                                id="shock-events-check",
                                options=events_options,
                                value=[e[0] for e in EVENTS],
                                style={"font-size": "9px"},
                                inputStyle={"margin-right": "3px"},
                                labelStyle={"display": "block", "margin-bottom": "3px"},
                            ),
                            style={"margin-top": "5px"},
                        ),
                        html.Hr(style={"margin": "8px 0"}),
                        html.Div([
                            html.Label("Vista:", style={"font-size": "10px", "font-weight": "bold"}),
                            dcc.RadioItems(
                                id="shock-view",
                                options=[
                                    {"label": " Assoluti",  "value": "abs"},
                                    {"label": " Δ% YoY",    "value": "yoy"},
                                    {"label": " Cumulata",  "value": "cum"},
                                ],
                                value="abs", inline=False,
                                style={"font-size": "10px"},
                                inputStyle={"margin-right": "3px"},
                                labelStyle={"display": "block", "margin-bottom": "3px"},
                            ),
                        ]),
                    ], style={"width": "200px", "min-width": "190px",
                              "padding": "12px", "border-right": "1px solid #ddd",
                              "height": "calc(100vh - 160px)",
                              "overflow-y": "auto", "background": "#fafafa"}),

                    html.Div([
                        html.Div([
                            dcc.RangeSlider(
                                id="shock-slider",
                                min=0, max=1, value=[0, 1],
                                marks={}, step=86400,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div(id="shock-slider-label",
                                     style={"font-size": "10px", "color": "#666",
                                            "text-align": "center", "margin-top": "2px"}),
                        ], style={"padding": "8px 28px 4px"}),

                        dcc.Loading(type="circle", children=[
                            dcc.Graph(id="chart-shock-main",
                                      figure=empty_fig("Carica i dati con 🔄 Ricarica"),
                                      style={"height": "55vh"},
                                      config={"responsive": True, "scrollZoom": True}),
                            dcc.Graph(id="chart-shock-corr",
                                      figure=empty_fig(""),
                                      style={"height": "28vh"},
                                      config={"responsive": True}),
                        ]),
                    ], style={"flex": "1", "min-width": "0",
                              "overflow-y": "auto",
                              "height": "calc(100vh - 160px)"}),

                ], style={"display": "flex"}),
            ]),

            dcc.Tab(label="📊  Modello Impatto", value="shock-impact", children=[
                html.Div([
                    html.Div([
                        html.B("⓪ Dati disponibili",
                               style={"font-size": "10px", "color": "#888",
                                      "display": "block", "margin-bottom": "3px"}),
                        html.Div(id="impact-source-label",
                                 children="— carica dati FRED o Eurostat —",
                                 style={"font-size": "9px", "color": "#aaa",
                                        "font-style": "italic",
                                        "margin-bottom": "8px"}),

                        html.Hr(style={"margin": "6px 0"}),
                        html.B("① Variabile dipendente Y",
                               style={"font-size": "10px", "color": "#1a3a5c",
                                      "display": "block", "margin-bottom": "4px"}),
                        dcc.Dropdown(id="impact-y-drop",
                                     placeholder="Seleziona Y…",
                                     clearable=False,
                                     style={"font-size": "10px", "margin-bottom": "5px"}),
                        dcc.RadioItems(
                            id="impact-y-tr",
                            options=[
                                {"label": " Livelli", "value": "levels"},
                                {"label": " YoY",     "value": "yoy"},
                                {"label": " Log",     "value": "log"},
                                {"label": " Δlog",    "value": "dlog"},
                            ],
                            value="yoy", inline=True,
                            style={"font-size": "9px"},
                            inputStyle={"margin-right": "2px"},
                            labelStyle={"margin-right": "8px"},
                        ),

                        html.Hr(style={"margin": "8px 0"}),
                        html.B("② Lag AR di Y",
                               style={"font-size": "10px", "color": "#1a3a5c",
                                      "display": "block", "margin-bottom": "4px"}),
                        dcc.Dropdown(
                            id="impact-ar",
                            options=[{"label": f"AR({k})", "value": k}
                                     for k in range(1, 13)],
                            value=[1], multi=True, clearable=False,
                            style={"font-size": "10px"},
                        ),

                        html.Hr(style={"margin": "8px 0"}),
                        html.B("③ Variabili X (attiva + trasf. + lag)",
                               style={"font-size": "10px", "color": "#1a3a5c",
                                      "background": "#eaf4fb", "display": "block",
                                      "padding": "3px 6px", "border-radius": "3px",
                                      "margin-bottom": "6px"}),
                        html.Div(id="impact-x-panel",
                                 children=html.Div(
                                     "— carica dati per vedere le variabili —",
                                     style={"font-size": "9px", "color": "#aaa",
                                            "font-style": "italic"})),

                        html.Hr(style={"margin": "8px 0"}),
                        html.B("④ Periodo campione",
                               style={"font-size": "10px", "color": "#1a3a5c",
                                      "display": "block", "margin-bottom": "4px"}),
                        dcc.RangeSlider(
                            id="impact-slider",
                            min=0, max=1, value=[0, 1],
                            marks={}, step=86400 * 30,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(id="impact-slider-label",
                                 style={"font-size": "9px", "color": "#666",
                                        "text-align": "center", "margin-bottom": "6px"}),

                        html.Hr(style={"margin": "8px 0"}),
                        html.B("⑤ Opzioni modello",
                               style={"font-size": "10px", "color": "#1a3a5c",
                                      "display": "block", "margin-bottom": "4px"}),
                        dcc.RadioItems(
                            id="impact-cov",
                            options=[
                                {"label": " OLS",  "value": "nonrobust"},
                                {"label": " HC3",  "value": "HC3"},
                                {"label": " HAC",  "value": "HAC"},
                            ],
                            value="HC3", inline=True,
                            style={"font-size": "9px"},
                            inputStyle={"margin-right": "2px"},
                            labelStyle={"margin-right": "8px"},
                        ),
                        dcc.Checklist(
                            id="impact-const",
                            options=[{"label": " Costante", "value": "const"}],
                            value=["const"],
                            style={"font-size": "9px", "margin-top": "4px"},
                            inputStyle={"margin-right": "3px"},
                        ),

                        html.Hr(style={"margin": "8px 0"}),
                        html.Button("▶  Stima impatto", id="btn-run-impact",
                                    n_clicks=0,
                                    style={"background": "#1b5e20", "color": "white",
                                           "border": "none", "padding": "7px 14px",
                                           "border-radius": "4px", "cursor": "pointer",
                                           "font-size": "12px", "font-weight": "bold",
                                           "width": "100%"}),
                        html.Div(id="impact-status",
                                 style={"font-size": "10px", "color": "#555",
                                        "margin-top": "5px", "font-style": "italic"}),

                        html.Hr(style={"margin": "8px 0"}),
                        html.B("⑥ Scenario prezzo petrolio",
                               style={"font-size": "10px", "color": "#7f2a00",
                                      "display": "block", "margin-bottom": "6px"}),
                        html.Div("Stima prima il modello, poi simula uno shock sul prezzo del petrolio.",
                                 style={"font-size": "9px", "color": "#999",
                                        "font-style": "italic", "margin-bottom": "8px"}),
                        html.Label("Prezzo corrente ($/bbl):",
                                   style={"font-size": "10px", "color": "#444",
                                          "display": "block", "margin-bottom": "2px"}),
                        dcc.Input(id="impact-oil-current", type="number", value=80,
                                  min=1, max=500, step=1,
                                  style={"width": "100%", "font-size": "11px",
                                         "padding": "3px 6px", "border": "1px solid #ccc",
                                         "border-radius": "3px", "margin-bottom": "6px"}),
                        html.Label("Prezzo scenario ($/bbl):",
                                   style={"font-size": "10px", "color": "#444",
                                          "display": "block", "margin-bottom": "2px"}),
                        dcc.Input(id="impact-oil-scenario", type="number", value=100,
                                  min=1, max=500, step=1,
                                  style={"width": "100%", "font-size": "11px",
                                         "padding": "3px 6px", "border": "1px solid #ccc",
                                         "border-radius": "3px", "margin-bottom": "6px"}),
                        html.Label("Orizzonte proiezione (mesi):",
                                   style={"font-size": "10px", "color": "#444",
                                          "display": "block", "margin-bottom": "4px"}),
                        dcc.Slider(id="impact-proj-months", min=1, max=24, value=12,
                                   step=1,
                                   marks={1:"1", 6:"6", 12:"12", 18:"18", 24:"24"},
                                   tooltip={"placement": "bottom", "always_visible": False}),
                        html.Button("▶  Simula scenario", id="btn-run-impact-scenario",
                                    n_clicks=0,
                                    style={"background": "#7f2a00", "color": "white",
                                           "border": "none", "padding": "7px 14px",
                                           "border-radius": "4px", "cursor": "pointer",
                                           "font-size": "12px", "font-weight": "bold",
                                           "width": "100%", "margin-top": "8px"}),
                        html.Div(id="impact-scenario-status",
                                 style={"font-size": "10px", "color": "#7f2a00",
                                        "margin-top": "5px", "font-style": "italic"}),
                        # campo nascosto cpi (legacy)
                        dcc.Input(id="impact-cpi-current", type="number", value=3.5,
                                  style={"display": "none"}),

                    ], style={"width": "240px", "min-width": "230px",
                              "padding": "12px", "border-right": "1px solid #ddd",
                              "height": "calc(100vh - 160px)",
                              "overflow-y": "auto", "background": "#fafafa"}),

                    html.Div([
                        dcc.Loading(type="circle", children=[
                            html.Div(id="impact-equation",
                                     style={"font-family": "monospace", "font-size": "11px",
                                            "background": "#f8f9fa", "border": "1px solid #dee2e6",
                                            "border-radius": "4px", "padding": "10px 16px",
                                            "margin": "10px 16px 0", "white-space": "pre-wrap",
                                            "color": "#1a3a5c"}),
                            html.Div(id="impact-stats-table",
                                     style={"margin": "8px 16px 0"}),
                            html.Div(id="impact-coef-table",
                                     style={"margin": "6px 16px 0"}),
                            dcc.Graph(id="chart-impact-fit",
                                      figure=empty_fig("Stima il modello"),
                                      style={"height": "35vh", "margin-top": "8px"},
                                      config={"responsive": True, "scrollZoom": True}),
                            dcc.Graph(id="chart-impact-irf",
                                      figure=empty_fig("IRF — 12 mesi"),
                                      style={"height": "42vh"},
                                      config={"responsive": True}),
                            dcc.Graph(id="chart-impact-projection",
                                      figure=empty_fig("Residui"),
                                      style={"height": "28vh"},
                                      config={"responsive": True}),
                        ]),
                    ], style={"flex": "1", "min-width": "0",
                              "overflow-y": "auto",
                              "height": "calc(100vh - 160px)"}),
                ], style={"display": "flex"}),
            ]),

            dcc.Tab(label="🔬  ADL Shock", value="shock-adl", children=[
                html.Div([

                    # ── Pannello sinistra ──────────────────────────────────────
                    html.Div([
                        html.B("⓪ Fonte dati attiva",
                               style={"font-size": "10px", "color": "#888",
                                      "display": "block", "margin-bottom": "4px"}),
                        html.Div(id="shock-adl-source-label",
                                 children="— carica dati FRED o Eurostat —",
                                 style={"font-size": "9px", "color": "#aaa",
                                        "font-style": "italic",
                                        "margin-bottom": "10px"}),

                        html.Hr(style={"margin": "6px 0"}),
                        html.B("① Variabile dipendente Y",
                               style={"font-size": "10px", "color": "#1a3a5c",
                                      "display": "block", "margin-bottom": "4px"}),
                        dcc.Dropdown(id="shock-adl-y",
                                     placeholder="Seleziona Y…",
                                     clearable=False,
                                     style={"font-size": "10px",
                                            "margin-bottom": "6px"}),
                        dcc.RadioItems(
                            id="shock-adl-y-tr",
                            options=[
                                {"label": " Livelli",  "value": "levels"},
                                {"label": " YoY",      "value": "yoy"},
                                {"label": " Log",      "value": "log"},
                                {"label": " Δlog",     "value": "dlog"},
                            ],
                            value="yoy", inline=True,
                            style={"font-size": "9px"},
                            inputStyle={"margin-right": "2px"},
                            labelStyle={"margin-right": "8px"},
                        ),

                        html.Hr(style={"margin": "8px 0"}),
                        html.B("② Lag AR di Y",
                               style={"font-size": "10px", "color": "#1a3a5c",
                                      "display": "block", "margin-bottom": "4px"}),
                        dcc.Dropdown(
                            id="shock-adl-ar",
                            options=[{"label": f"AR({k})", "value": k}
                                     for k in range(1, 13)],
                            value=[1], multi=True, clearable=False,
                            style={"font-size": "10px"},
                        ),

                        html.Hr(style={"margin": "8px 0"}),
                        html.B("③ Variabili X (attiva + trasf. + lag)",
                               style={"font-size": "10px", "color": "#1a3a5c",
                                      "background": "#eaf4fb", "display": "block",
                                      "padding": "3px 6px", "border-radius": "3px",
                                      "margin-bottom": "6px"}),
                        html.Div(id="shock-adl-x-panel",
                                 children=html.Div(
                                     "— carica dati per vedere le variabili —",
                                     style={"font-size": "9px", "color": "#aaa",
                                            "font-style": "italic"})),

                        html.Hr(style={"margin": "8px 0"}),
                        html.B("④ Periodo campione",
                               style={"font-size": "10px", "color": "#1a3a5c",
                                      "display": "block", "margin-bottom": "4px"}),
                        dcc.RangeSlider(
                            id="shock-adl-slider",
                            min=0, max=1, value=[0, 1],
                            marks={}, step=86400 * 30,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(id="shock-adl-slider-label",
                                 style={"font-size": "9px", "color": "#666",
                                        "text-align": "center",
                                        "margin-bottom": "6px"}),

                        html.Hr(style={"margin": "8px 0"}),
                        html.B("⑤ Opzioni modello",
                               style={"font-size": "10px", "color": "#1a3a5c",
                                      "display": "block", "margin-bottom": "4px"}),
                        dcc.RadioItems(
                            id="shock-adl-cov",
                            options=[
                                {"label": " OLS",  "value": "nonrobust"},
                                {"label": " HC3",  "value": "HC3"},
                                {"label": " HAC",  "value": "HAC"},
                            ],
                            value="HC3", inline=True,
                            style={"font-size": "9px"},
                            inputStyle={"margin-right": "2px"},
                            labelStyle={"margin-right": "8px"},
                        ),
                        dcc.Checklist(
                            id="shock-adl-const",
                            options=[{"label": " Costante", "value": "const"}],
                            value=["const"],
                            style={"font-size": "9px", "margin-top": "4px"},
                            inputStyle={"margin-right": "3px"},
                        ),

                        html.Hr(style={"margin": "8px 0"}),
                        html.Button("▶  Stima ADL",
                                    id="btn-run-shock-adl",
                                    n_clicks=0,
                                    style={"background": "#1b5e20", "color": "white",
                                           "border": "none", "padding": "8px 14px",
                                           "border-radius": "4px", "cursor": "pointer",
                                           "font-size": "12px", "font-weight": "bold",
                                           "width": "100%"}),
                        html.Div(id="shock-adl-status",
                                 style={"font-size": "10px", "color": "#555",
                                        "margin-top": "5px", "font-style": "italic"}),
                    ], style={"width": "230px", "min-width": "220px",
                              "padding": "12px", "border-right": "1px solid #ddd",
                              "height": "calc(100vh - 160px)",
                              "overflow-y": "auto", "background": "#fafafa"}),

                    # ── Area grafici ───────────────────────────────────────────
                    html.Div([
                        dcc.Loading(type="circle", children=[
                            html.Div(id="shock-adl-equation",
                                     style={"font-family": "monospace",
                                            "font-size": "11px",
                                            "background": "#f8f9fa",
                                            "border": "1px solid #dee2e6",
                                            "border-radius": "4px",
                                            "padding": "10px 16px",
                                            "margin": "10px 16px 0",
                                            "white-space": "pre-wrap",
                                            "color": "#1a3a5c"}),
                            html.Div(id="shock-adl-stats",
                                     style={"margin": "8px 16px 0"}),
                            html.Div(id="shock-adl-coef",
                                     style={"margin": "6px 16px 0"}),
                            dcc.Graph(id="chart-shock-adl-fit",
                                      figure=empty_fig("Stima il modello per vedere il fit"),
                                      style={"height": "38vh", "margin-top": "8px"},
                                      config={"responsive": True, "scrollZoom": True}),
                            dcc.Graph(id="chart-shock-adl-irf",
                                      figure=empty_fig("IRF — impulse response su 12 mesi"),
                                      style={"height": "42vh"},
                                      config={"responsive": True}),
                            dcc.Graph(id="chart-shock-adl-resid",
                                      figure=empty_fig(""),
                                      style={"height": "28vh"},
                                      config={"responsive": True}),
                        ]),
                    ], style={"flex": "1", "min-width": "0",
                              "overflow-y": "auto",
                              "height": "calc(100vh - 160px)"}),

                ], style={"display": "flex"}),
            ]),

            dcc.Tab(label="🎛  Simulatore Politica", value="shock-sim", children=[
                html.Div([
                    html.Div([
                        html.B("⚡ Shock esogeni", style={"font-size": "11px", "color": "#b71c1c"}),
                        html.Hr(style={"margin": "4px 0 8px"}),
                        html.Label("Petrolio WTI (var. %):", style={"font-size": "10px"}),
                        dcc.Slider(id="sim-oil-chg", min=-60, max=100, step=5, value=30,
                                   marks={-60: "-60%", -30: "-30%", 0: "0", 30: "+30%",
                                          60: "+60%", 100: "+100%"},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(style={"height": "10px"}),
                        html.Label("Gas Naturale (var. %):", style={"font-size": "10px"}),
                        dcc.Slider(id="sim-gas-chg", min=-60, max=200, step=10, value=50,
                                   marks={-60: "-60%", 0: "0", 50: "+50%",
                                          100: "+100%", 200: "+200%"},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(style={"height": "10px"}),
                        html.Label("Offerta petrolio russo (var. Mb/d):",
                                   style={"font-size": "10px"}),
                        dcc.Slider(id="sim-russia-supply", min=-4, max=2, step=0.5, value=-2,
                                   marks={-4: "-4Mb/d", -2: "-2", 0: "0", 2: "+2Mb/d"},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(style={"height": "10px"}),
                        html.Label("Riserve strategiche rilasciate (Mb):",
                                   style={"font-size": "10px"}),
                        dcc.Slider(id="sim-spr-release", min=0, max=180, step=10, value=0,
                                   marks={0: "0", 60: "60", 120: "120", 180: "180Mb"},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Hr(style={"margin": "10px 0 8px"}),
                        html.B("🏦 Politica monetaria", style={"font-size": "11px", "color": "#1a5276"}),
                        html.Hr(style={"margin": "4px 0 8px"}),
                        html.Label("Variazione tasso (bps):", style={"font-size": "10px"}),
                        dcc.Slider(id="sim-rate-chg", min=-300, max=400, step=25, value=0,
                                   marks={-300: "-300", -100: "-100", 0: "0",
                                          100: "+100", 200: "+200", 400: "+400"},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(style={"height": "10px"}),
                        html.Label("Orizzonte risposta (mesi):", style={"font-size": "10px"}),
                        dcc.Slider(id="sim-horizon", min=3, max=24, step=3, value=12,
                                   marks={3: "3m", 6: "6m", 12: "12m", 24: "24m"},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(style={"height": "10px"}),
                        html.Label("Variazione M2 (% annua):", style={"font-size": "10px"}),
                        dcc.Slider(id="sim-m2-growth", min=-5, max=15, step=0.5, value=4,
                                   marks={-5: "-5%", 0: "0", 4: "4%", 8: "8%", 15: "15%"},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(style={"height": "10px"}),
                        html.Label("Velocità M2 (var. %):", style={"font-size": "10px"}),
                        dcc.Slider(id="sim-velocity-chg", min=-10, max=10, step=0.5, value=-2,
                                   marks={-10: "-10%", -2: "-2%", 0: "0", 5: "+5%", 10: "+10%"},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Hr(style={"margin": "10px 0 8px"}),
                        html.Div([
                            html.Button("▶  Calcola scenario",
                                        id="btn-run-sim", n_clicks=0,
                                        style={"background": "#0d47a1", "color": "white",
                                               "border": "none", "padding": "8px 14px",
                                               "border-radius": "4px", "cursor": "pointer",
                                               "font-size": "12px", "font-weight": "bold",
                                               "width": "100%", "margin-bottom": "6px"}),
                            html.Button("🎯  Calcola tasso ottimale",
                                        id="btn-optimize-rate", n_clicks=0,
                                        style={"background": "#b71c1c", "color": "white",
                                               "border": "none", "padding": "8px 14px",
                                               "border-radius": "4px", "cursor": "pointer",
                                               "font-size": "12px", "font-weight": "bold",
                                               "width": "100%"}),
                        ]),
                        html.Details([
                            html.Summary("⚙ Funzione di perdita BC",
                                         style={"font-size": "10px", "cursor": "pointer",
                                                "color": "#555", "margin-top": "8px"}),
                            html.Div([
                                html.Label("λ inflazione:", style={"font-size": "10px"}),
                                dcc.Slider(id="loss-lambda-pi", min=0.1, max=3.0, step=0.1, value=1.0,
                                           marks={0.1: "0.1", 1.0: "1.0", 2.0: "2.0", 3.0: "3.0"},
                                           tooltip={"placement": "bottom", "always_visible": True}),
                                html.Div(style={"height": "8px"}),
                                html.Label("λ output gap:", style={"font-size": "10px"}),
                                dcc.Slider(id="loss-lambda-y", min=0.1, max=2.0, step=0.1, value=0.5,
                                           marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0", 2.0: "2.0"},
                                           tooltip={"placement": "bottom", "always_visible": True}),
                                html.Div(style={"height": "8px"}),
                                html.Label("Target inflazione (%):", style={"font-size": "10px"}),
                                dcc.Slider(id="loss-pi-target", min=0, max=4, step=0.5, value=2.0,
                                           marks={0: "0%", 2: "2%", 4: "4%"},
                                           tooltip={"placement": "bottom", "always_visible": True}),
                            ], style={"padding": "8px 4px"}),
                        ]),
                    ], style={"width": "280px", "min-width": "270px",
                              "padding": "12px", "border-right": "1px solid #ddd",
                              "height": "calc(100vh - 160px)",
                              "overflow-y": "auto", "background": "#fafafa"}),

                    html.Div([
                        dcc.Loading(type="circle", children=[
                            html.Div(id="sim-results-panel",
                                     style={"margin": "12px 16px 0"}),
                            html.Div(id="sim-optimal-panel",
                                     style={"margin": "8px 16px 0"}),
                            dcc.Graph(id="chart-sim-macro",
                                      figure=empty_fig("Configura lo scenario e clicca Calcola"),
                                      style={"height": "40vh"},
                                      config={"responsive": True}),
                            dcc.Graph(id="chart-sim-mvpq",
                                      figure=empty_fig(""),
                                      style={"height": "28vh"},
                                      config={"responsive": True}),
                            dcc.Graph(id="chart-sim-tradeoff",
                                      figure=empty_fig(""),
                                      style={"height": "30vh"},
                                      config={"responsive": True}),
                        ]),
                    ], style={"flex": "1", "min-width": "0",
                              "overflow-y": "auto",
                              "height": "calc(100vh - 160px)"}),
                ], style={"display": "flex"}),
            ]),
        ]),
    ])


def _slider_params(df: pd.DataFrame, step_years: int = 5):
    mn = int(df.index.min().timestamp())
    mx = int(df.index.max().timestamp())
    marks = {
        int(pd.Timestamp(yr, 1, 1).timestamp()): str(yr)
        for yr in range(df.index.min().year, df.index.max().year + 1, step_years)
    }
    return mn, mx, [mn, mx], marks


def _slider_params_daily(df: pd.DataFrame):
    return _slider_params(df, step_years=1)


# =============================================================================
# APP DASH
# =============================================================================

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[
               'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;600;700&display=swap',
               'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
           ],
           requests_pathname_prefix='/fred/', routes_pathname_prefix='/fred/')

app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
@keyframes pulse {
    0%   { opacity: 1;   transform: scale(1);    }
    50%  { opacity: 0.5; transform: scale(1.15); }
    100% { opacity: 1;   transform: scale(1);    }
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
/* embed: nasconde la barra delle tab → mostra solo la tab attiva */
html.fred-embed #main-tabs > div:first-child { display: none !important; }
</style>
<script>
  if (window.location.search.indexOf('embed=1') !== -1) {
    document.documentElement.classList.add('fred-embed');
  }
</script>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""

DEFAULT_LABELS = [v[0] for v in DEFAULT_SERIES.values()]


def _sidebar_default():
    return html.Div([
        html.Div(html.B("Serie attive", style={"font-size": "11px"}),
                 style={"padding-bottom": "6px", "margin-bottom": "8px",
                        "border-bottom": "2px solid #ccc"}),
        html.Div(id="mon-series-checklist",
                 children="— carica i dati per vedere le serie —",
                 style={"font-size": "10px", "color": "#aaa",
                        "font-style": "italic"}),
        html.Hr(style={"margin": "10px 0"}),
        html.Div([
            html.Button("✔ Tutto", id="sel-all", n_clicks=0,
                        style={"font-size": "9px", "padding": "2px 7px",
                               "margin-right": "4px", "cursor": "pointer"}),
            html.Button("✘ Niente", id="sel-none", n_clicks=0,
                        style={"font-size": "9px", "padding": "2px 7px",
                               "cursor": "pointer"}),
        ], style={"display": "flex"}),
        html.Div(id="custom-series-checklist"),
    ], style={"padding": "12px", "overflow-y": "auto"})


def _controls_bar():
    return html.Div([
        # Fonte dati
        html.Div([
            html.Label("Fonte:", style={"font-size": "11px", "font-weight": "bold",
                                        "margin-right": "8px", "white-space": "nowrap"}),
            dcc.RadioItems(
                id="mon-source-type",
                options=[
                    {"label": " 🇺🇸 USA (FRED)", "value": "usa"},
                    {"label": " 🇪🇺 Eurostat",   "value": "eur"},
                    {"label": " 🆚 Confronto",    "value": "both"},
                ],
                value="usa", inline=True,
                style={"font-size": "11px"},
                inputStyle={"margin-right": "3px"},
                labelStyle={"margin-right": "12px"},
            ),
            html.Div(
                dcc.Dropdown(
                    id="mon-eur-geo",
                    options=[{"label": v, "value": k}
                             for k, v in EUROSTAT_GEO.items()],
                    value="EA20", clearable=False,
                    style={"font-size": "10px", "min-width": "160px"},
                ),
                id="mon-geo-wrapper",
                style={"display": "none", "margin-left": "6px"},
            ),
        ], style={"display": "flex", "align-items": "center",
                  "background": "#f3e5f5", "border": "1px solid #ce93d8",
                  "border-radius": "4px", "padding": "5px 12px",
                  "margin-right": "14px"}),

        html.Div([
            html.Label("Vista:", style={"font-size": "11px", "font-weight": "bold",
                                        "margin-right": "8px",
                                        "white-space": "nowrap"}),
            dcc.RadioItems(
                id="view-mode",
                options=[
                    {"label": " Assoluta",            "value": "abs"},
                    {"label": " Δ% YoY",              "value": "yoy"},
                    {"label": " Cumulata % (da slider)", "value": "cumsum"},
                ],
                value="yoy", inline=True,
                style={"font-size": "11px"},
                inputStyle={"margin-right": "3px"},
                labelStyle={"margin-right": "14px"},
            ),
        ], style={"display": "flex", "align-items": "center",
                  "background": "#fff8e1", "border": "1px solid #ffe082",
                  "border-radius": "4px", "padding": "5px 12px",
                  "margin-right": "14px"}),

        html.Div([
            html.Label("MV=PQ:", style={"font-size": "11px", "font-weight": "bold",
                                         "margin-right": "8px", "white-space": "nowrap"}),
            dcc.Checklist(
                id="mvpq-show",
                options=[
                    {"label": " Livelli", "value": "abs"},
                    {"label": " YoY",     "value": "yoy"},
                    {"label": " CumProd", "value": "cum"},
                ],
                value=["abs", "yoy", "cum"],
                inline=True,
                style={"font-size": "11px"},
                inputStyle={"margin-right": "3px"},
                labelStyle={"margin-right": "10px"},
            ),
        ], style={"display": "flex", "align-items": "center",
                  "background": "#e8f5e9", "border": "1px solid #a5d6a7",
                  "border-radius": "4px", "padding": "5px 12px",
                  "margin-right": "14px"}),

        html.Div(
            html.Div([
                html.Label("Serie:", style={"font-size": "11px", "font-weight": "bold",
                                            "margin-right": "8px", "white-space": "nowrap"}),
                dcc.Checklist(
                    id="mvpq-series-show",
                    options=[
                        {"label": " M·V 🇺🇸", "value": "mv_usa"},
                        {"label": " P·Q 🇺🇸", "value": "pq_usa"},
                        {"label": " M·V 🇪🇺", "value": "mv_eur"},
                        {"label": " P·Q 🇪🇺", "value": "pq_eur"},
                    ],
                    value=["mv_usa", "pq_usa", "mv_eur", "pq_eur"],
                    inline=True,
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "3px"},
                    labelStyle={"margin-right": "10px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#fce4ec", "border": "1px solid #f48fb1",
                      "border-radius": "4px", "padding": "5px 12px"}),
            id="mvpq-series-wrapper",
            style={"display": "none", "margin-right": "14px"},
        ),

        html.Button(
            "🔄  Carica dati",
            id="btn-aggiorna", n_clicks=0,
            style={
                "background": "#1a5276", "color": "white",
                "border": "none", "padding": "8px 22px",
                "border-radius": "5px", "cursor": "pointer",
                "font-size": "13px", "font-weight": "bold",
                "letter-spacing": "0.5px",
                "box-shadow": "0 2px 4px rgba(0,0,0,0.3)",
            }
        ),

        html.Div(id="status-msg",
                 style={"font-size": "11px", "color": "#444",
                        "margin-left": "14px", "font-style": "italic"}),
    ], style={"display": "flex", "align-items": "center",
              "padding": "8px 16px", "background": "#f0f4fa",
              "border-bottom": "1px solid #dee2e6",
              "flex-wrap": "wrap", "gap": "8px"})


def _slider_area():
    return html.Div([
        dcc.RangeSlider(
            id="date-slider",
            min=0, max=1, value=[0, 1],
            marks={}, step=86400 * 30,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(id="slider-label",
                 style={"font-size": "10px", "color": "#666",
                        "text-align": "center", "margin-top": "2px"}),
    ], style={"padding": "8px 28px 2px"})


def _yields_tab_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Fonte:", style={"font-size": "11px", "font-weight": "bold",
                                             "margin-right": "8px", "white-space": "nowrap"}),
                dcc.RadioItems(
                    id="yields-source",
                    options=[
                        {"label": " 🇺🇸 USA",      "value": "usa"},
                        {"label": " 🇪🇺 Europa",   "value": "eur"},
                        {"label": " 🆚 Entrambi",  "value": "both"},
                    ],
                    value="usa", inline=True,
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "3px"},
                    labelStyle={"margin-right": "12px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#f3e5f5", "border": "1px solid #ce93d8",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "14px"}),

            html.Div([
                html.Label("Vista:", style={"font-size": "11px", "font-weight": "bold",
                                             "margin-right": "8px", "white-space": "nowrap"}),
                dcc.RadioItems(
                    id="yields-view",
                    options=[
                        {"label": " Storico",      "value": "history"},
                        {"label": " Curva tassi",  "value": "curve"},
                        {"label": " Entrambi",     "value": "both"},
                    ],
                    value="both", inline=True,
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "3px"},
                    labelStyle={"margin-right": "12px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#fff8e1", "border": "1px solid #ffe082",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "14px"}),

            html.Div([
                html.Label("Vista storico:", style={"font-size": "11px", "font-weight": "bold",
                                                     "margin-right": "8px", "white-space": "nowrap"}),
                dcc.RadioItems(
                    id="yields-hist-mode",
                    options=[
                        {"label": " Assoluti", "value": "abs"},
                        {"label": " Δ% YoY",   "value": "yoy"},
                    ],
                    value="abs", inline=True,
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "3px"},
                    labelStyle={"margin-right": "12px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#e8f5e9", "border": "1px solid #a5d6a7",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "14px"}),

            html.Button("🔄  Ricarica tassi", id="btn-reload-yields", n_clicks=0,
                        style={"background": "#1a5276", "color": "white",
                               "border": "none", "padding": "7px 16px",
                               "border-radius": "5px", "cursor": "pointer",
                               "font-size": "12px", "font-weight": "bold"}),

            html.Div(id="yields-status",
                     style={"font-size": "11px", "color": "#444",
                            "margin-left": "14px", "font-style": "italic"}),
        ], style={"display": "flex", "align-items": "center",
                  "padding": "8px 16px", "background": "#f0f4fa",
                  "border-bottom": "1px solid #dee2e6",
                  "flex-wrap": "wrap", "gap": "8px"}),

        html.Div([
            html.Div([
                html.Div(html.B("Tassi attivi", style={"font-size": "11px"}),
                         style={"padding-bottom": "6px", "margin-bottom": "8px",
                                "border-bottom": "2px solid #ccc"}),
                html.Div(id="yields-checklist-container",
                         children="— carica i dati —",
                         style={"font-size": "10px", "color": "#888"}),
                html.Hr(style={"margin": "10px 0"}),
                html.Div([
                    html.Button("✔ Tutto", id="yield-sel-all", n_clicks=0,
                                style={"font-size": "9px", "padding": "2px 7px",
                                       "margin-right": "4px", "cursor": "pointer"}),
                    html.Button("✘ Niente", id="yield-sel-none", n_clicks=0,
                                style={"font-size": "9px", "padding": "2px 7px",
                                       "cursor": "pointer"}),
                ], style={"display": "flex"}),
            ], style={"width": "210px", "min-width": "200px", "padding": "12px",
                      "border-right": "1px solid #ddd",
                      "height": "calc(100vh - 130px)",
                      "overflow-y": "auto", "background": "#fafafa"}),

            html.Div([
                html.Div([
                    dcc.RangeSlider(
                        id="yields-slider",
                        min=0, max=1, value=[0, 1],
                        marks={}, step=86400,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(id="yields-slider-label",
                             style={"font-size": "10px", "color": "#666",
                                    "text-align": "center", "margin-top": "2px"}),
                ], style={"padding": "8px 28px 2px"}),

                html.Div(id="wrap-yields-history", children=[
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-yields-history",
                                  figure=empty_fig("Caricamento..."),
                                  style={"height": "82vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ], style={"display": "block"}),

                html.Div(id="wrap-yields-curve", children=[
                    html.Div([
                        html.B("Snapshot curva dei tassi", style={"font-size": "11px", "color": "#1a5276"}),
                        html.Span("  Confronto fra ultimi 5 snapshot nel periodo selezionato",
                                  style={"font-size": "10px", "color": "#666", "margin-left": "8px"}),
                    ], id="wrap-yields-curve-header",
                       style={"padding": "5px 16px", "background": "#eaf4fb",
                              "border-top": "1px solid #aed6f1",
                              "border-bottom": "1px solid #aed6f1"}),
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-yields-curve",
                                  figure=empty_fig("Caricamento..."),
                                  style={"height": "82vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ], style={"display": "block"}),
            ], style={"flex": "1", "min-width": "0",
                      "overflow-y": "auto",
                      "height": "calc(100vh - 130px)"}),
        ], style={"display": "flex"}),
    ])


def _pil_unified_tab_layout():
    return html.Div([
        # ── Barra controlli ──────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Fonte:", style={"font-size": "11px", "font-weight": "bold",
                                             "margin-right": "8px", "white-space": "nowrap"}),
                dcc.RadioItems(
                    id="pil-source",
                    options=[
                        {"label": " 🇺🇸 USA",     "value": "usa"},
                        {"label": " 🇪🇺 Europa",  "value": "eur"},
                        {"label": " 🆚 Entrambi", "value": "both"},
                    ],
                    value="usa", inline=True,
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "3px"},
                    labelStyle={"margin-right": "12px"},
                ),
                html.Div(
                    dcc.Dropdown(
                        id="pil-geo",
                        options=[{"label": v, "value": k} for k, v in EUROSTAT_GEO.items()],
                        value="EA20", clearable=False,
                        style={"font-size": "10px", "min-width": "160px"},
                    ),
                    id="pil-geo-wrapper",
                    style={"display": "none", "margin-left": "8px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#f3e5f5", "border": "1px solid #ce93d8",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "14px"}),

            html.Div([
                html.Label("Vista:", style={"font-size": "11px", "font-weight": "bold",
                                             "margin-right": "8px", "white-space": "nowrap"}),
                dcc.Checklist(
                    id="pil-view",
                    options=[
                        {"label": " Valori Assoluti", "value": "abs"},
                        {"label": " Δ% YoY",          "value": "yoy"},
                        {"label": " Cumulata %",      "value": "cum"},
                    ],
                    value=["yoy"], inline=True,
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "3px"},
                    labelStyle={"margin-right": "12px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#fff8e1", "border": "1px solid #ffe082",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "14px"}),

            html.Button("🔄  Carica PIL", id="btn-reload-pil", n_clicks=0,
                        style={"background": "#1a5276", "color": "white",
                               "border": "none", "padding": "7px 16px",
                               "border-radius": "5px", "cursor": "pointer",
                               "font-size": "12px", "font-weight": "bold"}),

            html.Div(id="pil-status",
                     style={"font-size": "11px", "color": "#444",
                            "margin-left": "14px", "font-style": "italic"}),

        ], style={"display": "flex", "align-items": "center",
                  "padding": "8px 16px", "background": "#f0f4fa",
                  "border-bottom": "1px solid #dee2e6",
                  "flex-wrap": "wrap", "gap": "8px"}),

        html.Div([
            # ── Sidebar ──────────────────────────────────────────────────────
            html.Div([
                html.Div(html.B("Serie attive", style={"font-size": "11px"}),
                         style={"padding-bottom": "6px", "margin-bottom": "8px",
                                "border-bottom": "2px solid #ccc"}),
                html.Div(id="pil-checklist-container",
                         children="— carica i dati —",
                         style={"font-size": "10px", "color": "#888"}),
                html.Hr(style={"margin": "10px 0"}),
                html.Div([
                    html.Button("✔ Tutto", id="pil-sel-all", n_clicks=0,
                                style={"font-size": "9px", "padding": "2px 7px",
                                       "margin-right": "4px", "cursor": "pointer"}),
                    html.Button("✘ Niente", id="pil-sel-none", n_clicks=0,
                                style={"font-size": "9px", "padding": "2px 7px",
                                       "cursor": "pointer"}),
                ], style={"display": "flex"}),
            ], style={"width": "230px", "min-width": "220px", "padding": "12px",
                      "border-right": "1px solid #ddd",
                      "height": "calc(100vh - 130px)",
                      "overflow-y": "auto", "background": "#fafafa"}),

            # ── Grafici ──────────────────────────────────────────────────────
            html.Div([
                html.Div([
                    dcc.RangeSlider(
                        id="pil-slider", min=0, max=1, value=[0, 1],
                        marks={}, step=86400 * 30,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(id="pil-slider-label",
                             style={"font-size": "10px", "color": "#666",
                                    "text-align": "center", "margin-top": "2px"}),
                ], style={"padding": "8px 28px 2px"}),

                html.Div(id="wrap-pil-abs", children=[
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-pil-abs",
                                  figure=empty_fig("Carica i dati con 🔄 Carica PIL"),
                                  style={"height": "75vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ]),

                html.Div(id="wrap-pil-yoy", children=[
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-pil-yoy",
                                  figure=empty_fig("Carica i dati con 🔄 Carica PIL"),
                                  style={"height": "75vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ]),

                html.Div(id="wrap-pil-cum", children=[
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-pil-cum",
                                  figure=empty_fig("Carica i dati con 🔄 Carica PIL"),
                                  style={"height": "75vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ]),

            ], style={"flex": "1", "min-width": "0",
                      "overflow-y": "auto",
                      "height": "calc(100vh - 130px)"}),
        ], style={"display": "flex"}),
    ])


def _gdp_tab_layout():
    gdp_comp   = GDP_LABELS + [NET_EXP_LABEL]
    mon_labels = DEFAULT_LABELS
    mv_pq      = ["M·V", "P·Q"]

    def _check(lbl, default_on=True):
        return html.Div(
            dcc.Checklist(
                id={"type": "gdp-all-check", "index": lbl},
                options=[{"label": f" {lbl}", "value": lbl}],
                value=[lbl] if default_on else [],
                style={"font-size": "10px"},
                inputStyle={"margin-right": "4px"},
            ),
            style={"margin-bottom": "3px"}
        )

    sidebar = html.Div([
        html.Div(html.B("Serie attive", style={"font-size": "11px"}),
                 style={"padding-bottom": "6px", "margin-bottom": "8px",
                        "border-bottom": "2px solid #ccc"}),

        html.Div([
            html.Div("📦  Componenti PIL",
                     style={"font-size": "10px", "font-weight": "bold",
                            "color": "#1a5276", "background": "#eaf4fb",
                            "padding": "3px 6px", "border-radius": "3px",
                            "flex": "1"}),
            html.Button("✔", id="gdp-sel-all-gdp", n_clicks=0,
                        title="Seleziona tutto",
                        style={"font-size": "9px", "padding": "1px 5px",
                               "margin-left": "4px", "cursor": "pointer"}),
            html.Button("✘", id="gdp-sel-none-gdp", n_clicks=0,
                        title="Deseleziona tutto",
                        style={"font-size": "9px", "padding": "1px 5px",
                               "margin-left": "2px", "cursor": "pointer"}),
        ], style={"display": "flex", "align-items": "center",
                  "margin-bottom": "5px"}),
        *[_check(lbl) for lbl in gdp_comp],

        html.Hr(style={"margin": "8px 0"}),

        html.Div([
            html.Div("💰  Serie Monetarie",
                     style={"font-size": "10px", "font-weight": "bold",
                            "color": "#6a1b9a", "background": "#f3e5f5",
                            "padding": "3px 6px", "border-radius": "3px",
                            "flex": "1"}),
            html.Button("✔", id="gdp-sel-all-mon", n_clicks=0,
                        title="Seleziona tutto",
                        style={"font-size": "9px", "padding": "1px 5px",
                               "margin-left": "4px", "cursor": "pointer"}),
            html.Button("✘", id="gdp-sel-none-mon", n_clicks=0,
                        title="Deseleziona tutto",
                        style={"font-size": "9px", "padding": "1px 5px",
                               "margin-left": "2px", "cursor": "pointer"}),
        ], style={"display": "flex", "align-items": "center",
                  "margin-bottom": "5px"}),
        *[_check(lbl, default_on=False) for lbl in mon_labels],

        html.Hr(style={"margin": "8px 0"}),

        html.Div([
            html.Div("⚖️  MV = PQ",
                     style={"font-size": "10px", "font-weight": "bold",
                            "color": "#b71c1c", "background": "#ffebee",
                            "padding": "3px 6px", "border-radius": "3px",
                            "flex": "1"}),
            html.Button("✔", id="gdp-sel-all-mvpq", n_clicks=0,
                        title="Seleziona tutto",
                        style={"font-size": "9px", "padding": "1px 5px",
                               "margin-left": "4px", "cursor": "pointer"}),
            html.Button("✘", id="gdp-sel-none-mvpq", n_clicks=0,
                        title="Deseleziona tutto",
                        style={"font-size": "9px", "padding": "1px 5px",
                               "margin-left": "2px", "cursor": "pointer"}),
        ], style={"display": "flex", "align-items": "center",
                  "margin-bottom": "5px"}),
        *[_check(lbl, default_on=False) for lbl in mv_pq],

        html.Hr(style={"margin": "8px 0"}),

        html.Div([
            html.Button("✔ Tutto", id="gdp-sel-all", n_clicks=0,
                        style={"font-size": "9px", "padding": "2px 7px",
                               "margin-right": "4px", "cursor": "pointer"}),
            html.Button("✘ Niente", id="gdp-sel-none", n_clicks=0,
                        style={"font-size": "9px", "padding": "2px 7px",
                               "cursor": "pointer"}),
        ], style={"display": "flex"}),

    ], style={"width": "220px", "min-width": "210px", "padding": "12px",
              "border-right": "1px solid #ddd",
              "height": "calc(100vh - 130px)",
              "overflow-y": "auto", "background": "#fafafa"})

    return html.Div([
        html.Div([
            html.Div([
                html.Label("Vista:", style={"font-size": "11px", "font-weight": "bold",
                                             "margin-right": "8px", "white-space": "nowrap"}),
                dcc.Checklist(
                    id="gdp-view",
                    options=[
                        {"label": " Valori Assoluti", "value": "abs"},
                        {"label": " Δ% YoY",          "value": "yoy"},
                        {"label": " Cumulata %",      "value": "cum"},
                    ],
                    value=["yoy"],
                    inline=True,
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "3px"},
                    labelStyle={"margin-right": "12px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#fff8e1", "border": "1px solid #ffe082",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "14px"}),

            html.Button("🔄  Ricarica PIL", id="btn-reload-gdp", n_clicks=0,
                        style={"background": "#1a5276", "color": "white",
                               "border": "none", "padding": "7px 16px",
                               "border-radius": "5px", "cursor": "pointer",
                               "font-size": "12px", "font-weight": "bold"}),

            html.Div(id="gdp-status",
                     style={"font-size": "11px", "color": "#444",
                            "margin-left": "14px", "font-style": "italic"}),

        ], style={"display": "flex", "align-items": "center",
                  "padding": "8px 16px", "background": "#f0f4fa",
                  "border-bottom": "1px solid #dee2e6",
                  "flex-wrap": "wrap", "gap": "8px"}),

        html.Div([
            sidebar,
            html.Div([
                html.Div([
                    dcc.RangeSlider(
                        id="gdp-slider",
                        min=0, max=1, value=[0, 1],
                        marks={}, step=86400 * 30,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(id="gdp-slider-label",
                             style={"font-size": "10px", "color": "#666",
                                    "text-align": "center", "margin-top": "2px"}),
                ], style={"padding": "8px 28px 2px"}),

                html.Div(id="wrap-gdp-abs", children=[
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-gdp-abs",
                                  figure=empty_fig("Caricamento..."),
                                  style={"height": "75vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ], style={"display": "block"}),

                html.Div(id="wrap-gdp-yoy", children=[
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-gdp-yoy",
                                  figure=empty_fig("Caricamento..."),
                                  style={"height": "75vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ], style={"display": "block"}),

                html.Div(id="wrap-gdp-cum", children=[
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-gdp-cum",
                                  figure=empty_fig("Caricamento..."),
                                  style={"height": "75vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ], style={"display": "block"}),

            ], style={"flex": "1", "min-width": "0",
                      "overflow-y": "auto",
                      "height": "calc(100vh - 130px)"}),
        ], style={"display": "flex"}),
    ])


def _eurostat_tab_layout():
    eur_comp = [v[2] for v in EUROSTAT_SERIES.values()] + [NET_EXP_EUR_LABEL, NET_EXP_EUR_R_LABEL]

    def _check(lbl, default_on=True):
        return html.Div(
            dcc.Checklist(
                id={"type": "eur-check", "index": lbl},
                options=[{"label": f" {lbl}", "value": lbl}],
                value=[lbl] if default_on else [],
                style={"font-size": "10px"},
                inputStyle={"margin-right": "4px"},
            ),
            style={"margin-bottom": "3px"}
        )

    sidebar = html.Div([
        html.Div(html.B("Serie attive", style={"font-size": "11px"}),
                 style={"padding-bottom": "6px", "margin-bottom": "8px",
                        "border-bottom": "2px solid #ccc"}),

        # Selettore paese / area
        html.Div([
            html.Label("Paese / Area:",
                       style={"font-size": "10px", "font-weight": "bold",
                              "color": "#1a3a5c", "margin-bottom": "4px",
                              "display": "block"}),
            dcc.Dropdown(
                id="eur-geo",
                options=[{"label": v, "value": k} for k, v in EUROSTAT_GEO.items()],
                value="EA20",
                clearable=False,
                style={"font-size": "10px"},
            ),
        ], style={"margin-bottom": "10px"}),

        html.Div([
            html.Div("📦  Componenti PIL EUR",
                     style={"font-size": "10px", "font-weight": "bold",
                            "color": "#1a5276", "background": "#eaf4fb",
                            "padding": "3px 6px", "border-radius": "3px", "flex": "1"}),
            html.Button("✔", id="eur-sel-all", n_clicks=0,
                        style={"font-size": "9px", "padding": "1px 5px",
                               "margin-left": "4px", "cursor": "pointer"}),
            html.Button("✘", id="eur-sel-none", n_clicks=0,
                        style={"font-size": "9px", "padding": "1px 5px",
                               "margin-left": "2px", "cursor": "pointer"}),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "5px"}),
        *[_check(lbl) for lbl in eur_comp],

    ], style={"width": "220px", "min-width": "200px",
              "border-right": "1px solid #ddd",
              "height": "calc(100vh - 130px)",
              "overflow-y": "auto",
              "padding": "10px 8px",
              "background": "#fafafa"})

    return html.Div([
        # Barra controlli superiore
        html.Div([
            html.Div([
                html.Label("Vista:",
                           style={"font-size": "11px", "font-weight": "bold",
                                  "margin-right": "8px"}),
                dcc.Checklist(
                    id="eur-view",
                    options=[
                        {"label": " Valori Assoluti", "value": "abs"},
                        {"label": " Δ% YoY",          "value": "yoy"},
                        {"label": " Cumulata %",      "value": "cum"},
                    ],
                    value=["yoy"],
                    inline=True,
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "3px"},
                    labelStyle={"margin-right": "12px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#fff8e1", "border": "1px solid #ffe082",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "14px"}),

            html.Button("🔄  Carica Eurostat", id="btn-reload-eur", n_clicks=0,
                        style={"background": "#1a5276", "color": "white",
                               "border": "none", "padding": "7px 16px",
                               "border-radius": "5px", "cursor": "pointer",
                               "font-size": "12px", "font-weight": "bold"}),

            html.Div(id="eur-status",
                     style={"font-size": "11px", "color": "#444",
                            "margin-left": "14px", "font-style": "italic"}),

        ], style={"display": "flex", "align-items": "center",
                  "padding": "8px 16px", "background": "#f0f4fa",
                  "border-bottom": "1px solid #dee2e6",
                  "flex-wrap": "wrap", "gap": "8px"}),

        html.Div([
            sidebar,
            html.Div([
                html.Div([
                    dcc.RangeSlider(
                        id="eur-slider",
                        min=0, max=1, value=[0, 1],
                        marks={}, step=86400 * 30,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(id="eur-slider-label",
                             style={"font-size": "10px", "color": "#666",
                                    "text-align": "center", "margin-top": "2px"}),
                ], style={"padding": "8px 28px 2px"}),

                html.Div(id="wrap-eur-abs", children=[
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-eur-abs",
                                  figure=empty_fig("Clicca 🔄 Carica Eurostat"),
                                  style={"height": "75vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ], style={"display": "block"}),

                html.Div(id="wrap-eur-yoy", children=[
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-eur-yoy",
                                  figure=empty_fig("Clicca 🔄 Carica Eurostat"),
                                  style={"height": "75vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ], style={"display": "block"}),

                html.Div(id="wrap-eur-cum", children=[
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-eur-cum",
                                  figure=empty_fig("Clicca 🔄 Carica Eurostat"),
                                  style={"height": "75vh"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ], style={"display": "block"}),

            ], style={"flex": "1", "min-width": "0",
                      "overflow-y": "auto",
                      "height": "calc(100vh - 130px)"}),
        ], style={"display": "flex"}),
    ])


def _regression_tab_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Trasformazione:", style={"font-size": "11px", "font-weight": "bold",
                                                      "margin-right": "8px", "white-space": "nowrap"}),
                dcc.RadioItems(
                    id="reg-transform",
                    options=[
                        {"label": " Livelli",   "value": "levels"},
                        {"label": " Δ% YoY",    "value": "yoy"},
                        {"label": " Log",       "value": "log"},
                        {"label": " Δ Log",     "value": "dlog"},
                    ],
                    value="yoy", inline=True,
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "3px"},
                    labelStyle={"margin-right": "10px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#fff8e1", "border": "1px solid #ffe082",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "12px"}),

            html.Div([
                dcc.Checklist(
                    id="reg-add-const",
                    options=[{"label": " Includi costante (α)", "value": "const"}],
                    value=["const"],
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "4px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#e8f5e9", "border": "1px solid #a5d6a7",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "12px"}),

            html.Div([
                html.Label("Lag X (mesi):", style={"font-size": "11px", "font-weight": "bold",
                                                     "margin-right": "6px", "white-space": "nowrap"}),
                dcc.Input(id="reg-lag", type="number", value=0, min=0, max=24, step=1,
                          style={"width": "55px", "font-size": "11px"}),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#e3f2fd", "border": "1px solid #90caf9",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "12px"}),

            html.Div([
                html.Label("Std. Error:", style={"font-size": "11px", "font-weight": "bold",
                                                  "margin-right": "8px", "white-space": "nowrap"}),
                dcc.RadioItems(
                    id="reg-cov-type",
                    options=[
                        {"label": " OLS classico",    "value": "nonrobust"},
                        {"label": " HC3 (eterosch.)", "value": "HC3"},
                        {"label": " HAC Newey-West",  "value": "HAC"},
                    ],
                    value="nonrobust", inline=True,
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "3px"},
                    labelStyle={"margin-right": "10px"},
                ),
            ], style={"display": "flex", "align-items": "center",
                      "background": "#fce4ec", "border": "1px solid #f48fb1",
                      "border-radius": "4px", "padding": "5px 12px",
                      "margin-right": "12px"}),

            html.Button("▶  Stima modello", id="btn-run-reg", n_clicks=0,
                        style={"background": "#1b5e20", "color": "white",
                               "border": "none", "padding": "8px 22px",
                               "border-radius": "5px", "cursor": "pointer",
                               "font-size": "13px", "font-weight": "bold",
                               "box-shadow": "0 2px 4px rgba(0,0,0,0.3)"}),

            html.Div(id="reg-status",
                     style={"font-size": "11px", "color": "#444",
                            "margin-left": "14px", "font-style": "italic"}),

        ], style={"display": "flex", "align-items": "center",
                  "padding": "8px 16px", "background": "#f0f4fa",
                  "border-bottom": "1px solid #dee2e6",
                  "flex-wrap": "wrap", "gap": "8px"}),

        html.Div([
            html.Div([
                html.Div([
                    html.B("Periodo stima", style={"font-size": "10px"}),
                    dcc.RangeSlider(
                        id="reg-slider",
                        min=0, max=1, value=[0, 1],
                        marks={}, step=86400 * 30,
                        tooltip={"placement": "bottom", "always_visible": False},
                        vertical=False,
                    ),
                    html.Div(id="reg-slider-label",
                             style={"font-size": "9px", "color": "#666",
                                    "text-align": "center"}),
                ], style={"margin-bottom": "12px"}),

                html.Hr(style={"margin": "6px 0"}),

                html.Div([
                    html.B("Y — Variabile dipendente",
                           style={"font-size": "10px", "color": "#b71c1c",
                                  "background": "#ffebee", "display": "block",
                                  "padding": "3px 6px", "border-radius": "3px",
                                  "margin-bottom": "6px"}),
                    dcc.Dropdown(
                        id="reg-y",
                        options=[],
                        placeholder="Seleziona Y...",
                        clearable=True,
                        style={"font-size": "10px"},
                    ),
                ], style={"margin-bottom": "10px"}),

                html.Hr(style={"margin": "6px 0"}),

                html.Div([
                    html.B("X — Variabili indipendenti",
                           style={"font-size": "10px", "color": "#1a5276",
                                  "background": "#eaf4fb", "display": "block",
                                  "padding": "3px 6px", "border-radius": "3px",
                                  "margin-bottom": "6px"}),
                    html.Div(id="reg-x-checklist"),
                ]),

                html.Hr(style={"margin": "8px 0"}),
                html.Div([
                    html.Button("✔ Tutto X", id="reg-sel-all", n_clicks=0,
                                style={"font-size": "9px", "padding": "2px 7px",
                                       "margin-right": "4px", "cursor": "pointer"}),
                    html.Button("✘ Niente X", id="reg-sel-none", n_clicks=0,
                                style={"font-size": "9px", "padding": "2px 7px",
                                       "cursor": "pointer"}),
                ], style={"display": "flex"}),

            ], style={"width": "230px", "min-width": "220px", "padding": "12px",
                      "border-right": "1px solid #ddd",
                      "height": "calc(100vh - 130px)",
                      "overflow-y": "auto", "background": "#fafafa"}),

            html.Div([
                dcc.Loading(type="circle", children=[
                    html.Div(id="reg-equation",
                             style={"font-family": "monospace", "font-size": "12px",
                                    "background": "#f8f9fa", "border": "1px solid #dee2e6",
                                    "border-radius": "4px", "padding": "10px 16px",
                                    "margin": "10px 16px 0",
                                    "white-space": "pre-wrap", "color": "#1a3a5c"}),
                    html.Div(id="reg-stats-table",
                             style={"margin": "10px 16px 0"}),
                    html.Div(id="reg-coeff-table",
                             style={"margin": "10px 16px 0"}),
                    dcc.Graph(id="chart-reg-rolling-coef",
                              figure=empty_fig("Stima il modello per vedere i coefficienti rolling"),
                              style={"height": "45vh"},
                              config={"responsive": True}),
                    dcc.Graph(id="chart-reg-rolling-vif",
                              figure=empty_fig(""),
                              style={"height": "30vh"},
                              config={"responsive": True}),
                    dcc.Graph(id="chart-reg-fit",
                              figure=empty_fig("Stima il modello per vedere i risultati"),
                              style={"height": "35vh", "margin-top": "4px"},
                              config={"responsive": True}),
                    dcc.Graph(id="chart-reg-residuals",
                              figure=empty_fig(""),
                              style={"height": "30vh"},
                              config={"responsive": True}),
                    dcc.Graph(id="chart-reg-qqplot",
                              figure=empty_fig(""),
                              style={"height": "30vh"},
                              config={"responsive": True}),
                ]),
            ], style={"flex": "1", "min-width": "0",
                      "overflow-y": "auto",
                      "height": "calc(100vh - 130px)"}),
        ], style={"display": "flex"}),
    ])

# =============================================================================
# NUOVI TAB LAYOUT (ARIMA, ADL, DSGE)
# =============================================================================
"""
Tre nuovi tab da aggiungere alla dashboard economica.
Questo file contiene:
  - _all_series_with_shock()   helper store merger
  - _pstar()                   helper significatività
  - _arima_tab_layout()        layout tab ARIMA/SARIMA
  - _adl_tab_layout()          layout tab ADL
  - _dsge_tab_layout()         layout tab DSGE
  - register_new_tab_callbacks(app)  tutti i callback dei tre tab
"""


# =============================================================================
# HELPER
# =============================================================================

def _all_series_with_shock(data_mon, data_gdp, data_yields, data_shock, data_eur=None):
    """Unisce i quattro store (+ opzionale Eurostat) in un DataFrame mensile unico."""
    df = _all_series_df(data_mon, data_gdp, data_yields)
    if data_shock:
        dfs = pd.read_json(io.StringIO(data_shock), orient="split")
        dfs.index = pd.to_datetime(dfs.index)
        dfs = dfs.resample("MS").last()
        df = dfs if df.empty else pd.concat([df, dfs], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
    if data_eur:
        dfe = pd.read_json(io.StringIO(data_eur), orient="split")
        dfe.index = pd.to_datetime(dfe.index)
        dfe = dfe.resample("MS").last()
        df = dfe if df.empty else pd.concat([df, dfe], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _pstar(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "·"
    return ""


# =============================================================================
# LAYOUT — Tab ARIMA / SARIMA
# =============================================================================

def _arima_tab_layout():
    """
    Tab ARIMA / SARIMA — Workflow Box-Jenkins in 4 passi progressivi.

    Passo 1:  Trasformazione (log, differenza, ecc.) + detrend (HP, MA12, diff stagionale)
              → 4 grafici: originale / trasformata / trend / serie stazionaria
              → risultato salvato in store-arima-step1

    Passo 2:  Identificazione struttura sulla serie stazionaria:
              ACF, PACF, Periodogramma, ADF test
              → suggerisce automaticamente p, d, q (pre-compila i cursori del Passo 3)

    Passo 3:  Stima ARIMA(p,d,q) o SARIMA(p,d,q)(P,D,Q,s) — ordini modificabili
              → info modello, tabella statistiche, grafico fitted + forecast + IC 95%

    Passo 4:  Diagnostica residui (generata automaticamente dopo la stima)
              → verdetto, ACF residui, QQ plot, residui nel tempo, istogramma
    """
    # ── stili riutilizzati ────────────────────────────────────────────────────
    CARD = {
        "background": "#ffffff",
        "border": "1px solid #dee2e6",
        "border-radius": "6px",
        "padding": "14px 16px",
        "margin-bottom": "12px",
        "box-shadow": "0 1px 3px rgba(0,0,0,.06)",
    }
    HDR = {
        "font-size": "11px",
        "font-weight": "700",
        "color": "#1a3a5c",
        "background": "#eaf4fb",
        "padding": "4px 10px",
        "border-radius": "3px",
        "margin-bottom": "12px",
        "letter-spacing": ".4px",
    }
    BTN = {
        "background": "#1565c0",
        "color": "white",
        "border": "none",
        "padding": "7px 14px",
        "border-radius": "4px",
        "cursor": "pointer",
        "font-size": "12px",
        "font-weight": "bold",
        "width": "100%",
        "margin-top": "8px",
    }
    STATUS = {"font-size": "10px", "color": "#555",
              "margin-top": "4px", "font-style": "italic"}
    LBL = {"font-size": "10px", "color": "#333",
           "display": "block", "margin-bottom": "4px"}
    RI_LBL = {"display": "block", "font-size": "10px",
              "margin-bottom": "3px", "color": "#333"}
    RI_INP = {"margin-right": "4px"}

    def _num_row(label, id_, val, mn, mx):
        return html.Div([
            html.Label(label, style={**LBL, "width": "145px", "flex-shrink": "0",
                                     "margin-bottom": "0"}),
            dcc.Input(id=id_, type="number", value=val, min=mn, max=mx, step=1,
                      style={"width": "58px", "font-size": "11px",
                             "border": "1px solid #ced4da", "border-radius": "3px",
                             "padding": "2px 4px"}),
        ], style={"display": "flex", "align-items": "center",
                  "gap": "8px", "margin-bottom": "6px"})

    # ── sidebar ───────────────────────────────────────────────────────────────
    sidebar = html.Div([

        # ⓪ Fonte dati
        html.Div([
            html.B("⓪ Fonte dati",
                   style={"font-size": "10px", "color": "#1a5276",
                          "display": "block", "margin-bottom": "6px"}),
            dcc.RadioItems(
                id="arima-source-type",
                options=[
                    {"label": " 🇺🇸  USA (FRED)",  "value": "usa"},
                    {"label": " 🇪🇺  Eurostat",    "value": "eur"},
                ],
                value="usa", inline=True,
                style={"font-size": "10px", "margin-bottom": "6px"},
                inputStyle={"margin-right": "3px"},
                labelStyle={"margin-right": "18px"},
            ),
            html.Div(
                dcc.Dropdown(
                    id="arima-eur-geo",
                    options=[{"label": v, "value": k}
                             for k, v in EUROSTAT_GEO.items()],
                    value="EA20", clearable=False,
                    style={"font-size": "10px"},
                ),
                id="arima-geo-wrapper",
                style={"display": "none", "margin-bottom": "6px"},
            ),
            html.Div([
                html.Button("🔄  Carica dati", id="btn-arima-load",
                            n_clicks=0,
                            style={"background": "#1a5276",
                                   "color": "white", "border": "none",
                                   "padding": "5px 14px",
                                   "border-radius": "4px",
                                   "cursor": "pointer",
                                   "font-size": "10px",
                                   "font-weight": "bold"}),
                html.Div(id="arima-source-status",
                         style={"font-size": "9px", "color": "#555",
                                "font-style": "italic",
                                "margin-left": "10px"}),
            ], style={"display": "flex", "align-items": "center"}),
        ], style={"background": "#eaf4fb", "border": "1px solid #aed6f1",
                  "border-radius": "4px", "padding": "8px 10px",
                  "margin-bottom": "10px"}),

        html.B("Serie storica", style={**LBL, "color": "#b71c1c",
                                        "background": "#ffebee", "padding": "4px 8px",
                                        "border-radius": "3px"}),
        dcc.Dropdown(id="arima-y-var", options=[], placeholder="Seleziona serie…",
                     style={"font-size": "10px", "margin-bottom": "10px"}),

        html.B("Periodo di analisi", style={**LBL, "color": "#1a5276",
                                             "background": "#eaf4fb", "padding": "4px 8px",
                                             "border-radius": "3px"}),
        dcc.RangeSlider(id="arima-slider", min=0, max=1, value=[0, 1],
                        marks={}, step=86400 * 30,
                        tooltip={"placement": "bottom", "always_visible": False}),
        html.Div(id="arima-slider-label",
                 style={"font-size": "9px", "color": "#666",
                        "text-align": "center", "margin-bottom": "14px"}),

        html.Hr(style={"margin": "10px 0"}),

        # Step indicators
        html.Div([
            html.Div("📋  Workflow Box-Jenkins",
                     style={"font-size": "10px", "font-weight": "700",
                            "color": "#1a3a5c", "margin-bottom": "8px"}),
            *[html.Div([
                html.Span(num, style={"background": col, "color": "white",
                                      "border-radius": "50%", "width": "18px",
                                      "height": "18px", "display": "inline-flex",
                                      "align-items": "center", "justify-content": "center",
                                      "font-size": "10px", "font-weight": "700",
                                      "flex-shrink": "0"}),
                html.Span(label, style={"font-size": "10px", "color": "#333",
                                        "margin-left": "6px"}),
              ], style={"display": "flex", "align-items": "center",
                        "margin-bottom": "6px"})
              for num, label, col in [
                  ("①", "Trasformazione & Detrend", "#1565c0"),
                  ("②", "Identificazione struttura", "#2e7d32"),
                  ("③", "Stima modello", "#6a1b9a"),
                  ("④", "Diagnostica residui", "#bf360c"),
              ]],
        ], style={"background": "#f8f9fa", "border": "1px solid #dee2e6",
                  "border-radius": "5px", "padding": "10px"}),

    ], style={"width": "230px", "min-width": "220px",
              "padding": "12px", "border-right": "1px solid #ddd",
              "height": "calc(100vh - 100px)",
              "overflow-y": "auto", "background": "#fafafa"})

    # ── Step 1 ────────────────────────────────────────────────────────────────
    step1 = html.Div([
        html.Div("①  TRASFORMAZIONE & DETREND", style=HDR),
        html.Div([
            html.Div([
                html.Label("Trasformazione", style={**LBL, "font-weight": "600"}),
                dcc.RadioItems(
                    id="arima-step1-transform",
                    options=[
                        {"label": " Nessuna (livelli grezzi)",   "value": "none"},
                        {"label": " Logaritmo  ln(x)",           "value": "log"},
                        {"label": " Δ Differenza prima",         "value": "diff"},
                        {"label": " Δ ln(x)  — tasso crescita",  "value": "diff_log"},
                    ],
                    value="log",
                    labelStyle=RI_LBL, inputStyle=RI_INP,
                ),
            ], style={"flex": "1", "min-width": "200px"}),

            html.Div([
                html.Label("Metodo detrend", style={**LBL, "font-weight": "600"}),
                dcc.RadioItems(
                    id="arima-step1-detrend",
                    options=[
                        {"label": " HP Filter  (λ = 1600 mensile)",    "value": "hp"},
                        {"label": " Media mobile  12 mesi",             "value": "ma12"},
                        {"label": " Differenza stagionale  (lag 12)",   "value": "sdiff"},
                        {"label": " Solo trasformazione  (nessun detrend)", "value": "none"},
                    ],
                    value="hp",
                    labelStyle=RI_LBL, inputStyle=RI_INP,
                ),
            ], style={"flex": "1", "min-width": "220px"}),
        ], style={"display": "flex", "gap": "24px", "flex-wrap": "wrap",
                  "margin-bottom": "4px"}),

        html.Button("①  Applica Trasformazione & Detrend",
                    id="btn-arima-step1", n_clicks=0, style=BTN),
        html.Div(id="arima-step1-status", style=STATUS),
        html.Div(id="arima-step1-charts"),
        dcc.Store(id="store-arima-step1"),
    ], style=CARD)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    step2 = html.Div([
        html.Div("②  IDENTIFICAZIONE STRUTTURA  —  ACF / PACF / ADF / Periodogramma",
                 style={**HDR, "background": "#e8f5e9", "color": "#1b5e20"}),
        html.P(
            "Analizza la serie stazionaria prodotta al Passo ①.  "
            "I grafici ACF e PACF indicano gli ordini p e q ottimali.  "
            "L'ADF test verifica la stazionarietà e suggerisce d.  "
            "Gli ordini vengono pre-compilati automaticamente nel Passo ③.",
            style={"font-size": "10px", "color": "#555", "margin-bottom": "8px"}),
        html.Button("②  Analizza struttura e suggerisci ordini",
                    id="btn-arima-step2", n_clicks=0,
                    style={**BTN, "background": "#2e7d32"}),
        html.Div(id="arima-step2-status", style=STATUS),
        html.Div(id="arima-step2-output"),
    ], style=CARD)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    step3 = html.Div([
        html.Div("③  STIMA MODELLO  —  ARIMA / SARIMA",
                 style={**HDR, "background": "#f3e5f5", "color": "#4a148c"}),
        html.P(
            "Ordini pre-compilati dal Passo ②. Modificali se necessario, "
            "aggiungi eventualmente la componente stagionale.",
            style={"font-size": "10px", "color": "#555", "margin-bottom": "10px"}),

        html.Div([
            # Colonna ARIMA
            html.Div([
                html.B("Ordini ARIMA(p, d, q)",
                       style={"font-size": "10px", "color": "#4a148c",
                              "background": "#f3e5f5", "display": "block",
                              "padding": "4px 8px", "border-radius": "3px",
                              "margin-bottom": "8px"}),
                _num_row("p — lag AR:",           "arima-p", 1, 0, 12),
                _num_row("d — differenziazioni:", "arima-d", 0, 0,  2),
                _num_row("q — lag MA:",           "arima-q", 0, 0, 12),
            ], style={"flex": "1", "min-width": "180px"}),

            # Colonna stagionale
            html.Div([
                dcc.Checklist(
                    id="arima-seasonal-on",
                    options=[{"label": " Componente stagionale SARIMA", "value": "on"}],
                    value=[],
                    style={"font-size": "10px", "margin-bottom": "6px"},
                    inputStyle=RI_INP,
                ),
                html.Div([
                    html.B("Stagionale SARIMA(P, D, Q, s)",
                           style={"font-size": "10px", "color": "#2e7d32",
                                  "background": "#e8f5e9", "display": "block",
                                  "padding": "4px 8px", "border-radius": "3px",
                                  "margin-bottom": "8px"}),
                    _num_row("P — AR stagionale:",    "arima-P", 1, 0, 4),
                    _num_row("D — diff. stagionale:", "arima-D", 1, 0, 2),
                    _num_row("Q — MA stagionale:",    "arima-Q", 1, 0, 4),
                    _num_row("s — periodo (mesi):",   "arima-s", 12, 4, 24),
                ], id="arima-seasonal-params"),
            ], style={"flex": "1", "min-width": "200px"}),

            # Colonna previsione
            html.Div([
                html.B("Previsione & opzioni",
                       style={"font-size": "10px", "color": "#1a5276",
                              "background": "#eaf4fb", "display": "block",
                              "padding": "4px 8px", "border-radius": "3px",
                              "margin-bottom": "8px"}),
                html.Label("Passi avanti:", style=LBL),
                dcc.Slider(id="arima-forecast-steps", min=1, max=36, step=1, value=12,
                           marks={1: "1", 12: "12", 24: "24", 36: "36"},
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.Div(style={"height": "8px"}),
                html.Label("Standard error:", style=LBL),
                dcc.Dropdown(id="arima-cov-type",
                    options=[
                        {"label": "OPG (default)", "value": "opg"},
                        {"label": "Outer product", "value": "oim"},
                        {"label": "Robust HAC",    "value": "robust"},
                    ],
                    value="opg",
                    style={"font-size": "11px"}),
            ], style={"flex": "1", "min-width": "180px"}),
        ], style={"display": "flex", "gap": "20px", "flex-wrap": "wrap",
                  "margin-bottom": "4px"}),

            # Colonna dummy
            html.Div([
                html.B("Variabili esogene (dummy)",
                       style={"font-size": "10px", "color": "#e65100",
                              "background": "#fff3e0", "display": "block",
                              "padding": "4px 8px", "border-radius": "3px",
                              "margin-bottom": "8px"}),
                html.P("Aggiunte come regressori esogeni al SARIMAX.",
                       style={"font-size": "9px", "color": "#888",
                              "margin-bottom": "8px"}),
                dcc.Checklist(
                    id="arima-dummies",
                    options=[
                        {"label": " COVID (mar–mag 2020)",      "value": "dummy_covid"},
                        {"label": " Inflaz. USA (nov 2021–dic 2022)", "value": "dummy_inflaz"},
                        {"label": " Guerra Ucraina (feb–dic 2022)", "value": "dummy_ucraina"},
                        {"label": " GFC (set 2008–mar 2009)",   "value": "dummy_gfc"},
                        {"label": " Dot-com (mar 2000–ott 2002)","value": "dummy_dotcom"},
                    ],
                    value=[],
                    style={"font-size": "10px"},
                    inputStyle={"margin-right": "4px"},
                    labelStyle={"display": "block", "margin-bottom": "4px"},
                ),
            ], style={"flex": "1", "min-width": "180px"}),

        html.Button("③  Stima ARIMA / SARIMA",
                    id="btn-run-arima", n_clicks=0,
                    style={**BTN, "background": "#6a1b9a"}),
        html.Div(id="arima-status", style=STATUS),
        html.Div(id="arima-step3-output"),
    ], style=CARD)

    # ── Step 4 ────────────────────────────────────────────────────────────────
    step4 = html.Div([
        html.Div("④  DIAGNOSTICA RESIDUI",
                 style={**HDR, "background": "#fbe9e7", "color": "#bf360c"}),
        html.P("Generata automaticamente dopo la stima al Passo ③.",
               style={"font-size": "10px", "color": "#888", "margin-bottom": "6px"}),
        html.Div(id="arima-step4-output"),
    ], style=CARD)

    # ── assemblaggio ──────────────────────────────────────────────────────────
    main = html.Div([step1, step2, step3, step4],
                    style={"flex": "1", "min-width": "0",
                           "padding": "14px 16px",
                           "overflow-y": "auto",
                           "height": "calc(100vh - 100px)"})

    return html.Div([
        html.Div([
            html.H3("Modelli ARIMA / SARIMA",
                    style={"margin": "0 20px 0 0", "font-size": "15px",
                           "color": "#1a3a5c", "white-space": "nowrap"}),
            html.Span(
                "Workflow Box-Jenkins: Trasformazione → Identificazione struttura "
                "→ Stima → Diagnostica residui",
                style={"font-size": "11px", "color": "#666"}),
        ], style={"display": "flex", "align-items": "center",
                  "padding": "8px 16px", "background": "#f0f4fa",
                  "border-bottom": "1px solid #dee2e6",
                  "flex-wrap": "wrap", "gap": "8px"}),

        html.Div([sidebar, main], style={"display": "flex"}),
    ])



# =============================================================================
# Pipeline options usate nell'ADL (Y e ogni X) — stessa logica del tab Confronto
ADL_PIPE_OPTS = [
    {"label": "—",         "value": "none"},
    {"label": "log",       "value": "log"},
    {"label": "YoY %",     "value": "yoy"},
    {"label": "Σ cumsum",  "value": "cumsum"},
    {"label": "Δ¹",        "value": "diff1"},
    {"label": "Δ²",        "value": "diff2"},
    {"label": "× 100",     "value": "x100"},
    {"label": "MA 3m",     "value": "ma3"},
    {"label": "MA 6m",     "value": "ma6"},
    {"label": "MA 12m",    "value": "ma12"},
    {"label": "EMA 3m",    "value": "ema3"},
    {"label": "EMA 6m",    "value": "ema6"},
    {"label": "EMA 12m",   "value": "ema12"},
    {"label": "Sav-Gol",   "value": "sg"},
    {"label": "HP trend",  "value": "hp"},
    {"label": "Kalman",    "value": "kalman"},
]

# LAYOUT — Tab ADL
# =============================================================================

def _adl_tab_layout():
    return html.Div([

        # Header ────────────────────────────────────────────────────────────
        html.Div([
            html.H3("Modello ADL — Autoregressive Distributed Lag",
                    style={"margin": "0 20px 0 0", "font-size": "15px",
                           "color": "#1a3a5c", "white-space": "nowrap"}),
            html.Span(
                "Y(t) = α + β₁·Y(t-1) + ... + γ₀·X(t) + γ₁·X(t-1) + ... + ε",
                style={"font-size": "11px", "color": "#666"}),
        ], style={"display": "flex", "align-items": "center",
                  "padding": "8px 16px", "background": "#f0f4fa",
                  "border-bottom": "1px solid #dee2e6",
                  "flex-wrap": "wrap", "gap": "8px"}),

        html.Div([

            # ── Colonna sinistra: tutta la configurazione ─────────────────
            html.Div([

                # ⓪ Fonte dati
                html.Div([
                    html.B("⓪ Fonte dati",
                           style={"font-size": "10px", "color": "#1a5276",
                                  "display": "block", "margin-bottom": "6px"}),
                    dcc.RadioItems(
                        id="adl-source-type",
                        options=[
                            {"label": " 🇺🇸  USA (FRED)",  "value": "usa"},
                            {"label": " 🇪🇺  Eurostat",    "value": "eur"},
                        ],
                        value="usa", inline=True,
                        style={"font-size": "10px", "margin-bottom": "6px"},
                        inputStyle={"margin-right": "3px"},
                        labelStyle={"margin-right": "18px"},
                    ),
                    # dropdown paese — visibile solo con Eurostat
                    html.Div(
                        dcc.Dropdown(
                            id="adl-eur-geo",
                            options=[{"label": v, "value": k}
                                     for k, v in EUROSTAT_GEO.items()],
                            value="EA20", clearable=False,
                            style={"font-size": "10px"},
                        ),
                        id="adl-geo-wrapper",
                        style={"display": "none", "margin-bottom": "6px"},
                    ),
                    html.Div([
                        html.Button("🔄  Carica dati", id="btn-adl-load",
                                    n_clicks=0,
                                    style={"background": "#1a5276",
                                           "color": "white", "border": "none",
                                           "padding": "5px 14px",
                                           "border-radius": "4px",
                                           "cursor": "pointer",
                                           "font-size": "10px",
                                           "font-weight": "bold"}),
                        html.Div(id="adl-source-status",
                                 style={"font-size": "9px", "color": "#555",
                                        "font-style": "italic",
                                        "margin-left": "10px"}),
                    ], style={"display": "flex", "align-items": "center"}),
                ], style={"background": "#eaf4fb", "border": "1px solid #aed6f1",
                          "border-radius": "4px", "padding": "8px 10px",
                          "margin-bottom": "10px"}),

                html.Hr(style={"margin": "8px 0"}),

                # ➕ Aggiungi serie FRED aggiuntive
                html.Div([
                    html.B("➕ Aggiungi serie FRED",
                           style={"font-size": "10px", "color": "#1a5276",
                                  "display": "block", "margin-bottom": "5px"}),
                    dcc.Input(
                        id="adl-fred-input", type="text",
                        placeholder="es. FEDFUNDS, CPIAUCSL",
                        style={"width": "100%", "box-sizing": "border-box",
                               "font-size": "10px", "padding": "4px 7px",
                               "border": "1px solid #aed6f1", "border-radius": "3px"},
                    ),
                    html.Button(
                        "+ Aggiungi da FRED", id="btn-adl-fred", n_clicks=0,
                        style={"margin-top": "5px", "width": "100%",
                               "background": "#154360", "color": "white",
                               "border": "none", "padding": "5px",
                               "border-radius": "3px", "cursor": "pointer",
                               "font-size": "10px"},
                    ),
                    html.Div(id="adl-fred-status",
                             style={"font-size": "9px", "color": "#555",
                                    "margin-top": "3px", "font-style": "italic"}),
                    dcc.Store(id="store-adl-extra"),
                ], style={"background": "#eaf4fb", "border": "1px solid #aed6f1",
                          "border-radius": "4px", "padding": "8px 10px",
                          "margin-bottom": "10px"}),

                html.Hr(style={"margin": "8px 0"}),

                # ① Y + pipeline + lag AR
                html.B("① Variabile dipendente Y",
                       style={"font-size": "10px", "color": "#b71c1c",
                              "background": "#ffebee", "display": "block",
                              "padding": "4px 8px", "border-radius": "3px",
                              "margin-bottom": "6px"}),
                html.Div([
                    dcc.Dropdown(id="adl-y-var", options=[],
                                 placeholder="Seleziona Y…",
                                 style={"font-size": "10px"}),
                ], style={"margin-bottom": "5px"}),
                html.Div([
                    html.Span("Pipeline Y →",
                              style={"font-size": "9px", "color": "#b71c1c",
                                     "font-weight": "bold", "white-space": "nowrap",
                                     "margin-right": "5px"}),
                    *[html.Span([
                        html.Span("→" if k > 0 else "",
                                  style={"font-size": "10px", "color": "#bbb",
                                         "margin": "0 1px"}),
                        dcc.Dropdown(
                            id={"type": "adl-y-step", "index": k},
                            options=ADL_PIPE_OPTS,
                            value="none", clearable=False,
                            style={"font-size": "9px", "width": "74px",
                                   "display": "inline-block",
                                   "vertical-align": "middle"},
                        ),
                    ], style={"display": "inline-flex", "align-items": "center"})
                      for k in range(4)],
                ], style={"display": "flex", "align-items": "center",
                          "flex-wrap": "wrap", "gap": "2px",
                          "background": "#ffebee", "border": "1px solid #ef9a9a",
                          "border-radius": "4px", "padding": "5px 8px",
                          "margin-bottom": "8px"}),

                html.Div([
                    html.Label("Lag AR di Y:",
                               style={"font-size": "10px", "font-weight": "bold",
                                      "margin-right": "8px", "white-space": "nowrap"}),
                    dcc.Checklist(
                        id="adl-ar-lags",
                        options=[{"label": f" Y(t−{k})", "value": k}
                                 for k in range(1, 13)],
                        value=[1], inline=True,
                        style={"font-size": "10px"},
                        inputStyle={"margin-right": "3px"},
                        labelStyle={"margin-right": "8px", "margin-bottom": "3px"},
                    ),
                ], style={"display": "flex", "align-items": "flex-start",
                          "background": "#f0f4ff", "border": "1px solid #c5cae9",
                          "border-radius": "4px", "padding": "6px 10px",
                          "margin-bottom": "10px", "flex-wrap": "wrap"}),

                html.Hr(style={"margin": "8px 0"}),

                # ② Variabili esogene X — a tutta larghezza
                html.Div([
                    html.B("② Variabili esogene X",
                           style={"font-size": "10px", "color": "#1a5276",
                                  "flex": "1"}),
                    html.Span("Spunta → trasformazione → lag (L0=contemp., L1=lag 1m, …)",
                              style={"font-size": "9px", "color": "#888",
                                     "margin-left": "8px"}),
                ], style={"display": "flex", "align-items": "center",
                          "background": "#eaf4fb", "border": "1px solid #aed6f1",
                          "padding": "4px 8px", "border-radius": "3px",
                          "margin-bottom": "4px"}),

                # intestazione colonne X
                html.Div([
                    html.Div("Variabile",
                             style={"flex": "1", "font-size": "9px",
                                    "font-weight": "bold", "color": "#555"}),
                    html.Div("Lag",
                             style={"width": "180px", "font-size": "9px",
                                    "font-weight": "bold", "color": "#555"}),
                ], style={"display": "flex", "align-items": "center",
                          "padding": "3px 8px", "background": "#f5f5f5",
                          "border-bottom": "1px solid #ddd",
                          "margin-bottom": "2px"}),

                html.Div(id="adl-x-rows",
                         style={"overflow-y": "auto",
                                "max-height": "260px",
                                "border": "1px solid #e0e0e0",
                                "border-radius": "3px",
                                "margin-bottom": "10px"}),

                html.Hr(style={"margin": "8px 0"}),

                # ③ Dummy variables
                html.B("③ Variabili dummy",
                       style={"font-size": "10px", "color": "#e65100",
                              "background": "#fff3e0", "display": "block",
                              "padding": "4px 8px", "border-radius": "3px",
                              "margin-bottom": "6px"}),
                html.Div([
                    dcc.Checklist(
                        id="adl-dummies",
                        options=[
                            {"label": " COVID (mar–mag 2020)",           "value": "dummy_covid"},
                            {"label": " Inflaz. USA (nov 2021–dic 2022)","value": "dummy_inflaz"},
                            {"label": " Guerra Ucraina (feb–dic 2022)",  "value": "dummy_ucraina"},
                            {"label": " GFC (set 2008–mar 2009)",        "value": "dummy_gfc"},
                            {"label": " Dot-com (mar 2000–ott 2002)",    "value": "dummy_dotcom"},
                        ],
                        value=[],
                        inline=True,
                        style={"font-size": "10px"},
                        inputStyle={"margin-right": "3px"},
                        labelStyle={"margin-right": "12px", "margin-bottom": "4px"},
                    ),
                ], style={"background": "#fff8f0", "border": "1px solid #ffcc80",
                          "border-radius": "4px", "padding": "6px 10px",
                          "margin-bottom": "10px"}),

                html.Hr(style={"margin": "8px 0"}),

                # ④ Opzioni stima
                html.B("④ Opzioni stima",
                       style={"font-size": "10px", "color": "#555",
                              "display": "block", "margin-bottom": "6px"}),
                html.Div([
                    html.Div([
                        html.Label("Std Error:",
                                   style={"font-size": "10px",
                                          "font-weight": "bold",
                                          "margin-right": "6px"}),
                        dcc.RadioItems(
                            id="adl-cov-type",
                            options=[{"label": " OLS", "value": "nonrobust"},
                                     {"label": " HC3", "value": "HC3"},
                                     {"label": " HAC", "value": "HAC"}],
                            value="HAC", inline=True,
                            style={"font-size": "10px"},
                            inputStyle={"margin-right": "3px"},
                            labelStyle={"margin-right": "8px"},
                        ),
                    ], style={"display": "flex", "align-items": "center",
                              "margin-bottom": "6px"}),
                    dcc.Checklist(
                        id="adl-add-const",
                        options=[{"label": " Includi costante α", "value": "const"}],
                        value=["const"],
                        style={"font-size": "10px"},
                        inputStyle={"margin-right": "3px"},
                    ),
                ], style={"background": "#f8f9fa", "border": "1px solid #dee2e6",
                          "border-radius": "4px", "padding": "6px 10px",
                          "margin-bottom": "10px"}),

                html.Hr(style={"margin": "8px 0"}),

                # ⑤ Periodo stima
                html.B("⑤ Periodo stima",
                       style={"font-size": "10px", "display": "block",
                              "margin-bottom": "4px"}),
                dcc.RangeSlider(id="adl-slider", min=0, max=1, value=[0, 1],
                                marks={}, step=86400 * 30,
                                tooltip={"placement": "bottom",
                                         "always_visible": False}),
                html.Div(id="adl-slider-label",
                         style={"font-size": "9px", "color": "#666",
                                "text-align": "center", "margin-bottom": "10px"}),

                html.Button("▶  Stima modello ADL", id="btn-run-adl", n_clicks=0,
                            style={"background": "#1b5e20", "color": "white",
                                   "border": "none", "padding": "8px 14px",
                                   "border-radius": "4px", "cursor": "pointer",
                                   "font-size": "12px", "font-weight": "bold",
                                   "width": "100%"}),
                html.Div(id="adl-status",
                         style={"font-size": "10px", "color": "#555",
                                "margin-top": "5px", "font-style": "italic"}),

            ], style={"width": "420px", "min-width": "400px",
                      "padding": "12px", "border-right": "1px solid #ddd",
                      "height": "calc(100vh - 100px)",
                      "overflow-y": "auto", "background": "#fafafa"}),

            # ── Colonna destra: risultati ─────────────────────────────────
            html.Div([
                dcc.Loading(type="circle", children=[
                    html.Div(id="adl-equation",
                             style={"font-family": "monospace", "font-size": "11px",
                                    "background": "#f8f9fa", "border": "1px solid #dee2e6",
                                    "border-radius": "4px", "padding": "10px 16px",
                                    "margin": "10px 16px 0", "white-space": "pre-wrap",
                                    "color": "#1a3a5c"}),
                    html.Div(id="adl-stats-table", style={"margin": "8px 16px 0"}),
                    html.Div(id="adl-coef-table",  style={"margin": "8px 16px 0"}),

                    html.Div([
                        html.B("📈  Osservato vs Stimato",
                               style={"font-size": "11px", "color": "#1a3a5c"}),
                    ], style={"padding": "5px 16px", "background": "#eaf4fb",
                              "border-top": "1px solid #aed6f1",
                              "border-bottom": "1px solid #aed6f1",
                              "margin-top": "10px"}),
                    dcc.Graph(id="chart-adl-fit",
                              figure=empty_fig("Stima il modello"),
                              style={"height": "32vh"},
                              config={"responsive": True}),

                    html.Div([
                        html.B("📊  IRF — Risposta impulsiva",
                               style={"font-size": "11px", "color": "#1a3a5c"}),
                        html.Span("  Effetto di uno shock +1σ per ogni X, per lag",
                                  style={"font-size": "10px", "color": "#666",
                                         "margin-left": "8px"}),
                    ], style={"padding": "5px 16px", "background": "#e8f5e9",
                              "border-top": "1px solid #a5d6a7",
                              "border-bottom": "1px solid #a5d6a7"}),
                    dcc.Graph(id="chart-adl-irf",
                              figure=empty_fig(""),
                              style={"height": "35vh"},
                              config={"responsive": True}),

                    html.Div([
                        html.B("📐  Legenda σ — deviazione standard delle variabili X",
                               style={"font-size": "11px", "color": "#1a5276"}),
                        html.Span("  Lo shock +1σ corrisponde a una variazione tipica (= 1 dev. std.) "
                                  "della variabile X nel periodo stimato",
                                  style={"font-size": "10px", "color": "#555",
                                         "margin-left": "8px"}),
                    ], style={"padding": "5px 16px", "background": "#eaf4fb",
                              "border-top": "1px solid #aed6f1",
                              "border-bottom": "1px solid #aed6f1"}),
                    html.Div(id="adl-irf-sigma-table",
                             style={"margin": "6px 16px 10px"}),

                    html.Div([
                        html.B("🔬  Diagnostica residui",
                               style={"font-size": "11px", "color": "#1a3a5c"}),
                    ], style={"padding": "5px 16px", "background": "#fff8e1",
                              "border-top": "1px solid #ffe082",
                              "border-bottom": "1px solid #ffe082"}),
                    dcc.Graph(id="chart-adl-resid",
                              figure=empty_fig(""),
                              style={"height": "28vh"},
                              config={"responsive": True}),
                    dcc.Graph(id="chart-adl-qq",
                              figure=empty_fig(""),
                              style={"height": "28vh"},
                              config={"responsive": True}),
                ]),
            ], style={"flex": "1", "min-width": "0",
                      "overflow-y": "auto",
                      "height": "calc(100vh - 100px)"}),

        ], style={"display": "flex"}),
    ])


# =============================================================================
# LAYOUT — Tab Curva di Phillips
# =============================================================================

def _phillips_tab_layout():
    """
    Tab per la stima empirica della Curva di Phillips Neo-Keynesiana:

      π_t = α + γ·π^e_t + δ·ỹ_t + ε_t

    Metodi output gap: HP filter, NAIRU gap (legge di Okun)
    Aspettative:       adattive (π_{t-1}), MA(N), breakeven TIPS, survey Michigan/SPF
    Modalità:          livelli o inflation gap (π_t − π*)
    """
    PAD = {"padding": "10px"}
    SL  = {"margin-bottom": "14px"}
    HDR = {"font-size": "11px", "font-weight": "700",
           "text-transform": "uppercase", "color": "#555",
           "margin-bottom": "4px", "margin-top": "12px"}

    def _sl(id_, lo, hi, step, val, lbl, marks=None):
        return html.Div([
            html.Div(lbl, style=HDR),
            dcc.Slider(id=id_, min=lo, max=hi, step=step, value=val,
                       marks=marks or {lo: str(lo), hi: str(hi)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style=SL)

    # ── sidebar ──────────────────────────────────────────────────────────────
    sidebar = html.Div([
        html.H4("Curva di Phillips", style={"margin": "0 0 16px",
                                             "font-size": "15px",
                                             "border-bottom": "2px solid #1f77b4",
                                             "padding-bottom": "6px"}),

        html.Div("Fonte dati", style=HDR),
        dcc.RadioItems(id="pc-source",
                       options=[{"label": " 🇺🇸 USA",    "value": "usa"},
                                 {"label": " 🇪🇺 EUR",    "value": "eur"}],
                       value="usa",
                       labelStyle={"display": "block", "margin-bottom": "4px",
                                   "font-size": "13px"}),

        html.Div("Frequenza", style={**HDR, "margin-top": "14px"}),
        dcc.RadioItems(id="pc-freq",
                       options=[{"label": " Mensile (M)",     "value": "M"},
                                 {"label": " Trimestrale (Q)", "value": "Q"},
                                 {"label": " Annuale (Y)",    "value": "Y"}],
                       value="Q",
                       labelStyle={"display": "block", "margin-bottom": "4px",
                                   "font-size": "13px"}),

        html.Hr(style={"margin": "14px 0 10px"}),

        html.Div("Output Gap — Metodi", style=HDR),
        dcc.Checklist(id="pc-gap-methods",
                      options=[
                          {"label": " HP Filter (PIL reale)",      "value": "hp"},
                          {"label": " NAIRU Gap (disoccupazione)",  "value": "nairu"},
                          {"label": " Trend lineare (PIL reale)",   "value": "linear"},
                      ],
                      value=["hp", "nairu"],
                      labelStyle={"display": "block", "margin-bottom": "5px",
                                  "font-size": "13px"}),

        _sl("pc-hp-lambda", 100, 129600, 100, 1600,
            "HP Lambda (129600=M, 1600=Q, 100=Y)",
            {100: "100", 1600: "1600", 14400: "14400", 129600: "129600"}),

        html.Hr(style={"margin": "14px 0 10px"}),

        html.Div("Aspettative d'inflazione", style=HDR),
        dcc.Checklist(id="pc-exp-methods",
                      options=[
                          {"label": " Adattive  π_{t-1}",         "value": "adaptive"},
                          {"label": " Media mobile MA(N)",         "value": "ma"},
                          {"label": " Breakeven TIPS (🇺🇸 only)",  "value": "breakeven"},
                          {"label": " Survey Michigan (🇺🇸 only)", "value": "survey"},
                      ],
                      value=["adaptive"],
                      labelStyle={"display": "block", "margin-bottom": "5px",
                                  "font-size": "13px"}),

        _sl("pc-ma-window", 2, 12, 1, 4,
            "Finestra MA (trimestri)",
            {2: "2", 4: "4", 8: "8", 12: "12"}),

        _sl("pc-nairu-window", 8, 120, 4, 40,
            "Finestra NAIRU forma ridotta (periodi)",
            {8: "8", 20: "20", 40: "40", 80: "80", 120: "120"}),

        html.Hr(style={"margin": "14px 0 10px"}),

        html.Div("Regressione", style=HDR),
        dcc.Checklist(id="pc-gap-mode",
                      options=[{"label": " Usa Inflation Gap  (π_t − π*)",
                                "value": "gap"}],
                      value=[],
                      labelStyle={"font-size": "13px"}),

        html.Div("π* target (%)", style={**HDR, "margin-top": "10px"}),
        dcc.Input(id="pc-pi-star", type="number", value=2.0, step=0.5,
                  min=0, max=10,
                  style={"width": "80px", "font-size": "13px",
                         "border": "1px solid #ccc", "border-radius": "4px",
                         "padding": "4px 6px"}),

        html.Div("Variabile aspettative per regressione", style={**HDR, "margin-top": "12px"}),
        dcc.Dropdown(id="pc-exp-for-reg",
                     options=[{"label": "Adattive (π_{t-1})", "value": "adaptive"},
                               {"label": "MA(N)",              "value": "ma"},
                               {"label": "Breakeven TIPS",     "value": "breakeven"},
                               {"label": "Survey Michigan",    "value": "survey"}],
                     value="adaptive",
                     clearable=False,
                     style={"font-size": "13px"}),

        html.Div("Output gap per regressione", style={**HDR, "margin-top": "10px"}),
        dcc.Dropdown(id="pc-gap-for-reg",
                     options=[{"label": "HP Filter",     "value": "hp"},
                               {"label": "NAIRU Gap",    "value": "nairu"},
                               {"label": "Trend lineare","value": "linear"}],
                     value="hp",
                     clearable=False,
                     style={"font-size": "13px"}),

        html.Hr(style={"margin": "14px 0 10px"}),

        html.Button("▶  Carica & Stima", id="btn-run-phillips", n_clicks=0,
                    style={"width": "100%", "background": "#1f77b4",
                           "color": "white", "border": "none",
                           "border-radius": "6px", "padding": "10px",
                           "font-size": "13px", "cursor": "pointer",
                           "font-weight": "bold"}),

        html.Div(id="pc-status", style={"margin-top": "10px",
                                         "font-size": "11px",
                                         "color": "#555",
                                         "white-space": "pre-wrap"}),

        html.Hr(style={"margin": "16px 0 10px"}),
        html.Button("→ Invia κ e β al DSGE", id="btn-pc-to-dsge", n_clicks=0,
                    style={"width": "100%", "background": "#2ca02c",
                           "color": "white", "border": "none",
                           "border-radius": "6px", "padding": "8px",
                           "font-size": "12px", "cursor": "pointer"}),
        html.Div(id="pc-to-dsge-status",
                 style={"font-size": "11px", "color": "#2ca02c",
                        "margin-top": "6px", "text-align": "center"}),
        html.Button("→ Invia σ, φπ, φy al DSGE", id="btn-pc-is-to-dsge", n_clicks=0,
                    style={"width": "100%", "background": "#9467bd",
                           "color": "white", "border": "none",
                           "border-radius": "6px", "padding": "8px",
                           "font-size": "12px", "cursor": "pointer",
                           "margin-top": "6px"}),
        html.Div(id="pc-is-to-dsge-status",
                 style={"font-size": "11px", "color": "#9467bd",
                        "margin-top": "6px", "text-align": "center"}),

    ], style={"width": "290px", "min-width": "290px", "padding": "16px",
              "background": "#fafafa", "border-right": "1px solid #ddd",
              "overflow-y": "auto", "height": "calc(100vh - 100px)"})

    # ── pannello risultati ────────────────────────────────────────────────────
    results = html.Div([
        dcc.Tabs(id="pc-result-tabs", value="pc-tab-prices",
                 children=[
                     dcc.Tab(label="📉 Prezzi & Inflazione", value="pc-tab-prices"),
                     dcc.Tab(label="📊 Output Gap",          value="pc-tab-gap"),
                     dcc.Tab(label="📐 NAIRU",               value="pc-tab-nairu"),
                     dcc.Tab(label="🔮 Aspettative",          value="pc-tab-exp"),
                     dcc.Tab(label="⭕ Phillips Scatter",     value="pc-tab-scatter"),
                     dcc.Tab(label="📋 Regressione",          value="pc-tab-reg"),
                     dcc.Tab(label="⚗ GMM Galí-Gertler",     value="pc-tab-gmm"),
                     dcc.Tab(label="⚙ IS + Taylor Rule",     value="pc-tab-is-taylor"),
                 ],
                 style={"font-size": "12px"}),

        dcc.Loading(
            id="pc-loading",
            type="circle",
            color="#1f77b4",
            children=html.Div(id="pc-tab-content",
                              style={"padding": "10px", "height": "calc(100vh - 160px)",
                                     "overflow-y": "auto"}),
        ),

        # Hidden stores
        dcc.Store(id="store-phillips",  storage_type="local"),
        dcc.Store(id="store-pc-kappa",  storage_type="local"),   # stima κ per DSGE
        dcc.Store(id="store-pc-beta",   storage_type="local"),   # stima β (=γ) per DSGE
        dcc.Store(id="store-pc-sigma",  storage_type="local"),   # stima σ IS curve
        dcc.Store(id="store-pc-phi-pi", storage_type="local"),   # stima φπ Taylor Rule
        dcc.Store(id="store-pc-phi-y",  storage_type="local"),   # stima φy Taylor Rule
    ], style={"flex": "1", "overflow": "hidden"})

    return html.Div([sidebar, results],
                    style={"display": "flex", "height": "calc(100vh - 100px)"})


# =============================================================================
# LAYOUT — Tab DSGE
# =============================================================================

def _dsge_tab_layout():
    """
    Tab per la simulazione di un modello DSGE New Keynesian semplificato a 3 equazioni:

      IS  (domanda):     x(t) = x(t-1) − (1/σ)·(i(t-1) − π(t-1) − r*)  + shock_domanda
      NKPC (inflazione): π(t) = π* + β·(π(t-1)−π*) + κ·x(t)              + shock_costi
      TR  (banca centr.): i(t) = r* + π* + φπ·(π(t)−π*) + φy·x(t)        + shock_monetario

    Come usarlo:
      1. Calibra i parametri strutturali (σ, κ, β, φπ, φy)
         - φπ > 1 è la condizione di Blanchard-Kahn per la stabilità
      2. Imposta i target (π*, r*) e le condizioni iniziali
      3. Scegli gli shock: domanda, costi (inflazionistico), monetario
      4. Imposta la persistenza ρ (quanto dura lo shock) e i periodi da simulare
      5. Clicca Simula e osserva:
         - IRF: come reagiscono output gap, inflazione e tasso
         - Diagramma di fase: convergenza all'equilibrio
         - Tasso nominale vs reale nel tempo
    """
    def _sl(id_, mn, mx, step, val, marks, label):
        return html.Div([
            html.Label(label, style={"font-size": "10px"}),
            dcc.Slider(id=id_, min=mn, max=mx, step=step, value=val,
                       marks=marks,
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(style={"height": "8px"}),
        ])

    return html.Div([

        # Header ────────────────────────────────────────────────────────────
        html.Div([
            html.H3("Modello DSGE — New Keynesian",
                    style={"margin": "0 20px 0 0", "font-size": "15px",
                           "color": "#1a3a5c", "white-space": "nowrap"}),
            html.Span(
                "Tre equazioni: IS (domanda aggregata), NKPC (curva di Phillips), "
                "Taylor Rule (banca centrale). "
                "Calibra i parametri strutturali e simula la risposta agli shock.",
                style={"font-size": "11px", "color": "#666"}),
        ], style={"display": "flex", "align-items": "center",
                  "padding": "8px 16px", "background": "#f0f4fa",
                  "border-bottom": "1px solid #dee2e6",
                  "flex-wrap": "wrap", "gap": "8px"}),

        html.Div([

            # Sidebar: parametri ──────────────────────────────────────────
            html.Div([

                html.B("📐 Parametri strutturali",
                       style={"font-size": "10px", "color": "#1a5276",
                              "background": "#eaf4fb", "display": "block",
                              "padding": "4px 8px", "border-radius": "3px",
                              "margin-bottom": "8px"}),
                html.Div("Curva IS",
                         style={"font-size": "9px", "color": "#888",
                                "font-weight": "bold", "margin-bottom": "4px"}),
                _sl("dsge-sigma", 0.1, 3.0, 0.1, 1.0,
                    {0.1: "0.1", 1.0: "1", 3.0: "3"},
                    "σ — elasticità intertemporale"),
                html.Div("Curva di Phillips",
                         style={"font-size": "9px", "color": "#888",
                                "font-weight": "bold", "margin-bottom": "4px"}),
                _sl("dsge-kappa", 0.01, 0.5, 0.01, 0.15,
                    {0.01: "0.01", 0.15: "0.15", 0.5: "0.5"},
                    "κ — slope della Phillips curve"),
                _sl("dsge-beta", 0.90, 0.999, 0.001, 0.99,
                    {0.90: "0.9", 0.99: "0.99", 0.999: "0.999"},
                    "β — fattore di sconto"),
                html.Div("Taylor Rule",
                         style={"font-size": "9px", "color": "#888",
                                "font-weight": "bold", "margin-bottom": "4px"}),
                _sl("dsge-phi-pi", 0.5, 4.0, 0.1, 1.5,
                    {0.5: "0.5", 1.5: "1.5", 4.0: "4"},
                    "φπ — risposta a inflazione (deve essere > 1)"),
                _sl("dsge-phi-y", 0.0, 2.0, 0.1, 0.5,
                    {0.0: "0", 0.5: "0.5", 2.0: "2"},
                    "φy — risposta a output gap"),

                html.Hr(style={"margin": "8px 0"}),

                html.B("🎯 Target e stato stazionario",
                       style={"font-size": "10px", "color": "#555",
                              "display": "block", "margin-bottom": "8px"}),
                _sl("dsge-pi-star", 0.0, 4.0, 0.5, 2.0,
                    {0.0: "0%", 2.0: "2%", 4.0: "4%"},
                    "π* — target inflazione (%)"),
                _sl("dsge-r-star", 0.0, 4.0, 0.25, 1.0,
                    {0.0: "0%", 1.0: "1%", 4.0: "4%"},
                    "r* — tasso naturale reale (%)"),

                html.Hr(style={"margin": "8px 0"}),

                html.B("⚡ Shock esogeni",
                       style={"font-size": "10px", "color": "#b71c1c",
                              "background": "#ffebee", "display": "block",
                              "padding": "4px 8px", "border-radius": "3px",
                              "margin-bottom": "8px"}),
                _sl("dsge-demand-shock", -5.0, 5.0, 0.25, 0.0,
                    {-5: "-5", 0: "0", 5: "5"},
                    "Shock domanda — output gap iniziale"),
                _sl("dsge-cost-push", -3.0, 5.0, 0.25, 1.0,
                    {-3: "-3", 0: "0", 5: "+5%"},
                    "Shock costi — impulso inflazionistico"),
                _sl("dsge-monetary-shock", -3.0, 3.0, 0.25, 0.0,
                    {-3: "-3", 0: "0", 3: "+3"},
                    "Shock monetario — deviazione dalla TR"),
                _sl("dsge-persistence", 0.0, 0.95, 0.05, 0.5,
                    {0.0: "0", 0.5: "0.5", 0.95: "0.95"},
                    "ρ — persistenza shock (AR(1))"),

                html.Hr(style={"margin": "8px 0"}),

                _sl("dsge-periods", 4, 60, 4, 20,
                    {4: "4", 20: "20", 40: "40", 60: "60"},
                    "Periodi da simulare (trimestri)"),

                html.Button("▶  Simula DSGE", id="btn-run-dsge", n_clicks=0,
                            style={"background": "#4e342e", "color": "white",
                                   "border": "none", "padding": "8px 14px",
                                   "border-radius": "4px", "cursor": "pointer",
                                   "font-size": "12px", "font-weight": "bold",
                                   "width": "100%"}),
                html.Div(id="dsge-status",
                         style={"font-size": "10px", "color": "#555",
                                "margin-top": "5px", "font-style": "italic"}),

            ], style={"width": "280px", "min-width": "270px",
                      "padding": "12px", "border-right": "1px solid #ddd",
                      "height": "calc(100vh - 100px)",
                      "overflow-y": "auto", "background": "#fafafa"}),

            # Risultati ───────────────────────────────────────────────────
            html.Div([
                dcc.Loading(type="circle", children=[

                    html.Div(id="dsge-equations",
                             style={"font-family": "monospace", "font-size": "11px",
                                    "background": "#f8f9fa", "border": "1px solid #dee2e6",
                                    "border-radius": "4px", "padding": "10px 16px",
                                    "margin": "10px 16px 0", "white-space": "pre-wrap",
                                    "color": "#1a3a5c"}),

                    html.Div([
                        html.B("📊  Risposta impulsiva (IRF)",
                               style={"font-size": "11px", "color": "#1a3a5c"}),
                        html.Span("  Output gap (%)  |  Inflazione (%)  |  Tasso nominale (%)",
                                  style={"font-size": "10px", "color": "#666",
                                         "margin-left": "8px"}),
                    ], style={"padding": "5px 16px", "background": "#eaf4fb",
                              "border-top": "1px solid #aed6f1",
                              "border-bottom": "1px solid #aed6f1", "margin-top": "10px"}),
                    dcc.Graph(id="chart-dsge-irf",
                              figure=empty_fig("Configura i parametri e clicca Simula"),
                              style={"height": "40vh"}, config={"responsive": True}),

                    html.Div([
                        html.B("🔄  Diagramma di fase — Output gap vs Inflazione",
                               style={"font-size": "11px", "color": "#1a3a5c"}),
                        html.Span("  La traiettoria mostra la convergenza all'equilibrio",
                                  style={"font-size": "10px", "color": "#666",
                                         "margin-left": "8px"}),
                    ], style={"padding": "5px 16px", "background": "#e8f5e9",
                              "border-top": "1px solid #a5d6a7",
                              "border-bottom": "1px solid #a5d6a7"}),
                    dcc.Graph(id="chart-dsge-phase",
                              figure=empty_fig(""),
                              style={"height": "38vh"}, config={"responsive": True}),

                    html.Div([
                        html.B("📈  Tasso nominale vs reale nel tempo",
                               style={"font-size": "11px", "color": "#1a3a5c"}),
                    ], style={"padding": "5px 16px", "background": "#fff8e1",
                              "border-top": "1px solid #ffe082",
                              "border-bottom": "1px solid #ffe082"}),
                    dcc.Graph(id="chart-dsge-rates",
                              figure=empty_fig(""),
                              style={"height": "28vh"}, config={"responsive": True}),
                ]),
            ], style={"flex": "1", "min-width": "0",
                      "overflow-y": "auto",
                      "height": "calc(100vh - 100px)"}),

        ], style={"display": "flex"}),
    ])




# =============================================================================
# CONFRONTO SERIE STORICHE — Layout
# =============================================================================

def _compare_tab_layout():
    """Tab confronto serie storiche — overlay o subplots con transform per serie."""
    _sb = {
        "width": "290px", "min-width": "290px",
        "padding": "14px 12px",
        "background": "#f4f6fa",
        "border-right": "1px solid #dee2e6",
        "overflow-y": "auto",
        "font-family": "sans-serif",
    }
    _lbl = {"font-size": "11px", "font-weight": "bold",
            "color": "#1a3a5c", "margin-bottom": "4px", "display": "block"}
    _sec = {"margin-bottom": "14px"}

    return html.Div([
        # ── SIDEBAR ─────────────────────────────────────────────────────────
        html.Div([
            html.H5("📊 Confronto Serie Storiche", style={
                "color": "#1a3a5c", "margin": "0 0 14px",
                "border-bottom": "2px solid #2e6da4", "padding-bottom": "8px",
                "font-size": "13px",
            }),

            # Serie dai dati caricati
            html.Div([
                html.Label("Serie dai dati caricati:", style=_lbl),
                dcc.Dropdown(
                    id="csr-series-dropdown",
                    options=[],
                    multi=True,
                    placeholder="Seleziona…",
                    style={"font-size": "11px"},
                ),
            ], style=_sec),

            # Aggiungi serie FRED
            html.Div([
                html.Label("Aggiungi serie FRED (ID, virgola):", style=_lbl),
                dcc.Input(
                    id="csr-fred-input",
                    type="text",
                    placeholder="es. FEDFUNDS, CPIAUCSL",
                    debounce=False,
                    style={
                        "width": "100%", "box-sizing": "border-box",
                        "font-size": "11px", "padding": "5px 8px",
                        "background": "#ffffff", "color": "#1a1a1a",
                        "border": "1px solid #aed6f1", "border-radius": "4px",
                    },
                ),
                html.Button(
                    "+ Aggiungi da FRED", id="csr-fred-btn", n_clicks=0,
                    style={
                        "margin-top": "6px", "width": "100%",
                        "background": "#1a5276", "color": "#ffffff",
                        "border": "none", "padding": "6px",
                        "border-radius": "4px", "cursor": "pointer",
                        "font-size": "11px",
                    },
                ),
                html.Div(id="csr-fred-status",
                         style={"font-size": "10px", "color": "#555", "margin-top": "4px"}),
            ], style=_sec),

            html.Hr(style={"border-color": "#dee2e6", "margin": "8px 0"}),

            # Controlli trasformazione per serie
            html.Div(
                id="csr-transform-controls",
                children=html.Div(
                    "— seleziona serie per configurare le trasformazioni —",
                    style={"font-size": "10px", "color": "#888", "font-style": "italic"},
                ),
                style={"margin-bottom": "14px"},
            ),

            html.Hr(style={"border-color": "#dee2e6", "margin": "8px 0"}),

            # Intervallo temporale
            html.Div([
                html.Label("Intervallo temporale:", style=_lbl),
                html.Div(
                    id="csr-date-label",
                    style={"font-size": "10px", "color": "#2e6da4",
                           "margin-bottom": "8px", "text-align": "center"},
                ),
                dcc.RangeSlider(
                    id="csr-date-slider",
                    min=0, max=1, value=[0, 1],
                    marks={}, step=None,
                    allowCross=False,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style={"margin-bottom": "14px", "padding-bottom": "4px"}),

            html.Hr(style={"border-color": "#dee2e6", "margin": "8px 0"}),

            # Opzioni layout
            html.Div([
                html.Label("Layout grafico:", style=_lbl),
                dcc.RadioItems(
                    id="csr-layout-mode",
                    options=[
                        {"label": " Subplots (una riga per serie)", "value": "subplots"},
                        {"label": " Sovrapposto (asse singolo)",    "value": "overlay"},
                    ],
                    value="subplots",
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "4px"},
                    labelStyle={"color": "#333", "margin-bottom": "4px",
                                "display": "block"},
                ),
            ], style=_sec),

            html.Div([
                dcc.Checklist(
                    id="csr-normalize",
                    options=[{"label": " Normalizza a 100 all'inizio", "value": "norm"}],
                    value=[],
                    style={"font-size": "11px"},
                    inputStyle={"margin-right": "4px"},
                    labelStyle={"color": "#333"},
                ),
            ], style={"margin-bottom": "16px"}),

            html.Button(
                "🔄 Aggiorna grafico", id="csr-update-btn", n_clicks=0,
                style={
                    "width": "100%", "background": "#1b5e20",
                    "color": "#ffffff", "border": "none",
                    "padding": "8px", "border-radius": "4px",
                    "cursor": "pointer", "font-size": "12px", "font-weight": "bold",
                },
            ),

            dcc.Store(id="store-csr-extra"),

        ], style=_sb),

        # ── GRAFICO PRINCIPALE ───────────────────────────────────────────────
        html.Div([
            dcc.Graph(
                id="csr-graph",
                figure={
                    "data": [],
                    "layout": {
                        "template": "plotly_white",
                        "paper_bgcolor": "#ffffff",
                        "plot_bgcolor":  "#ffffff",
                        "annotations": [{
                            "text": "Seleziona serie e clicca <b>Aggiorna grafico</b>",
                            "xref": "paper", "yref": "paper",
                            "x": 0.5, "y": 0.5,
                            "showarrow": False,
                            "font": {"color": "#555", "size": 16},
                        }],
                    },
                },
                style={"height": "calc(100vh - 110px)", "min-height": "380px"},
                config={
                    "displayModeBar": True,
                    "toImageButtonOptions": {"format": "svg", "width": 1400, "height": 900},
                },
            ),
        ], style={"flex": "1", "padding": "6px", "overflow-y": "auto",
                   "overflow-x": "hidden"}),

    ], style={"display": "flex", "height": "calc(100vh - 95px)", "overflow": "hidden"})


# =============================================================================
# VALUATION TAB LAYOUT
# =============================================================================

def _valuation_tab_layout():
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

        html.B("⚙ Parametri DCF", style={"font-size": "10px", "color": "#1a5276",
                                          "background": "#eaf4fb", "display": "block",
                                          "padding": "4px 8px", "border-radius": "3px",
                                          "margin-bottom": "8px"}),
        _sl("val-wacc",    4.0, 20.0, 0.5,  9.0, "WACC (%)"),
        _sl("val-g1",      0.0, 40.0, 0.5, 12.0, "Crescita fase 1 — anni 1-5 (%)"),
        _sl("val-g2",      0.0, 20.0, 0.5,  6.0, "Crescita fase 2 — anni 6-10 (%)"),
        _sl("val-gterm",   0.0,  6.0, 0.25, 2.5, "Crescita terminale g (%)"),
        _sl("val-fcf-margin", 1.0, 50.0, 0.5, 15.0, "Margine FCF/Revenue (%)"),

        html.Hr(style={"margin": "10px 0"}),

        html.B("⚙ Parametri multipli", style={"font-size": "10px", "color": "#1a5276",
                                               "background": "#eaf4fb", "display": "block",
                                               "padding": "4px 8px", "border-radius": "3px",
                                               "margin-bottom": "8px"}),
        _sl("val-pe-sector",    5.0, 60.0, 1.0, 22.0, "P/E settore (multiplo)"),
        _sl("val-ev-ebitda",    3.0, 30.0, 0.5, 12.0, "EV/EBITDA settore (multiplo)"),
        _sl("val-ke",           4.0, 20.0, 0.5, 10.0, "Ke — costo equity DDM (%)"),
        _sl("val-bond-yield",   1.0, 10.0, 0.25, 4.5, "Rendimento AAA bond (Graham, %)"),

    ], style={"width": "270px", "min-width": "270px", "padding": "14px",
              "background": "#fafafa", "border-right": "1px solid #ddd",
              "overflow-y": "auto", "height": "calc(100vh - 100px)"})

    results = html.Div([
        dcc.Tabs(id="val-result-tabs", value="val-tab-summary",
                 children=[
                     dcc.Tab(label="📊 Riepilogo",       value="val-tab-summary"),
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
                                     "height": "calc(100vh - 160px)",
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
                 style={"display": "flex", "height": "calc(100vh - 100px)"}),
    ])


# =============================================================================
# APP LAYOUT
# =============================================================================

def _fred_navbar():
    from navbar import make_navbar
    return make_navbar(current='macro')


app.layout = html.Div([

    dcc.Location(id='fred-url', refresh=False),
    dcc.Interval(id='fred-tab-once', interval=250, n_intervals=0, max_intervals=1),

    _fred_navbar(),
    html.Div(style={'height': '102px'}),   # spazio per la navbar fissa del sito

    # ── HEADER ───────────────────────────────────────────────────────────────
    html.Div([
        html.H1("Macro Economia",
                style={"margin": "0 16px 0 0", "font-size": "20px",
                       "white-space": "nowrap", "color": "#1a3a5c"}),
        html.Div([
            html.Label("API Key FRED:",
                       style={"font-size": "11px", "margin-right": "5px"}),
            dcc.Input(id="api-key", type="password", value=FRED_API_KEY,
                      debounce=True,
                      style={"width": "200px", "font-size": "11px"}),
        ], style={"display": "flex", "align-items": "center",
                  "margin-right": "14px"}),

        dcc.Upload(
            id="upload-file",
            children=html.Div("📂  Carica xlsx (opzionale)"),
            style={"min-width": "170px", "height": "30px",
                   "lineHeight": "30px", "border": "1px dashed #888",
                   "border-radius": "4px", "text-align": "center",
                   "font-size": "11px", "cursor": "pointer",
                   "margin-right": "10px", "color": "#555"},
            multiple=False,
        ),
        html.Div(id="upload-label",
                 style={"font-size": "10px", "color": "#888",
                        "font-style": "italic", "min-width": "100px"}),
    ], style={"display": "flex", "align-items": "center",
              "padding": "8px 16px", "background": "#f0f4fa",
              "border-bottom": "2px solid #2e6da4",
              "flex-wrap": "wrap", "gap": "6px"}),

    # ── STORES ───────────────────────────────────────────────────────────────
    dcc.Store(id="store-data"),
    dcc.Store(id="store-upload-meta"),
    dcc.Store(id="store-yields"),
    dcc.Store(id="store-yields-eur"),
    dcc.Store(id="store-gdp"),
    dcc.Store(id="store-shock"),
    dcc.Store(id="store-eurostat"),
    dcc.Store(id="store-pil-groups"),
    dcc.Store(id="store-shock-eur", storage_type="session"),
    dcc.Store(id="store-shock-loading-state", data={"active": False}),
    dcc.Store(id="store-impact-model"),
    dcc.Store(id="store-mon-source-type", data="usa"),
    dcc.Store(id="store-mon-loading-state", data={"active": False}),
    dcc.Store(id="store-arima-source", storage_type="session"),
    dcc.Store(id="store-arima-loading-state", data={"active": False}),
    dcc.Store(id="store-adl-source", storage_type="session"),
    dcc.Store(id="store-adl-loading-state", data={"active": False}),

    # ── Overlay caricamento Analisi Monetaria ────────────────────────────────
    dcc.Interval(id="mon-progress-tick", interval=350,
                 disabled=True, n_intervals=0),
    html.Div([
        html.Div([
            html.Div(
                html.Div(style={
                    "width": "56px", "height": "56px",
                    "border": "6px solid rgba(255,255,255,0.15)",
                    "border-top": "6px solid #ffffff",
                    "border-radius": "50%",
                    "animation": "spin 0.9s linear infinite",
                }),
                style={"margin-bottom": "20px"}
            ),
            html.Div(id="mon-loading-title",
                     children="Caricamento dati in corso...",
                     style={"font-size": "22px", "font-weight": "bold",
                            "color": "white", "margin-bottom": "6px"}),
            html.Div(id="mon-loading-source",
                     children="",
                     style={"font-size": "14px", "color": "#aed6f1",
                            "margin-bottom": "20px"}),
            html.Div([
                html.Div(id="mon-progress-bar",
                         style={"width": "0%", "height": "100%",
                                "background": "linear-gradient(90deg,#7b0000,#e53935)",
                                "border-radius": "6px",
                                "transition": "width 0.3s ease"}),
            ], style={"width": "420px", "height": "14px",
                      "background": "rgba(255,255,255,0.15)",
                      "border-radius": "7px", "overflow": "hidden",
                      "margin-bottom": "10px"}),
            html.Div(id="mon-progress-pct",
                     children="0%",
                     style={"font-size": "28px", "font-weight": "bold",
                            "color": "#ef9a9a", "margin-bottom": "6px"}),
            html.Div(id="mon-progress-detail",
                     children="Connessione ai server...",
                     style={"font-size": "12px", "color": "#aaa",
                            "font-style": "italic"}),
        ], style={"display": "flex", "flex-direction": "column",
                  "align-items": "center", "justify-content": "center",
                  "background": "rgba(10,20,40,0.95)",
                  "border-radius": "16px", "padding": "50px 60px",
                  "box-shadow": "0 8px 40px rgba(0,0,0,0.6)"}),
    ], id="mon-loading-overlay",
       style={"display": "none", "position": "fixed",
              "top": "0", "left": "0", "width": "100%", "height": "100%",
              "background": "rgba(0,0,0,0.75)",
              "z-index": "9999",
              "align-items": "center", "justify-content": "center"}),

    # ── Overlay caricamento Shock Eurostat ────────────────────────────────────
    dcc.Interval(id="shock-progress-tick", interval=350,
                 disabled=True, n_intervals=0),
    html.Div([
        html.Div([
            html.Div(
                html.Div(style={
                    "width": "56px", "height": "56px",
                    "border": "6px solid rgba(255,255,255,0.15)",
                    "border-top": "6px solid #ffffff",
                    "border-radius": "50%",
                    "animation": "spin 0.9s linear infinite",
                }),
                style={"margin-bottom": "20px"}
            ),
            html.Div(id="shock-loading-title",
                     children="Caricamento dati in corso...",
                     style={"font-size": "22px", "font-weight": "bold",
                            "color": "white", "margin-bottom": "6px"}),
            html.Div(id="shock-loading-source",
                     children="",
                     style={"font-size": "14px", "color": "#aed6f1",
                            "margin-bottom": "20px"}),
            html.Div([
                html.Div(id="shock-progress-bar",
                         style={"width": "0%", "height": "100%",
                                "background": "linear-gradient(90deg,#7b2d00,#d4651a)",
                                "border-radius": "6px",
                                "transition": "width 0.3s ease"}),
            ], style={"width": "420px", "height": "14px",
                      "background": "rgba(255,255,255,0.15)",
                      "border-radius": "7px", "overflow": "hidden",
                      "margin-bottom": "10px"}),
            html.Div(id="shock-progress-pct",
                     children="0%",
                     style={"font-size": "28px", "font-weight": "bold",
                            "color": "#d4651a", "margin-bottom": "6px"}),
            html.Div(id="shock-progress-detail",
                     children="Connessione ai server...",
                     style={"font-size": "12px", "color": "#aaa",
                            "font-style": "italic"}),
        ], style={"display": "flex", "flex-direction": "column",
                  "align-items": "center", "justify-content": "center",
                  "background": "rgba(10,20,40,0.95)",
                  "border-radius": "16px", "padding": "50px 60px",
                  "box-shadow": "0 8px 40px rgba(0,0,0,0.6)"}),
    ], id="shock-loading-overlay",
       style={"display": "none", "position": "fixed",
              "top": "0", "left": "0", "width": "100%", "height": "100%",
              "background": "rgba(0,0,0,0.75)",
              "z-index": "9999",
              "align-items": "center", "justify-content": "center"}),

    # ── Overlay caricamento ADL ───────────────────────────────────────────────
    dcc.Interval(id="adl-progress-tick", interval=350,
                 disabled=True, n_intervals=0),
    html.Div([
        html.Div([
            html.Div(
                html.Div(style={
                    "width": "56px", "height": "56px",
                    "border": "6px solid rgba(255,255,255,0.15)",
                    "border-top": "6px solid #ffffff",
                    "border-radius": "50%",
                    "animation": "spin 0.9s linear infinite",
                }),
                style={"margin-bottom": "20px"}
            ),
            html.Div(id="adl-loading-title",
                     children="Caricamento dati in corso...",
                     style={"font-size": "22px", "font-weight": "bold",
                            "color": "white", "margin-bottom": "6px"}),
            html.Div(id="adl-loading-source",
                     children="",
                     style={"font-size": "14px", "color": "#aed6f1",
                            "margin-bottom": "20px"}),
            # barra progresso
            html.Div([
                html.Div(id="adl-progress-bar",
                         style={"width": "0%", "height": "100%",
                                "background": "linear-gradient(90deg,#1a5276,#2e86c1)",
                                "border-radius": "6px",
                                "transition": "width 0.3s ease"}),
            ], style={"width": "420px", "height": "14px",
                      "background": "rgba(255,255,255,0.15)",
                      "border-radius": "7px", "overflow": "hidden",
                      "margin-bottom": "10px"}),
            html.Div(id="adl-progress-pct",
                     children="0%",
                     style={"font-size": "28px", "font-weight": "bold",
                            "color": "#2e86c1", "margin-bottom": "6px"}),
            html.Div(id="adl-progress-detail",
                     children="Connessione ai server...",
                     style={"font-size": "12px", "color": "#aaa",
                            "font-style": "italic"}),
        ], style={"display": "flex", "flex-direction": "column",
                  "align-items": "center", "justify-content": "center",
                  "background": "rgba(10,20,40,0.95)",
                  "border-radius": "16px", "padding": "50px 60px",
                  "box-shadow": "0 8px 40px rgba(0,0,0,0.6)"}),
    ], id="adl-loading-overlay",
       style={"display": "none", "position": "fixed",
              "top": "0", "left": "0", "width": "100%", "height": "100%",
              "background": "rgba(0,0,0,0.75)",
              "z-index": "9999",
              "align-items": "center", "justify-content": "center"}),

    # ── Overlay caricamento ARIMA ─────────────────────────────────────────────
    dcc.Interval(id="arima-progress-tick", interval=350,
                 disabled=True, n_intervals=0),
    html.Div([
        html.Div([
            html.Div(
                html.Div(style={
                    "width": "56px", "height": "56px",
                    "border": "6px solid rgba(255,255,255,0.15)",
                    "border-top": "6px solid #ffffff",
                    "border-radius": "50%",
                    "animation": "spin 0.9s linear infinite",
                }),
                style={"margin-bottom": "20px"}
            ),
            html.Div(id="arima-loading-title",
                     children="Caricamento dati in corso...",
                     style={"font-size": "22px", "font-weight": "bold",
                            "color": "white", "margin-bottom": "6px"}),
            html.Div(id="arima-loading-source",
                     children="",
                     style={"font-size": "14px", "color": "#aed6f1",
                            "margin-bottom": "20px"}),
            html.Div([
                html.Div(id="arima-progress-bar",
                         style={"width": "0%", "height": "100%",
                                "background": "linear-gradient(90deg,#7b0000,#e53935)",
                                "border-radius": "6px",
                                "transition": "width 0.3s ease"}),
            ], style={"width": "420px", "height": "14px",
                      "background": "rgba(255,255,255,0.15)",
                      "border-radius": "7px", "overflow": "hidden",
                      "margin-bottom": "10px"}),
            html.Div(id="arima-progress-pct",
                     children="0%",
                     style={"font-size": "28px", "font-weight": "bold",
                            "color": "#ef9a9a", "margin-bottom": "6px"}),
            html.Div(id="arima-progress-detail",
                     children="Connessione ai server...",
                     style={"font-size": "12px", "color": "#aaa",
                            "font-style": "italic"}),
        ], style={"display": "flex", "flex-direction": "column",
                  "align-items": "center", "justify-content": "center",
                  "background": "rgba(10,20,40,0.95)",
                  "border-radius": "16px", "padding": "50px 60px",
                  "box-shadow": "0 8px 40px rgba(0,0,0,0.6)"}),
    ], id="arima-loading-overlay",
       style={"display": "none", "position": "fixed",
              "top": "0", "left": "0", "width": "100%", "height": "100%",
              "background": "rgba(0,0,0,0.75)",
              "z-index": "9999",
              "align-items": "center", "justify-content": "center"}),

    # ── TABS ─────────────────────────────────────────────────────────────────
    dcc.Tabs(id="main-tabs", value="tab1", children=[

        dcc.Tab(label="📊  Analisi Monetaria", value="tab1", children=[
            _controls_bar(),
            html.Div([
                html.Div(_sidebar_default(),
                         style={"width": "210px", "min-width": "200px",
                                "border-right": "1px solid #ddd",
                                "height": "calc(100vh - 120px)",
                                "overflow-y": "auto",
                                "background": "#fafafa"}),
                html.Div([
                    _slider_area(),
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-main",
                                  figure=empty_fig("Clicca ▶ AGGIORNA per caricare i dati"),
                                  style={"height": "42vh", "width": "100%"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                    html.Div([
                        html.B("MV = PQ — Teoria Quantitativa della Moneta",
                               style={"font-size": "11px", "color": "#1a5276"}),
                        html.Span("  M·V = Moneta × Velocità  |  P·Q = CPI × PIL Reale",
                                  style={"font-size": "10px", "color": "#666",
                                         "margin-left": "10px"}),
                    ], style={"padding": "5px 16px",
                              "background": "#eaf4fb",
                              "border-top": "1px solid #aed6f1",
                              "border-bottom": "1px solid #aed6f1"}),
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="chart-mvpq",
                                  figure=empty_fig("Clicca ▶ AGGIORNA per caricare i dati"),
                                  style={"height": "43vh", "width": "100%"},
                                  config={"responsive": True, "scrollZoom": True}),
                    ]),
                ], style={"flex": "1", "min-width": "0",
                          "overflow-y": "auto",
                          "height": "calc(100vh - 120px)"}),
            ], style={"display": "flex"}),
        ]),

        dcc.Tab(label="📈  Curva dei Tassi", value="tab2", children=[
            _yields_tab_layout(),
        ]),

        dcc.Tab(label="🏭  PIL", value="tab3", children=[
            _pil_unified_tab_layout(),
        ]),

        dcc.Tab(label="📐  Regressione", value="tab5", children=[
            _regression_tab_layout(),
        ]),

        dcc.Tab(label="🛢  Shock & Banche Centrali", value="tab-shock", children=[
            _shock_tab_layout(),
        ]),

        dcc.Tab(label="〜  ARIMA / SARIMA", value="tab-arima", children=[
            _arima_tab_layout(),
        ]),

        dcc.Tab(label="📐  Modello ADL", value="tab-adl", children=[
            _adl_tab_layout(),
        ]),

        dcc.Tab(label="📐  Phillips", value="tab-phillips", children=[
            _phillips_tab_layout(),
        ]),

        dcc.Tab(label="⚙  DSGE", value="tab-dsge", children=[
            _dsge_tab_layout(),
        ]),

        dcc.Tab(label="💹  Valutazione", value="tab-valuation", children=[
            _valuation_tab_layout(),
        ]),

        dcc.Tab(label="📊  Confronto Serie", value="tab-compare", children=[
            _compare_tab_layout(),
        ]),

        dcc.Tab(label="ℹ  Guida", value="tab4", children=[
            html.Div([

                html.H3("Guida al flusso analitico", style={"border-bottom":"3px solid #1f77b4","padding-bottom":"8px","margin-bottom":"20px"}),

                # ── WORKFLOW GENERALE ─────────────────────────────────────────
                html.Div([
                    html.H4("Workflow consigliato", style={"color":"#1f77b4","margin-bottom":"10px"}),
                    html.Ol([
                        html.Li("Scarica i dati nel tab Phillips (USA o EUR, frequenza Q)"),
                        html.Li("Stima la Curva di Phillips con OLS e/o GMM → ottieni κ e β"),
                        html.Li("Invia κ e β al DSGE con il bottone verde"),
                        html.Li("Calibra σ, φπ, φy manualmente nel DSGE"),
                        html.Li("Simula shock e valuta la risposta della politica monetaria"),
                    ], style={"line-height":"2.0","font-size":"13px"}),
                ], style={"background":"#e8f0fe","padding":"14px","border-radius":"8px","margin-bottom":"24px"}),

                # ── CURVA DI PHILLIPS ─────────────────────────────────────────
                html.H4("📐 Curva di Phillips — concetti chiave", style={"color":"#d62728","margin-bottom":"8px"}),

                html.H5("Versione backward-looking (OLS)", style={"margin":"12px 0 4px"}),
                html.Pre("π_t = α + γ·π_{t-1} + κ·ỹ_t + ε_t",
                         style={"background":"#f5f5f5","padding":"8px","border-radius":"4px","font-size":"12px"}),
                html.Ul([
                    html.Li([html.B("α"), " — inflazione strutturale (≈ π* se usi Inflation Gap)"]),
                    html.Li([html.B("γ"), " — inerzia: quanto l'inflazione passata spinge quella attuale. Alto = disinflazione lenta e costosa"]),
                    html.Li([html.B("κ"), " — pendenza Phillips: quanto 1pp di output gap genera inflazione. Negli ultimi 20 anni è diventato molto piccolo (0.03–0.10) — curva 'piatta'"]),
                    html.Li([html.B("ỹ_t"), " — output gap: PIL effettivo meno PIL potenziale (HP filter o NAIRU gap)"]),
                ], style={"font-size":"12px","line-height":"1.8","margin-bottom":"12px"}),

                html.H5("Versione GMM Galí-Gertler 1999 (forward-looking ibrida)", style={"margin":"12px 0 4px"}),
                html.Pre("π_t = α + θ·π_{t-1} + γ·E_t[π_{t+1}] + λ·mc_t + ε_t",
                         style={"background":"#f5f5f5","padding":"8px","border-radius":"4px","font-size":"12px"}),
                html.Ul([
                    html.Li([html.B("θ (backward)"), " — peso dell'inflazione passata"]),
                    html.Li([html.B("γ (forward)"), " — peso delle aspettative future. Se γ > θ: economia forward-looking → la BC deve agire sulle aspettative, non solo sui dati passati"]),
                    html.Li([html.B("λ (mc_t)"), " — sensibilità ai costi marginali reali (Labor Share HP-detrended). Se piccolo: i prezzi dell'energia non si trasmettono direttamente, passano solo attraverso i salari"]),
                    html.Li([html.B("π_{t+1} è endogena"), " → uso del GMM con strumenti Z = {π_{t-1}, π_{t-2}, π_{t-3}, mc_{t-1}, mc_{t-2}}"]),
                    html.Li([html.B("θ + γ < 1"), " — condizione necessaria per stabilità. Se ≥ 1 l'inflazione è esplosiva"]),
                ], style={"font-size":"12px","line-height":"1.8","margin-bottom":"12px"}),

                html.Div([
                    html.B("Perché usare GMM invece di OLS? ", style={"color":"#d62728"}),
                    "OLS su π_{t+1} darebbe stime distorte perché π_{t+1} è correlata con l'errore di oggi (endogeneità). "
                    "Il GMM usa variabili strumentali (valori ritardati) per isolare la componente 'razionale' delle aspettative, "
                    "separandola dal rumore delle survey o dall'errore di previsione.",
                ], style={"background":"#fff9e6","padding":"10px","border-radius":"4px","font-size":"12px","margin-bottom":"8px","border-left":"4px solid #ff7f0e"}),

                html.Div([
                    html.B("Sul p-value in macroeconomia: ", style={"color":"#555"}),
                    "p < 0.10 è accettabile in economia macro. Le serie macroeconomiche sono rumorose per natura e il GMM "
                    "ha meno potere statistico dell'OLS perché usa strumenti imperfetti. "
                    "La coerenza economica (segni giusti, θ+γ<1, λ>0) conta quanto la significatività statistica.",
                ], style={"background":"#f5f5f5","padding":"10px","border-radius":"4px","font-size":"12px","margin-bottom":"20px"}),

                # ── NAIRU ──────────────────────────────────────────────────────
                html.H4("📐 NAIRU — tasso naturale di disoccupazione", style={"color":"#9467bd","margin-bottom":"8px"}),
                html.Ul([
                    html.Li([html.B("HP Filter"), " — trend di Hodrick-Prescott sulla disoccupazione. Semplice, ma soffre di end-point bias (distorto alla fine del campione) e λ arbitrario"]),
                    html.Li([html.B("Forma ridotta"), " — stima da Δπ = α + β·u: NAIRU = −α/β. Lega direttamente la disoccupazione alla variazione dell'inflazione. Rolling = varia nel tempo"]),
                    html.Li([html.B("Kalman Filter"), " — gold standard BCE/Fed. NAIRU evolve come random walk, aggiornato ogni periodo dalla sorpresa inflazionistica. Produce banda di incertezza ±1σ"]),
                ], style={"font-size":"12px","line-height":"1.8","margin-bottom":"12px"}),
                html.Div([
                    html.B("Interpretazione del gap: "),
                    "u > NAIRU → mercato del lavoro slack → pressione disinflazionistica. "
                    "u < NAIRU → mercato teso → pressione inflazionistica. "
                    "La BCE usa il Kalman perché quantifica l'incertezza sulla stima — il NAIRU non è osservabile.",
                ], style={"background":"#f3e8ff","padding":"10px","border-radius":"4px","font-size":"12px","margin-bottom":"20px","border-left":"4px solid #9467bd"}),

                # ── DSGE ──────────────────────────────────────────────────────
                html.H4("⚙ Modello DSGE — raccordo con i dati empirici", style={"color":"#2ca02c","margin-bottom":"8px"}),

                html.H5("Le tre equazioni", style={"margin":"12px 0 4px"}),
                html.Pre(
                    "IS:   x(t) = x(t-1) − (1/σ)·(i(t-1) − π(t-1) − r*) + d(t)\n"
                    "NKPC: π(t) = π* + β·(π(t-1) − π*) + κ·x(t) + u(t)\n"
                    "TR:   i(t) = r* + π* + φπ·(π(t-1)−π*) + φy·x(t-1) + v(t)",
                    style={"background":"#f5f5f5","padding":"10px","border-radius":"4px","font-size":"12px"}),

                html.H5("Mappa parametri empirici → DSGE", style={"margin":"14px 0 6px"}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Stima empirica"), html.Th("Parametro DSGE"), html.Th("Slider"), html.Th("Come ottenerlo")
                    ], style={"background":"#e8f5e9","font-size":"12px","text-align":"left"})),
                    html.Tbody([
                        html.Tr([html.Td("κ (OLS) o λ (GMM)"), html.Td("κ — pendenza Phillips"), html.Td("dsge-kappa"), html.Td("Tab Phillips → Regressione o GMM → bottone verde")]),
                        html.Tr([html.Td("γ (OLS) o θ (GMM)"), html.Td("β — persistenza inflazione"), html.Td("dsge-beta"), html.Td("Tab Phillips → Regressione o GMM → bottone verde")], style={"background":"#f9f9f9"}),
                        html.Tr([html.Td("Media tasso reale (i−π)"), html.Td("r* — tasso naturale"), html.Td("dsge-r-star"), html.Td("Media storica di FedFunds − CPI (≈ 0.5–1% post-2010)")]),
                        html.Tr([html.Td("Media inflazione"), html.Td("π* — target"), html.Td("dsge-pi-star"), html.Td("2% (mandato BCE/Fed) oppure media campione")], style={"background":"#f9f9f9"}),
                        html.Tr([html.Td("Stima IS curve"), html.Td("σ — elasticità domanda"), html.Td("dsge-sigma"), html.Td("Non ancora stimata: usa 1.5 (USA) o 1.0 (EUR)")]),
                        html.Tr([html.Td("Stima Taylor Rule"), html.Td("φπ — risposta inflazione"), html.Td("dsge-phi-pi"), html.Td("Non ancora stimata: usa 1.5 (standard Taylor 1993)")], style={"background":"#f9f9f9"}),
                        html.Tr([html.Td("Stima Taylor Rule"), html.Td("φy — risposta output"), html.Td("dsge-phi-y"), html.Td("Non ancora stimata: usa 0.5 (standard Taylor 1993)")]),
                    ], style={"font-size":"12px"}),
                ], style={"width":"100%","border-collapse":"collapse","border":"1px solid #ddd","margin-bottom":"12px"}),

                html.Div([
                    html.B("Condizione di Blanchard-Kahn: ", style={"color":"#2ca02c"}),
                    "φπ > 1 è obbligatoria per la stabilità del modello. "
                    "Con φπ < 1 la banca centrale non risponde abbastanza all'inflazione → sistema instabile (inflazione esplosiva). "
                    "Questo è il 'principio di Taylor'.",
                ], style={"background":"#e8f5e9","padding":"10px","border-radius":"4px","font-size":"12px","margin-bottom":"12px","border-left":"4px solid #2ca02c"}),

                html.H5("Calibrazione pratica — scenario inflazione energetica (tipo 2022)", style={"margin":"14px 0 6px"}),
                html.Pre(
                    "κ  = 0.07   (da Phillips OLS/GMM — curva piatta)\n"
                    "β  = 0.70   (da OLS — inflazione moderatamente persistente)\n"
                    "σ  = 1.5    (stima standard USA)\n"
                    "φπ = 1.5    (Taylor 1993)\n"
                    "φy = 0.5    (Taylor 1993)\n"
                    "π* = 2.0%   (mandato BCE/Fed)\n"
                    "r* = 0.5%   (tasso naturale post-2010)\n"
                    "Shock costi: +2.0  ρ=0.7  Periodi: 20\n"
                    "→ Atteso: tassi salgono ~100-150bp, inflazione rientra in 6-8 trimestri",
                    style={"background":"#f5f5f5","padding":"10px","border-radius":"4px","font-size":"11px","margin-bottom":"20px"}),

                html.H5("Come leggere i grafici DSGE", style={"margin":"14px 0 6px"}),
                html.Ul([
                    html.Li([html.B("IRF (Impulse Response Function)"), " — risposta di output gap, inflazione e tasso nominale allo shock. Il tasso deve salire, l'inflazione scendere in 6-12 periodi (trimestri)"]),
                    html.Li([html.B("Diagramma di fase"), " — la traiettoria (x, π) deve convergere al punto (0, π*). Se fa spirali divergenti: φπ < 1 o β troppo alto"]),
                    html.Li([html.B("Tasso nominale vs reale"), " — in risposta a shock inflazionistico il tasso reale deve salire (politica restrittiva). Se il tasso reale scende durante l'inflazione: la BC è accomodante"]),
                ], style={"font-size":"12px","line-height":"1.8","margin-bottom":"20px"}),

                # ── BREAKEVEN TIPS ────────────────────────────────────────────
                html.H4("🔮 Breakeven TIPS — aspettative di mercato", style={"color":"#ff7f0e","margin-bottom":"8px"}),
                html.Div([
                    html.Pre("Breakeven = Rendimento Treasury nominale − Rendimento TIPS (inflation-linked)",
                             style={"background":"#f5f5f5","padding":"8px","border-radius":"4px","font-size":"12px"}),
                    html.P(["Es. Treasury 5Y = 4.5%, TIPS 5Y = 2.0% → Breakeven = 2.5%: ",
                            html.B("il mercato si aspetta 2.5% di inflazione media nei prossimi 5 anni."),
                            " È la misura più 'pura' perché non dipende da opinioni ma da soldi reali investiti. "
                            "Usata come proxy di E_t[π_{t+1}] nella NKPC forward-looking."],
                           style={"font-size":"12px","line-height":"1.6","margin-top":"8px"}),
                ], style={"background":"#fff3e0","padding":"12px","border-radius":"6px","border-left":"4px solid #ff7f0e","margin-bottom":"20px"}),

            ], style={"max-width": "900px", "margin": "24px auto",
                      "font-size": "13px", "line-height": "1.7",
                      "padding": "0 24px 40px"}),
        ]),
    ]),
], style={"font-family": "system-ui, -apple-system, sans-serif"})


# =============================================================================
# CALLBACKS — Tab 1: Analisi Monetaria
# =============================================================================

@app.callback(
    Output("upload-label",       "children"),
    Output("store-upload-meta",  "data"),
    Input("upload-file",         "contents"),
    State("upload-file",         "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if not contents:
        return "", None
    series_dict, msg = parse_excel_upload(contents, filename)
    if series_dict is None:
        return f"❌ {msg}", None
    meta = {sid: list(v) for sid, v in series_dict.items()}
    return f"📄 {filename} ({len(meta)} serie)", meta


@app.callback(
    Output("mon-geo-wrapper", "style"),
    Input("mon-source-type",  "value"),
)
def mon_toggle_geo(source_type):
    base = {"margin-left": "6px"}
    return base if source_type in ("eur", "both") else {**base, "display": "none"}


@app.callback(
    Output("mvpq-series-wrapper", "style"),
    Input("mon-source-type", "value"),
)
def mon_toggle_series_check(source_type):
    base = {"margin-right": "14px"}
    return base if source_type == "both" else {**base, "display": "none"}


# ── Clientside: segnala inizio caricamento al click del bottone ──────────────
app.clientside_callback(
    """
    function(n, source_type, geo) {
        if (!n) return window.dash_clientside.no_update;
        var src;
        if (source_type === "eur") {
            src = "Eurostat \u2014 " + (geo || "EA20");
        } else if (source_type === "both") {
            src = "FRED USA + Eurostat \u2014 " + (geo || "EA20");
        } else {
            src = "FRED \u2014 USA";
        }
        return {"active": true, "src": src, "source_type": source_type || "usa"};
    }
    """,
    Output("store-mon-loading-state", "data"),
    Input("btn-aggiorna",    "n_clicks"),
    State("mon-source-type", "value"),
    State("mon-eur-geo",     "value"),
    prevent_initial_call=True,
)


@app.callback(
    Output("mon-loading-overlay", "style"),
    Output("mon-loading-title",   "children"),
    Output("mon-loading-source",  "children"),
    Output("mon-progress-bar",    "style"),
    Output("mon-progress-pct",    "style"),
    Output("mon-progress-tick",   "disabled"),
    Output("mon-progress-tick",   "n_intervals"),
    Input("store-mon-loading-state", "data"),
    prevent_initial_call=True,
)
def mon_toggle_overlay(state):
    _hidden = {"display": "none"}
    _bar_red  = {"width": "0%", "height": "100%",
                 "background": "linear-gradient(90deg,#7b0000,#e53935)",
                 "border-radius": "6px", "transition": "width 0.3s ease"}
    _bar_blue = {"width": "0%", "height": "100%",
                 "background": "linear-gradient(90deg,#1a5276,#2e86c1)",
                 "border-radius": "6px", "transition": "width 0.3s ease"}
    _pct_red  = {"font-size": "28px", "font-weight": "bold",
                 "color": "#ef9a9a", "margin-bottom": "6px"}
    _pct_blue = {"font-size": "28px", "font-weight": "bold",
                 "color": "#90caf9", "margin-bottom": "6px"}
    if state and state.get("active"):
        overlay_style = {
            "display": "flex", "position": "fixed",
            "top": "0", "left": "0",
            "width": "100%", "height": "100%",
            "background": "rgba(0,0,0,0.75)",
            "z-index": "9999",
            "align-items": "center", "justify-content": "center",
        }
        src_type = state.get("source_type")
        if src_type == "eur":
            bar_style, pct_style = _bar_blue, _pct_blue
        elif src_type == "both":
            _bar_both = {"width": "0%", "height": "100%",
                         "background": "linear-gradient(90deg,#1a5276,#6a1b9a)",
                         "border-radius": "6px", "transition": "width 0.3s ease"}
            _pct_both = {"font-size": "28px", "font-weight": "bold",
                         "color": "#ce93d8", "margin-bottom": "6px"}
            bar_style, pct_style = _bar_both, _pct_both
        else:
            bar_style, pct_style = _bar_red, _pct_red
        return overlay_style, "Caricamento dati in corso...", state.get("src", ""), bar_style, pct_style, False, 0
    return _hidden, "", "", _bar_red, _pct_red, True, 0


@app.callback(
    Output("mon-progress-pct",    "children"),
    Output("mon-progress-bar",    "style", allow_duplicate=True),
    Output("mon-progress-detail", "children"),
    Input("mon-progress-tick",    "n_intervals"),
    State("store-mon-loading-state", "data"),
    prevent_initial_call=True,
)
def mon_tick_progress(n, state):
    src_type = state.get("source_type") if state else "usa"
    if src_type == "eur":
        grad = "linear-gradient(90deg,#1a5276,#2e86c1)"
    elif src_type == "both":
        grad = "linear-gradient(90deg,#1a5276,#6a1b9a)"
    else:
        grad = "linear-gradient(90deg,#7b0000,#e53935)"
    pct = int(95 * (1 - math.exp(-n * 0.09)))
    pct = min(pct, 93)
    if pct < 20:
        detail = "Connessione ai server..."
    elif pct < 45:
        detail = "Download serie in corso..."
    elif pct < 70:
        detail = "Elaborazione dati temporali..."
    else:
        detail = "Quasi pronto..."
    bar_style = {"width": f"{pct}%", "height": "100%",
                 "background": grad,
                 "border-radius": "6px", "transition": "width 0.3s ease"}
    return f"{pct}%", bar_style, detail


@app.callback(
    Output("store-data",             "data"),
    Output("status-msg",             "children"),
    Output("date-slider",            "min"),
    Output("date-slider",            "max"),
    Output("date-slider",            "value"),
    Output("date-slider",            "marks"),
    Output("store-mon-source-type",  "data"),
    Output("store-mon-loading-state","data", allow_duplicate=True),
    Input("btn-aggiorna",            "n_clicks"),
    State("api-key",                 "value"),
    State("store-upload-meta",       "data"),
    State("mon-source-type",         "value"),
    State("mon-eur-geo",             "value"),
    prevent_initial_call="initial_duplicate",
)
def aggiorna(n_clicks, api_key, upload_meta, source_type, eur_geo):
    _done = {"active": False}
    api_key = (api_key or FRED_API_KEY).strip()

    if source_type == "both":
        geo = eur_geo or "EA20"
        print(f"\n▶ Download monetario confronto USA + EUR [{geo}]...")
        df_usa = build_dataframe(DEFAULT_SERIES, api_key)
        df_eur = build_monetary_eur_df(geo, api_key)
        if df_usa.empty and df_eur.empty:
            return None, "❌ Nessun dato — controlla la chiave API o la connessione", 0, 1, [0, 1], {}, source_type, _done
        df_usa = df_usa.rename(columns={c: f"{c} 🇺🇸" for c in df_usa.columns})
        df_eur = df_eur.rename(columns={c: f"{c} 🇪🇺" for c in df_eur.columns})
        df = pd.concat([df_usa, df_eur], axis=1).sort_index()
        geo_lbl = EUROSTAT_GEO.get(geo, geo)
        source_lbl = f"Confronto USA vs {geo_lbl}"
    elif source_type == "eur":
        geo = eur_geo or "EA20"
        print(f"\n▶ Download monetario EUR [{geo}]...")
        df = build_monetary_eur_df(geo, api_key)
        source_lbl = f"Area Euro / {EUROSTAT_GEO.get(geo, geo)}"
    elif upload_meta:
        series_dict = {sid: tuple(v) for sid, v in upload_meta.items()}
        source_lbl  = f"File xlsx ({len(series_dict)} serie)"
        print(f"\n▶ Download monetario — {source_lbl}")
        df = build_dataframe(series_dict, api_key)
        source_type = "usa"
    else:
        series_dict = DEFAULT_SERIES
        source_lbl  = "USA (FRED)"
        print(f"\n▶ Download monetario — {source_lbl}")
        df = build_dataframe(series_dict, api_key)
        source_type = "usa"

    if df.empty:
        return None, "❌ Nessun dato — controlla la chiave API o la connessione", 0, 1, [0, 1], {}, source_type, _done
    d1  = df.index.min().strftime("%m/%Y")
    d2  = df.index.max().strftime("%m/%Y")
    msg = f"✅  {source_lbl}  |  {len(df.columns)} serie  |  {len(df)} obs  ({d1} → {d2})"
    return df.to_json(date_format="iso", orient="split"), msg, *_slider_params(df), source_type, _done


@app.callback(
    Output("slider-label", "children"),
    Input("date-slider",   "value"),
)
def slider_label(val):
    if not val or (val[1] - val[0]) < 86400:
        return ""
    s = pd.to_datetime(val[0], unit="s").strftime("%b %Y")
    e = pd.to_datetime(val[1], unit="s").strftime("%b %Y")
    return f"📅  {s}  →  {e}"


@app.callback(
    Output("mon-series-checklist", "children"),
    Input("store-data", "data"),
    State("store-mon-source-type", "data"),
    prevent_initial_call=False,
)
def mon_populate_checklist(data, source_type):
    if not data:
        return "— carica i dati per vedere le serie —"
    df = pd.read_json(io.StringIO(data), orient="split")
    cols = sorted(df.columns.tolist())
    rows = []
    for col in cols:
        if source_type == "both":
            dot_color = "#1565c0" if col.endswith("🇪🇺") else "#b71c1c"
        elif source_type == "eur":
            dot_color = "#1565c0"
        else:
            dot_color = "#b71c1c"
        short = col[:32] + "…" if len(col) > 32 else col
        rows.append(html.Div([
            html.Span("●", style={"color": dot_color, "font-size": "9px",
                                   "margin-right": "4px", "vertical-align": "middle"}),
            dcc.Checklist(
                id={"type": "series-check", "index": col},
                options=[{"label": f" {short}", "value": col}],
                value=[col],
                style={"font-size": "10px", "display": "inline"},
                inputStyle={"margin-right": "3px"},
            ),
        ], style={"margin-bottom": "4px", "display": "flex", "align-items": "center"}))
    return rows


@app.callback(
    Output({"type": "series-check", "index": ALL}, "value"),
    Input("sel-all",  "n_clicks"),
    Input("sel-none", "n_clicks"),
    State({"type": "series-check", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def sel_desel(a, b, ids):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered_id == "sel-none":
        return [[] for _ in ids]
    return [[i["index"]] for i in ids]


@app.callback(
    Output("chart-main", "figure"),
    Output("chart-mvpq", "figure"),
    Input("store-data",  "data"),
    Input("date-slider", "value"),
    Input({"type": "series-check", "index": ALL}, "value"),
    Input("view-mode",   "value"),
    Input("mvpq-show", "value"),
    Input("mvpq-series-show", "value"),
    State("store-mon-source-type", "data"),
    prevent_initial_call=False,
)
def update_charts(data, slider_val, checks, view_mode, mvpq_show, mvpq_series_show, source_type):
    if not data:
        f = empty_fig("Clicca  ▶ AGGIORNA  per scaricare i dati")
        return f, f
    df = pd.read_json(io.StringIO(data), orient="split")
    df.index = pd.to_datetime(df.index)
    if slider_val and (slider_val[1] - slider_val[0]) > 86400:
        start = pd.to_datetime(slider_val[0], unit="s").normalize()
        end   = pd.to_datetime(slider_val[1], unit="s").normalize()
    else:
        start = df.index.min()
        end   = df.index.max()
    selected = [v[0] for v in (checks or []) if v]
    avail    = [c for c in selected if c in df.columns]
    if source_type == "eur":
        title_suffix = " — Area Euro (Eurostat)"
    elif source_type == "both":
        title_suffix = " — USA 🇺🇸 vs Europa 🇪🇺"
    else:
        title_suffix = " — USA (FRED)"
    if not avail:
        fig1 = empty_fig("Seleziona almeno una serie nel pannello di sinistra")
    else:
        df_plot = transform_df(df[avail], view_mode, start, end)
        if df_plot.empty:
            fig1 = empty_fig("Nessun dato nel range selezionato")
        else:
            if view_mode == "abs":
                title = "Serie Monetarie — Valori Assoluti" + title_suffix
                ylabel, zero = "Valore", False
            elif view_mode == "yoy":
                title = "Serie Monetarie — Δ% Anno su Anno" + title_suffix
                ylabel, zero = "Δ% YoY", True
            else:
                title = "Serie Monetarie — Crescita % Cumulata" + title_suffix
                ylabel, zero = "Crescita % cumulata", True
            fig1 = make_line_chart(df_plot, title, ylabel, zero)
    if source_type == "both":
        fig2 = make_mvpq_both_chart(df, start, end, mvpq_show, mvpq_series_show)
    else:
        fig2 = make_mvpq_chart(df, "both", start, end, mvpq_show)
    return fig1, fig2


# =============================================================================
# CALLBACKS — Tab 2: Curva dei Tassi
# =============================================================================

@app.callback(
    Output("store-yields",        "data"),
    Output("store-yields-eur",    "data"),
    Output("yields-status",       "children"),
    Output("yields-slider",       "min"),
    Output("yields-slider",       "max"),
    Output("yields-slider",       "value"),
    Output("yields-slider",       "marks"),
    Input("btn-reload-yields",    "n_clicks"),
    State("api-key",              "value"),
    prevent_initial_call=False,
)
def load_yields(n_clicks, api_key):
    # Cache su disco (12h): il download live FRED+BCE è lento (~40s), qui è istantaneo.
    import os as _os, time as _t, pickle as _pk
    _cache_f = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'yields_cache.pkl')
    df_usa = df_eur = None
    try:
        if _os.path.exists(_cache_f) and (_t.time() - _os.path.getmtime(_cache_f) < 43200):
            with open(_cache_f, 'rb') as _f:
                df_usa, df_eur = _pk.load(_f)
            print("  ✓ Curva tassi da cache")
    except Exception:
        df_usa = df_eur = None
    if df_usa is None:
        api_key = (api_key or FRED_API_KEY).strip()
        print("\n▶ Download tassi da FRED...")
        df_usa = build_daily_dataframe(YIELD_SERIES, api_key)
        print("\n▶ Download curva rendimenti dalla BCE...")
        df_eur = bce_get_yields_df()
        try:
            with open(_cache_f, 'wb') as _f:
                _pk.dump((df_usa, df_eur), _f)
        except Exception:
            pass

    if df_usa.empty and df_eur.empty:
        return None, None, "❌ Download fallito — verifica connessione", 0, 1, [0, 1], {}

    # slider basato su USA se disponibile, altrimenti EUR
    df_ref = df_usa if not df_usa.empty else df_eur
    mn, mx, val, marks = _slider_params_daily(df_ref)

    parts = []
    if not df_usa.empty:
        parts.append(f"USA: {len(df_usa.columns)} serie")
    if not df_eur.empty:
        parts.append(f"EUR: {len(df_eur.columns)} serie")
    d1 = df_ref.index.min().strftime("%d/%m/%Y")
    d2 = df_ref.index.max().strftime("%d/%m/%Y")
    msg = f"✅  {' | '.join(parts)}  ({d1} → {d2})"

    usa_json = df_usa.to_json(date_format="iso", orient="split") if not df_usa.empty else None
    eur_json = df_eur.to_json(date_format="iso", orient="split") if not df_eur.empty else None
    return usa_json, eur_json, msg, mn, mx, val, marks


@app.callback(
    Output("yields-slider-label", "children"),
    Input("yields-slider",        "value"),
)
def yields_slider_label(val):
    if not val or (val[1] - val[0]) < 86400:
        return ""
    s = pd.to_datetime(val[0], unit="s").strftime("%b %Y")
    e = pd.to_datetime(val[1], unit="s").strftime("%b %Y")
    return f"📅  {s}  →  {e}"


@app.callback(
    Output("yields-checklist-container", "children"),
    Input("store-yields",     "data"),
    Input("store-yields-eur", "data"),
    Input("yields-source",    "value"),
)
def yields_populate_checklist(usa_data, eur_data, source):
    labels = []
    if source in ("usa", "both") and usa_data:
        df = pd.read_json(io.StringIO(usa_data), orient="split")
        labels += df.columns.tolist()
    if source in ("eur", "both") and eur_data:
        df = pd.read_json(io.StringIO(eur_data), orient="split")
        labels += df.columns.tolist()
    if not labels:
        return "— carica i dati —"
    rows = []
    for lbl in labels:
        color = "#1565c0" if "🇪🇺" in lbl else "#b71c1c"
        rows.append(html.Div([
            html.Span("●", style={"color": color, "font-size": "9px",
                                   "margin-right": "4px", "vertical-align": "middle"}),
            dcc.Checklist(
                id={"type": "yield-check", "index": lbl},
                options=[{"label": f" {lbl}", "value": lbl}],
                value=[lbl],
                style={"font-size": "10px", "display": "inline"},
                inputStyle={"margin-right": "3px"},
            ),
        ], style={"margin-bottom": "4px", "display": "flex", "align-items": "center"}))
    return rows


@app.callback(
    Output({"type": "yield-check", "index": ALL}, "value"),
    Input("yield-sel-all",  "n_clicks"),
    Input("yield-sel-none", "n_clicks"),
    State({"type": "yield-check", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def yield_sel_desel(a, b, ids):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered_id == "yield-sel-none":
        return [[] for _ in ids]
    return [[i["index"]] for i in ids]


@app.callback(
    Output("chart-yields-history",     "figure"),
    Output("chart-yields-curve",       "figure"),
    Output("wrap-yields-history",      "style"),
    Output("wrap-yields-curve",        "style"),
    Input("store-yields",                          "data"),
    Input("store-yields-eur",                      "data"),
    Input("yields-slider",                         "value"),
    Input({"type": "yield-check", "index": ALL},   "value"),
    Input("yields-view",                           "value"),
    Input("yields-hist-mode",                      "value"),
    Input("yields-source",                         "value"),
    prevent_initial_call=False,
)
def update_yields(usa_data, eur_data, slider_val, checks, view, hist_mode, source):
    # ── costruisce df combinato in base alla fonte selezionata ────────────────
    frames = {}
    if source in ("usa", "both") and usa_data:
        df_usa = pd.read_json(io.StringIO(usa_data), orient="split")
        df_usa.index = pd.to_datetime(df_usa.index)
        for c in df_usa.columns:
            frames[c] = df_usa[c]
    if source in ("eur", "both") and eur_data:
        df_eur = pd.read_json(io.StringIO(eur_data), orient="split")
        df_eur.index = pd.to_datetime(df_eur.index)
        for c in df_eur.columns:
            frames[c] = df_eur[c]

    if not frames:
        f = empty_fig("Carica i tassi con 🔄 Ricarica tassi")
        return f, f, {"display": "block"}, {"display": "block"}

    df = pd.DataFrame(frames).sort_index()

    if slider_val and (slider_val[1] - slider_val[0]) > 86400:
        start = pd.to_datetime(slider_val[0], unit="s").normalize()
        end   = pd.to_datetime(slider_val[1], unit="s").normalize()
    else:
        start = df.index.min()
        end   = df.index.max()

    selected = [v[0] for v in (checks or []) if v]
    avail    = [c for c in selected if c in df.columns]

    show_hist  = view in ("history", "both")
    show_curve = view in ("curve",   "both")

    # ── Grafico storico ───────────────────────────────────────────────────────
    if not show_hist:
        fig_hist = empty_fig("Vista storico disattivata")
    elif not avail:
        fig_hist = empty_fig("Seleziona almeno un tasso")
    else:
        df_slice = df[avail].loc[start:end]
        if hist_mode == "yoy":
            plot_df = pd.DataFrame({
                col: ((df[col] - df[col].shift(252)) / df[col].shift(252).abs() * 100).loc[start:end]
                for col in avail
            })
            fig_hist = make_line_chart(plot_df, "Tassi — Δ% YoY", "Δ% YoY", True)
        else:
            fig_hist = make_line_chart(df_slice, "Tassi — Valori Assoluti (%)", "Tasso (%)", False)

    # ── Snapshot curva dei rendimenti ─────────────────────────────────────────
    if not show_curve:
        fig_curve = empty_fig("Vista curva disattivata")
    else:
        fig_curve = go.Figure()
        blues  = ["#c6dbef", "#6baed6", "#2171b5", "#084594", "#08306b"]
        greens = ["#c7e9c0", "#74c476", "#238b45", "#006d2c", "#00441b"]

        def _shade(dark, light, t):
            # interpola dark→light (t in [0,1]); ritorna "rgb(r,g,b)"
            def _rgb(h):
                h = h.lstrip('#'); return [int(h[i:i+2], 16) for i in (0, 2, 4)]
            d, l = _rgb(dark), _rgb(light)
            return "rgb(%d,%d,%d)" % tuple(int(d[i] + (l[i] - d[i]) * t) for i in range(3))

        def _add_curve_snapshots(df_sub, mat_x, dark, light, region_label):
            # Snapshot a lag MENSILE fisso: oggi + ogni mese indietro fino a 12 mesi.
            # Ogni curva è una trace: si mostra/nasconde dalla legenda (click/doppio-click).
            df_c = df_sub.loc[start:end].ffill().dropna(how="all")
            if df_c.empty:
                return
            idx = df_c.dropna().index
            if len(idx) == 0:
                return
            last = idx[-1]
            cols = df_sub.columns.tolist()
            # 1° anno: mensile (0…-12M). 2° anno: bimestrale (-14M…-24M).
            months = list(range(0, 13)) + [14, 16, 18, 20, 22, 24]
            for k in months:
                target = last - pd.DateOffset(months=k)
                avail  = idx[idx <= target]
                if len(avail) == 0:
                    continue
                ts  = avail[-1]      # data disponibile più vicina (≤ target)
                row = df_c.loc[ts].dropna()
                x_use = [mat_x[j] for j, c in enumerate(cols) if c in row.index and pd.notna(row[c])]
                y_use = [row[c]   for c in cols if c in row.index and pd.notna(row[c])]
                if not y_use:
                    continue
                label = "oggi" if k == 0 else f"-{k}M"
                nm = f"{region_label} {label} · {ts.strftime('%d/%m/%y')}"
                fig_curve.add_trace(go.Scatter(
                    x=x_use, y=y_use, mode="lines+markers",
                    name=nm,
                    line=dict(color=_shade(dark, light, k / 24.0),
                              width=2.6 if k == 0 else 1.3),
                    hovertemplate=f"<b>{nm}</b><br>Scadenza %{{x}}Y · %{{y:.2f}}%<extra></extra>",
                ))

        if source in ("usa", "both") and usa_data:
            df_usa_c = pd.read_json(io.StringIO(usa_data), orient="split")
            df_usa_c.index = pd.to_datetime(df_usa_c.index)
            tsy_cols = [c for c in YIELD_LABELS if c in df_usa_c.columns]   # incl. Fed Funds, 3M, 6M
            tsy_mats = [YIELD_MATURITIES[YIELD_LABELS.index(c)] for c in tsy_cols]
            _add_curve_snapshots(df_usa_c[tsy_cols], tsy_mats, "#08306b", "#9ecae1", "🇺🇸")

        if source in ("eur", "both") and eur_data:
            df_eur_c = pd.read_json(io.StringIO(eur_data), orient="split")
            df_eur_c.index = pd.to_datetime(df_eur_c.index)
            eur_cols = [c for c in BCE_YIELD_LABELS if c in df_eur_c.columns]
            eur_mats = [BCE_YIELD_SERIES[k][1] for k in BCE_YIELD_SERIES if BCE_YIELD_SERIES[k][0] in eur_cols]
            _add_curve_snapshots(df_eur_c[eur_cols], eur_mats, "#00441b", "#a1d99b", "🇪🇺")

        if not fig_curve.data:
            fig_curve = empty_fig("Nessun dato nel range per la curva")
        else:
            source_lbl = {"usa": "USA 🇺🇸", "eur": "Europa 🇪🇺", "both": "USA 🇺🇸 vs Europa 🇪🇺"}.get(source, "")
            fig_curve.update_layout(
                title=dict(text=f"Yield Curve — {source_lbl}  |  snapshot nel periodo selezionato",
                           font=dict(size=12, color="#1a3a5c"), x=0.01),
                xaxis_title="Scadenza (anni)", yaxis_title="Tasso (%)",
                hovermode="closest",
                legend=dict(orientation="v", yanchor="middle", y=0.5,
                            xanchor="left", x=1.02, font=dict(size=11),
                            bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc", borderwidth=1),
                updatemenus=[dict(
                    type="buttons", direction="right", showactive=False,
                    x=1.0, xanchor="right", y=1.16, yanchor="top",
                    pad=dict(r=0, t=0), font=dict(size=10),
                    bgcolor="#ffffff", bordercolor="#ccc",
                    buttons=[
                        dict(label="✓ Tutte",   method="restyle", args=[{"visible": True}]),
                        dict(label="✕ Nessuna", method="restyle", args=[{"visible": "legendonly"}]),
                    ],
                )],
                margin=dict(t=72, b=40, l=60, r=220),
                paper_bgcolor="white", plot_bgcolor="#f8f8f8",
            )
            fig_curve.update_xaxes(showgrid=True, gridcolor="#e8e8e8",
                                   tickvals=[0,0.25,0.5,1,2,3,5,7,10,15,20,25,30],
                                   ticktext=["FF","3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","15Y","20Y","25Y","30Y"])
            fig_curve.update_yaxes(showgrid=True, gridcolor="#e8e8e8")

    show = {"display": "block"}
    hide = {"display": "none"}
    if view == "history":  w_hist, w_curve = show, hide
    elif view == "curve":  w_hist, w_curve = hide, show
    else:                  w_hist, w_curve = show, show
    return fig_hist, fig_curve, w_hist, w_curve


# =============================================================================
# CALLBACKS — Tab PIL unificato (USA + Europa)
# =============================================================================

@app.callback(
    Output("pil-geo-wrapper", "style"),
    Input("pil-source", "value"),
)
def pil_toggle_geo(source):
    base = {"margin-left": "8px"}
    return base if source in ("eur", "both") else {**base, "display": "none"}


@app.callback(
    Output("store-gdp",       "data"),
    Output("store-eurostat",  "data"),
    Output("pil-status",      "children"),
    Output("pil-slider",      "min"),
    Output("pil-slider",      "max"),
    Output("pil-slider",      "value"),
    Output("pil-slider",      "marks"),
    Input("btn-reload-pil",   "n_clicks"),
    State("api-key",          "value"),
    State("pil-source",       "value"),
    State("pil-geo",          "value"),
    prevent_initial_call=False,
)
def load_pil(n_clicks, api_key, source, geo):
    api_key  = (api_key or FRED_API_KEY).strip()
    geo      = geo or "EA20"
    usa_json = no_update
    eur_json = no_update
    parts    = []

    if source in ("usa", "both"):
        print("\n▶ Download PIL USA da FRED...")
        df_usa = build_dataframe(GDP_SERIES, api_key)
        if not df_usa.empty:
            exp_c = next((c for c in df_usa.columns if "Esportazioni" in c), None)
            imp_c = next((c for c in df_usa.columns if "Importazioni" in c), None)
            if exp_c and imp_c:
                df_usa[NET_EXP_LABEL] = df_usa[exp_c] - df_usa[imp_c]
            usa_json = df_usa.to_json(date_format="iso", orient="split")
            parts.append(f"USA: {len(df_usa.columns)} serie")

    if source in ("eur", "both"):
        print(f"\n▶ Download Eurostat [{geo}]...")
        df_eur = build_eurostat_dataframe(geo)
        if not df_eur.empty:
            exp_n = next((c for c in df_eur.columns if "Esportazioni EUR Nom." in c), None)
            imp_n = next((c for c in df_eur.columns if "Importazioni EUR Nom." in c), None)
            if exp_n and imp_n:
                df_eur[NET_EXP_EUR_LABEL] = df_eur[exp_n] - df_eur[imp_n]
            exp_r = next((c for c in df_eur.columns if "Esportazioni EUR Reali" in c), None)
            imp_r = next((c for c in df_eur.columns if "Importazioni EUR Reali" in c), None)
            if exp_r and imp_r:
                df_eur[NET_EXP_EUR_R_LABEL] = df_eur[exp_r] - df_eur[imp_r]
            eur_json = df_eur.to_json(date_format="iso", orient="split")
            geo_label = EUROSTAT_GEO.get(geo, geo)
            parts.append(f"EUR ({geo_label}): {len(df_eur.columns)} serie")

    if not parts:
        return no_update, no_update, "❌ Nessun dato scaricato", 0, 1, [0, 1], {}

    # slider: usa il df più ricco per l'intervallo
    ref_json = usa_json if usa_json is not no_update else eur_json
    df_ref   = pd.read_json(io.StringIO(ref_json), orient="split")
    df_ref.index = pd.to_datetime(df_ref.index)
    mn, mx, val, marks = _slider_params(df_ref)
    msg = f"✅  {' | '.join(parts)}"
    return usa_json, eur_json, msg, mn, mx, val, marks


@app.callback(
    Output("pil-slider-label", "children"),
    Input("pil-slider",        "value"),
)
def pil_slider_label(val):
    if not val or (val[1] - val[0]) < 86400:
        return ""
    s = pd.to_datetime(val[0], unit="s").strftime("%b %Y")
    e = pd.to_datetime(val[1], unit="s").strftime("%b %Y")
    return f"📅  {s}  →  {e}"


@app.callback(
    Output("pil-checklist-container", "children"),
    Output("store-pil-groups",        "data"),
    Input("store-gdp",                "data"),
    Input("store-eurostat",           "data"),
    Input("store-data",               "data"),
    Input("store-mon-source-type",    "data"),
    Input("pil-source",               "value"),
)
def pil_populate_checklist(usa_gdp, eur_gdp, mon_data, mon_source, pil_source):

    def _header(label, color, bg, key):
        return html.Div([
            html.Div(label, style={"font-size": "10px", "font-weight": "bold",
                                   "color": color, "background": bg,
                                   "padding": "2px 6px", "border-radius": "3px", "flex": "1"}),
            html.Button("✔", id={"type": "pil-grp-all",  "index": key}, n_clicks=0,
                        title="Seleziona gruppo",
                        style={"font-size": "9px", "padding": "1px 5px",
                               "margin-left": "4px", "cursor": "pointer"}),
            html.Button("✘", id={"type": "pil-grp-none", "index": key}, n_clicks=0,
                        title="Deseleziona gruppo",
                        style={"font-size": "9px", "padding": "1px 5px",
                               "margin-left": "2px", "cursor": "pointer"}),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "5px"})

    def _check_rows(cols, dot_color):
        return [html.Div([
            html.Span("●", style={"color": dot_color, "font-size": "9px",
                                   "margin-right": "4px", "vertical-align": "middle"}),
            dcc.Checklist(
                id={"type": "pil-check", "index": col},
                options=[{"label": f" {col}", "value": col}],
                value=[col],
                style={"font-size": "10px", "display": "inline"},
                inputStyle={"margin-right": "3px"},
            ),
        ], style={"margin-bottom": "3px", "display": "flex", "align-items": "center"})
        for col in cols]

    groups = {}  # group_key -> [col, ...]
    rows   = []

    # ── PIL USA ──────────────────────────────────────────────────────────────
    if pil_source in ("usa", "both") and usa_gdp:
        df = pd.read_json(io.StringIO(usa_gdp), orient="split")
        cols = df.columns.tolist()
        groups["gdp-usa"] = cols
        rows.append(_header("📦 PIL USA 🇺🇸", "#b71c1c", "#ffebee", "gdp-usa"))
        rows += _check_rows(cols, "#b71c1c")
        rows.append(html.Hr(style={"margin": "6px 0"}))

    # ── Serie Monetarie USA ──────────────────────────────────────────────────
    if pil_source in ("usa", "both") and mon_data:
        df_m = pd.read_json(io.StringIO(mon_data), orient="split")
        mon_source = mon_source or "usa"
        if mon_source == "both":
            cols = [c for c in df_m.columns if c.endswith("🇺🇸")]
        elif mon_source == "usa":
            cols = df_m.columns.tolist()
        else:
            cols = []
        if cols:
            groups["mon-usa"] = cols
            rows.append(_header("💰 Monetarie USA 🇺🇸", "#e65100", "#fff3e0", "mon-usa"))
            rows += _check_rows(cols, "#e65100")
            rows.append(html.Hr(style={"margin": "6px 0"}))

    # ── PIL Europa ───────────────────────────────────────────────────────────
    if pil_source in ("eur", "both") and eur_gdp:
        df = pd.read_json(io.StringIO(eur_gdp), orient="split")
        cols = df.columns.tolist()
        groups["gdp-eur"] = cols
        rows.append(_header("📦 PIL Europa 🇪🇺", "#1565c0", "#e3f2fd", "gdp-eur"))
        rows += _check_rows(cols, "#1565c0")
        rows.append(html.Hr(style={"margin": "6px 0"}))

    # ── Serie Monetarie EUR ──────────────────────────────────────────────────
    if pil_source in ("eur", "both") and mon_data:
        df_m = pd.read_json(io.StringIO(mon_data), orient="split")
        mon_source = mon_source or "usa"
        if mon_source == "both":
            cols = [c for c in df_m.columns if c.endswith("🇪🇺")]
        elif mon_source == "eur":
            cols = df_m.columns.tolist()
        else:
            cols = []
        if cols:
            groups["mon-eur"] = cols
            rows.append(_header("💰 Monetarie EUR 🇪🇺", "#2e7d32", "#e8f5e9", "mon-eur"))
            rows += _check_rows(cols, "#2e7d32")

    if not rows:
        return "— carica i dati —", {}

    return rows, groups


@app.callback(
    Output({"type": "pil-check", "index": ALL}, "value"),
    Input({"type": "pil-grp-all",  "index": ALL}, "n_clicks"),
    Input({"type": "pil-grp-none", "index": ALL}, "n_clicks"),
    Input("pil-sel-all",  "n_clicks"),
    Input("pil-sel-none", "n_clicks"),
    State({"type": "pil-grp-all",  "index": ALL}, "id"),
    State({"type": "pil-grp-none", "index": ALL}, "id"),
    State({"type": "pil-check",    "index": ALL}, "id"),
    State({"type": "pil-check",    "index": ALL}, "value"),
    State("store-pil-groups", "data"),
    prevent_initial_call=True,
)
def pil_sel_desel(grp_all, grp_none, sel_all, sel_none,
                  grp_all_ids, grp_none_ids, check_ids, check_vals, groups):
    ctx = callback_context
    if not ctx.triggered or not check_ids:
        raise PreventUpdate

    groups    = groups or {}
    tid       = ctx.triggered_id  # Dash 3: stringa o dict

    # Globale ✔ tutto
    if tid == "pil-sel-all":
        return [[i["index"]] for i in check_ids]

    # Globale ✘ niente
    if tid == "pil-sel-none":
        return [[] for _ in check_ids]

    # Bottone di gruppo (tid è un dict {"type": ..., "index": ...})
    if not isinstance(tid, dict):
        raise PreventUpdate

    key    = tid.get("index")
    action = tid.get("type")   # "pil-grp-all" o "pil-grp-none"
    group_cols = set(groups.get(key, []))

    result = []
    for id_dict, cur_val in zip(check_ids, check_vals or []):
        col = id_dict["index"]
        if col in group_cols:
            result.append([col] if action == "pil-grp-all" else [])
        else:
            result.append(cur_val if cur_val is not None else [])
    return result


@app.callback(
    Output("chart-pil-abs",  "figure"),
    Output("chart-pil-yoy",  "figure"),
    Output("chart-pil-cum",  "figure"),
    Output("wrap-pil-abs",   "style"),
    Output("wrap-pil-yoy",   "style"),
    Output("wrap-pil-cum",   "style"),
    Input("store-gdp",            "data"),
    Input("store-eurostat",       "data"),
    Input("store-data",           "data"),
    Input("store-mon-source-type","data"),
    Input("pil-slider",           "value"),
    Input({"type": "pil-check", "index": ALL}, "value"),
    Input("pil-view",             "value"),
    Input("pil-source",           "value"),
    prevent_initial_call=False,
)
def update_pil(usa_data, eur_data, mon_data, mon_source,
               slider_val, checks, view, source):
    show = {"display": "block"}
    hide = {"display": "none"}
    view = view or []

    w_abs = show if "abs" in view else hide
    w_yoy = show if "yoy" in view else hide
    w_cum = show if "cum" in view else hide

    # ── Costruisce df combinato ───────────────────────────────────────────────
    frames = {}

    if source in ("usa", "both") and usa_data:
        df_usa = pd.read_json(io.StringIO(usa_data), orient="split")
        df_usa.index = pd.to_datetime(df_usa.index)
        for c in df_usa.columns:
            frames[c] = df_usa[c]

    if source in ("eur", "both") and eur_data:
        df_eur = pd.read_json(io.StringIO(eur_data), orient="split")
        df_eur.index = pd.to_datetime(df_eur.index)
        for c in df_eur.columns:
            frames[c] = df_eur[c]

    if mon_data:
        df_m = pd.read_json(io.StringIO(mon_data), orient="split")
        df_m.index = pd.to_datetime(df_m.index)
        mon_source = mon_source or "usa"
        if mon_source == "both":
            if source in ("usa", "both"):
                for c in [x for x in df_m.columns if x.endswith("🇺🇸")]:
                    frames[c] = df_m[c]
            if source in ("eur", "both"):
                for c in [x for x in df_m.columns if x.endswith("🇪🇺")]:
                    frames[c] = df_m[c]
        elif mon_source == "usa" and source in ("usa", "both"):
            for c in df_m.columns:
                frames[c] = df_m[c]
        elif mon_source == "eur" and source in ("eur", "both"):
            for c in df_m.columns:
                frames[c] = df_m[c]

    if not frames:
        f = empty_fig("Carica i dati con 🔄 Carica PIL")
        return f, f, f, w_abs, w_yoy, w_cum

    df = pd.DataFrame(frames).sort_index()
    df.index = pd.to_datetime(df.index)

    if slider_val and (slider_val[1] - slider_val[0]) > 86400:
        start = pd.to_datetime(slider_val[0], unit="s").normalize()
        end   = pd.to_datetime(slider_val[1], unit="s").normalize()
    else:
        start, end = df.index.min(), df.index.max()

    selected = [v[0] for v in (checks or []) if v]
    avail    = [c for c in selected if c in df.columns]

    if not avail:
        f = empty_fig("Seleziona almeno una serie")
        return f, f, f, w_abs, w_yoy, w_cum

    src_lbl = {"usa": "USA 🇺🇸", "eur": "Europa 🇪🇺", "both": "USA 🇺🇸 vs Europa 🇪🇺"}.get(source, "")

    df_sl = df[avail].loc[start:end]

    fig_abs = (
        make_line_chart(df_sl, f"PIL — Valori Assoluti  |  {src_lbl}", "Valore", False)
        if "abs" in view else empty_fig()
    )

    if "yoy" in view:
        yoy_d = {
            col: ((df[col] - df[col].shift(4)) / df[col].shift(4).abs() * 100).loc[start:end]
            for col in avail
        }
        fig_yoy = make_line_chart(
            pd.DataFrame(yoy_d).dropna(how="all"),
            f"PIL — Δ% Anno su Anno  |  {src_lbl}", "Δ% YoY", True)
    else:
        fig_yoy = empty_fig()

    if "cum" in view:
        cum_d = {
            col: ((1 + df[col].loc[start:end].dropna().pct_change().fillna(0)).cumprod() - 1) * 100
            for col in avail
            if not df[col].loc[start:end].dropna().empty
        }
        fig_cum = (
            make_line_chart(pd.DataFrame(cum_d),
                            f"PIL — Crescita % Cumulata  |  {src_lbl}", "Crescita % cum.", True)
            if cum_d else empty_fig("Dati insufficienti")
        )
    else:
        fig_cum = empty_fig()

    return fig_abs, fig_yoy, fig_cum, w_abs, w_yoy, w_cum


# =============================================================================
# CALLBACKS — Tab 5: Regressione OLS
# =============================================================================

def _all_series_df(data_mon, data_gdp, data_yields, data_shock=None):
    frames = []
    if data_mon:
        df = pd.read_json(io.StringIO(data_mon), orient="split")
        df.index = pd.to_datetime(df.index)
        frames.append(df)
    if data_gdp:
        df = pd.read_json(io.StringIO(data_gdp), orient="split")
        df.index = pd.to_datetime(df.index)
        exp_col = next((c for c in df.columns if "Esportazioni (mld" in c), None)
        imp_col = next((c for c in df.columns if "Importazioni (mld" in c), None)
        if exp_col and imp_col:
            df[NET_EXP_LABEL] = df[exp_col] - df[imp_col]
        frames.append(df)
    if data_yields:
        df = pd.read_json(io.StringIO(data_yields), orient="split")
        df.index = pd.to_datetime(df.index)
        df = df.resample("MS").last()
        frames.append(df)
    if data_shock:
        df = pd.read_json(io.StringIO(data_shock), orient="split")
        df.index = pd.to_datetime(df.index)
        df = df.resample("MS").last()
        # prendi solo le colonne utili per la regressione
        shock_cols = [c for c in df.columns if any(k in c for k in [
            "WTI", "Gas Naturale", "S&P 500", "VIX",
            "PPI", "Inflazione attesa", "EUR/USD",
        ])]
        if shock_cols:
            frames.append(df[shock_cols])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _apply_transform(s: pd.Series, transform: str) -> pd.Series:
    s = s.dropna()
    if transform == "levels": return s
    elif transform == "yoy":  return ((s - s.shift(12)) / s.shift(12).abs()) * 100
    elif transform == "log":  return np.log(s.clip(lower=1e-9))
    elif transform == "dlog": return np.log(s.clip(lower=1e-9)).diff()
    return s


def _make_stat_table(rows, header_bg="#1a3a5c"):
    return html.Table([
        html.Thead(html.Tr([
            html.Th(c, style={"background": header_bg, "color": "white",
                              "padding": "6px 10px", "font-size": "11px",
                              "text-align": "left", "white-space": "nowrap"})
            for c in rows[0]
        ])),
        html.Tbody([
            html.Tr([
                html.Td(cell, style={"padding": "5px 10px", "font-size": "11px",
                                     "border-bottom": "1px solid #eee",
                                     "font-family": "monospace",
                                     "background": "#fff" if ri % 2 == 0 else "#f8f9fa"})
                for cell in row
            ])
            for ri, row in enumerate(rows[1:])
        ])
    ], style={"border-collapse": "collapse", "width": "100%",
              "border": "1px solid #dee2e6", "border-radius": "4px"})


@app.callback(
    Output("reg-y",           "options"),
    Output("reg-x-checklist", "children"),
    Output("reg-slider",      "min"),
    Output("reg-slider",      "max"),
    Output("reg-slider",      "value"),
    Output("reg-slider",      "marks"),
    Input("store-data",       "data"),
    Input("store-gdp",        "data"),
    Input("store-yields",     "data"),
    Input("store-shock",      "data"),   # ← aggiunto
    prevent_initial_call=False,
)
def reg_populate(data_mon, data_gdp, data_yields, data_shock):
    df = _all_series_df(data_mon, data_gdp, data_yields, data_shock)
    if df.empty:
        return [], [], 0, 1, [0, 1], {}
    cols    = df.columns.tolist()
    options = [{"label": c, "value": c} for c in cols]
    x_checks = [
        html.Div(dcc.Checklist(id={"type": "reg-x-check", "index": c},
                               options=[{"label": f" {c}", "value": c}],
                               value=[], style={"font-size": "10px"},
                               inputStyle={"margin-right": "4px"}),
                 style={"margin-bottom": "3px"})
        for c in cols
    ]
    mn, mx, val, marks = _slider_params(df)
    return options, x_checks, mn, mx, val, marks


@app.callback(
    Output("reg-slider-label", "children"),
    Input("reg-slider",        "value"),
)
def reg_slider_label(val):
    if not val or (val[1] - val[0]) < 86400:
        return ""
    s = pd.to_datetime(val[0], unit="s").strftime("%b %Y")
    e = pd.to_datetime(val[1], unit="s").strftime("%b %Y")
    return f"📅  {s}  →  {e}"


@app.callback(
    Output({"type": "reg-x-check", "index": ALL}, "value"),
    Input("reg-sel-all",  "n_clicks"),
    Input("reg-sel-none", "n_clicks"),
    State({"type": "reg-x-check", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def reg_sel_x(a, b, ids):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered_id == "reg-sel-none":
        return [[] for _ in ids]
    return [[i["index"]] for i in ids]


@app.callback(
    Output("reg-equation",           "children"),
    Output("reg-stats-table",        "children"),
    Output("reg-coeff-table",        "children"),
    Output("chart-reg-rolling-coef", "figure"),
    Output("chart-reg-rolling-vif",  "figure"),
    Output("chart-reg-fit",          "figure"),
    Output("chart-reg-residuals",    "figure"),
    Output("chart-reg-qqplot",       "figure"),
    Output("reg-status",             "children"),
    Input("btn-run-reg",    "n_clicks"),
    State("store-data",     "data"),
    State("store-gdp",      "data"),
    State("store-yields",   "data"),
    State("store-shock",    "data"),   # ← aggiunto
    State("reg-y",          "value"),
    State({"type": "reg-x-check", "index": ALL}, "value"),
    State({"type": "reg-x-check", "index": ALL}, "id"),
    State("reg-transform",  "value"),
    State("reg-add-const",  "value"),
    State("reg-lag",        "value"),
    State("reg-cov-type",   "value"),
    State("reg-slider",     "value"),
    prevent_initial_call=True,
)
def run_regression(n, data_mon, data_gdp, data_yields, data_shock,
                   y_col, x_vals, x_ids,
                   transform, add_const, lag, cov_type, slider_val):
    err = lambda msg: (msg, None, None,
                       empty_fig("Nessun risultato"), empty_fig(""),
                       empty_fig(""), empty_fig(""), empty_fig(""), f"❌ {msg}")
    if not y_col:
        return err("Seleziona una variabile Y")
    x_selected = [ids["index"] for vals, ids in zip(x_vals, x_ids) if vals]
    if not x_selected:
        return err("Seleziona almeno una variabile X")
    if y_col in x_selected:
        return err("Y non può essere anche tra le X")
    df = _all_series_df(data_mon, data_gdp, data_yields, data_shock)
    if df.empty:
        return err("Nessun dato disponibile")
    if slider_val and (slider_val[1] - slider_val[0]) > 86400:
        start = pd.to_datetime(slider_val[0], unit="s").normalize()
        end   = pd.to_datetime(slider_val[1], unit="s").normalize()
    else:
        start, end = df.index.min(), df.index.max()
    df = df.loc[start:end]
    y = _apply_transform(df[y_col], transform).dropna()
    lag = int(lag or 0)
    X_dict = {}
    for col in x_selected:
        sx = _apply_transform(df[col], transform)
        if lag > 0: sx = sx.shift(lag)
        X_dict[col] = sx
    combined = pd.DataFrame({"__y__": y, **X_dict}).dropna()
    if len(combined) < len(x_selected) + 5:
        return err(f"Osservazioni insufficienti ({len(combined)})")
    y_fit = combined["__y__"]
    X_fit = combined[x_selected]
    if "const" in (add_const or []):
        X_fit = sm.add_constant(X_fit)
    cov_type = cov_type or "nonrobust"
    n_lag_nw = None
    try:
        ols_res = sm.OLS(y_fit, X_fit)
        if cov_type == "HAC":
            n_lag_nw = max(1, int(4 * (len(y_fit) / 100) ** (2/9)))
            model = ols_res.fit(cov_type="HAC", cov_kwds={"maxlags": n_lag_nw})
        elif cov_type == "HC3":
            model = ols_res.fit(cov_type="HC3")
        else:
            model = ols_res.fit()
    except Exception as e:
        return err(f"Errore stima: {e}")
    n_obs    = int(model.nobs)
    n_params = int(model.df_model + 1)
    transform_lbl = {"levels": "", "yoy": "ΔYoY", "log": "log", "dlog": "Δlog"}[transform]
    def fmt_col(c): return f"{transform_lbl}({c})" if transform_lbl else c
    terms = []
    for cn, cv in model.params.items():
        sign = "+" if cv >= 0 else "−"
        if cn == "const": terms.append(f"  α = {cv:+.4f}")
        else: terms.append(f"  {sign} {abs(cv):.4f} · {fmt_col(cn)}")
    lag_note = f"  [X lag = {lag} mesi]" if lag else ""
    eq_text = f"{fmt_col(y_col)} =\n" + "\n".join(terms) + f"  + ε{lag_note}"
    dw = float(sm.stats.stattools.durbin_watson(model.resid))
    jb_stat, jb_p, jb_skew, jb_kurt = sm.stats.stattools.jarque_bera(model.resid)
    bp_lm, bp_p, *_ = sm.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)
    cond_num = float(np.linalg.cond(X_fit.values.astype(float)))
    def pstar(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "·"
        return ""
    cov_label = {"nonrobust": "OLS classico", "HC3": "HC3 eterosch.",
                  "HAC": "HAC Newey-West"}.get(cov_type, cov_type)
    stats_rows = [
        ["Statistica", "Valore", "Note"],
        ["N osservazioni", f"{n_obs}", ""],
        ["N parametri",    f"{n_params}", "incl. costante"],
        ["Std. Error",     cov_label, ""],
        ["R²",             f"{model.rsquared:.6f}", ""],
        ["R² adj.",        f"{model.rsquared_adj:.6f}", ""],
        ["F-stat",         f"{model.fvalue:.4f}",
         f"p={model.f_pvalue:.4e} {pstar(model.f_pvalue)}"],
        ["AIC",            f"{model.aic:.4f}", ""],
        ["BIC",            f"{model.bic:.4f}", ""],
        ["Durbin-Watson",  f"{dw:.4f}", "~2 = no autocorr."],
        ["Jarque-Bera",    f"{jb_stat:.4f}",
         f"p={jb_p:.4e} {pstar(jb_p)} | sk={jb_skew:.3f} ku={jb_kurt:.3f}"],
        ["Breusch-Pagan",  f"{bp_lm:.4f}", f"p={bp_p:.4e} {pstar(bp_p)}"],
        ["Cond. number",   f"{cond_num:.2f}", ">30 → multicollinearità"],
    ]
    if n_lag_nw:
        stats_rows.append(["HAC maxlags", f"{n_lag_nw}", "Newey-West"])
    stat_table = _make_stat_table(stats_rows, "#1a3a5c")
    conf = model.conf_int(alpha=0.05)
    x_cols_vif = [c for c in X_fit.columns if c != "const"]
    vif_dict = {}
    if len(x_cols_vif) > 1:
        for xc in x_cols_vif:
            other = [c for c in x_cols_vif if c != xc]
            try:
                r2 = sm.OLS(X_fit[xc], sm.add_constant(X_fit[other])).fit().rsquared
                vif_dict[xc] = 1 / (1 - r2) if r2 < 1 else np.inf
            except: vif_dict[xc] = np.nan
    else:
        vif_dict = {c: np.nan for c in x_cols_vif}
    coef_rows = [["Variabile", "Coeff.", "Std Err", "t-stat", "p-val", "Sig.",
                   "IC95 inf", "IC95 sup", "VIF"]]
    for var in model.params.index:
        p   = model.pvalues[var]
        vif = vif_dict.get(var, np.nan)
        vif_s = f"{vif:.2f}" if isinstance(vif, float) and not np.isnan(vif) else "—"
        coef_rows.append([var, f"{model.params[var]:.6f}", f"{model.bse[var]:.6f}",
                          f"{model.tvalues[var]:.4f}", f"{p:.4e}", pstar(p),
                          f"{conf.loc[var, 0]:.6f}", f"{conf.loc[var, 1]:.6f}", vif_s])
    coef_table = html.Div([
        html.Div("Coefficienti di regressione",
                 style={"font-size": "11px", "font-weight": "bold", "color": "#1a3a5c",
                        "background": "#eaf4fb", "padding": "5px 10px",
                        "border-radius": "4px 4px 0 0", "border": "1px solid #aed6f1",
                        "border-bottom": "none", "margin-top": "12px"}),
        _make_stat_table(coef_rows, "#2e6da4"),
        html.Div("*** p<0.001  ** p<0.01  * p<0.05  · p<0.10",
                 style={"font-size": "10px", "color": "#777",
                        "margin-top": "4px", "font-style": "italic"}),
    ])
    ROLL_WIN = 24
    x_cols_only = [c for c in X_fit.columns if c != "const"]
    has_const   = "const" in X_fit.columns
    n_total     = len(combined)
    roll_dates, roll_coefs, roll_ci_lo, roll_ci_hi, roll_vif = [], {}, {}, {}, {}
    for c in x_cols_only:
        roll_coefs[c] = []; roll_ci_lo[c] = []; roll_ci_hi[c] = []; roll_vif[c] = []
    for i in range(ROLL_WIN, n_total + 1):
        w_idx = combined.index[i - ROLL_WIN : i]
        y_w   = combined.loc[w_idx, "__y__"]
        X_w   = combined.loc[w_idx, x_cols_only]
        if has_const: X_w = sm.add_constant(X_w)
        try:
            if cov_type == "HAC":
                lag_w = max(1, int(4 * (ROLL_WIN / 100) ** (2/9)))
                m_w   = sm.OLS(y_w, X_w).fit(cov_type="HAC", cov_kwds={"maxlags": lag_w})
            elif cov_type == "HC3":
                m_w = sm.OLS(y_w, X_w).fit(cov_type="HC3")
            else:
                m_w = sm.OLS(y_w, X_w).fit()
        except: continue
        roll_dates.append(w_idx[-1])
        for c in x_cols_only:
            if c in m_w.params.index:
                cv = float(m_w.params[c]); se = float(m_w.bse[c])
                roll_coefs[c].append(cv); roll_ci_lo[c].append(cv-1.96*se); roll_ci_hi[c].append(cv+1.96*se)
            else:
                roll_coefs[c].append(np.nan); roll_ci_lo[c].append(np.nan); roll_ci_hi[c].append(np.nan)
        x_roll_cols = [c for c in X_w.columns if c != "const"]
        if len(x_roll_cols) > 1:
            for c in x_roll_cols:
                other = [o for o in x_roll_cols if o != c]
                try:
                    r2_c = sm.OLS(X_w[c], sm.add_constant(X_w[other])).fit().rsquared
                    roll_vif[c].append(1 / (1 - r2_c) if r2_c < 1 else np.inf)
                except: roll_vif[c].append(np.nan)
        else:
            for c in x_roll_cols: roll_vif[c].append(np.nan)
    if not roll_dates:
        fig_rolling_coef = empty_fig("Dati insufficienti per rolling (< 24 obs)")
        fig_rolling_vif  = empty_fig("")
    else:
        roll_dates = pd.DatetimeIndex(roll_dates)
        n_vars = len(x_cols_only)
        if n_vars == 1:
            c = x_cols_only[0]
            fig_rolling_coef = go.Figure()
            coef_arr  = np.array(roll_coefs[c], dtype=float)
            ci_lo_arr = np.array(roll_ci_lo[c], dtype=float)
            ci_hi_arr = np.array(roll_ci_hi[c], dtype=float)
            fig_rolling_coef.add_trace(go.Scatter(
                x=list(roll_dates)+list(roll_dates[::-1]),
                y=list(ci_hi_arr)+list(ci_lo_arr[::-1]),
                fill="toself", fillcolor="rgba(31,119,180,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="IC 95%", hoverinfo="skip"))
            fig_rolling_coef.add_trace(go.Scatter(x=roll_dates, y=coef_arr, name=f"β {c}",
                                                    line=dict(color=COLORS[0], width=2)))
            fig_rolling_coef.add_hline(y=0, line_color="#888", line_dash="dot", line_width=1)
            mean_c = float(np.nanmean(coef_arr))
            fig_rolling_coef.add_hline(y=mean_c, line_color=COLORS[0], line_dash="dash",
                                        line_width=1, annotation_text=f"μ={mean_c:.4f}",
                                        annotation_position="right")
            fig_rolling_coef.update_layout(
                title=dict(text=f"Coefficienti rolling — finestra {ROLL_WIN} mesi",
                           font=dict(size=11), x=0.01),
                hovermode="x unified", margin=dict(t=45, b=30, l=60, r=80),
                paper_bgcolor="white", plot_bgcolor="#f8f8f8")
        else:
            fig_rolling_coef = make_subplots(rows=n_vars, cols=1, shared_xaxes=True,
                                              vertical_spacing=0.06,
                                              subplot_titles=[f"β — {c}" for c in x_cols_only])
            for vi, c in enumerate(x_cols_only):
                row_i = vi+1; col_c = COLORS[vi % len(COLORS)]
                coef_arr  = np.array(roll_coefs[c], dtype=float)
                ci_lo_arr = np.array(roll_ci_lo[c], dtype=float)
                ci_hi_arr = np.array(roll_ci_hi[c], dtype=float)
                mean_c    = float(np.nanmean(coef_arr))
                rgba_fill = f"rgba({int(col_c[1:3],16)},{int(col_c[3:5],16)},{int(col_c[5:7],16)},0.15)"
                fig_rolling_coef.add_trace(go.Scatter(
                    x=list(roll_dates)+list(roll_dates[::-1]),
                    y=list(ci_hi_arr)+list(ci_lo_arr[::-1]),
                    fill="toself", fillcolor=rgba_fill, line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False, hoverinfo="skip"), row=row_i, col=1)
                fig_rolling_coef.add_trace(go.Scatter(x=roll_dates, y=coef_arr, name=f"β {c}",
                                                       line=dict(color=col_c, width=2)), row=row_i, col=1)
                fig_rolling_coef.add_hline(y=0, line_color="#aaa", line_dash="dot", line_width=1, row=row_i, col=1)
                fig_rolling_coef.add_hline(y=mean_c, line_color=col_c, line_dash="dash", line_width=1,
                                            annotation_text=f"μ={mean_c:.4f}", annotation_position="right",
                                            row=row_i, col=1)
            fig_rolling_coef.update_layout(
                title=dict(text=f"Coefficienti rolling — {ROLL_WIN} mesi  |  banda=IC 95%",
                           font=dict(size=11), x=0.01),
                hovermode="x unified", margin=dict(t=50, b=30, l=65, r=90),
                paper_bgcolor="white", plot_bgcolor="#f8f8f8")
        has_vif = any(any(not np.isnan(v) for v in roll_vif[c]) for c in x_cols_only)
        if not has_vif or len(x_cols_only) < 2:
            fig_rolling_vif = empty_fig("VIF rolling: serve almeno 2 variabili X")
        else:
            fig_rolling_vif = go.Figure()
            for vi, c in enumerate(x_cols_only):
                fig_rolling_vif.add_trace(go.Scatter(x=roll_dates, y=np.array(roll_vif[c],dtype=float),
                                                       name=f"VIF {c}", line=dict(color=COLORS[vi%len(COLORS)], width=1.8)))
            fig_rolling_vif.add_hline(y=5,  line_color="#ff7f0e", line_dash="dash", line_width=1, annotation_text="VIF=5",  annotation_position="right")
            fig_rolling_vif.add_hline(y=10, line_color="#d62728", line_dash="dash", line_width=1, annotation_text="VIF=10", annotation_position="right")
            fig_rolling_vif.update_layout(title=dict(text=f"VIF rolling — {ROLL_WIN} mesi", font=dict(size=11), x=0.01),
                                           hovermode="x unified", yaxis_title="VIF",
                                           margin=dict(t=45, b=30, l=60, r=90),
                                           paper_bgcolor="white", plot_bgcolor="#f8f8f8")
    fitted = model.fittedvalues
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(x=y_fit.index, y=y_fit.values, name="Osservato",
                                  line=dict(color="#1f77b4", width=1.5)))
    fig_fit.add_trace(go.Scatter(x=fitted.index, y=fitted.values, name="Stimato",
                                  line=dict(color="#d62728", width=1.5, dash="dot")))
    fig_fit.update_layout(title=dict(text=f"Osservato vs Stimato — {fmt_col(y_col)}", font=dict(size=11), x=0.01),
                           hovermode="x unified", margin=dict(t=40, b=30, l=55, r=20),
                           paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                           legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)))
    fig_fit.add_annotation(text=f"R²={model.rsquared:.4f}  |  R²adj={model.rsquared_adj:.4f}",
                            xref="paper", yref="paper", x=0.99, y=0.98, showarrow=False,
                            font=dict(size=10), bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="#ccc", borderwidth=1, xanchor="right", yanchor="top")
    resid = model.resid; std_r = resid.std()
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=resid.index, y=resid.values, name="Residui",
                                  mode="lines", line=dict(color="#2ca02c", width=1)))
    fig_res.add_hline(y=0, line_color="#555", line_dash="dot", line_width=1)
    fig_res.add_hline(y= 2*std_r, line_color="#ff7f0e", line_dash="dash", line_width=1, annotation_text="+2σ")
    fig_res.add_hline(y=-2*std_r, line_color="#ff7f0e", line_dash="dash", line_width=1, annotation_text="−2σ")
    fig_res.update_layout(title=dict(text="Residui (±2σ)", font=dict(size=11), x=0.01),
                           hovermode="x unified", margin=dict(t=40, b=30, l=55, r=20),
                           paper_bgcolor="white", plot_bgcolor="#f8f8f8")
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(resid.values)
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                 marker=dict(size=4, color="#9467bd"), name="Quantili campione"))
    x_line = np.array([min(osm), max(osm)])
    fig_qq.add_trace(go.Scatter(x=x_line, y=slope*x_line+intercept, mode="lines",
                                 line=dict(color="#d62728", width=1.5), name="Normale teorica"))
    fig_qq.update_layout(title=dict(text="Q-Q plot residui", font=dict(size=11), x=0.01),
                          xaxis_title="Quantili teorici", yaxis_title="Quantili campione",
                          margin=dict(t=40, b=35, l=55, r=20),
                          paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                          legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)))
    status_msg = (f"✅  OLS stimato — {n_obs} obs  |  R²={model.rsquared:.4f}  |  "
                  f"F={model.fvalue:.2f} (p={model.f_pvalue:.2e})  |  DW={dw:.3f}")
    return eq_text, stat_table, coef_table, fig_rolling_coef, fig_rolling_vif, fig_fit, fig_res, fig_qq, status_msg



# =============================================================================
# SHOCK EVENTS + CALLBACKS
# =============================================================================

SHOCK_EVENTS = [
    ("2020-03-01", "COVID-19 lockdown", "#d62728"),
    ("2021-11-01", "Inflazione picco USA", "#ff7f0e"),
    ("2022-02-24", "Invasione Ucraina", "#9467bd"),
    ("2022-06-01", "Fed +75bps", "#2ca02c"),
    ("2023-07-01", "OPEC+ tagli", "#8c564b"),
    ("2024-01-01", "Recessione Europa", "#e377c2"),
]



def register_shock_callbacks(app):
    """
    Registra tutti i callback del tab shock sull'istanza app.
    Chiama questa funzione PRIMA di if __name__ == '__main__':
    Esempio: register_shock_callbacks(app)
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import statsmodels.api as sm
    from scipy.optimize import minimize_scalar
    from dash import callback_context
    from dash.exceptions import PreventUpdate
    import io

    # ── D.1: Carica dati shock ──────────────────────────────────────────────
    @app.callback(
        Output("store-shock",     "data"),
        Output("shock-status",    "children"),
        Output("shock-slider",    "min"),
        Output("shock-slider",    "max"),
        Output("shock-slider",    "value"),
        Output("shock-slider",    "marks"),
        Input("btn-reload-shock", "n_clicks"),
        State("api-key",          "value"),
        prevent_initial_call=False,
    )
    def load_shock(n_clicks, api_key):
        api_key = (api_key or FRED_API_KEY).strip()

        print("\n▶ Download shock da FRED...")
        df = build_daily_dataframe(SHOCK_SERIES, api_key)

        if df is None or df.empty:
            return (None, "❌ Dati non disponibili — verifica API key",
                    0, 1, [0, 1], {})

        mn, mx, val, marks = _slider_params_daily(df)
        d1 = df.index.min().strftime("%d/%m/%Y")
        d2 = df.index.max().strftime("%d/%m/%Y")
        msg = f"✅  {len(df.columns)} serie  |  {d1} → {d2}"
        return (df.to_json(date_format="iso", orient="split"), msg,
                mn, mx, val, marks)

    # ── D.1b: Carica dati Eurostat per shock ───────────────────────────────────
    app.clientside_callback(
        """
        function(n, geo) {
            if (!n) return window.dash_clientside.no_update;
            var g = geo || "EA20";
            return {"active": true, "src": "Eurostat \u2014 " + g};
        }
        """,
        Output("store-shock-loading-state", "data"),
        Input("btn-shock-eur-load", "n_clicks"),
        State("shock-eur-geo",      "value"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("shock-loading-overlay", "style"),
        Output("shock-loading-title",   "children"),
        Output("shock-loading-source",  "children"),
        Output("shock-progress-tick",   "disabled"),
        Output("shock-progress-tick",   "n_intervals"),
        Input("store-shock-loading-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_shock_overlay(state):
        if state and state.get("active"):
            overlay_style = {
                "display": "flex", "position": "fixed",
                "top": "0", "left": "0",
                "width": "100%", "height": "100%",
                "background": "rgba(0,0,0,0.75)",
                "z-index": "9999",
                "align-items": "center", "justify-content": "center",
            }
            return overlay_style, "Caricamento dati Eurostat...", state.get("src", ""), False, 0
        return {"display": "none"}, "", "", True, 0

    @app.callback(
        Output("shock-progress-pct",    "children"),
        Output("shock-progress-bar",    "style"),
        Output("shock-progress-detail", "children"),
        Input("shock-progress-tick",    "n_intervals"),
        prevent_initial_call=True,
    )
    def tick_shock_progress(n):
        pct = min(int(95 * (1 - math.exp(-n * 0.08))), 93)
        if pct < 20:
            detail = "Connessione ai server Eurostat..."
        elif pct < 50:
            detail = "Download serie macroeconomiche..."
        elif pct < 75:
            detail = "Scaricamento HICP e tassi d'interesse..."
        else:
            detail = "Elaborazione e allineamento date..."
        bar_style = {
            "width": f"{pct}%", "height": "100%",
            "background": "linear-gradient(90deg,#7b2d00,#d4651a)",
            "border-radius": "6px", "transition": "width 0.3s ease",
        }
        return f"{pct}%", bar_style, detail

    @app.callback(
        Output("store-shock-eur",          "data"),
        Output("shock-eur-status",         "children"),
        Output("shock-eur-checklist-container", "children"),
        Output("store-shock-loading-state","data",  allow_duplicate=True),
        Input("btn-shock-eur-load",        "n_clicks"),
        State("shock-eur-geo",             "value"),
        State("api-key",                   "value"),
        prevent_initial_call=True,
    )
    def load_shock_eur(n_clicks, geo, api_key):
        _done = {"active": False}

        def _err(msg):
            no_data_msg = html.Div("— errore caricamento —",
                                   style={"font-size": "9px", "color": "#c0392b",
                                          "font-style": "italic"})
            return None, msg, no_data_msg, _done

        if not _has_internet():
            return _err("⚠️  Nessuna connessione internet")

        geo = geo or "EA20"
        print(f"\n▶ Shock — Download Eurostat [{geo}]...")
        df = build_eurostat_dataframe(geo)

        # Aggiungi Brent da FRED (comune a entrambe le analisi)
        _ak = (api_key or FRED_API_KEY).strip()
        brent = fred_get("MCOILBRENTEU", _ak)
        if brent is not None:
            idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
            df["Brent Petrolio ($/barile)"] = to_monthly(brent, "M").reindex(idx).ffill()

        # Indice azionario del paese
        eq_info = EUROSTAT_EQUITY.get(geo)
        if eq_info:
            eq_ticker, eq_name = eq_info
            eq_s = yfinance_monthly(eq_ticker, eq_name)
            if eq_s is not None:
                idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
                df[eq_name] = eq_s.reindex(idx).ffill()

        if df.empty:
            return _err("❌ Download Eurostat fallito")

        d1  = df.index.min().strftime("%m/%Y")
        d2  = df.index.max().strftime("%m/%Y")
        geo_label = EUROSTAT_GEO.get(geo, geo)
        msg = f"✅  {geo_label}  |  {len(df.columns)} serie  |  {d1} → {d2}"

        # Costruisci checklist dinamica con le colonne disponibili
        DEFAULT_ON = {
            "HICP Indice (2015=100)", "Tasso Disoccupazione EUR (%)",
            "Brent Petrolio ($/barile)", "Tasso 3M Euribor (%)",
        }
        checklist_items = [
            html.Div(
                dcc.Checklist(
                    id={"type": "shock-eur-check", "index": col},
                    options=[{"label": f" {col}", "value": col}],
                    value=[col] if col in DEFAULT_ON else [],
                    style={"font-size": "9px"},
                    inputStyle={"margin-right": "3px"},
                ),
                style={"margin-bottom": "3px"}
            )
            for col in sorted(df.columns)
        ]

        return (df.to_json(date_format="iso", orient="split"), msg,
                checklist_items, _done)

    # ── D.2: Label slider shock ─────────────────────────────────────────────
    @app.callback(
        Output("shock-slider-label", "children"),
        Input("shock-slider", "value"),
    )
    def shock_slider_lbl(val):
        if not val or (val[1] - val[0]) < 86400:
            return ""
        s = pd.to_datetime(val[0], unit="s").strftime("%b %Y")
        e = pd.to_datetime(val[1], unit="s").strftime("%b %Y")
        return f"📅  {s}  →  {e}"


    # ── D.3: Monitor shock ──────────────────────────────────────────────────
    @app.callback(
        Output("chart-shock-main", "figure"),
        Output("chart-shock-corr", "figure"),
        Input("store-shock",     "data"),
        Input("store-data",      "data"),
        Input("store-shock-eur", "data"),
        Input("shock-slider",    "value"),
        Input({"type": "shock-check",     "index": ALL}, "value"),
        Input({"type": "shock-eur-check", "index": ALL}, "value"),
        Input("shock-events-check", "value"),
        Input("shock-view",      "value"),
        prevent_initial_call=False,
    )
    def update_shock_monitor(shock_data, mon_data, eur_data, slider_val,
                             checks, eur_checks, events_sel, view):
        if not shock_data and not eur_data:
            f = empty_fig("Carica i dati con 🔄 Ricarica dati shock  o  📥 Carica Eurostat")
            return f, f

        frames_all = []

        if shock_data:
            df_s = pd.read_json(io.StringIO(shock_data), orient="split")
            df_s.index = pd.to_datetime(df_s.index)
            frames_all.append(df_s.resample("MS").last())

        if mon_data:
            df_m = pd.read_json(io.StringIO(mon_data), orient="split")
            df_m.index = pd.to_datetime(df_m.index)
            frames_all.append(df_m.resample("MS").last())

        if eur_data:
            df_e = pd.read_json(io.StringIO(eur_data), orient="split")
            df_e.index = pd.to_datetime(df_e.index)
            frames_all.append(df_e.resample("MS").last())

        if not frames_all:
            f = empty_fig("Nessun dato disponibile")
            return f, f

        df_all = pd.concat(frames_all, axis=1)
        df_all = df_all.loc[:, ~df_all.columns.duplicated()]

        # Range slider
        if slider_val and (slider_val[1] - slider_val[0]) > 86400:
            start = pd.to_datetime(slider_val[0], unit="s").normalize()
            end   = pd.to_datetime(slider_val[1], unit="s").normalize()
        else:
            start = df_all.index.min()
            end   = df_all.index.max()

        selected_fred = [v[0] for v in (checks or []) if v]
        selected_eur  = [v[0] for v in (eur_checks or []) if v]
        selected = selected_fred + selected_eur
        avail    = [c for c in selected if c in df_all.columns]

        if not avail:
            f = empty_fig("Seleziona almeno una variabile")
            return f, f

        # Trasformazione
        plot_frames = {}
        for col in avail:
            s_full = df_all[col].dropna()
            if view == "abs":
                plot_frames[col] = s_full.loc[start:end]
            elif view == "yoy":
                yoy_s = ((s_full - s_full.shift(252 if col in ["S&P 500", "Petrolio WTI ($/barile)",
                                                                "Gas Naturale ($/MMBtu)", "VIX (Volatilità)"]
                          else 12)) / s_full.shift(
                              252 if col in ["S&P 500", "Petrolio WTI ($/barile)",
                                             "Gas Naturale ($/MMBtu)", "VIX (Volatilità)"] else 12
                          ).abs() * 100).loc[start:end]
                plot_frames[col] = yoy_s
            else:  # cum
                sl = s_full.loc[start:end].dropna()
                r  = sl.pct_change().fillna(0)
                plot_frames[col] = ((1 + r).cumprod() - 1) * 100

        df_plot = pd.DataFrame(plot_frames).dropna(how="all")
        if df_plot.empty:
            f = empty_fig("Nessun dato nel range")
            return f, f

        # ── Grafico principale ─────────────────────────────────────────────
        # Normalizza a base 100 se assoluto e scale molto diverse
        if view == "abs" and len(avail) > 1:
            # Normalizza solo se gli ordini di grandezza differiscono di >10x
            col_ranges = {c: (df_plot[c].dropna().max() - df_plot[c].dropna().min())
                          for c in df_plot.columns
                          if not df_plot[c].dropna().empty}
            max_range = max(col_ranges.values()) if col_ranges else 1
            min_range = min(col_ranges.values()) if col_ranges else 1
            use_secondary = (max_range / max(min_range, 1e-10)) > 50
        else:
            use_secondary = False

        if use_secondary and len(avail) >= 2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            # Primo gruppo (es. S&P500): asse secondario
            sp500_col = next((c for c in df_plot.columns if "S&P" in c), None)
            for i, col in enumerate(df_plot.columns):
                s = df_plot[col].dropna()
                is_sp = (col == sp500_col)
                fig.add_trace(
                    go.Scatter(x=s.index, y=s.values, name=col,
                               line=dict(color=COLORS[i % len(COLORS)], width=1.8),
                               hovertemplate=f"<b>{col}</b><br>%{{x|%d %b %Y}}: %{{y:.2f}}<extra></extra>"),
                    secondary_y=is_sp
                )
            fig.update_yaxes(title_text="Prezzo / Valore", secondary_y=False)
            fig.update_yaxes(title_text="S&P 500", secondary_y=True)
        else:
            fig = go.Figure()
            for i, col in enumerate(df_plot.columns):
                s = df_plot[col].dropna()
                if s.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=col,
                    line=dict(color=COLORS[i % len(COLORS)], width=1.8),
                    hovertemplate=f"<b>{col}</b><br>%{{x|%d %b %Y}}: %{{y:.2f}}<extra></extra>",
                ))

        # Marcatori eventi storici
        events_sel = events_sel or []
        for ev_date_str, ev_label, ev_color in SHOCK_EVENTS:
            if ev_date_str in events_sel:
                ev_ts = pd.Timestamp(ev_date_str)
                if start <= ev_ts <= end:
                    fig.add_shape(type="line",
                                  x0=ev_date_str, x1=ev_date_str,
                                  y0=0, y1=1, yref="paper",
                                  line=dict(color=ev_color, dash="dash", width=1.5))
                    fig.add_annotation(x=ev_date_str, y=1, yref="paper",
                                       text=ev_label, showarrow=False,
                                       xanchor="left", yanchor="top",
                                       font=dict(size=9, color=ev_color),
                                       bgcolor="rgba(255,255,255,0.7)")

        ylabel = {"abs": "Valore", "yoy": "Δ% YoY", "cum": "Crescita % cum."}[view]
        title  = {"abs": "Serie Shock — Valori Assoluti",
                   "yoy": "Serie Shock — Δ% YoY",
                   "cum": "Serie Shock — Crescita % Cumulata"}[view]

        fig.update_layout(
            title=dict(text=title, font=dict(size=11), x=0.01),
            yaxis_title=ylabel,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(size=9),
                        bgcolor="rgba(255,255,255,0.8)"),
            margin=dict(t=50, b=30, l=55, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8",
        )
        if view in ("yoy", "cum"):
            fig.add_hline(y=0, line_color="#666", line_dash="dot", line_width=1)
        fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
        fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")

        # ── Grafico correlazione rolling ───────────────────────────────────
        # Correlazione rolling 12m: petrolio vs S&P500 e petrolio vs CPI
        fig_corr = go.Figure()
        oil_col  = next((c for c in df_all.columns if "WTI" in c), None)
        sp_col   = next((c for c in df_all.columns if "S&P" in c), None)
        cpi_col  = next((c for c in df_all.columns if "CPI All" in c), None)

        df_monthly = df_all.resample("MS").last()
        roll_w = 24

        if oil_col and sp_col and oil_col in df_monthly.columns and sp_col in df_monthly.columns:
            oil_r = df_monthly[oil_col].pct_change()
            sp_r  = df_monthly[sp_col].pct_change()
            combined = pd.DataFrame({"oil": oil_r, "sp": sp_r}).dropna()
            combined = combined.loc[start:end]
            if len(combined) >= roll_w:
                corr_oil_sp = combined["oil"].rolling(roll_w).corr(combined["sp"])
                fig_corr.add_trace(go.Scatter(
                    x=corr_oil_sp.index, y=corr_oil_sp.values,
                    name=f"Corr. Petrolio↔S&P500 ({roll_w}m)",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="Corr Petrolio/S&P: %{y:.3f}<extra></extra>",
                ))

        if oil_col and cpi_col and oil_col in df_monthly.columns and cpi_col in df_monthly.columns:
            oil_r  = df_monthly[oil_col].pct_change()
            cpi_yy = df_monthly[cpi_col]
            cpi_yy = ((cpi_yy - cpi_yy.shift(12)) / cpi_yy.shift(12).abs() * 100)
            combined = pd.DataFrame({"oil": oil_r, "cpi": cpi_yy}).dropna()
            combined = combined.loc[start:end]
            if len(combined) >= roll_w:
                corr_oil_cpi = combined["oil"].rolling(roll_w).corr(combined["cpi"])
                fig_corr.add_trace(go.Scatter(
                    x=corr_oil_cpi.index, y=corr_oil_cpi.values,
                    name=f"Corr. Petrolio↔CPI ({roll_w}m)",
                    line=dict(color="#d62728", width=2),
                    hovertemplate="Corr Petrolio/CPI: %{y:.3f}<extra></extra>",
                ))

        if not fig_corr.data:
            fig_corr = empty_fig("Correlazione non disponibile (dati insufficienti)")
        else:
            fig_corr.add_hline(y=0, line_color="#888", line_dash="dot", line_width=1)
            fig_corr.add_hline(y=0.5,  line_color="#2ca02c", line_dash="dash",
                                line_width=1, annotation_text="Corr=0.5")
            fig_corr.add_hline(y=-0.5, line_color="#d62728", line_dash="dash",
                                line_width=1, annotation_text="Corr=-0.5")
            fig_corr.update_layout(
                title=dict(text=f"Correlazione rolling {roll_w} mesi — Petrolio vs Macro",
                           font=dict(size=11), x=0.01),
                yaxis_title="Coefficiente di correlazione",
                yaxis=dict(range=[-1.05, 1.05]),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0, font=dict(size=9)),
                margin=dict(t=45, b=30, l=55, r=20),
                paper_bgcolor="white", plot_bgcolor="#f8f8f8",
            )
            fig_corr.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
            fig_corr.update_yaxes(showgrid=True, gridcolor="#e8e8e8")

        return fig, fig_corr

    # ── D.4a: Popola Modello Impatto (Y dropdown + X panel + slider) ───────────
    @app.callback(
        Output("impact-y-drop",       "options"),
        Output("impact-y-drop",       "value"),
        Output("impact-x-panel",      "children"),
        Output("impact-source-label", "children"),
        Output("impact-slider",       "min"),
        Output("impact-slider",       "max"),
        Output("impact-slider",       "value"),
        Output("impact-slider",       "marks"),
        Input("store-shock",          "data"),
        Input("store-shock-eur",      "data"),
        prevent_initial_call=False,
    )
    def impact_populate(shock_data, eur_data):
        df = _shock_adl_combined(shock_data, eur_data)
        _empty = (0, 1, [0, 1], {})
        if df.empty:
            no_data = html.Div("— carica dati FRED o Eurostat —",
                               style={"font-size": "9px", "color": "#aaa",
                                      "font-style": "italic"})
            return [], None, no_data, "— nessun dato —", *_empty
        cols = list(df.columns)
        opts = [{"label": c, "value": c} for c in cols]
        d1 = df.index.min().strftime("%m/%Y")
        d2 = df.index.max().strftime("%m/%Y")
        rows = _build_x_panel(cols, "imp-x-active", "imp-x-tr", "imp-x-lags")
        mn, mx, val, marks = _slider_params(df)
        return opts, cols[0], rows, f"{len(cols)} serie  |  {d1} → {d2}", mn, mx, val, marks

    @app.callback(
        Output("impact-slider-label", "children"),
        Input("impact-slider",        "value"),
    )
    def impact_slider_lbl(val):
        if not val or val[1] <= val[0]:
            return ""
        s = pd.Timestamp(val[0], unit="s").strftime("%b %Y")
        e = pd.Timestamp(val[1], unit="s").strftime("%b %Y")
        return f"📅  {s}  →  {e}"

    # ── D.4b: Stima Modello Impatto ─────────────────────────────────────────
    @app.callback(
        Output("impact-equation",        "children"),
        Output("impact-stats-table",     "children"),
        Output("impact-coef-table",      "children"),
        Output("chart-impact-fit",       "figure"),
        Output("chart-impact-irf",       "figure"),
        Output("chart-impact-projection","figure"),
        Output("impact-status",          "children"),
        Output("store-impact-model",     "data"),
        Input("btn-run-impact",          "n_clicks"),
        State("store-shock",             "data"),
        State("store-shock-eur",         "data"),
        State("impact-y-drop",           "value"),
        State("impact-y-tr",             "value"),
        State("impact-ar",               "value"),
        State({"type": "imp-x-active",  "index": ALL}, "value"),
        State({"type": "imp-x-active",  "index": ALL}, "id"),
        State({"type": "imp-x-tr",      "index": ALL}, "value"),
        State({"type": "imp-x-lags",    "index": ALL}, "value"),
        State("impact-cov",              "value"),
        State("impact-const",            "value"),
        State("impact-slider",           "value"),
        prevent_initial_call=True,
    )
    def run_impact_model(n, shock_data, eur_data, y_col, y_tr, ar_lags,
                         x_active, x_ids, x_trs, x_lags_list,
                         cov_type, add_const, slider_val):

        def err(msg):
            ef = empty_fig(msg)
            return msg, "", "", ef, ef, ef, f"❌ {msg}", no_update

        if not shock_data and not eur_data:
            return err("Carica i dati FRED o Eurostat prima")
        if not y_col:
            return err("Seleziona la variabile dipendente Y")

        df = _shock_adl_combined(shock_data, eur_data)
        if df.empty:
            return err("Nessun dato disponibile")

        if slider_val and len(slider_val) == 2 and slider_val[1] > slider_val[0]:
            t0 = pd.Timestamp(slider_val[0], unit="s").normalize()
            t1 = pd.Timestamp(slider_val[1], unit="s").normalize()
            df = df.loc[t0:t1]

        if y_col not in df.columns:
            return err(f"Colonna Y '{y_col}' non trovata")


        # ── Trasforma Y ──────────────────────────────────────────────────────
        y_series = _apply_transform(df[y_col], y_tr or "yoy").dropna()
        y_label  = f"{'ΔYoY' if y_tr=='yoy' else y_tr}({y_col})"

        # ── Costruisce dizionario X attive ────────────────────────────────────
        active_x = {}
        for chk, xid, xtr, xlags in zip(x_active, x_ids, x_trs, x_lags_list):
            col = xid["index"]
            if chk and col in df.columns:
                active_x[col] = (xtr or "yoy", xlags or [0])

        if not active_x and not ar_lags:
            return err("Seleziona almeno una variabile X o un lag AR")

        # ── Allinea indice temporale ──────────────────────────────────────────
        combined = pd.DataFrame({"__y__": y_series})
        for col, (xtr, xlags) in active_x.items():
            x_base = _apply_transform(df[col], xtr).dropna()
            for lag in sorted(set(xlags)):
                col_name = col if lag == 0 else f"{col}_L{lag}"
                combined[col_name] = x_base.shift(lag)

        ar_lags = sorted(set(ar_lags or []))
        for k in ar_lags:
            combined[f"Y(t-{k})"] = combined["__y__"].shift(k)

        combined = combined.dropna()
        if len(combined) < 10:
            return err(f"Osservazioni insufficienti: {len(combined)}")

        # ── OLS ──────────────────────────────────────────────────────────────
        y_vec  = combined["__y__"]
        X_cols = [c for c in combined.columns if c != "__y__"]
        X_mat  = combined[X_cols].copy()
        if add_const and "const" in (add_const or []):
            X_mat = sm.add_constant(X_mat, has_constant="add")

        try:
            res  = sm.OLS(y_vec, X_mat).fit()
            cov  = cov_type or "HC3"
            if cov == "HAC":
                rob = res.get_robustcov_results(cov_type="HAC",
                                                maxlags=int(len(combined)**0.25))
            elif cov == "HC3":
                rob = res.get_robustcov_results(cov_type="HC3")
            else:
                rob = res
            param_names = X_mat.columns.tolist()
            _p  = rob.params;  _pv = rob.pvalues
            _tv = rob.tvalues; _bs = rob.bse
            params  = pd.Series(_p  if hasattr(_p,  "index") else _p,  index=param_names)
            pvalues = pd.Series(_pv if hasattr(_pv, "index") else _pv, index=param_names)
            tvalues = pd.Series(_tv if hasattr(_tv, "index") else _tv, index=param_names)
            bse     = pd.Series(_bs if hasattr(_bs, "index") else _bs, index=param_names)
            model   = res
        except Exception as e:
            return err(f"Errore OLS: {e}")

        # ── Equazione ────────────────────────────────────────────────────────
        terms = []
        for v, cv in params.items():
            if v == "const":
                terms.append(f"  α = {cv:+.6f}")
            else:
                terms.append(f"  {'+'if cv>=0 else '−'} {abs(cv):.6f} · {v}")
        equation = f"{y_label} =\n" + "\n".join(terms) + "\n  + ε"

        def pstar(p):
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            if p < 0.10:  return "·"
            return ""

        # ── Diagnostiche ─────────────────────────────────────────────────────
        try:
            dw = float(sm.stats.stattools.durbin_watson(model.resid))
        except Exception:
            dw = float("nan")
        try:
            jb_stat, jb_p, jb_skew, jb_kurt = sm.stats.stattools.jarque_bera(model.resid)
        except Exception:
            jb_stat = jb_p = jb_skew = jb_kurt = float("nan")
        try:
            bp_lm, bp_p, *_ = sm.stats.diagnostic.het_breuschpagan(
                model.resid, model.model.exog)
        except Exception:
            bp_lm = bp_p = float("nan")
        try:
            cond_num = float(np.linalg.cond(X_mat.values.astype(float)))
        except Exception:
            cond_num = float("nan")

        cov_label = {"nonrobust": "OLS classico", "HC3": "HC3",
                     "HAC": "HAC Newey-West"}.get(cov, cov)

        stats_rows = [
            ["Statistica", "Valore", "Interpretazione"],
            ["N osservazioni", str(len(combined)), ""],
            ["Std. Error",     cov_label, ""],
            ["R²",             f"{model.rsquared:.6f}", ""],
            ["R² adj.",        f"{model.rsquared_adj:.6f}", ""],
            ["F-stat",         f"{model.fvalue:.4f}",
             f"p={model.f_pvalue:.4e} {pstar(model.f_pvalue)}"],
            ["AIC",            f"{model.aic:.4f}", "↓ migliore"],
            ["BIC",            f"{model.bic:.4f}", "↓ migliore"],
            ["Durbin-Watson",  f"{dw:.4f}",
             "✓ ok" if 1.5 <= dw <= 2.5 else "⚠ autocorrelazione"],
            ["Jarque-Bera",    f"{jb_stat:.4f}",
             f"p={jb_p:.4e} {pstar(jb_p)}"],
            ["  → Asimmetria", f"{jb_skew:.4f}", "~0 = simmetrico"],
            ["  → Curtosi",    f"{jb_kurt:.4f}", "normale=3"],
            ["Breusch-Pagan",  f"{bp_lm:.4f}",
             f"p={bp_p:.4e} {pstar(bp_p)}"],
            ["Cond. number",   f"{cond_num:.2f}", ">30 multicollin."],
        ]
        stats_table = _make_stat_table(stats_rows, "#1a3a5c")

        # ── Coefficienti con IC95 e VIF ───────────────────────────────────────
        try:
            conf = model.conf_int(alpha=0.05)
            if not hasattr(conf, "loc"):
                conf = pd.DataFrame(conf, index=params.index, columns=[0, 1])
        except Exception:
            conf = pd.DataFrame({"0": params*np.nan, "1": params*np.nan})

        x_cols_vif = [c for c in X_mat.columns if c != "const"]
        vif_dict = {}
        if len(x_cols_vif) > 1:
            for xc in x_cols_vif:
                other = [c for c in x_cols_vif if c != xc]
                try:
                    r2v = sm.OLS(X_mat[xc],
                                 sm.add_constant(X_mat[other])).fit().rsquared
                    vif_dict[xc] = 1 / (1 - r2v) if r2v < 1 else np.inf
                except Exception:
                    vif_dict[xc] = np.nan
        coef_rows = [["Variabile", "Coeff.", "Std Err", "t-stat",
                       "p-val", "Sig.", "IC95 inf", "IC95 sup", "VIF"]]
        for v in params.index:
            p   = pvalues[v]
            vif = vif_dict.get(v, np.nan)
            vif_s = f"{vif:.2f}" if (isinstance(vif, float) and not np.isnan(vif)) else "—"
            try:
                ic_lo = f"{conf.loc[v, 0]:.6f}"
                ic_hi = f"{conf.loc[v, 1]:.6f}"
            except Exception:
                ic_lo = ic_hi = "—"
            coef_rows.append([v, f"{params[v]:+.6f}", f"{bse[v]:.6f}",
                               f"{tvalues[v]:.4f}", f"{p:.4e}", pstar(p),
                               ic_lo, ic_hi, vif_s])
        coef_table = html.Div([
            html.Div("Coefficienti di regressione",
                     style={"font-size": "11px", "font-weight": "bold",
                            "color": "#1a3a5c", "background": "#eaf4fb",
                            "padding": "5px 10px",
                            "border-radius": "4px 4px 0 0",
                            "border": "1px solid #aed6f1",
                            "border-bottom": "none", "margin-top": "10px"}),
            _make_stat_table(coef_rows, "#2e6da4"),
            html.Div("*** p<0.001  ** p<0.01  * p<0.05  · p<0.10",
                     style={"font-size": "10px", "color": "#777",
                            "margin-top": "4px", "font-style": "italic"}),
        ])

        # ── Fit ──────────────────────────────────────────────────────────────
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=y_vec.index, y=y_vec.values,
                                     name=y_label, line=dict(color="#1f77b4", width=1.5)))
        fig_fit.add_trace(go.Scatter(x=y_vec.index, y=model.fittedvalues.values,
                                     name="Fitted",
                                     line=dict(color="#d62728", width=1.5, dash="dot")))
        fig_fit.add_hline(y=0, line_color="#999", line_width=0.8)
        fig_fit.update_layout(
            title=dict(text=f"Fit  |  R²={model.rsquared:.4f}  R²adj={model.rsquared_adj:.4f}",
                       font=dict(size=11), x=0.01),
            hovermode="x unified", margin=dict(t=45, b=30, l=55, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8",
            legend=dict(orientation="h", y=1.02, font=dict(size=9)),
        )

        # ── IRF dinamica 12 mesi ─────────────────────────────────────────────
        H_IRF = 12
        _PCT_TR = {"yoy", "ldiff", "log_diff", "pct", "pct_change", "dlog"}
        irf_scale  = 100.0 if (y_tr or "yoy") in _PCT_TR else 1.0
        irf_suffix = "%" if irf_scale == 100.0 else ""

        ar_coefs = {}
        for v in params.index:
            if v.startswith("Y(t-"):
                try:
                    ar_coefs[int(v[4:-1])] = float(params[v])
                except ValueError:
                    pass

        x_groups = {}
        for v in params.index:
            if v == "const" or v.startswith("Y(t-"):
                continue
            if "_L" in v:
                parts = v.rsplit("_L", 1)
                try:
                    base, lag = parts[0], int(parts[1])
                except ValueError:
                    base, lag = v, 0
            else:
                base, lag = v, 0
            x_groups.setdefault(base, []).append((lag, float(params[v])))

        x_sigmas = {c: float(combined[c].std())
                    for c in combined.columns if c != "__y__"}
        n_g = len(x_groups)
        if n_g == 0:
            fig_irf = empty_fig("Nessuna variabile X")
        else:
            cpr = min(3, n_g)
            nri = (n_g + cpr - 1) // cpr
            sp_titles = []
            for base in x_groups:
                sigma_b = next((x_sigmas.get(k) for k in combined.columns
                                if k != "__y__" and (k == base or
                                   k.startswith(base + "_L"))), None)
                slbl = (f"  [σ={sigma_b*100:.2f}%]"
                        if sigma_b is not None and irf_scale == 100.0
                        else (f"  [σ={sigma_b:.4f}]" if sigma_b is not None else ""))
                sp_titles.append(f"{base[:24]}{slbl}")
            fig_irf = make_subplots(rows=nri, cols=cpr, subplot_titles=sp_titles,
                                     vertical_spacing=0.14, horizontal_spacing=0.08)
            fmt = (lambda v: f"{v:+.2f}%") if irf_suffix else (lambda v: f"{v:+.4f}")
            for gi, (base, lc) in enumerate(x_groups.items()):
                ri = gi // cpr + 1; ci = gi % cpr + 1
                color = COLORS[gi % len(COLORS)]
                sigma = next((x_sigmas.get(k, 1.0) for k in combined.columns
                              if k != "__y__" and (k == base or
                                 k.startswith(base + "_L"))), 1.0)
                direct = {lag: coef * sigma for lag, coef in lc}
                irf_vals = []
                for h in range(H_IRF):
                    d = direct.get(h, 0.0)
                    ar_part = sum(ar_coefs.get(j, 0.0) * irf_vals[h - j]
                                  for j in ar_coefs if 0 < j <= h)
                    irf_vals.append(d + ar_part)
                horizons = list(range(H_IRF))
                irf_sc   = [v * irf_scale for v in irf_vals]
                cum_sc   = list(np.cumsum(irf_sc))
                fig_irf.add_trace(go.Bar(
                    x=horizons, y=irf_sc, name=base,
                    marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in irf_sc],
                    marker_line_width=0, showlegend=False,
                    text=[fmt(v) for v in irf_sc],
                    textposition="outside", textfont=dict(size=7)),
                    row=ri, col=ci)
                fig_irf.add_trace(go.Scatter(
                    x=horizons, y=cum_sc, mode="lines+markers",
                    name=f"Cum. {base}",
                    line=dict(color=color, width=2, dash="dot"),
                    marker=dict(size=4), showlegend=False),
                    row=ri, col=ci)
                fig_irf.add_hline(y=0, line_dash="solid", line_color="#bbb",
                                   line_width=0.8, row=ri, col=ci)
            fig_irf.update_layout(
                title=dict(text=f"IRF — +1σ per variabile  |  Y: {y_label}",
                           font=dict(size=11), x=0.01),
                height=max(280, 230 * nri),
                margin=dict(t=50, b=30, l=45, r=15),
                paper_bgcolor="white", plot_bgcolor="#f8f8f8",
            )
            fig_irf.update_xaxes(title_text="Mesi", showgrid=True, gridcolor="#e8e8e8")
            fig_irf.update_yaxes(title_text=f"Δ Y ({irf_suffix or 'unità'})",
                                  showgrid=True, gridcolor="#e8e8e8")

        # ── Residui ───────────────────────────────────────────────────────────
        resid = model.resid
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(x=resid.index, y=resid.values,
                                        mode="lines", line=dict(color="#7f7f7f", width=1),
                                        name="Residui"))
        fig_resid.add_hline(y=0, line_dash="solid", line_color="#aaa", line_width=0.8)
        fig_resid.update_layout(
            title=dict(text="Residui", font=dict(size=11), x=0.01),
            margin=dict(t=40, b=30, l=55, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8",
        )

        status = (f"✅  N={len(combined)}  |  R²={model.rsquared:.4f}  "
                  f"R²adj={model.rsquared_adj:.4f}  |  DW={dw:.3f}  |  "
                  f"JB p={jb_p:.3e} sk={jb_skew:.2f} ku={jb_kurt:.2f}  |  {cov_label}")

        # ── Salva modello per scenario petrolio ───────────────────────────────
        x_transforms = {col: xtr for col, (xtr, _) in active_x.items()}
        x_sigmas_dict = {c: float(combined[c].std())
                         for c in combined.columns if c != "__y__"}
        store_model = {
            "params":      params.to_dict(),
            "ar_coefs":    {str(k): v for k, v in ar_coefs.items()},
            "y_tr":        y_tr or "yoy",
            "y_label":     y_label,
            "x_transforms": x_transforms,
            "x_sigmas":    x_sigmas_dict,
        }

        return (equation, stats_table, coef_table, fig_fit, fig_irf, fig_resid,
                status, store_model)


    # ── D.4c: Scenario prezzo petrolio ─────────────────────────────────────
    @app.callback(
        Output("chart-impact-projection", "figure", allow_duplicate=True),
        Output("impact-scenario-status",  "children"),
        Input("btn-run-impact-scenario",  "n_clicks"),
        State("store-impact-model",       "data"),
        State("impact-oil-current",       "value"),
        State("impact-oil-scenario",      "value"),
        State("impact-proj-months",       "value"),
        prevent_initial_call=True,
    )
    def run_impact_scenario(n, model_data, oil_current, oil_scenario, n_months):
        if not model_data:
            return empty_fig("Stima prima il modello"), "❌ Modello non stimato"
        if not oil_current or not oil_scenario:
            return empty_fig("Inserisci i prezzi"), "❌ Prezzi mancanti"
        if oil_current <= 0:
            return empty_fig("Prezzo corrente non valido"), "❌ Prezzo corrente ≤ 0"

        params_dict  = model_data.get("params", {})
        ar_coefs_raw = model_data.get("ar_coefs", {})
        y_label      = model_data.get("y_label", "Y")
        y_tr         = model_data.get("y_tr", "yoy")
        x_transforms = model_data.get("x_transforms", {})

        ar_coefs = {int(k): float(v) for k, v in ar_coefs_raw.items()}
        params   = {k: float(v) for k, v in params_dict.items()}

        # Trova variabili petrolio nel modello (cerca parole chiave)
        OIL_KW = ["brent", "crude", "wti", "oil", "petrolio", "petrol",
                   "energy", "ener", "nrg"]
        oil_vars = []
        for name in params:
            name_lo = name.lower()
            if any(kw in name_lo for kw in OIL_KW):
                oil_vars.append(name)

        if not oil_vars:
            return (empty_fig("Nessuna variabile petrolio trovata nel modello"),
                    "⚠ Aggiungi una variabile tipo Brent o WTI tra le X")

        H = int(n_months or 12)

        # ── Calcola la dimensione dello shock in unità trasformate ────────────
        # Per ogni variabile oil, determina la trasformazione usata
        def oil_shock_size(col_name):
            """Restituisce il delta dello shock nella trasformazione applicata."""
            # Risali al nome colonna base (senza _L suffix)
            base = col_name.rsplit("_L", 1)[0] if "_L" in col_name else col_name
            xtr = x_transforms.get(base, "yoy")
            ratio = oil_scenario / oil_current
            if xtr in ("yoy", "pct"):
                return (ratio - 1.0) * 100.0   # % change
            elif xtr == "levels":
                return float(oil_scenario - oil_current)
            elif xtr == "log":
                return float(np.log(oil_scenario) - np.log(oil_current))
            elif xtr in ("dlog", "ldiff", "log_diff"):
                return float(np.log(ratio))
            else:
                return (ratio - 1.0) * 100.0

        # ── IRF-style propagation per il baseline e per lo scenario ──────────
        # Raggruppa le variabili oil per lag
        oil_groups = {}
        for name in oil_vars:
            if "_L" in name:
                parts = name.rsplit("_L", 1)
                try:
                    base_n, lag = parts[0], int(parts[1])
                except ValueError:
                    base_n, lag = name, 0
            else:
                base_n, lag = name, 0
            oil_groups.setdefault(base_n, []).append((lag, params[name]))

        _PCT_TR = {"yoy", "ldiff", "log_diff", "pct", "pct_change", "dlog"}
        irf_scale  = 100.0 if y_tr in _PCT_TR else 1.0
        irf_suffix = "%" if irf_scale == 100.0 else ""

        # Propagazione: effetto cumulativo su Y per H mesi
        effect = [0.0] * H
        for base_n, lags_coefs in oil_groups.items():
            delta = oil_shock_size(base_n)
            direct = {lag: coef * delta for lag, coef in lags_coefs}
            local_irf = []
            for h in range(H):
                d = direct.get(h, 0.0)
                ar_part = sum(ar_coefs.get(j, 0.0) * local_irf[h - j]
                              for j in ar_coefs if 0 < j <= h)
                local_irf.append(d + ar_part)
            for h in range(H):
                effect[h] += local_irf[h] * irf_scale

        horizons = list(range(H))
        cum_effect = list(np.cumsum(effect))

        pct_chg = (oil_scenario / oil_current - 1) * 100
        direction = "↑" if oil_scenario > oil_current else "↓"
        sign_color_fn = lambda v: "#c0392b" if v < 0 else "#27ae60"

        fig = go.Figure()

        # Barre per effetto mensile
        fig.add_trace(go.Bar(
            x=horizons, y=effect,
            name=f"Effetto mensile su {y_label}",
            marker_color=[sign_color_fn(v) for v in effect],
            marker_line_width=0,
            text=[f"{v:+.3f}{irf_suffix}" for v in effect],
            textposition="outside",
            textfont=dict(size=8),
            yaxis="y1",
        ))

        # Linea effetto cumulativo
        fig.add_trace(go.Scatter(
            x=horizons, y=cum_effect,
            name="Effetto cumulativo",
            mode="lines+markers",
            line=dict(color="#e67e22", width=2.5, dash="dot"),
            marker=dict(size=5, color="#e67e22"),
            yaxis="y1",
        ))

        fig.add_hline(y=0, line_color="#999", line_width=0.8)

        oil_vars_str = ", ".join(sorted(set(
            v.rsplit("_L", 1)[0] if "_L" in v else v for v in oil_vars)))
        fig.update_layout(
            title=dict(
                text=(f"Scenario petrolio: {oil_current:.0f} → {oil_scenario:.0f} $/bbl "
                      f"({direction}{abs(pct_chg):.1f}%)  |  "
                      f"Variabili: {oil_vars_str[:60]}"),
                font=dict(size=11), x=0.01),
            xaxis=dict(title="Mesi dall'inizio dello shock", showgrid=True,
                       gridcolor="#e8e8e8"),
            yaxis=dict(title=f"Δ {y_label} ({irf_suffix or 'unità'})",
                       showgrid=True, gridcolor="#e8e8e8"),
            hovermode="x unified",
            margin=dict(t=55, b=40, l=55, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8",
            legend=dict(orientation="h", y=1.02, font=dict(size=9)),
            barmode="relative",
        )

        cum_final = cum_effect[-1] if cum_effect else 0.0
        status = (f"✅  Scenario: {oil_current:.0f}→{oil_scenario:.0f} $/bbl  "
                  f"({direction}{abs(pct_chg):.1f}%)  |  "
                  f"Impatto cumulativo a {H} mesi: {cum_final:+.3f}{irf_suffix}  |  "
                  f"Variabili oil: {oil_vars_str[:40]}")
        return fig, status


    # ── D.5: Simulatore di policy ───────────────────────────────────────────
    @app.callback(
        Output("sim-results-panel",  "children"),
        Output("sim-optimal-panel",  "children"),
        Output("chart-sim-macro",    "figure"),
        Output("chart-sim-mvpq",     "figure"),
        Output("chart-sim-tradeoff", "figure"),
        Input("btn-run-sim",         "n_clicks"),
        Input("btn-optimize-rate",   "n_clicks"),
        State("store-shock",         "data"),
        State("store-data",          "data"),
        State("store-gdp",           "data"),
        State("sim-oil-chg",         "value"),
        State("sim-gas-chg",         "value"),
        State("sim-russia-supply",   "value"),
        State("sim-spr-release",     "value"),
        State("sim-rate-chg",        "value"),
        State("sim-horizon",         "value"),
        State("sim-m2-growth",       "value"),
        State("sim-velocity-chg",    "value"),
        State("loss-lambda-pi",      "value"),
        State("loss-lambda-y",       "value"),
        State("loss-pi-target",      "value"),
        prevent_initial_call=True,
    )
    def run_policy_sim(n_sim, n_opt,
                       shock_data, mon_data, gdp_data,
                       oil_chg, gas_chg, russia_supply, spr_release,
                       rate_chg_bps, horizon, m2_growth, vel_chg,
                       lambda_pi, lambda_y, pi_target):

        ctx = callback_context
        optimize_mode = ctx.triggered and "optimize" in str(ctx.triggered_id)

        # ── Stima sensitività dai dati storici ─────────────────────────────
        # Calibrazione empirica con regressione semplice sulle ultime serie disponibili
        def _get_empirical_coefs(shock_data, mon_data, gdp_data):
            """Restituisce dict con coefficienti stimati da OLS rolling."""
            coefs = {
                "beta_oil_cpi":   0.025,  # +10% oil → +0.25% CPI
                "beta_oil_gdp":  -0.015,  # +10% oil → -0.15% GDP
                "beta_oil_sp":    0.050,  # +10% oil → +0.5% SP500 (supply side, ambiguous)
                "beta_rate_cpi": -0.030,  # +100bps → -0.30% CPI (dopo 12m)
                "beta_rate_gdp": -0.060,  # +100bps → -0.60% GDP (dopo 12m)
                "beta_rate_sp":  -0.120,  # +100bps → -1.2% SP500
                "beta_m2_cpi":    0.040,  # +1% M2 → +0.04% CPI
                "beta_gas_cpi":   0.008,  # +10% gas → +0.08% CPI
                "beta_gas_gdp":  -0.008,  # +10% gas → -0.08% GDP
            }

            # Migliora con dati reali se disponibili
            try:
                frames = []
                if shock_data:
                    df_s = pd.read_json(io.StringIO(shock_data), orient="split")
                    df_s.index = pd.to_datetime(df_s.index)
                    frames.append(df_s.resample("MS").last())
                if mon_data:
                    df_m = pd.read_json(io.StringIO(mon_data), orient="split")
                    df_m.index = pd.to_datetime(df_m.index)
                    frames.append(df_m)
                if not frames:
                    return coefs

                df_all = pd.concat(frames, axis=1)
                df_all = df_all.loc[:, ~df_all.columns.duplicated()]

                oil_col = next((c for c in df_all.columns if "WTI" in c), None)
                cpi_col = next((c for c in df_all.columns if "CPI All" in c), None)
                fed_col = next((c for c in df_all.columns if "Fed Funds" in c), None)
                sp_col  = next((c for c in df_all.columns if "S&P" in c), None)

                if oil_col and cpi_col:
                    oil_yoy = ((df_all[oil_col] - df_all[oil_col].shift(12)) /
                                df_all[oil_col].shift(12).abs() * 100).dropna()
                    cpi_yoy = ((df_all[cpi_col] - df_all[cpi_col].shift(12)) /
                                df_all[cpi_col].shift(12).abs() * 100).dropna()
                    comb = pd.DataFrame({"oil": oil_yoy, "cpi": cpi_yoy.shift(-3)}).dropna()
                    if len(comb) >= 40:
                        res = sm.OLS(comb["cpi"], sm.add_constant(comb["oil"])).fit()
                        # coefficiente: per 1pp YoY in oil → pp YoY in CPI
                        # normalizza a "per 10pp"
                        coefs["beta_oil_cpi"] = float(res.params.get("oil", 0.025)) * 10 / 100

                if oil_col and fed_col and sp_col:
                    oil_yoy = ((df_all[oil_col] - df_all[oil_col].shift(12)) /
                                df_all[oil_col].shift(12).abs() * 100).dropna()
                    sp_yoy  = ((df_all[sp_col] - df_all[sp_col].shift(12)) /
                                df_all[sp_col].shift(12).abs() * 100).dropna()
                    fed_chg = df_all[fed_col].diff(3).dropna()
                    comb = pd.DataFrame({"oil": oil_yoy, "sp": sp_yoy, "fed": fed_chg}).dropna()
                    if len(comb) >= 40:
                        res = sm.OLS(comb["sp"],
                                     sm.add_constant(comb[["oil", "fed"]])).fit()
                        coefs["beta_oil_sp"]  = float(res.params.get("oil", 0.05)) * 10 / 100
                        coefs["beta_rate_sp"] = float(res.params.get("fed", -0.12)) * 1 / 100

            except Exception:
                pass
            return coefs

        coefs = _get_empirical_coefs(shock_data, mon_data, gdp_data)

        # ── Valori correnti M2 / CPI / GDP (ultima osservazione disponibile) ─
        m2_current   = None
        vel_current  = None
        cpi_current  = None
        gdp_current  = None
        fed_current  = None

        if mon_data:
            df_m = pd.read_json(io.StringIO(mon_data), orient="split")
            df_m.index = pd.to_datetime(df_m.index)
            m2_col  = next((c for c in df_m.columns if "M2 Money" in c), None)
            vel_col = next((c for c in df_m.columns if "Velocity" in c), None)
            cpi_col = next((c for c in df_m.columns if "CPI All" in c), None)
            fed_col = next((c for c in df_m.columns if "Fed Funds" in c), None)
            if m2_col:
                m2_current  = float(df_m[m2_col].dropna().iloc[-1])
            if vel_col:
                vel_current = float(df_m[vel_col].dropna().iloc[-1])
            if cpi_col:
                cpi_current = float(df_m[cpi_col].dropna().iloc[-1])
            if fed_col:
                fed_current = float(df_m[fed_col].dropna().iloc[-1])
        if gdp_data:
            df_g = pd.read_json(io.StringIO(gdp_data), orient="split")
            df_g.index = pd.to_datetime(df_g.index)
            gdp_col = next((c for c in df_g.columns if "PIL Reale" in c), None)
            if gdp_col:
                gdp_current = float(df_g[gdp_col].dropna().iloc[-1])

        # ── Calcolo effetti scenario ───────────────────────────────────────
        oil_chg      = float(oil_chg or 0)
        gas_chg      = float(gas_chg or 0)
        russia_supp  = float(russia_supply or 0)
        spr          = float(spr_release or 0)
        rate_chg     = float(rate_chg_bps or 0) / 100.0  # in punti percentuali
        hor          = int(horizon or 12)
        m2_gr        = float(m2_growth or 4)
        vel_gr       = float(vel_chg or -2)

        # Effetto netto petrolio: shock diretto ± supply russo ± SPR
        # SPR 1Mb rilasciato ≈ -0.03% prezzo WTI (rule of thumb appross.)
        oil_supply_offset = russia_supp * (-3.5) + spr * 0.03  # in % prezzo
        oil_net_chg = oil_chg + oil_supply_offset

        # ── Impatti diretti (effetti cumulati in `horizon` mesi) ──────────
        # Scala dai coefficienti mensili a orizzonte horizon
        scale = hor / 12.0

        delta_cpi = (
            coefs["beta_oil_cpi"]  * (oil_net_chg / 10) * 10 +
            coefs["beta_gas_cpi"]  * (gas_chg / 10) * 10 +
            coefs["beta_rate_cpi"] * rate_chg +
            coefs["beta_m2_cpi"]   * (m2_gr - 4)  # scostamento da crescita M2 normale
        ) * scale

        delta_gdp = (
            coefs["beta_oil_gdp"]  * (oil_net_chg / 10) * 10 +
            coefs["beta_gas_gdp"]  * (gas_chg / 10) * 10 +
            coefs["beta_rate_gdp"] * rate_chg
        ) * scale

        delta_sp = (
            coefs["beta_oil_sp"]  * (oil_net_chg / 10) * 10 +
            coefs["beta_rate_sp"] * rate_chg
        ) * scale

        # ── MV = PQ ─────────────────────────────────────────────────────
        # Crescita MV = m2_gr + vel_gr
        # Crescita PQ = Δinflazione + Δcrescita_reale
        # (appross: Δ%PQ ≈ ΔπYoY + ΔGDPYoY)
        if m2_current and vel_current:
            mv_growth = m2_gr + vel_gr
        else:
            mv_growth = m2_gr + vel_gr

        pq_growth = (delta_cpi / scale) * 12 + (delta_gdp / scale) * 12
        mvpq_gap  = mv_growth - pq_growth  # positivo = eccesso monetario
        mvpq_label = "eccesso" if mvpq_gap > 0 else "carenza"

        # ── Tasso ottimale (Taylor Rule modificata con loss function) ──────
        if optimize_mode:
            lambda_pi_ = float(lambda_pi or 1.0)
            lambda_y_  = float(lambda_y  or 0.5)
            pi_target_ = float(pi_target or 2.0)

            def loss_fn(r_chg):
                """Funzione di perdita BC su orizzonte horizon."""
                d_pi = (
                    coefs["beta_oil_cpi"]  * (oil_net_chg / 10) * 10 +
                    coefs["beta_gas_cpi"]  * (gas_chg / 10) * 10 +
                    coefs["beta_rate_cpi"] * r_chg +
                    coefs["beta_m2_cpi"]   * (m2_gr - 4)
                ) * scale

                d_gdp = (
                    coefs["beta_oil_gdp"]  * (oil_net_chg / 10) * 10 +
                    coefs["beta_gas_gdp"]  * (gas_chg / 10) * 10 +
                    coefs["beta_rate_gdp"] * r_chg
                ) * scale

                # Inflazione attesa = corrente + shock
                cpi_yoy_now = 3.5  # default se non disponibile
                pi_expected = cpi_yoy_now + d_pi

                return lambda_pi_ * (pi_expected - pi_target_) ** 2 + lambda_y_ * d_gdp ** 2

            res_opt = minimize_scalar(loss_fn, bounds=(-5, 8), method="bounded")
            opt_rate_chg = res_opt.x
            opt_loss     = res_opt.fun

            opt_delta_cpi = (
                coefs["beta_oil_cpi"]  * (oil_net_chg / 10) * 10 +
                coefs["beta_gas_cpi"]  * (gas_chg / 10) * 10 +
                coefs["beta_rate_cpi"] * opt_rate_chg +
                coefs["beta_m2_cpi"]   * (m2_gr - 4)
            ) * scale

            opt_delta_gdp = (
                coefs["beta_oil_gdp"]  * (oil_net_chg / 10) * 10 +
                coefs["beta_gas_gdp"]  * (gas_chg / 10) * 10 +
                coefs["beta_rate_gdp"] * opt_rate_chg
            ) * scale

            opt_sign = "rialzo" if opt_rate_chg > 0 else "taglio"
            optimal_panel = html.Div([
                html.Div([
                    html.B("🎯 Intervento ottimale stimato",
                           style={"font-size": "13px", "color": "#fff",
                                  "display": "block", "margin-bottom": "6px"}),
                    html.Div([
                        html.Div([
                            html.Span("Variazione tassi raccomandata",
                                      style={"font-size": "11px", "color": "#aaa"}),
                            html.Span(f"{opt_rate_chg * 100:+.0f} bps ({opt_sign})",
                                      style={"font-size": "22px", "font-weight": "bold",
                                             "color": "#4caf50" if opt_rate_chg < 0 else "#ef5350",
                                             "display": "block"}),
                        ], style={"flex": "1", "text-align": "center", "padding": "10px"}),
                        html.Div([
                            html.Span("CPI atteso",
                                      style={"font-size": "11px", "color": "#aaa"}),
                            html.Span(f"{3.5 + opt_delta_cpi:+.2f}%",
                                      style={"font-size": "18px", "font-weight": "bold",
                                             "color": "#ff8a65" if (3.5 + opt_delta_cpi) > pi_target_ else "#66bb6a",
                                             "display": "block"}),
                        ], style={"flex": "1", "text-align": "center", "padding": "10px"}),
                        html.Div([
                            html.Span("PIL atteso",
                                      style={"font-size": "11px", "color": "#aaa"}),
                            html.Span(f"{opt_delta_gdp:+.2f}%",
                                      style={"font-size": "18px", "font-weight": "bold",
                                             "color": "#ef5350" if opt_delta_gdp < -1 else "#66bb6a",
                                             "display": "block"}),
                        ], style={"flex": "1", "text-align": "center", "padding": "10px"}),
                        html.Div([
                            html.Span("Loss function BC",
                                      style={"font-size": "11px", "color": "#aaa"}),
                            html.Span(f"{opt_loss:.4f}",
                                      style={"font-size": "18px", "font-weight": "bold",
                                             "color": "#fff", "display": "block"}),
                        ], style={"flex": "1", "text-align": "center", "padding": "10px"}),
                    ], style={"display": "flex"}),
                    html.Div([
                        html.Span(
                            f"Parametri: λπ={lambda_pi_}  λy={lambda_y_}  π*={pi_target_}%  "
                            f"| Orizzonte={hor} mesi  |  "
                            f"Δ Petrolio netto={oil_net_chg:+.1f}% (russo={russia_supp:+.1f}Mb/d, SPR={spr:.0f}Mb)",
                            style={"font-size": "9px", "color": "#888"},
                        ),
                    ], style={"padding": "4px 0"}),
                ], style={"background": "#1a2a1a", "border": "1px solid #2a5a2a",
                           "border-radius": "8px", "padding": "12px 16px"}),
            ])
        else:
            optimal_panel = html.Div()

        # ── Pannello risultati scenario ───────────────────────────────────
        def _metric_card(label, value_str, color, note=""):
            return html.Div([
                html.Div(label, style={"font-size": "10px", "color": "#888"}),
                html.Div(value_str, style={"font-size": "22px", "font-weight": "bold",
                                            "color": color}),
                html.Div(note, style={"font-size": "9px", "color": "#777",
                                       "margin-top": "2px"}),
            ], style={"flex": "1", "text-align": "center", "padding": "10px",
                       "background": "#1a1a2a", "border-radius": "6px",
                       "margin": "4px"})

        cpi_color = "#ef5350" if delta_cpi > 1 else ("#ffb74d" if delta_cpi > 0.3 else "#66bb6a")
        gdp_color = "#66bb6a" if delta_gdp > 0 else "#ef5350"
        sp_color  = "#66bb6a" if delta_sp > 0 else "#ef5350"
        gap_color = "#ff8a65" if mvpq_gap > 2 else ("#ffb74d" if mvpq_gap > 0 else "#66bb6a")

        results_panel = html.Div([
            html.Div([
                html.B("📊 Risultati scenario corrente",
                       style={"font-size": "12px", "color": "#ddd"}),
                html.Span(f"  (orizzonte {hor} mesi  |  Δ tassi {rate_chg * 100:+.0f} bps  |  "
                           f"Petrolio {oil_net_chg:+.1f}%  |  Gas {gas_chg:+.1f}%)",
                           style={"font-size": "10px", "color": "#888"}),
            ], style={"margin-bottom": "8px"}),
            html.Div([
                _metric_card("Impatto CPI", f"{delta_cpi:+.2f} pp",
                             cpi_color, f"su orizzonte {hor}m"),
                _metric_card("Impatto PIL reale", f"{delta_gdp:+.2f} pp",
                             gdp_color, f"cumulato {hor}m"),
                _metric_card("Impatto S&P 500", f"{delta_sp:+.2f} pp",
                             sp_color, f"cumulato {hor}m"),
                _metric_card("Gap MV−PQ", f"{mvpq_gap:+.2f} pp",
                             gap_color, f"{mvpq_label} monetario"),
                _metric_card("Petrolio netto", f"{oil_net_chg:+.1f}%",
                             "#ef5350" if oil_net_chg > 20 else "#ffb74d",
                             f"russo {russia_supp:+.1f}Mb/d | SPR {spr:.0f}Mb"),
            ], style={"display": "flex", "flex-wrap": "wrap"}),
        ], style={"background": "#1a1a2a", "border": "1px solid #2a2a4a",
                   "border-radius": "8px", "padding": "12px 16px"})

        # ── Grafico macro: traiettorie per diversi livelli di tassi ──────
        rate_range = np.linspace(-3.0, 5.0, 50)  # -300bps a +500bps

        def _path(delta_r):
            months = np.arange(1, hor + 1)
            phi = np.minimum(months / hor, 1.0)  # rampa temporale

            d_cpi_ = (coefs["beta_oil_cpi"] * (oil_net_chg / 10) * 10 +
                      coefs["beta_gas_cpi"] * (gas_chg / 10) * 10 +
                      coefs["beta_rate_cpi"] * delta_r +
                      coefs["beta_m2_cpi"] * (m2_gr - 4)) * phi

            d_gdp_ = (coefs["beta_oil_gdp"] * (oil_net_chg / 10) * 10 +
                      coefs["beta_gas_gdp"] * (gas_chg / 10) * 10 +
                      coefs["beta_rate_gdp"] * delta_r) * phi

            d_sp_  = (coefs["beta_oil_sp"] * (oil_net_chg / 10) * 10 +
                      coefs["beta_rate_sp"] * delta_r) * phi

            return months, d_cpi_, d_gdp_, d_sp_

        fig_macro = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Impatto su CPI (pp)", "Impatto su PIL (%)", "Impatto su S&P 500 (%)"],
            horizontal_spacing=0.10,
        )

        # Scenario utente
        mos, d_cpi_path, d_gdp_path, d_sp_path = _path(rate_chg)

        fig_macro.add_trace(go.Scatter(
            x=mos, y=d_cpi_path, name=f"CPI (tasso {rate_chg*100:+.0f}bps)",
            line=dict(color="#ef5350", width=2.5),
            hovertemplate="Mese %{x}: %{y:+.3f} pp<extra></extra>",
        ), row=1, col=1)

        fig_macro.add_trace(go.Scatter(
            x=mos, y=d_gdp_path, name=f"PIL (tasso {rate_chg*100:+.0f}bps)",
            line=dict(color="#42a5f5", width=2.5),
            hovertemplate="Mese %{x}: %{y:+.3f} pp<extra></extra>",
        ), row=1, col=2)

        fig_macro.add_trace(go.Scatter(
            x=mos, y=d_sp_path, name=f"S&P (tasso {rate_chg*100:+.0f}bps)",
            line=dict(color="#66bb6a", width=2.5),
            hovertemplate="Mese %{x}: %{y:+.3f} pp<extra></extra>",
        ), row=1, col=3)

        # Scenario zero (nessun intervento)
        mos0, d_cpi_0, d_gdp_0, d_sp_0 = _path(0.0)

        for col_i, (data, name) in enumerate([
            (d_cpi_0, "CPI (no interv.)"),
            (d_gdp_0, "PIL (no interv.)"),
            (d_sp_0,  "S&P (no interv.)"),
        ], start=1):
            fig_macro.add_trace(go.Scatter(
                x=mos0, y=data, name=name,
                line=dict(color="#888", width=1.2, dash="dot"),
                showlegend=True,
                hovertemplate=f"Mese %{{x}}: %{{y:+.3f}} pp<extra>{name}</extra>",
            ), row=1, col=col_i)

        for r, c in [(1, 1), (1, 2), (1, 3)]:
            fig_macro.add_hline(y=0, line_color="#555", line_dash="dot",
                                 line_width=1, row=r, col=c)

        fig_macro.update_layout(
            title=dict(text=f"Traiettorie macro — scenario: Δtassi={rate_chg*100:+.0f}bps  "
                             f"Petrolio={oil_net_chg:+.1f}%  Gas={gas_chg:+.1f}%",
                       font=dict(size=11), x=0.01),
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=9)),
            margin=dict(t=65, b=35, l=50, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8",
        )
        for r in [1]:
            for c in [1, 2, 3]:
                fig_macro.update_xaxes(title_text="Mesi", showgrid=True,
                                        gridcolor="#e8e8e8", row=r, col=c)
                fig_macro.update_yaxes(showgrid=True, gridcolor="#e8e8e8", row=r, col=c)

        # ── Grafico MV = PQ ────────────────────────────────────────────────
        fig_mvpq = go.Figure()
        months = np.arange(1, hor + 1)
        # MV crescita accumulata (composta semplificata)
        mv_cum = [(1 + (m2_gr + vel_gr) / 100) ** (t / 12) - 1 for t in months]
        # PQ crescita accumulata
        d_cpi_ann = delta_cpi / scale * 12
        d_gdp_ann = delta_gdp / scale * 12
        pq_cum    = [(1 + (d_cpi_ann + d_gdp_ann) / 100) ** (t / 12) - 1 for t in months]

        fig_mvpq.add_trace(go.Scatter(
            x=months, y=[v * 100 for v in mv_cum],
            name="M·V cumulata",
            line=dict(color="#1f77b4", width=2.5),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.08)",
            hovertemplate="Mese %{x}: M·V = %{y:+.2f}%<extra></extra>",
        ))
        fig_mvpq.add_trace(go.Scatter(
            x=months, y=[v * 100 for v in pq_cum],
            name="P·Q cumulata",
            line=dict(color="#d62728", width=2.5),
            fill="tozeroy", fillcolor="rgba(214,39,40,0.08)",
            hovertemplate="Mese %{x}: P·Q = %{y:+.2f}%<extra></extra>",
        ))
        gap_vals = [(mv - pq) * 100 for mv, pq in zip(mv_cum, pq_cum)]
        fig_mvpq.add_trace(go.Bar(
            x=months, y=gap_vals, name="Gap MV−PQ",
            marker_color=["rgba(44,160,44,0.4)" if v > 0 else "rgba(214,39,40,0.35)"
                          for v in gap_vals],
            hovertemplate="Mese %{x}: gap = %{y:+.2f} pp<extra></extra>",
        ))
        fig_mvpq.add_hline(y=0, line_color="#555", line_dash="dot", line_width=1)
        fig_mvpq.update_layout(
            title=dict(text=(f"MV = PQ — scenario: M2 +{m2_gr:.1f}% | V {vel_gr:+.1f}% | "
                              f"CPI atteso {d_cpi_ann:+.2f}%/a | PIL atteso {d_gdp_ann:+.2f}%/a  →  "
                              f"Gap MV−PQ = {mvpq_gap:+.2f} pp ({mvpq_label} monetario)"),
                       font=dict(size=10), x=0.01),
            yaxis_title="Crescita % cumulata",
            barmode="overlay",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=9)),
            margin=dict(t=55, b=30, l=55, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8",
        )
        fig_mvpq.update_xaxes(title_text="Mesi", showgrid=True, gridcolor="#e8e8e8")
        fig_mvpq.update_yaxes(showgrid=True, gridcolor="#e8e8e8")

        # ── Frontiera trade-off (Taylor frontier) ─────────────────────────
        # Per ogni livello di tasso: calcola CPI e PIL attesi e plotta
        rates_bps = np.linspace(-300, 500, 80)
        cpi_outcomes = []
        gdp_outcomes = []
        sp_outcomes  = []
        losses_vals  = []

        lambda_pi_ = float(lambda_pi or 1.0)
        lambda_y_  = float(lambda_y  or 0.5)
        pi_target_ = float(pi_target or 2.0)
        cpi_base   = 3.5  # valore corrente stimato

        for r_bps in rates_bps:
            r_pp = r_bps / 100
            d_c = (coefs["beta_oil_cpi"] * (oil_net_chg / 10) * 10 +
                   coefs["beta_gas_cpi"] * (gas_chg / 10) * 10 +
                   coefs["beta_rate_cpi"] * r_pp +
                   coefs["beta_m2_cpi"] * (m2_gr - 4)) * scale
            d_g = (coefs["beta_oil_gdp"] * (oil_net_chg / 10) * 10 +
                   coefs["beta_gas_gdp"] * (gas_chg / 10) * 10 +
                   coefs["beta_rate_gdp"] * r_pp) * scale
            d_s = (coefs["beta_oil_sp"] * (oil_net_chg / 10) * 10 +
                   coefs["beta_rate_sp"] * r_pp) * scale

            pi_exp = cpi_base + d_c
            loss_v = lambda_pi_ * (pi_exp - pi_target_) ** 2 + lambda_y_ * d_g ** 2

            cpi_outcomes.append(cpi_base + d_c)
            gdp_outcomes.append(d_g)
            sp_outcomes.append(d_s)
            losses_vals.append(loss_v)

        # Punto scenario corrente
        curr_cpi_pt = cpi_base + delta_cpi
        curr_gdp_pt = delta_gdp
        curr_loss   = (lambda_pi_ * (curr_cpi_pt - pi_target_) ** 2 +
                        lambda_y_ * curr_gdp_pt ** 2)

        fig_tt = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Trade-off CPI vs PIL (curva di Taylor)",
                             "Funzione di perdita BC vs Δtassi"],
            horizontal_spacing=0.12,
        )

        # Colori della curva in base alla loss
        min_loss = min(losses_vals)
        max_loss = max(losses_vals)
        norm_loss = [(l - min_loss) / max(max_loss - min_loss, 1e-10) for l in losses_vals]
        curve_colors = [
            f"rgba({int(255 * n)},{int(150 * (1-n))},{int(50 * (1-n))},0.8)"
            for n in norm_loss
        ]

        # Plot colorato per livello di loss
        for i in range(len(rates_bps) - 1):
            fig_tt.add_trace(go.Scatter(
                x=[cpi_outcomes[i], cpi_outcomes[i+1]],
                y=[gdp_outcomes[i], gdp_outcomes[i+1]],
                mode="lines", showlegend=False,
                line=dict(color=curve_colors[i], width=3),
                hovertemplate=f"CPI: %{{x:.2f}}%  PIL: %{{y:+.2f}}%  "
                               f"(tasso {rates_bps[i]:+.0f}bps)<extra></extra>",
            ), row=1, col=1)

        # Punto scenario corrente
        fig_tt.add_trace(go.Scatter(
            x=[curr_cpi_pt], y=[curr_gdp_pt],
            mode="markers", name="Scenario corrente",
            marker=dict(size=14, color="#1f77b4", symbol="star",
                        line=dict(width=2, color="white")),
            hovertemplate=f"Scenario: CPI={curr_cpi_pt:.2f}%  PIL={curr_gdp_pt:+.2f}%<extra></extra>",
        ), row=1, col=1)

        # Linea target inflazione
        fig_tt.add_vline(x=pi_target_, line_color="#2ca02c", line_dash="dash",
                          line_width=1.5, row=1, col=1,
                          annotation_text=f"π*={pi_target_}%")

        # Linea zero PIL
        fig_tt.add_hline(y=0, line_color="#888", line_dash="dot", line_width=1, row=1, col=1)

        # Loss function
        fig_tt.add_trace(go.Scatter(
            x=list(rates_bps), y=losses_vals,
            name="Loss function",
            line=dict(color="#9467bd", width=2),
            hovertemplate="Tasso %{x:+.0f}bps: loss=%{y:.4f}<extra></extra>",
        ), row=1, col=2)

        # Minimo
        min_idx = int(np.argmin(losses_vals))
        fig_tt.add_trace(go.Scatter(
            x=[rates_bps[min_idx]], y=[losses_vals[min_idx]],
            mode="markers", name=f"Minimo ({rates_bps[min_idx]:+.0f}bps)",
            marker=dict(size=12, color="#d62728", symbol="star",
                        line=dict(width=2, color="white")),
            hovertemplate=f"Ottimo: {rates_bps[min_idx]:+.0f}bps  loss={losses_vals[min_idx]:.4f}<extra></extra>",
        ), row=1, col=2)

        # Scenario corrente sulla loss
        fig_tt.add_trace(go.Scatter(
            x=[rate_chg * 100], y=[curr_loss],
            mode="markers", name="Loss scenario corrente",
            marker=dict(size=10, color="#1f77b4", symbol="diamond",
                        line=dict(width=2, color="white")),
            hovertemplate=f"Corrente: {rate_chg*100:+.0f}bps  loss={curr_loss:.4f}<extra></extra>",
        ), row=1, col=2)

        fig_tt.update_layout(
            title=dict(
                text=(f"Frontiera di Taylor — λπ={lambda_pi_}  λy={lambda_y_}  π*={pi_target_}%  |  "
                       f"★ ottimo stimato = {rates_bps[min_idx]:+.0f} bps"),
                font=dict(size=11), x=0.01
            ),
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=9),
                        bgcolor="rgba(255,255,255,0.85)"),
            margin=dict(t=65, b=40, l=55, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8",
        )
        fig_tt.update_xaxes(title_text="CPI atteso (%)", showgrid=True,
                              gridcolor="#e8e8e8", row=1, col=1)
        fig_tt.update_yaxes(title_text="PIL atteso (pp)", showgrid=True,
                              gridcolor="#e8e8e8", row=1, col=1)
        fig_tt.update_xaxes(title_text="Δ tassi (bps)", showgrid=True,
                              gridcolor="#e8e8e8", row=1, col=2)
        fig_tt.update_yaxes(title_text="Loss BC", showgrid=True,
                              gridcolor="#e8e8e8", row=1, col=2)

        return results_panel, optimal_panel, fig_macro, fig_mvpq, fig_tt


# =============================================================================
# ADL SHOCK — callback per il subtab 🔬 ADL Shock
# Popola il pannello X e stima il modello ADL usando i dati shock (FRED + EUR)
# =============================================================================

def _shock_adl_combined(shock_data, eur_data):
    """Unisce store-shock e store-shock-eur in un unico DataFrame mensile."""
    frames = []
    if shock_data:
        df_s = pd.read_json(io.StringIO(shock_data), orient="split")
        df_s.index = pd.to_datetime(df_s.index)
        frames.append(df_s.resample("MS").last())
    if eur_data:
        df_e = pd.read_json(io.StringIO(eur_data), orient="split")
        df_e.index = pd.to_datetime(df_e.index)
        frames.append(df_e.resample("MS").last())
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.sort_index()


def _build_x_panel(cols, active_type, tr_type, lags_type):
    """Costruisce il pannello variabili X con checkbox + trasf. + lag."""
    TR_OPTS  = [{"label": "liv",  "value": "levels"},
                {"label": "yoy",  "value": "yoy"},
                {"label": "log",  "value": "log"},
                {"label": "Δlog", "value": "dlog"}]
    LAG_OPTS = [{"label": f"L{k}", "value": k} for k in range(13)]
    rows = []
    for col in cols:
        short = col[:28] + "…" if len(col) > 28 else col
        rows.append(html.Div([
            dcc.Checklist(
                id={"type": active_type, "index": col},
                options=[{"label": f" {short}", "value": col}],
                value=[],
                style={"font-size": "9px", "flex": "1"},
                inputStyle={"margin-right": "3px"},
            ),
            dcc.Dropdown(
                id={"type": tr_type, "index": col},
                options=TR_OPTS, value="yoy", clearable=False,
                style={"font-size": "8px", "width": "52px"},
            ),
            dcc.Dropdown(
                id={"type": lags_type, "index": col},
                options=LAG_OPTS, value=[0], multi=True, clearable=False,
                style={"font-size": "8px", "width": "80px"},
            ),
        ], style={"display": "flex", "align-items": "center", "gap": "4px",
                  "margin-bottom": "4px", "padding": "2px 4px",
                  "background": "#f8f9fa", "border-radius": "3px"}))
    return rows


@app.callback(
    Output("shock-adl-y",            "options"),
    Output("shock-adl-y",            "value"),
    Output("shock-adl-x-panel",      "children"),
    Output("shock-adl-source-label", "children"),
    Output("shock-adl-slider",       "min"),
    Output("shock-adl-slider",       "max"),
    Output("shock-adl-slider",       "value"),
    Output("shock-adl-slider",       "marks"),
    Input("store-shock",             "data"),
    Input("store-shock-eur",         "data"),
    prevent_initial_call=False,
)
def shock_adl_populate(shock_data, eur_data):
    df = _shock_adl_combined(shock_data, eur_data)
    _empty_slider = (0, 1, [0, 1], {})
    if df.empty:
        no_data = html.Div("— carica dati FRED o Eurostat —",
                           style={"font-size": "9px", "color": "#aaa",
                                  "font-style": "italic"})
        return [], None, no_data, "— nessun dato caricato —", *_empty_slider

    cols = list(df.columns)
    opts = [{"label": c, "value": c} for c in cols]
    d1 = df.index.min().strftime("%m/%Y")
    d2 = df.index.max().strftime("%m/%Y")
    src_label = f"{len(cols)} serie  |  {d1} → {d2}"

    rows = _build_x_panel(cols, "sadl-x-active", "sadl-x-tr", "sadl-x-lags")
    mn, mx, val, marks = _slider_params(df)
    return opts, cols[0] if cols else None, rows, src_label, mn, mx, val, marks


@app.callback(
    Output("shock-adl-slider-label", "children"),
    Input("shock-adl-slider",        "value"),
)
def shock_adl_slider_lbl(val):
    if not val or val[1] <= val[0]:
        return ""
    s = pd.Timestamp(val[0], unit="s").strftime("%b %Y")
    e = pd.Timestamp(val[1], unit="s").strftime("%b %Y")
    return f"📅  {s}  →  {e}"


@app.callback(
    Output("shock-adl-equation",      "children"),
    Output("shock-adl-stats",         "children"),
    Output("shock-adl-coef",          "children"),
    Output("chart-shock-adl-fit",     "figure"),
    Output("chart-shock-adl-irf",     "figure"),
    Output("chart-shock-adl-resid",   "figure"),
    Output("shock-adl-status",        "children"),
    Input("btn-run-shock-adl",        "n_clicks"),
    State("store-shock",              "data"),
    State("store-shock-eur",          "data"),
    State("shock-adl-y",             "value"),
    State("shock-adl-y-tr",          "value"),
    State("shock-adl-ar",            "value"),
    State({"type": "sadl-x-active",  "index": ALL}, "value"),
    State({"type": "sadl-x-active",  "index": ALL}, "id"),
    State({"type": "sadl-x-tr",      "index": ALL}, "value"),
    State({"type": "sadl-x-lags",    "index": ALL}, "value"),
    State("shock-adl-cov",           "value"),
    State("shock-adl-const",         "value"),
    State("shock-adl-slider",        "value"),
    prevent_initial_call=True,
)
def run_shock_adl(n_clicks, shock_data, eur_data, y_col, y_tr, ar_lags,
                  x_active, x_ids, x_trs, x_lags_list,
                  cov_type, add_const, slider_val):

    def err8(msg):
        ef = empty_fig(msg)
        return msg, "", "", ef, ef, ef, f"❌ {msg}"

    if not shock_data and not eur_data:
        return err8("Carica i dati FRED o Eurostat prima")
    if not y_col:
        return err8("Seleziona la variabile dipendente Y")

    df = _shock_adl_combined(shock_data, eur_data)
    if df.empty:
        return err8("DataFrame combinato vuoto")

    # ── Filtro periodo slider ─────────────────────────────────────────────────
    if slider_val and len(slider_val) == 2 and slider_val[1] > slider_val[0]:
        t0 = pd.Timestamp(slider_val[0], unit="s").normalize()
        t1 = pd.Timestamp(slider_val[1], unit="s").normalize()
        df = df.loc[t0:t1]

    if y_col not in df.columns:
        return err8(f"Colonna Y '{y_col}' non trovata")

    # ── Trasforma Y ──────────────────────────────────────────────────────────
    y_series = _apply_transform(df[y_col], y_tr or "yoy").dropna()
    y_label  = f"{'ΔYoY' if y_tr=='yoy' else y_tr}({y_col})"

    # ── Costruisce dizionario X attive ────────────────────────────────────────
    active_x = {}
    for chk, xid, xtr, xlags in zip(x_active, x_ids, x_trs, x_lags_list):
        col = xid["index"]
        if chk and col in df.columns:
            active_x[col] = (xtr or "yoy", xlags or [0])

    if not active_x and not ar_lags:
        return err8("Seleziona almeno una variabile X o un lag AR")

    # ── Allinea indice temporale ──────────────────────────────────────────────
    combined = pd.DataFrame({"__y__": y_series})
    for col, (xtr, xlags) in active_x.items():
        x_base = _apply_transform(df[col], xtr).dropna()
        for lag in sorted(set(xlags)):
            col_name = col if lag == 0 else f"{col}_L{lag}"
            combined[col_name] = x_base.shift(lag)

    ar_lags = sorted(set(ar_lags or []))
    for k in ar_lags:
        combined[f"Y(t-{k})"] = combined["__y__"].shift(k)

    combined = combined.dropna()
    if len(combined) < 10:
        return err8(f"Osservazioni insufficienti dopo NA: {len(combined)}")

    # ── OLS ──────────────────────────────────────────────────────────────────
    y_vec = combined["__y__"]
    X_cols = [c for c in combined.columns if c != "__y__"]
    X_mat  = combined[X_cols].copy()
    if add_const and "const" in (add_const or []):
        X_mat = sm.add_constant(X_mat, has_constant="add")

    try:
        res = sm.OLS(y_vec, X_mat).fit()
        cov = cov_type or "HC3"
        if cov == "HAC":
            rob = res.get_robustcov_results(cov_type="HAC",
                                             maxlags=int(len(combined)**0.25))
        elif cov == "HC3":
            rob = res.get_robustcov_results(cov_type="HC3")
        else:
            rob = res

        # Estrai sempre come pandas Series indicizzate per nome variabile
        param_names = X_mat.columns.tolist()
        _p   = rob.params
        _pv  = rob.pvalues
        _tv  = rob.tvalues
        _bse = rob.bse
        params  = pd.Series(_p   if hasattr(_p,  "index") else _p,   index=param_names)
        pvalues = pd.Series(_pv  if hasattr(_pv, "index") else _pv,  index=param_names)
        tvalues = pd.Series(_tv  if hasattr(_tv, "index") else _tv,  index=param_names)
        bse     = pd.Series(_bse if hasattr(_bse,"index") else _bse, index=param_names)
        model   = res   # usiamo res per rsquared, fvalue, aic, bic, fittedvalues, resid
    except Exception as e:
        return err8(f"Errore OLS: {e}")

    # ── Equazione ────────────────────────────────────────────────────────────
    terms = []
    for v, cv in params.items():
        if v == "const":
            terms.append(f"  α = {cv:+.6f}")
        else:
            terms.append(f"  {'+'if cv>=0 else '−'} {abs(cv):.6f} · {v}")
    equation = f"{y_label} =\n" + "\n".join(terms) + "\n  + ε"

    # ── Helper significatività ────────────────────────────────────────────────
    def pstar(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "·"
        return ""

    # ── Diagnostiche complete ─────────────────────────────────────────────────
    try:
        dw = float(sm.stats.stattools.durbin_watson(model.resid))
    except Exception:
        dw = float("nan")
    try:
        jb_stat, jb_p, jb_skew, jb_kurt = sm.stats.stattools.jarque_bera(model.resid)
    except Exception:
        jb_stat = jb_p = jb_skew = jb_kurt = float("nan")
    try:
        bp_lm, bp_p, *_ = sm.stats.diagnostic.het_breuschpagan(
            model.resid, model.model.exog)
    except Exception:
        bp_lm = bp_p = float("nan")
    try:
        cond_num = float(np.linalg.cond(X_mat.values.astype(float)))
    except Exception:
        cond_num = float("nan")

    cov_label = {"nonrobust": "OLS classico",
                 "HC3": "HC3 eteroschedasticità",
                 "HAC": "HAC Newey-West"}.get(cov, cov)

    # Interpretazione Durbin-Watson
    if dw < 1.5:
        dw_note = "⚠ autocorr. positiva"
    elif dw > 2.5:
        dw_note = "⚠ autocorr. negativa"
    else:
        dw_note = "✓ nessuna autocorrelazione"

    # Interpretazione JB
    jb_note = (f"p={jb_p:.4e} {pstar(jb_p)} | "
               f"asimmetria={jb_skew:.3f} | curtosi={jb_kurt:.3f}")
    jb_diag = ("✓ residui normali" if jb_p > 0.05
                else "⚠ residui non normali")

    stats_rows = [
        ["Statistica", "Valore", "Interpretazione"],
        ["N osservazioni",  str(len(combined)),              ""],
        ["Std. Error",      cov_label,                       ""],
        ["R²",              f"{model.rsquared:.6f}",         "varianza spiegata"],
        ["R² adj.",         f"{model.rsquared_adj:.6f}",     "penalizza parametri"],
        ["F-stat",          f"{model.fvalue:.4f}",
         f"p={model.f_pvalue:.4e} {pstar(model.f_pvalue)}"],
        ["AIC",             f"{model.aic:.4f}",              "↓ migliore"],
        ["BIC",             f"{model.bic:.4f}",              "↓ migliore"],
        ["Log-likelihood",  f"{model.llf:.4f}",              "↑ migliore"],
        ["Durbin-Watson",   f"{dw:.4f}",                     dw_note],
        ["Jarque-Bera",     f"{jb_stat:.4f}",                jb_note],
        ["  → test normalità", jb_diag,                      "H₀: residui normali"],
        ["  → Asimmetria",  f"{jb_skew:.4f}",
         "~0 = sim.; >0 coda dx; <0 coda sx"],
        ["  → Curtosi",     f"{jb_kurt:.4f}",
         "normale=3; >3 leptocurtica; <3 platicurtica"],
        ["Breusch-Pagan",   f"{bp_lm:.4f}",
         f"p={bp_p:.4e} {pstar(bp_p)} — H₀: omoschedasticità"],
        ["Cond. number",    f"{cond_num:.2f}",
         ">30 multicollinearità; >1000 grave"],
    ]
    stats_table = _make_stat_table(stats_rows, "#1a3a5c")

    # ── Tabella coefficienti con IC95 e VIF ───────────────────────────────────
    try:
        conf = model.conf_int(alpha=0.05)
        # conf può essere ndarray — riconvertiamo
        if not hasattr(conf, "loc"):
            conf = pd.DataFrame(conf, index=params.index, columns=[0, 1])
    except Exception:
        conf = pd.DataFrame({"0": params * np.nan, "1": params * np.nan})

    x_cols_vif = [c for c in X_mat.columns if c != "const"]
    vif_dict = {}
    if len(x_cols_vif) > 1:
        for xc in x_cols_vif:
            other = [c for c in x_cols_vif if c != xc]
            try:
                r2v = sm.OLS(X_mat[xc],
                             sm.add_constant(X_mat[other])).fit().rsquared
                vif_dict[xc] = 1 / (1 - r2v) if r2v < 1 else np.inf
            except Exception:
                vif_dict[xc] = np.nan
    else:
        vif_dict = {c: np.nan for c in x_cols_vif}

    coef_rows = [["Variabile", "Coeff.", "Std Err", "t-stat",
                   "p-val", "Sig.", "IC95 inf", "IC95 sup", "VIF"]]
    for v in params.index:
        p   = pvalues[v]
        vif = vif_dict.get(v, np.nan)
        vif_s = f"{vif:.2f}" if (isinstance(vif, float) and not np.isnan(vif)) else "—"
        try:
            ic_lo = f"{conf.loc[v, 0]:.6f}"
            ic_hi = f"{conf.loc[v, 1]:.6f}"
        except Exception:
            ic_lo = ic_hi = "—"
        coef_rows.append([
            v,
            f"{params[v]:+.6f}",
            f"{bse[v]:.6f}",
            f"{tvalues[v]:.4f}",
            f"{p:.4e}",
            pstar(p),
            ic_lo, ic_hi, vif_s,
        ])
    coef_table = html.Div([
        html.Div("Coefficienti di regressione",
                 style={"font-size": "11px", "font-weight": "bold",
                        "color": "#1a3a5c", "background": "#eaf4fb",
                        "padding": "5px 10px",
                        "border-radius": "4px 4px 0 0",
                        "border": "1px solid #aed6f1",
                        "border-bottom": "none", "margin-top": "10px"}),
        _make_stat_table(coef_rows, "#2e6da4"),
        html.Div("*** p<0.001  ** p<0.01  * p<0.05  · p<0.10",
                 style={"font-size": "10px", "color": "#777",
                        "margin-top": "4px", "font-style": "italic"}),
    ])

    # ── Grafico fit ───────────────────────────────────────────────────────────
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(x=y_vec.index, y=y_vec.values,
                                  name=y_label, line=dict(color="#1f77b4", width=1.5)))
    fig_fit.add_trace(go.Scatter(x=y_vec.index, y=model.fittedvalues.values,
                                  name="Fitted", line=dict(color="#d62728", width=1.5,
                                                            dash="dot")))
    fig_fit.add_hline(y=0, line_dash="solid", line_color="#999", line_width=0.8)
    fig_fit.update_layout(
        title=dict(text=f"ADL Shock — Fit  |  R²={model.rsquared:.4f}  R²adj={model.rsquared_adj:.4f}",
                   font=dict(size=11), x=0.01),
        hovermode="x unified", margin=dict(t=45, b=30, l=55, r=20),
        paper_bgcolor="white", plot_bgcolor="#f8f8f8",
        legend=dict(orientation="h", y=1.02, font=dict(size=9)),
    )

    # ── IRF dinamica ─────────────────────────────────────────────────────────
    H_IRF = 12
    _PCT_TR = {"yoy", "ldiff", "log_diff", "pct", "pct_change", "dlog"}
    irf_scale  = 100.0 if (y_tr or "yoy") in _PCT_TR else 1.0
    irf_suffix = "%" if irf_scale == 100.0 else ""

    ar_coefs = {}
    for v in params.index:
        if v.startswith("Y(t-"):
            try:
                ar_coefs[int(v[4:-1])] = float(params[v])
            except ValueError:
                pass

    x_groups = {}
    for v in params.index:
        if v == "const" or v.startswith("Y(t-"):
            continue
        if "_L" in v:
            parts = v.rsplit("_L", 1)
            try:
                base, lag = parts[0], int(parts[1])
            except ValueError:
                base, lag = v, 0
        else:
            base, lag = v, 0
        x_groups.setdefault(base, []).append((lag, float(params[v])))

    x_sigmas = {c: float(combined[c].std())
                for c in combined.columns if c != "__y__"}

    n_g = len(x_groups)
    if n_g == 0:
        fig_irf = empty_fig("Nessuna variabile X (solo componente AR)")
    else:
        cpr = min(3, n_g)
        nri = (n_g + cpr - 1) // cpr
        sp_titles = []
        for base in x_groups:
            sigma_b = next((x_sigmas.get(k) for k in combined.columns
                            if k != "__y__" and (k == base or
                               k.startswith(base + "_L"))), None)
            slbl = (f"  [σ={sigma_b*100:.2f}%]" if sigma_b is not None and irf_scale == 100.0
                    else (f"  [σ={sigma_b:.4f}]" if sigma_b is not None else ""))
            sp_titles.append(f"{base[:24]}{slbl}")

        fig_irf = make_subplots(rows=nri, cols=cpr, subplot_titles=sp_titles,
                                 vertical_spacing=0.14, horizontal_spacing=0.08)
        fmt = (lambda v: f"{v:+.2f}%") if irf_suffix else (lambda v: f"{v:+.4f}")

        for gi, (base, lc) in enumerate(x_groups.items()):
            ri = gi // cpr + 1
            ci = gi % cpr + 1
            color = COLORS[gi % len(COLORS)]
            sigma = next((x_sigmas.get(k, 1.0) for k in combined.columns
                          if k != "__y__" and (k == base or
                             k.startswith(base + "_L"))), 1.0)
            direct = {lag: coef * sigma for lag, coef in lc}

            irf_vals = []
            for h in range(H_IRF):
                d = direct.get(h, 0.0)
                ar_part = sum(ar_coefs.get(j, 0.0) * irf_vals[h - j]
                              for j in ar_coefs if 0 < j <= h)
                irf_vals.append(d + ar_part)

            horizons  = list(range(H_IRF))
            irf_sc    = [v * irf_scale for v in irf_vals]
            cum_sc    = list(np.cumsum(irf_sc))

            fig_irf.add_trace(go.Bar(
                x=horizons, y=irf_sc, name=base,
                marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in irf_sc],
                marker_line_width=0, showlegend=False,
                text=[fmt(v) for v in irf_sc],
                textposition="outside", textfont=dict(size=7)),
                row=ri, col=ci)
            fig_irf.add_trace(go.Scatter(
                x=horizons, y=cum_sc, mode="lines+markers",
                name=f"Cum. {base}",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=5), showlegend=False),
                row=ri, col=ci)
            fig_irf.add_hline(y=0, line_dash="solid", line_color="#bbb",
                               line_width=0.8, row=ri, col=ci)

        fig_irf.update_layout(
            title=dict(text=f"IRF — Impatto di +1σ per variabile  |  Y: {y_label}",
                       font=dict(size=11), x=0.01),
            height=max(280, 230 * nri),
            margin=dict(t=50, b=30, l=45, r=15),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8",
        )
        fig_irf.update_xaxes(title_text="Mesi", showgrid=True, gridcolor="#e8e8e8")
        fig_irf.update_yaxes(title_text=f"Δ Y ({irf_suffix or 'unità'})",
                              showgrid=True, gridcolor="#e8e8e8")

    # ── Residui ───────────────────────────────────────────────────────────────
    resid = model.resid
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(x=resid.index, y=resid.values,
                                    mode="lines",
                                    line=dict(color="#7f7f7f", width=1),
                                    name="Residui"))
    fig_resid.add_hline(y=0, line_dash="solid", line_color="#aaa", line_width=0.8)
    fig_resid.update_layout(
        title=dict(text="Residui", font=dict(size=11), x=0.01),
        margin=dict(t=40, b=30, l=55, r=20),
        paper_bgcolor="white", plot_bgcolor="#f8f8f8",
    )

    status = (f"✅  N={len(combined)}  |  R²={model.rsquared:.4f}  "
              f"R²adj={model.rsquared_adj:.4f}  |  DW={dw:.3f}  |  "
              f"JB p={jb_p:.3e} sk={jb_skew:.2f} ku={jb_kurt:.2f}  |  {cov_label}")
    return equation, stats_table, coef_table, fig_fit, fig_irf, fig_resid, status


register_shock_callbacks(app)


# =============================================================================
# MAIN
# =============================================================================



def register_new_tab_callbacks(app):
    """
    Registra tutti i callback dei tre nuovi tab:
      - ARIMA/SARIMA (arima_populate, arima_slider_label, run_arima)
      - ADL          (adl_populate,   adl_slider_label,   run_adl)
      - DSGE         (run_dsge)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import acf as sm_acf, pacf as sm_pacf
    from scipy import stats as sp_stats

    # =========================================================================
    # ARIMA / SARIMA  — callbacks (workflow Box-Jenkins 4 passi)
    # =========================================================================
    import statsmodels.api as sm2
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import acf as _acf_fn, pacf as _pacf_fn, adfuller
    from scipy import stats as sp_stats
    from scipy import signal as sp_signal

    # ── helper: grafico in-sample fit ────────────────────────────────────────
    def _make_insample_fig(series, fit, cond_vol, DL):
        fig = go.Figure()
        fig.add_scatter(x=series.index, y=series.values, mode="lines",
                        name="Osservato", line=dict(color="#1565c0", width=1.5))
        fig.add_scatter(x=fit.index, y=fit.values, mode="lines",
                        name="Stimato ARIMA", line=dict(color="#2e7d32", width=1.2, dash="dot"))
        if cond_vol is not None:
            cv = cond_vol.reindex(fit.index).ffill().bfill()
            hi = (fit + 1.96 * cv).values
            lo = (fit - 1.96 * cv).values
            fig.add_scatter(
                x=list(fit.index) + list(fit.index[::-1]),
                y=list(hi) + list(lo[::-1]),
                fill="toself", fillcolor="rgba(103,58,183,.12)",
                line=dict(color="rgba(0,0,0,0)"), name="±1.96σ GARCH")
        fig.update_layout(**DL, height=240,
                          title=dict(text="In-sample fit  —  osservato vs stimato ARIMA"
                                         + ("  +  banda ±1.96σ GARCH" if cond_vol is not None else ""),
                                     font=dict(size=10, color="#1a3a5c"), x=0.01),
                          legend=dict(orientation="h", y=1.18, font=dict(size=8)))
        return fig

    # ── toggle geo dropdown (ARIMA) ──────────────────────────────────────────
    @app.callback(
        Output("arima-geo-wrapper", "style"),
        Input("arima-source-type",  "value"),
    )
    def arima_toggle_geo(source_type):
        base = {"margin-bottom": "6px"}
        return base if source_type == "eur" else {**base, "display": "none"}

    # ── Clientside: segnala inizio caricamento ARIMA ─────────────────────────
    app.clientside_callback(
        """
        function(n, source_type, geo) {
            if (!n) return window.dash_clientside.no_update;
            var src = source_type === "eur"
                ? "Eurostat \u2014 " + (geo || "EA20")
                : "FRED \u2014 USA";
            return {"active": true, "src": src, "source_type": source_type || "usa"};
        }
        """,
        Output("store-arima-loading-state", "data"),
        Input("btn-arima-load",    "n_clicks"),
        State("arima-source-type", "value"),
        State("arima-eur-geo",     "value"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("arima-loading-overlay", "style"),
        Output("arima-loading-title",   "children"),
        Output("arima-loading-source",  "children"),
        Output("arima-progress-bar",    "style"),
        Output("arima-progress-pct",    "style"),
        Output("arima-progress-tick",   "disabled"),
        Output("arima-progress-tick",   "n_intervals"),
        Input("store-arima-loading-state", "data"),
        prevent_initial_call=True,
    )
    def arima_toggle_overlay(state):
        _hidden   = {"display": "none"}
        _bar_red  = {"width": "0%", "height": "100%",
                     "background": "linear-gradient(90deg,#7b0000,#e53935)",
                     "border-radius": "6px", "transition": "width 0.3s ease"}
        _bar_blue = {"width": "0%", "height": "100%",
                     "background": "linear-gradient(90deg,#1a5276,#2e86c1)",
                     "border-radius": "6px", "transition": "width 0.3s ease"}
        _pct_red  = {"font-size": "28px", "font-weight": "bold",
                     "color": "#ef9a9a", "margin-bottom": "6px"}
        _pct_blue = {"font-size": "28px", "font-weight": "bold",
                     "color": "#90caf9", "margin-bottom": "6px"}
        if state and state.get("active"):
            overlay_style = {
                "display": "flex", "position": "fixed",
                "top": "0", "left": "0", "width": "100%", "height": "100%",
                "background": "rgba(0,0,0,0.75)", "z-index": "9999",
                "align-items": "center", "justify-content": "center",
            }
            is_eur = state.get("source_type") == "eur"
            bar_style = _bar_blue if is_eur else _bar_red
            pct_style = _pct_blue if is_eur else _pct_red
            return overlay_style, "Caricamento dati in corso...", state.get("src", ""), bar_style, pct_style, False, 0
        return _hidden, "", "", _bar_red, _pct_red, True, 0

    @app.callback(
        Output("arima-progress-pct",    "children"),
        Output("arima-progress-bar",    "style", allow_duplicate=True),
        Output("arima-progress-detail", "children"),
        Input("arima-progress-tick",    "n_intervals"),
        State("store-arima-loading-state", "data"),
        prevent_initial_call=True,
    )
    def arima_tick_progress(n, state):
        is_eur = state and state.get("source_type") == "eur"
        grad = ("linear-gradient(90deg,#1a5276,#2e86c1)" if is_eur
                else "linear-gradient(90deg,#7b0000,#e53935)")
        pct = min(int(95 * (1 - math.exp(-n * 0.09))), 93)
        if pct < 20:
            detail = "Connessione ai server..."
        elif pct < 45:
            detail = "Download serie in corso..."
        elif pct < 70:
            detail = "Elaborazione dati temporali..."
        else:
            detail = "Quasi pronto..."
        bar_style = {"width": f"{pct}%", "height": "100%", "background": grad,
                     "border-radius": "6px", "transition": "width 0.3s ease"}
        return f"{pct}%", bar_style, detail

    # ── carica dati ARIMA (USA o Eurostat) ───────────────────────────────────
    @app.callback(
        Output("store-arima-source",        "data"),
        Output("arima-source-status",       "children"),
        Output("store-arima-loading-state", "data", allow_duplicate=True),
        Input("btn-arima-load",             "n_clicks"),
        State("arima-source-type",          "value"),
        State("arima-eur-geo",              "value"),
        State("api-key",                    "value"),
        prevent_initial_call=True,
    )
    def load_arima_source(n_clicks, source_type, geo, api_key):
        _done = {"active": False}
        def _err(msg):
            return None, msg, _done

        if not _has_internet():
            return _err("⚠️  Nessuna connessione internet")

        if source_type == "eur":
            geo = geo or "EA20"
            print(f"\n▶ ARIMA — Download Eurostat [{geo}]...")
            df = build_eurostat_dataframe(geo)
            if df.empty:
                return _err("❌ Download Eurostat fallito — server non raggiungibile")
            # Esportazioni nette
            exp_n = next((c for c in df.columns if "Esportazioni EUR Nom." in c), None)
            imp_n = next((c for c in df.columns if "Importazioni EUR Nom." in c), None)
            if exp_n and imp_n:
                df[NET_EXP_EUR_LABEL] = df[exp_n] - df[imp_n]
            exp_r = next((c for c in df.columns if "Esportazioni EUR Reali" in c), None)
            imp_r = next((c for c in df.columns if "Importazioni EUR Reali" in c), None)
            if exp_r and imp_r:
                df[NET_EXP_EUR_R_LABEL] = df[exp_r] - df[imp_r]
            # Brent
            _ak = (api_key or FRED_API_KEY).strip()
            brent = fred_get("MCOILBRENTEU", _ak)
            if brent is not None:
                df["Brent Petrolio ($/barile)"] = to_monthly(brent, "M").reindex(
                    pd.date_range(df.index.min(), df.index.max(), freq="MS")
                ).ffill()
            # Indice azionario
            eq_info = EUROSTAT_EQUITY.get(geo)
            if eq_info:
                eq_ticker, eq_name = eq_info
                eq_s = yfinance_monthly(eq_ticker, eq_name)
                if eq_s is not None:
                    df[eq_name] = eq_s.reindex(
                        pd.date_range(df.index.min(), df.index.max(), freq="MS")
                    ).ffill()
            label = EUROSTAT_GEO.get(geo, geo)
        else:
            api_key = (api_key or FRED_API_KEY).strip()
            print(f"\n▶ ARIMA — Download FRED USA...")
            df = build_dataframe(ADL_USA_SERIES, api_key)
            if df.empty:
                return _err("❌ Download FRED fallito — verifica API key o connessione")
            exp_col = next((c for c in df.columns if "Esportazioni USA" in c), None)
            imp_col = next((c for c in df.columns if "Importazioni USA" in c), None)
            if exp_col and imp_col:
                df[ADL_NET_EXP_LABEL] = df[exp_col] - df[imp_col]
            label = "USA (FRED)"

        d1  = df.index.min().strftime("%m/%Y")
        d2  = df.index.max().strftime("%m/%Y")
        msg = f"✅  {label}  |  {len(df.columns)} serie  |  {d1} → {d2}"
        return df.to_json(date_format="iso", orient="split"), msg, _done

    # ── populate dropdown + slider ────────────────────────────────────────────
    @app.callback(
        Output("arima-y-var",  "options"),
        Output("arima-slider", "min"),
        Output("arima-slider", "max"),
        Output("arima-slider", "value"),
        Output("arima-slider", "marks"),
        Input("store-data",         "data"),
        Input("store-gdp",          "data"),
        Input("store-yields",       "data"),
        Input("store-shock",        "data"),
        Input("store-arima-source", "data"),
        State("arima-source-type",  "value"),
        prevent_initial_call=False,
    )
    def arima_populate(mon, gdp, yld, shk, arima_src, source_type):
        if source_type == "eur" and arima_src:
            df = _all_series_with_shock(None, None, None, None, data_eur=arima_src)
        else:
            df = _all_series_with_shock(mon, gdp, yld, shk)
        if df.empty:
            return [], 0, 1, [0, 1], {}
        cols = sorted(df.columns.tolist())
        mn, mx, val, marks = _slider_params(df)
        return [{"label": c, "value": c} for c in cols], mn, mx, val, marks

    @app.callback(
        Output("arima-slider-label", "children"),
        Input("arima-slider",        "value"),
    )
    def arima_slider_label(val):
        if not val or (val[1] - val[0]) < 86400:
            return ""
        s = pd.to_datetime(val[0], unit="s").strftime("%b %Y")
        e = pd.to_datetime(val[1], unit="s").strftime("%b %Y")
        return f"📅  {s}  →  {e}"

    # ── toggle stagionale ─────────────────────────────────────────────────────
    @app.callback(
        Output("arima-seasonal-params", "style"),
        Input("arima-seasonal-on",      "value"),
    )
    def arima_toggle_seasonal(val):
        base = {"margin-top": "6px"}
        return base if val else {**base, "display": "none"}

    # ── PASSO 1: trasformazione + detrend ────────────────────────────────────
    @app.callback(
        Output("arima-step1-charts",  "children"),
        Output("store-arima-step1",   "data"),
        Output("arima-step1-status",  "children"),
        Input("btn-arima-step1",      "n_clicks"),
        State("arima-y-var",          "value"),
        State("arima-step1-transform","value"),
        State("arima-step1-detrend",  "value"),
        State("arima-slider",         "value"),
        State("store-data",           "data"),
        State("store-gdp",            "data"),
        State("store-yields",         "data"),
        State("store-shock",          "data"),
        State("store-arima-source",   "data"),
        State("arima-source-type",    "value"),
        prevent_initial_call=True,
    )
    def arima_step1(n_clicks, y_var, transform, detrend,
                    slider_val, mon, gdp, yld, shk,
                    arima_src, source_type):
        _err = lambda m: (None, None, f"❌ {m}")
        if not y_var:
            return _err("Seleziona una serie.")
        if source_type == "eur" and arima_src:
            df = _all_series_with_shock(None, None, None, None, data_eur=arima_src)
        else:
            df = _all_series_with_shock(mon, gdp, yld, shk)
        if df.empty or y_var not in df.columns:
            return _err("Serie non disponibile — clicca 🔄 Carica dati per scaricare la fonte selezionata.")

        # Slicing periodo
        if slider_val and (slider_val[1] - slider_val[0]) > 86400:
            s0 = pd.to_datetime(slider_val[0], unit="s").normalize()
            s1 = pd.to_datetime(slider_val[1], unit="s").normalize()
            df = df.loc[s0:s1]

        orig = df[y_var].dropna()
        if len(orig) < 24:
            return _err("Osservazioni insufficienti (< 24). Espandi il periodo.")

        # Trasformazione
        transform = transform or "log"
        if transform == "log":
            if (orig <= 0).any():
                return _err("La serie contiene valori ≤ 0: impossibile applicare il logaritmo.")
            trans = np.log(orig)
            trans_lbl = f"ln({y_var})"
        elif transform == "diff":
            trans = orig.diff().dropna()
            trans_lbl = f"Δ{y_var}"
        elif transform == "diff_log":
            if (orig <= 0).any():
                return _err("La serie contiene valori ≤ 0.")
            trans = np.log(orig).diff().dropna()
            trans_lbl = f"Δln({y_var})"
        else:
            trans = orig.copy()
            trans_lbl = y_var

        # Detrend
        detrend = detrend or "hp"
        if detrend == "hp" and transform not in ("diff", "diff_log"):
            from statsmodels.tsa.filters.hp_filter import hpfilter
            cycle, trend_s = hpfilter(trans.dropna(), lamb=1600)
            stat = pd.Series(cycle, index=trans.dropna().index)
            trend_s = pd.Series(trend_s, index=trans.dropna().index)
            detrend_lbl = "HP cycle"
        elif detrend == "ma12":
            trend_s = trans.rolling(window=12, center=True).mean()
            stat = trans - trend_s
            detrend_lbl = f"{trans_lbl} − MA12"
        elif detrend == "sdiff":
            stat = trans.diff(12).dropna()
            trend_s = trans
            detrend_lbl = f"Δ₁₂ {trans_lbl}"
        else:
            trend_s = None
            stat = trans.copy()
            detrend_lbl = trans_lbl

        stat = stat.dropna()
        if len(stat) < 12:
            return _err("Serie stazionaria troppo corta dopo detrend. "
                        "Prova un metodo diverso o un periodo più lungo.")

        # ── 4 grafici ─────────────────────────────────────────────────────────
        CFG = {"displayModeBar": False}
        DL = dict(paper_bgcolor="white", plot_bgcolor="#f8f9f9",
                  margin=dict(l=44, r=12, t=32, b=28), height=200,
                  font=dict(size=9, color="#444"),
                  xaxis=dict(showgrid=True, gridcolor="#ebebeb",
                             linecolor="#ccc", tickfont=dict(size=8)),
                  yaxis=dict(showgrid=True, gridcolor="#ebebeb",
                             linecolor="#ccc", tickfont=dict(size=8)))
        def _line(x, y, color, title, dash="solid"):
            fig = go.Figure(go.Scatter(x=x, y=y, mode="lines",
                                       line=dict(color=color, width=1.3, dash=dash)))
            fig.update_layout(**DL, title=dict(text=title, font=dict(size=10, color="#1a3a5c"), x=0.01))
            return fig

        f1 = _line(orig.index, orig.values, "#1565c0", f"Originale: {y_var}")

        f2 = _line(trans.index, trans.values, "#2e7d32", f"Trasformata: {trans_lbl}")

        if trend_s is not None:
            td = trend_s.dropna()
            f3 = go.Figure()
            f3.add_scatter(x=trans.index, y=trans.values, mode="lines",
                           line=dict(color="#aec6e8", width=1, dash="dot"), name="Trasformata")
            f3.add_scatter(x=td.index, y=td.values, mode="lines",
                           line=dict(color="#e65100", width=1.6), name="Trend")
            f3.update_layout(**DL, showlegend=True,
                             legend=dict(orientation="h", y=1.15, font=dict(size=8)),
                             title=dict(text="Trend estratto", font=dict(size=10, color="#1a3a5c"), x=0.01))
        else:
            f3 = _line(trans.index, trans.values, "#e65100", "Trend (nessuno estratto)")

        f4 = _line(stat.index, stat.values, "#880e4f", f"Serie stazionaria: {detrend_lbl}")
        f4.add_hline(y=0, line_dash="dash", line_color="#bbb", line_width=0.8)

        charts = html.Div([
            html.Div([
                dcc.Graph(figure=f1, config=CFG),
                dcc.Graph(figure=f2, config=CFG),
            ], style={"display": "grid", "grid-template-columns": "1fr 1fr", "gap": "6px"}),
            html.Div([
                dcc.Graph(figure=f3, config=CFG),
                dcc.Graph(figure=f4, config=CFG),
            ], style={"display": "grid", "grid-template-columns": "1fr 1fr", "gap": "6px"}),
        ], style={"margin-top": "10px"})

        store = {
            "series_json":  stat.to_json(date_format="iso"),
            "orig_json":    orig.to_json(date_format="iso"),
            "trans_json":   trans.to_json(date_format="iso"),
            "trend_json":   trend_s.dropna().to_json(date_format="iso") if trend_s is not None else None,
            "y_var":        y_var,
            "transform":    transform,
            "detrend":      detrend,
            "trans_lbl":    trans_lbl,
            "detrend_lbl":  detrend_lbl,
            "n":            len(stat),
        }
        status = (f"✅  Trasformazione '{trans_lbl}'  +  detrend '{detrend}' applicati.  "
                  f"{len(stat)} osservazioni nella serie stazionaria.")
        return charts, store, status

    # ── PASSO 2: ACF / PACF / ADF / Periodogramma ────────────────────────────
    @app.callback(
        Output("arima-step2-output", "children"),
        Output("arima-p",            "value"),
        Output("arima-d",            "value"),
        Output("arima-q",            "value"),
        Output("arima-step2-status", "children"),
        Input("btn-arima-step2",     "n_clicks"),
        State("store-arima-step1",   "data"),
        prevent_initial_call=True,
    )
    def arima_step2(n_clicks, step1_data):
        _err = lambda m: (None, dash.no_update, dash.no_update, dash.no_update, f"❌ {m}")
        if not step1_data:
            return _err("Esegui prima il Passo ①.")

        series = pd.read_json(io.StringIO(step1_data["series_json"]), typ="series")
        series = series.sort_index().dropna()
        detrend_lbl = step1_data.get("detrend_lbl", "serie")

        if len(series) < 20:
            return _err("Serie troppo corta per analisi ACF/PACF.")

        nlags = min(40, len(series) // 3)

        # ACF
        acf_vals, acf_ci = _acf_fn(series, nlags=nlags, alpha=0.05, fft=True)
        # PACF
        try:
            pacf_vals, pacf_ci = _pacf_fn(series, nlags=nlags, alpha=0.05, method="ywm")
        except Exception:
            pacf_vals, pacf_ci = _pacf_fn(series, nlags=min(nlags, len(series)//2 - 1),
                                           alpha=0.05)
        # ADF
        adf_result = adfuller(series, autolag="AIC")
        adf_stat, adf_p, adf_lag = adf_result[0], adf_result[1], adf_result[2]
        adf_cv = adf_result[4]

        # Periodogramma
        f_arr, pxx = sp_signal.periodogram(series.values, fs=1.0)
        with np.errstate(divide="ignore"):
            periods = np.where(f_arr > 0, 1.0 / f_arr, np.nan)

        # ── Suggerimento ordini ───────────────────────────────────────────────
        # d: da ADF
        suggested_d = 0 if adf_p < 0.10 else 1

        # p: primo lag PACF fuori dalla banda CI
        pacf_half_ci = np.abs(pacf_ci[:, 1] - pacf_ci[:, 0]) / 2
        sig_pacf = [i for i in range(1, len(pacf_vals))
                    if abs(pacf_vals[i]) > pacf_half_ci[min(i, len(pacf_half_ci)-1)]]
        suggested_p = min(sig_pacf[0] if sig_pacf else 1, 5)

        # q: da ACF — se i lag significativi sono pochi (<= 3) → MA puro
        acf_half_ci = np.abs(acf_ci[:, 1] - acf_ci[:, 0]) / 2
        sig_acf = [i for i in range(1, len(acf_vals))
                   if abs(acf_vals[i]) > acf_half_ci[min(i, len(acf_half_ci)-1)]]
        if len(sig_acf) <= 3:
            suggested_q = len(sig_acf)
        else:
            suggested_q = 0  # ACF lentamente decrescente → AR, non MA

        # ── ADF card ─────────────────────────────────────────────────────────
        if adf_p < 0.05:
            adf_color = "#1b5e20"; adf_bg = "#e8f5e9"
            adf_msg = f"✅  ADF stat={adf_stat:.3f}  p={adf_p:.4f}  →  serie stazionaria  (d = 0 consigliato)"
        elif adf_p < 0.10:
            adf_color = "#e65100"; adf_bg = "#fff3e0"
            adf_msg = f"⚠  ADF stat={adf_stat:.3f}  p={adf_p:.4f}  →  borderline  (d = 0 o d = 1)"
        else:
            adf_color = "#b71c1c"; adf_bg = "#ffebee"
            adf_msg = f"❌  ADF stat={adf_stat:.3f}  p={adf_p:.4f}  →  serie NON stazionaria  (d = 1 consigliato)"

        # ── grafici ──────────────────────────────────────────────────────────
        DL = dict(paper_bgcolor="white", plot_bgcolor="#f8f9f9",
                  margin=dict(l=44, r=12, t=34, b=28), height=210,
                  font=dict(size=9, color="#444"),
                  xaxis=dict(showgrid=True, gridcolor="#ebebeb",
                             linecolor="#ccc", tickfont=dict(size=8)),
                  yaxis=dict(showgrid=True, gridcolor="#ebebeb",
                             linecolor="#ccc", tickfont=dict(size=8)))
        CFG = {"displayModeBar": False}

        def _acf_fig(vals, ci, title, bar_color):
            fig = go.Figure()
            half = np.abs(ci[:, 1] - ci[:, 0]) / 2
            lags = list(range(len(vals)))
            for i, v in enumerate(vals):
                h = half[min(i, len(half)-1)]
                c = "#d62728" if i > 0 and abs(v) > h else bar_color
                fig.add_shape(type="line", x0=i, x1=i, y0=0, y1=v,
                              line=dict(color=c, width=2.5))
            fig.add_scatter(x=lags, y=[h for h in [half[min(i, len(half)-1)] for i in lags]],
                            mode="lines", line=dict(color="#ff7f0e", dash="dash", width=1),
                            showlegend=False)
            fig.add_scatter(x=lags, y=[-half[min(i, len(half)-1)] for i in lags],
                            mode="lines", line=dict(color="#ff7f0e", dash="dash", width=1),
                            showlegend=False)
            fig.add_hline(y=0, line_color="#888", line_width=0.6)
            fig.update_layout(**DL, title=dict(text=title,
                                               font=dict(size=10, color="#1a3a5c"), x=0.01),
                              yaxis_range=[-1.05, 1.05])
            return fig

        f_acf  = _acf_fig(acf_vals,  acf_ci,  f"ACF  —  {detrend_lbl}  (barre rosse = significativo → q)", "#1f77b4")
        f_pacf = _acf_fig(pacf_vals, pacf_ci, f"PACF  —  {detrend_lbl}  (barre rosse = significativo → p)", "#2ca02c")

        # Periodogramma
        mask = (periods >= 2) & (periods <= 72) & np.isfinite(periods)
        fp = go.Figure()
        fp.add_scatter(x=periods[mask], y=pxx[mask], mode="lines",
                       line=dict(color="#9467bd", width=1.2),
                       fill="tozeroy", fillcolor="rgba(148,103,189,.12)")
        # annotate top 3 peaks
        sort_idx = np.argsort(pxx[mask])[::-1][:3]
        per_masked = periods[mask]
        for i in sort_idx:
            fp.add_vline(x=per_masked[i], line_color="#d62728",
                         line_dash="dot", line_width=1)
            fp.add_annotation(x=per_masked[i], y=pxx[mask][i],
                               text=f"{per_masked[i]:.0f}m",
                               font=dict(size=8, color="#d62728"),
                               showarrow=False, yshift=8)
        fp.update_layout(**DL, title=dict(text="Periodogramma  (periodo dominante in mesi)",
                                          font=dict(size=10, color="#1a3a5c"), x=0.01),
                         xaxis_title="Periodo (mesi)", yaxis_title="Potenza spettrale")

        # ── box suggerimenti ──────────────────────────────────────────────────
        def _sug_box(letter, val, descr, color):
            return html.Div([
                html.Div(f"{letter} = {val}",
                         style={"font-size": "22px", "font-weight": "700",
                                "color": color, "text-align": "center"}),
                html.Div(descr,
                         style={"font-size": "9px", "color": "#555",
                                "text-align": "center", "margin-top": "2px"}),
            ], style={"flex": "1", "background": "#f8f9fa", "border": f"2px solid {color}",
                      "border-radius": "6px", "padding": "10px 6px"})

        suggestions = html.Div([
            html.Div("📊  Ordini suggeriti dall'analisi",
                     style={"font-size": "11px", "font-weight": "700",
                            "color": "#1a3a5c", "margin-bottom": "10px"}),
            html.Div([
                _sug_box("p", suggested_p, "lag AR  (da PACF)", "#1f77b4"),
                _sug_box("d", suggested_d, "differenziazioni  (da ADF)", "#2e7d32"),
                _sug_box("q", suggested_q, "lag MA  (da ACF)", "#9467bd"),
            ], style={"display": "flex", "gap": "10px", "margin-bottom": "10px"}),
            html.Div(adf_msg,
                     style={"font-size": "10px", "color": adf_color,
                            "background": adf_bg, "padding": "7px 10px",
                            "border-radius": "4px", "margin-bottom": "8px"}),
            html.Div([
                html.Span("Valori critici ADF:  ", style={"font-size": "9px", "color": "#666"}),
                *[html.Span(f"{k}: {v:.3f}  ", style={"font-size": "9px", "color": "#333"})
                  for k, v in adf_cv.items()],
            ]),
            html.P("→ Gli ordini suggeriti sono stati precompilati nel Passo ③.  "
                   "Puoi modificarli manualmente prima di stimare.",
                   style={"font-size": "9px", "color": "#888",
                          "margin-top": "6px", "margin-bottom": "0"}),
        ], style={"background": "#f0f4fa", "border": "1px solid #c8d8ea",
                  "border-radius": "6px", "padding": "14px", "margin-top": "12px"})

        output = html.Div([
            html.Div([
                dcc.Graph(figure=f_acf,  config=CFG),
                dcc.Graph(figure=f_pacf, config=CFG),
                dcc.Graph(figure=fp,     config=CFG),
            ], style={"display": "grid",
                      "grid-template-columns": "1fr 1fr 1fr",
                      "gap": "6px", "margin-top": "10px"}),
            suggestions,
        ])

        status = (f"✅  Analisi completata.  "
                  f"ADF stat={adf_stat:.3f}  p={adf_p:.4f}  lag={adf_lag}.  "
                  f"Ordini suggeriti: p={suggested_p}  d={suggested_d}  q={suggested_q}")
        return output, suggested_p, suggested_d, suggested_q, status

    # ── PASSO 3: stima + PASSO 4: diagnostica ────────────────────────────────
    @app.callback(
        Output("arima-step3-output", "children"),
        Output("arima-step4-output", "children"),
        Output("arima-status",       "children"),
        Input("btn-run-arima",       "n_clicks"),
        State("store-arima-step1",   "data"),
        State("arima-p",             "value"),
        State("arima-d",             "value"),
        State("arima-q",             "value"),
        State("arima-seasonal-on",   "value"),
        State("arima-P",             "value"),
        State("arima-D",             "value"),
        State("arima-Q",             "value"),
        State("arima-s",             "value"),
        State("arima-forecast-steps","value"),
        State("arima-cov-type",      "value"),
        State("arima-dummies",       "value"),
        prevent_initial_call=True,
    )
    def run_arima(n_clicks, step1_data,
                  p, d, q, seasonal_on, P_, D_, Q_, s_,
                  fc_steps, cov_type, dummy_vars):          # ← aggiunto
        _err = lambda m: (None, None, f"❌ {m}")
        if not step1_data:
            return _err("Esegui prima i Passi ① e ②.")

        series = pd.read_json(io.StringIO(step1_data["series_json"]), typ="series")
        series = series.sort_index().dropna()
        y_var       = step1_data.get("y_var", "")
        detrend_lbl = step1_data.get("detrend_lbl", "serie")

        p  = int(p  or 1); d  = int(d  or 0); q  = int(q  or 0)
        P_ = int(P_ or 1); D_ = int(D_ or 1); Q_ = int(Q_ or 1); s_ = int(s_ or 12)
        use_seasonal = bool(seasonal_on)
        steps = int(fc_steps or 12)
        cov   = cov_type or "opg"

        # ── costruisci matrice esogena dummy ──────────────────────────────
        dummy_vars = dummy_vars or []
        DUMMY_RANGES = {
            "dummy_covid":   ("2020-03-01", "2020-05-01"),
            "dummy_inflaz":  ("2021-11-01", "2022-12-01"),
            "dummy_ucraina": ("2022-02-01", "2022-12-01"),
            "dummy_gfc":     ("2008-09-01", "2009-03-01"),
            "dummy_dotcom":  ("2000-03-01", "2002-10-01"),
        }
        DUMMY_LABELS = {
            "dummy_covid":   "D_COVID",
            "dummy_inflaz":  "D_Inflaz21",
            "dummy_ucraina": "D_Ucraina",
            "dummy_gfc":     "D_GFC",
            "dummy_dotcom":  "D_Dotcom",
        }

        exog_train = None
        exog_fore  = None
        exog_cols  = []

        if dummy_vars:
            exog_df = pd.DataFrame(index=series.index)
            for dv in dummy_vars:
                s0, s1 = DUMMY_RANGES[dv]
                col = DUMMY_LABELS[dv]
                exog_df[col] = 0.0
                exog_df.loc[
                    (exog_df.index >= s0) & (exog_df.index <= s1), col
                ] = 1.0
                exog_cols.append(col)
            exog_train = exog_df.values

            # per la previsione: future index con tutte dummy = 0
            last_date = series.index[-1]
            future_idx = pd.date_range(
                last_date + pd.DateOffset(months=1),
                periods=steps, freq="MS"
            )
            exog_fore_df = pd.DataFrame(0.0,
                                        index=future_idx,
                                        columns=exog_cols)
            exog_fore = exog_fore_df.values

        # ── stima modello ─────────────────────────────────────────────────
        try:
            if use_seasonal and s_ > 1:
                model = SARIMAX(series,
                                order=(p, d, q),
                                seasonal_order=(P_, D_, Q_, s_),
                                exog=exog_train,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                mname = f"SARIMA({p},{d},{q})({P_},{D_},{Q_},{s_})"
            else:
                model = SARIMAX(series,
                                order=(p, d, q),
                                exog=exog_train,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                mname = f"ARIMA({p},{d},{q})"
            if exog_cols:
                mname += f" + {', '.join(exog_cols)}"
            result = model.fit(disp=False, cov_type=cov)
        except Exception as ex:
            return _err(f"Errore stima: {ex}")

              # ── definisci DL e CFG prima di tutto ────────────────────────────
        DL = dict(
            paper_bgcolor="white", plot_bgcolor="#f8f9f9",
            margin=dict(l=50, r=12, t=34, b=28),
            font=dict(size=9, color="#444"),
            xaxis=dict(showgrid=True, gridcolor="#ebebeb",
                       linecolor="#ccc", tickfont=dict(size=8)),
            yaxis=dict(showgrid=True, gridcolor="#ebebeb",
                       linecolor="#ccc", tickfont=dict(size=8))
        )
        CFG = {"displayModeBar": False}

        # diagnostics
        resid   = result.resid.dropna()
        sigma   = float(np.std(resid))
        dw      = float(sm2.stats.stattools.durbin_watson(resid))
        jb_s, jb_p, jb_sk, jb_ku = sm2.stats.stattools.jarque_bera(resid)
        lb_df   = sm2.stats.diagnostic.acorr_ljungbox(resid, lags=[10], return_df=True)
        lb_stat = float(lb_df["lb_stat"].iloc[0])
        lb_p    = float(lb_df["lb_pvalue"].iloc[0])

        # ── ARCH test + GARCH(1,1) sui residui ───────────────────────────
        try:
            from statsmodels.stats.diagnostic import het_arch as _het_arch
            _arch_stat, _arch_p, _, _ = _het_arch(resid.values, nlags=5)
        except Exception:
            _arch_p, _arch_stat = 1.0, 0.0

        _garch_ok     = False
        _garch_params = {}
        _cond_vol     = None
        _garch_fc_std = None
        try:
            from arch import arch_model as _arch_model
            _gm = _arch_model(resid.values * 100, vol='Garch', p=1, q=1, rescale=False)
            _gr = _gm.fit(disp='off', options={'maxiter': 300})
            _o = float(_gr.params.get('omega',    0))
            _a = float(_gr.params.get('alpha[1]', 0))
            _b = float(_gr.params.get('beta[1]',  0))
            if _a + _b < 1.0 and _o > 0:
                _garch_ok     = True
                _garch_params = {'omega': _o, 'alpha': _a, 'beta': _b,
                                 'persistence': _a + _b,
                                 'uncon_vol': float(np.sqrt(_o / (1 - _a - _b))) / 100}
                _cond_vol     = pd.Series(_gr.conditional_volatility / 100,
                                          index=resid.index)
                _gf           = _gr.forecast(horizon=steps, reindex=False)
                _garch_fc_std = np.sqrt(_gf.variance.values[-1]) / 100
        except Exception:
            pass

        # ── forecast ─────────────────────────────────────────────────────
        fc   = result.get_forecast(steps=steps, exog=exog_fore)
        fc_m = fc.predicted_mean
        fc_c = fc.conf_int(alpha=0.05)
        fit  = result.fittedvalues

        # ── grafico forecast ──────────────────────────────────────────────
        fig_fc = go.Figure()
        fig_fc.add_scatter(x=series.index, y=series.values, mode="lines",
                           name="Osservato", line=dict(color="#1565c0", width=1.5))
        fig_fc.add_scatter(x=fit.index, y=fit.values, mode="lines",
                           name="Stimato (in-sample)",
                           line=dict(color="#2e7d32", width=1.2, dash="dot"))
        fig_fc.add_scatter(x=fc_m.index, y=fc_m.values, mode="lines",
                           name="Previsione", line=dict(color="#e65100", width=1.5))
        fig_fc.add_scatter(
            x=list(fc_c.index) + list(fc_c.index[::-1]),
            y=list(fc_c.iloc[:, 1]) + list(fc_c.iloc[:, 0][::-1]),
            fill="toself", fillcolor="rgba(230,81,0,.10)",
            line=dict(color="rgba(0,0,0,0)"), name="IC 95% ARIMA")

        if _garch_ok and _garch_fc_std is not None:
            _ghi = fc_m.values + 1.96 * _garch_fc_std
            _glo = fc_m.values - 1.96 * _garch_fc_std
            fig_fc.add_scatter(
                x=list(fc_m.index) + list(fc_m.index[::-1]),
                y=list(_ghi) + list(_glo[::-1]),
                fill="toself", fillcolor="rgba(103,58,183,.15)",
                line=dict(color="rgba(0,0,0,0)"), name="IC 95% GARCH")

        # aggiunge bande colorate per le dummy attive
        DUMMY_RANGES = {
            "dummy_covid":   ("2020-03-01", "2020-05-01",  "rgba(214,39,40,0.10)",  "COVID"),
            "dummy_inflaz":  ("2021-11-01", "2022-12-01",  "rgba(255,127,14,0.10)", "Inflaz."),
            "dummy_ucraina": ("2022-02-01", "2022-12-01",  "rgba(148,103,189,0.10)","Ucraina"),
            "dummy_gfc":     ("2008-09-01", "2009-03-01",  "rgba(44,160,44,0.10)",  "GFC"),
            "dummy_dotcom":  ("2000-03-01", "2002-10-01",  "rgba(140,86,75,0.10)",  "Dot-com"),
        }
        for dv in (dummy_vars or []):
            if dv in DUMMY_RANGES:
                s0, s1, col, lbl = DUMMY_RANGES[dv]
                fig_fc.add_vrect(x0=s0, x1=s1,
                                 fillcolor=col, line_width=0,
                                 annotation_text=lbl,
                                 annotation_position="top left",
                                 annotation_font=dict(size=8))

        fig_fc.update_layout(**DL, height=260,
                             title=dict(
                                 text=f"{mname}  —  {detrend_lbl}  |  IC 95%",
                                 font=dict(size=10, color="#1a3a5c"), x=0.01),
                             legend=dict(orientation="h", y=1.17, font=dict(size=8)))

        # ── back-transformation → livelli originali ───────────────────────────
        _transform   = step1_data.get("transform", "none")
        _detrend     = step1_data.get("detrend", "none")
        orig_s       = (pd.read_json(io.StringIO(step1_data["orig_json"]),  typ="series")
                          .sort_index() if step1_data.get("orig_json")  else None)
        trans_s      = (pd.read_json(io.StringIO(step1_data["trans_json"]), typ="series")
                          .sort_index() if step1_data.get("trans_json") else None)
        trend_s_back = (pd.read_json(io.StringIO(step1_data["trend_json"]), typ="series")
                          .sort_index() if step1_data.get("trend_json") else None)

        fig_orig = None
        bt_note  = ""
        try:
            # ── STEP A: aggiungi trend ────────────────────────────────────────
            if _detrend in ("hp", "ma12") and trend_s_back is not None:
                # estrapola il trend come random walk (ultimo valore noto)
                last_trend = float(trend_s_back.iloc[-1])
                trend_fc   = pd.Series(last_trend, index=fc_m.index)
                fc_trans   = fc_m   + trend_fc
                fc_lo_t    = fc_c.iloc[:, 0].values + last_trend
                fc_hi_t    = fc_c.iloc[:, 1].values + last_trend
                bt_note    = "trend estrapolato come random walk (ultimo valore)"
            elif _detrend == "sdiff" and trans_s is not None:
                # Δ₁₂ invertita: y_t = stat_t + y_{t-12}
                # usiamo gli ultimi 12 valori di trans_s come base
                s_lag       = 12
                base_tail   = trans_s.dropna().values[-s_lag:]
                fc_vals     = []
                fc_lo_arr   = []
                fc_hi_arr   = []
                extended    = list(base_tail)
                for i in range(len(fc_m)):
                    base    = extended[i]           # valore a t-12 (già accumulato)
                    fc_vals.append(fc_m.iloc[i]   + base)
                    fc_lo_arr.append(fc_c.iloc[i, 0] + base)
                    fc_hi_arr.append(fc_c.iloc[i, 1] + base)
                    extended.append(fc_vals[-1])    # aggiunge per i prossimi lag
                fc_trans = pd.Series(fc_vals,  index=fc_m.index)
                fc_lo_t  = np.array(fc_lo_arr)
                fc_hi_t  = np.array(fc_hi_arr)
                bt_note  = "ricostruzione Δ₁₂ inversa — cumsum su base t−12"
            else:
                fc_trans = fc_m.copy()
                fc_lo_t  = fc_c.iloc[:, 0].values
                fc_hi_t  = fc_c.iloc[:, 1].values
                bt_note  = "nessun detrend da invertire"

            # ── STEP B: inverso della trasformazione ──────────────────────────
            if _transform == "log":
                fc_orig_vals = np.exp(fc_trans.values)
                fc_lo_orig   = np.exp(fc_lo_t)
                fc_hi_orig   = np.exp(fc_hi_t)
                orig_lbl     = y_var
            elif _transform == "diff":
                if orig_s is not None:
                    last_orig_val = float(orig_s.dropna().iloc[-1])
                    fc_orig_vals  = np.cumsum(fc_trans.values) + last_orig_val
                    fc_lo_orig    = np.cumsum(fc_lo_t)         + last_orig_val
                    fc_hi_orig    = np.cumsum(fc_hi_t)         + last_orig_val
                    orig_lbl      = y_var
                else:
                    raise ValueError("orig_s mancante per inverso diff")
            elif _transform == "diff_log":
                if orig_s is not None:
                    last_log_val = float(np.log(orig_s.dropna().iloc[-1]))
                    fc_log_vals  = np.cumsum(fc_trans.values) + last_log_val
                    fc_lo_log    = np.cumsum(fc_lo_t)         + last_log_val
                    fc_hi_log    = np.cumsum(fc_hi_t)         + last_log_val
                    fc_orig_vals = np.exp(fc_log_vals)
                    fc_lo_orig   = np.exp(fc_lo_log)
                    fc_hi_orig   = np.exp(fc_hi_log)
                    orig_lbl     = y_var
                else:
                    raise ValueError("orig_s mancante per inverso diff_log")
            else:
                fc_orig_vals = fc_trans.values
                fc_lo_orig   = fc_lo_t
                fc_hi_orig   = fc_hi_t
                orig_lbl     = y_var

            fc_orig_idx = fc_m.index

            # ── fit in-sample → livelli originali ────────────────────────────
            try:
                if _detrend in ("hp", "ma12") and trend_s_back is not None:
                    _trend_al = trend_s_back.reindex(fit.index).ffill().bfill()
                    _fit_dt   = fit + _trend_al
                elif _detrend == "sdiff" and trans_s is not None:
                    _fit_dt = fit + trans_s.shift(12).reindex(fit.index)
                else:
                    _fit_dt = fit.copy()

                if _transform == "log":
                    _fit_orig = pd.Series(np.exp(_fit_dt.values), index=fit.index)
                elif _transform == "diff" and orig_s is not None:
                    _fit_orig = orig_s.shift(1).reindex(fit.index) + _fit_dt
                elif _transform == "diff_log" and orig_s is not None:
                    _orig_lag = orig_s.shift(1).reindex(fit.index)
                    _fit_orig = pd.Series(
                        (_orig_lag * np.exp(_fit_dt.values)).values, index=fit.index)
                else:
                    _fit_orig = _fit_dt.copy()
                _fit_orig = _fit_orig.dropna()
            except Exception:
                _fit_orig = None

            # ── grafico livelli ───────────────────────────────────────────────
            if orig_s is not None:
                n_hist = min(len(orig_s), max(60, steps * 3))
                orig_plot = orig_s.dropna().iloc[-n_hist:]
            else:
                orig_plot = None

            fig_orig = go.Figure()
            if orig_plot is not None:
                fig_orig.add_scatter(x=orig_plot.index, y=orig_plot.values,
                                     mode="lines", name="Storico",
                                     line=dict(color="#1565c0", width=1.6))
            if _fit_orig is not None and len(_fit_orig) > 0:
                _fit_plot = (_fit_orig[_fit_orig.index >= orig_plot.index[0]]
                             if orig_plot is not None else _fit_orig)
                fig_orig.add_scatter(x=_fit_plot.index, y=_fit_plot.values,
                                     mode="lines", name="Stimato (livelli)",
                                     line=dict(color="#2e7d32", width=1.2, dash="dot"))
            fig_orig.add_scatter(x=fc_orig_idx, y=fc_orig_vals,
                                 mode="lines", name="Previsione (livelli)",
                                 line=dict(color="#e65100", width=2))
            fig_orig.add_scatter(
                x=list(fc_orig_idx) + list(fc_orig_idx[::-1]),
                y=list(fc_hi_orig)  + list(fc_lo_orig[::-1]),
                fill="toself", fillcolor="rgba(230,81,0,.12)",
                line=dict(color="rgba(0,0,0,0)"), name="IC 95%")
            # linea verticale al confine storico/previsione
            if orig_plot is not None:
                fig_orig.add_vline(x=orig_plot.index[-1], line_dash="dash",
                                   line_color="#888", line_width=1)
            fig_orig.update_layout(
                **DL, height=290,
                title=dict(
                    text=f"Previsione in livelli originali  —  {orig_lbl}  |  "
                         f"{steps} passi  |  IC 95%  ({bt_note})",
                    font=dict(size=10, color="#1a3a5c"), x=0.01),
                legend=dict(orientation="h", y=1.17, font=dict(size=8)))

        except Exception as bt_err:
            fig_orig = None
            bt_note  = f"back-transformation non disponibile: {bt_err}"

        # ── tabella statistiche ───────────────────────────────────────────────
        def _ok_color(ok):
            return {"ok": "#1b5e20", "warn": "#e65100", "bad": "#b71c1c", None: "#333"}[ok]

        def _row(label, val, note, ok=None):
            return html.Tr([
                html.Td(label, style={"font-size": "11px", "padding": "4px 8px",
                                      "color": "#555", "border-bottom": "1px solid #f0f0f0"}),
                html.Td(val,   style={"font-size": "11px", "padding": "4px 8px",
                                      "font-weight": "600", "color": _ok_color(ok),
                                      "border-bottom": "1px solid #f0f0f0"}),
                html.Td(note,  style={"font-size": "10px", "padding": "4px 8px",
                                      "color": "#888", "border-bottom": "1px solid #f0f0f0"}),
            ])

        dw_ok  = "ok" if 1.5 < dw < 2.5 else "warn" if 1.0 < dw < 3.0 else "bad"
        jb_ok  = "ok" if jb_p > 0.05 else "warn" if jb_p > 0.01 else "bad"
        lb_ok  = "ok" if lb_p > 0.05 else "warn" if lb_p > 0.01 else "bad"

        stat_tbl = html.Table([
            html.Thead(html.Tr([
                html.Th(h, style={"font-size": "10px", "color": "#4a148c",
                                   "padding": "5px 8px", "background": "#f3e5f5",
                                   "text-align": "left"})
                for h in ["Statistica", "Valore", "Note"]
            ])),
            html.Tbody([
                _row("Modello",        mname, ""),
                _row("N osservazioni", str(len(series)), ""),
                _row("Log-Likelihood", f"{result.llf:.4f}", ""),
                _row("AIC",            f"{result.aic:.4f}", "minore = migliore"),
                _row("BIC",            f"{result.bic:.4f}", ""),
                _row("σ residui",      f"{sigma:.6f}", ""),
                _row("Durbin-Watson",  f"{dw:.4f}", "~2 = no autocorr.", dw_ok),
                _row("Jarque-Bera",    f"{jb_s:.4f}",
                     f"p={jb_p:.4e} {_pstar(jb_p)}  |  sk={jb_sk:.3f}  ku={jb_ku:.3f}", jb_ok),
                _row("Ljung-Box (10)", f"{lb_stat:.4f}",
                     f"p={lb_p:.4e} {_pstar(lb_p)}  (H0: no autocorr. residui)", lb_ok),
            ])
        ], style={"width": "100%", "border-collapse": "collapse",
                  "background": "#fff", "border-radius": "4px",
                  "margin-bottom": "10px"})

        # secondo grafico (livelli originali) opzionale
        orig_chart_el = []
        if fig_orig is not None:
            orig_chart_el = [
                html.Div([
                    html.Span("📍  Previsione in livelli originali",
                              style={"font-size": "11px", "font-weight": "700",
                                     "color": "#1a3a5c"}),
                    html.Span(f"  {bt_note}",
                              style={"font-size": "9px", "color": "#888",
                                     "margin-left": "8px"}),
                ], style={"padding": "5px 0", "margin-top": "8px",
                          "border-top": "1px solid #e0e0e0"}),
                dcc.Graph(figure=fig_orig, config=CFG),
            ]
        else:
            orig_chart_el = [
                html.Div(f"⚠  {bt_note}",
                         style={"font-size": "10px", "color": "#e65100",
                                "margin-top": "8px", "padding": "6px 0"}),
            ]

        step3_out = html.Div([
            html.Div([
                html.Div(mname,
                         style={"font-size": "15px", "font-weight": "700", "color": "#4a148c"}),
                html.Div(f"Serie: {detrend_lbl}  |  N = {len(series)}  |  "
                         f"AIC: {result.aic:.2f}  |  BIC: {result.bic:.2f}",
                         style={"font-size": "10px", "color": "#888", "margin-top": "2px"}),
            ], style={"background": "#f3e5f5", "border-radius": "5px",
                      "padding": "10px 12px", "margin-bottom": "10px"}),
            stat_tbl,
            html.Div([
                html.Span("📊  Fitted & Forecast (spazio stazionario)",
                          style={"font-size": "11px", "font-weight": "700",
                                 "color": "#1a3a5c"}),
            ], style={"padding": "4px 0", "border-top": "1px solid #e0e0e0",
                      "margin-bottom": "2px"}),
            dcc.Graph(figure=fig_fc, config=CFG),
            *orig_chart_el,
            # ── grafico in-sample fit ─────────────────────────────────────
            html.Div([
                html.Span("🔍  In-sample fit  —  osservato vs stimato",
                          style={"font-size": "11px", "font-weight": "700",
                                 "color": "#1a3a5c"}),
            ], style={"padding": "6px 0 2px", "border-top": "1px solid #e0e0e0",
                      "margin-top": "4px"}),
            dcc.Graph(figure=_make_insample_fig(
                series, fit, _cond_vol if _garch_ok else None, DL), config=CFG),
        ])

        # ── STEP 4: diagnostica residui ───────────────────────────────────────
        # ACF residui
        r_acf, r_ci = _acf_fn(resid, nlags=30, alpha=0.05, fft=True)
        r_lags = list(range(len(r_acf)))
        r_half = np.abs(r_ci[:, 1] - r_ci[:, 0]) / 2

        fig_racf = go.Figure()
        for i, v in enumerate(r_acf):
            h = r_half[min(i, len(r_half)-1)]
            c = "#d62728" if i > 0 and abs(v) > h else "#7f7f7f"
            fig_racf.add_shape(type="line", x0=i, x1=i, y0=0, y1=v,
                               line=dict(color=c, width=2))
        fig_racf.add_scatter(x=r_lags, y=[r_half[min(i, len(r_half)-1)] for i in r_lags],
                             mode="lines", line=dict(color="#ff7f0e", dash="dash", width=1),
                             showlegend=False)
        fig_racf.add_scatter(x=r_lags, y=[-r_half[min(i, len(r_half)-1)] for i in r_lags],
                             mode="lines", line=dict(color="#ff7f0e", dash="dash", width=1),
                             showlegend=False)
        fig_racf.add_hline(y=0, line_color="#888", line_width=0.6)
        fig_racf.update_layout(**DL, height=200,
                               title=dict(text="ACF Residui  (rosso = lag significativo → autocorr. residua)",
                                          font=dict(size=10, color="#1a3a5c"), x=0.01),
                               yaxis_range=[-1.05, 1.05])

        # QQ plot
        qq  = sp_stats.probplot(resid, dist="norm")
        qx  = qq[0][0]; qy = qq[0][1]
        slope, intercept = qq[1][0], qq[1][1]
        fig_qq = go.Figure()
        fig_qq.add_scatter(x=qx, y=qy, mode="markers",
                           marker=dict(color="#9467bd", size=3, opacity=0.7), name="Quantili")
        fig_qq.add_scatter(x=[qx[0], qx[-1]],
                           y=[intercept + slope * qx[0], intercept + slope * qx[-1]],
                           mode="lines", line=dict(color="#ff7f0e", dash="dash", width=1.3),
                           name="Normale teor.")
        fig_qq.update_layout(**DL, height=200,
                             title=dict(text="QQ Plot residui  (deviazioni dalla normale)",
                                        font=dict(size=10, color="#1a3a5c"), x=0.01),
                             xaxis_title="Quantili teorici",
                             yaxis_title="Quantili osservati",
                             legend=dict(orientation="h", y=1.2, font=dict(size=8)))

        # Residui nel tempo
        fig_rt = go.Figure()
        fig_rt.add_scatter(x=resid.index, y=resid.values, mode="lines",
                           line=dict(color="#e91e63", width=0.9))
        fig_rt.add_hline(y=0, line_dash="dash", line_color="#aaa", line_width=0.8)
        fig_rt.add_hline(y= 2*sigma, line_dash="dot", line_color="#ff9800", line_width=0.8)
        fig_rt.add_hline(y=-2*sigma, line_dash="dot", line_color="#ff9800", line_width=0.8)
        fig_rt.update_layout(**DL, height=200,
                             title=dict(text="Residui nel tempo  (linee arancio = ±2σ)",
                                        font=dict(size=10, color="#1a3a5c"), x=0.01))

        # Istogramma residui
        x_rng = np.linspace(resid.min(), resid.max(), 120)
        norm_y = (sp_stats.norm.pdf(x_rng, np.mean(resid), sigma)
                  * len(resid) * (resid.max() - resid.min()) / 30)
        fig_hist = go.Figure()
        fig_hist.add_histogram(x=resid.values, nbinsx=30,
                               marker_color="#1565c0", opacity=0.65, name="Residui")
        fig_hist.add_scatter(x=x_rng, y=norm_y, mode="lines",
                             line=dict(color="#e65100", width=1.5), name="Normale attesa")
        fig_hist.update_layout(**DL, height=200,
                               title=dict(text="Distribuzione residui  vs  Normale",
                                          font=dict(size=10, color="#1a3a5c"), x=0.01),
                               legend=dict(orientation="h", y=1.2, font=dict(size=8)))

        # ── ACF residui² + volatilità condizionale GARCH ─────────────────────
        r2_acf, r2_ci = _acf_fn(resid**2, nlags=20, alpha=0.05, fft=True)
        r2_lags = list(range(len(r2_acf)))
        r2_half = np.abs(r2_ci[:, 1] - r2_ci[:, 0]) / 2

        fig_r2acf = go.Figure()
        for i, v in enumerate(r2_acf):
            h = r2_half[min(i, len(r2_half)-1)]
            c = "#d62728" if i > 0 and abs(v) > h else "#7f7f7f"
            fig_r2acf.add_shape(type="line", x0=i, x1=i, y0=0, y1=v,
                                line=dict(color=c, width=2))
        fig_r2acf.add_scatter(x=r2_lags,
                              y=[ r2_half[min(i, len(r2_half)-1)] for i in r2_lags],
                              mode="lines", line=dict(color="#ff7f0e", dash="dash", width=1),
                              showlegend=False)
        fig_r2acf.add_scatter(x=r2_lags,
                              y=[-r2_half[min(i, len(r2_half)-1)] for i in r2_lags],
                              mode="lines", line=dict(color="#ff7f0e", dash="dash", width=1),
                              showlegend=False)
        fig_r2acf.add_hline(y=0, line_color="#888", line_width=0.6)
        fig_r2acf.update_layout(**DL, height=200,
                                title=dict(
                                    text="ACF Residui²  (presenza di effetti ARCH → eteroscedasticità)",
                                    font=dict(size=10, color="#1a3a5c"), x=0.01),
                                yaxis_range=[-1.05, 1.05])

        # grafico volatilità condizionale GARCH
        garch_el = []
        if _garch_ok and _cond_vol is not None:
            fig_cvol = go.Figure()
            fig_cvol.add_scatter(x=_cond_vol.index, y=_cond_vol.values, mode="lines",
                                 line=dict(color="#6a1b9a", width=1.2), name="σ_t GARCH")
            fig_cvol.add_hline(y=_garch_params['uncon_vol'],
                               line_dash="dash", line_color="#ff7f0e", line_width=1,
                               annotation_text=f"σ incondiz. = {_garch_params['uncon_vol']:.4f}",
                               annotation_position="right",
                               annotation_font=dict(size=8))
            fig_cvol.update_layout(**DL, height=200,
                                   title=dict(
                                       text=(f"Volatilità condizionale GARCH(1,1)  —  "
                                             f"ω={_garch_params['omega']:.4f}  "
                                             f"α={_garch_params['alpha']:.4f}  "
                                             f"β={_garch_params['beta']:.4f}  "
                                             f"α+β={_garch_params['persistence']:.4f}"),
                                       font=dict(size=10, color="#1a3a5c"), x=0.01))
            garch_el = [
                html.Div([
                    html.Span("📈  GARCH(1,1) sui residui ARIMA",
                              style={"font-size": "11px", "font-weight": "700",
                                     "color": "#4a148c"}),
                ], style={"padding": "4px 0", "border-top": "1px solid #e0e0e0",
                          "margin": "8px 0 2px"}),
                dcc.Graph(figure=fig_cvol, config=CFG),
            ]

        # ── verdetto ──────────────────────────────────────────────────────────
        issues = []
        if not (1.5 < dw < 2.5):
            issues.append(f"DW = {dw:.2f}  →  autocorrelazione residua al lag 1  "
                          f"(ottimale: 1.5–2.5)")
        if jb_p < 0.05:
            issues.append(f"Jarque-Bera p = {jb_p:.4f}  →  residui non normali  "
                          f"(sk = {jb_sk:.2f}, ku = {jb_ku:.2f})")
        if lb_p < 0.05:
            issues.append(f"Ljung-Box p = {lb_p:.4f}  →  autocorrelazione residua lag 1-10")
        if _arch_p < 0.05:
            arch_msg = (f"ARCH test p = {_arch_p:.4f}  →  eteroscedasticità condizionale rilevata")
            if _garch_ok:
                arch_msg += (f"  (GARCH(1,1) stimato: α+β = {_garch_params['persistence']:.3f}  ✓  "
                             f"IC forecast aggiornato)")
            issues.append(arch_msg)

        if not issues:
            verdict = html.Div([
                html.Span("✅  ", style={"font-size": "14px"}),
                html.Span("Diagnostica superata — i residui sono rumore bianco.  "
                          "Il modello è adeguato.",
                          style={"font-size": "11px", "font-weight": "600",
                                 "color": "#1b5e20"}),
            ], style={"background": "#e8f5e9", "border": "1px solid #a5d6a7",
                      "border-radius": "5px", "padding": "10px 14px",
                      "margin-bottom": "10px"})
        else:
            verdict = html.Div([
                html.Div("⚠  Problemi diagnostici rilevati:",
                         style={"font-size": "11px", "font-weight": "700",
                                "color": "#b71c1c", "margin-bottom": "6px"}),
                html.Ul([html.Li(iss, style={"font-size": "10px", "color": "#c62828",
                                             "margin-bottom": "3px"})
                         for iss in issues]),
                html.Div("→  Considera: aumentare p o q  |  aggiungere componente "
                         "stagionale  |  verificare outlier strutturali post-2020  |  "
                         "cambiare trasformazione al Passo ①",
                         style={"font-size": "10px", "color": "#555", "margin-top": "4px"}),
            ], style={"background": "#ffebee", "border": "1px solid #ef9a9a",
                      "border-radius": "5px", "padding": "10px 14px",
                      "margin-bottom": "10px"})

        step4_out = html.Div([
            verdict,
            html.Div([
                dcc.Graph(figure=fig_racf,  config=CFG),
                dcc.Graph(figure=fig_qq,    config=CFG),
            ], style={"display": "grid", "grid-template-columns": "1fr 1fr",
                      "gap": "6px", "margin-bottom": "6px"}),
            html.Div([
                dcc.Graph(figure=fig_rt,    config=CFG),
                dcc.Graph(figure=fig_hist,  config=CFG),
            ], style={"display": "grid", "grid-template-columns": "1fr 1fr",
                      "gap": "6px", "margin-bottom": "6px"}),
            dcc.Graph(figure=fig_r2acf, config=CFG),
            *garch_el,
        ])

        status = (f"✅  {mname} stimato.  "
                  f"Log-lik = {result.llf:.2f}  |  AIC = {result.aic:.2f}  |  "
                  f"BIC = {result.bic:.2f}  |  DW = {dw:.2f}")
        return step3_out, step4_out, status


        # =========================================================================
    # ADL — Autoregressive Distributed Lag
    # =========================================================================

    @app.callback(
        Output("adl-y-var",   "options"),
        Output("adl-x-rows",  "children"),
        Output("adl-slider",  "min"),
        Output("adl-slider",  "max"),
        Output("adl-slider",  "value"),
        Output("adl-slider",  "marks"),
        Input("store-adl-source", "data"),
        Input("store-adl-extra",  "data"),
        prevent_initial_call=False,
    )
    def adl_populate(source_data, extra_data):
        frames = []
        if source_data:
            try:
                df0 = pd.read_json(io.StringIO(source_data), orient="split")
                df0.index = pd.to_datetime(df0.index)
                frames.append(df0)
            except Exception:
                pass
        if extra_data:
            for sid, meta in extra_data.items():
                idx  = pd.to_datetime(meta["dates"])
                vals = pd.array(meta["values"], dtype="Float64")
                s    = pd.Series(vals, index=idx, name=sid)
                frames.append(s.to_frame())
        if not frames:
            return [], [], 0, 1, [0, 1], {}
        df = pd.concat(frames, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.index = pd.to_datetime(df.index)
        if df.empty:
            return [], [], 0, 1, [0, 1], {}

        cols = sorted(df.columns.tolist())
        y_options = [{"label": c, "value": c} for c in cols]
        _dd_s = {"font-size": "9px", "width": "65px", "display": "inline-block"}
        x_rows = []
        for ci, col in enumerate(cols):
            bg  = "#ffffff" if ci % 2 == 0 else "#f9f9f9"
            tag = "🆕 " if (extra_data and col in extra_data) else ""
            pipe_steps = [
                html.Span([
                    html.Span("→" if k > 0 else "",
                              style={"font-size": "10px", "color": "#bbb",
                                     "margin": "0 1px"}),
                    dcc.Dropdown(
                        id={"type": "adl-x-step", "index": f"{col}___{k}"},
                        options=ADL_PIPE_OPTS,
                        value="none", clearable=False,
                        style=_dd_s,
                    ),
                ], style={"display": "inline-flex", "align-items": "center"})
                for k in range(4)
            ]
            x_rows.append(html.Div([
                # Riga 1: nome + lag
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id={"type": "adl-x-active", "index": col},
                            options=[{"label": f" {tag}{col}", "value": col}],
                            value=[],
                            style={"font-size": "10px"},
                            inputStyle={"margin-right": "4px"},
                        ),
                    ], style={"flex": "1", "overflow": "hidden",
                               "text-overflow": "ellipsis", "white-space": "nowrap"}),
                    html.Div([
                        dcc.Dropdown(
                            id={"type": "adl-x-lags", "index": col},
                            options=[{"label": f"L{k}", "value": k}
                                     for k in range(0, 13)],
                            value=[0], multi=True, clearable=False,
                            placeholder="lag...",
                            style={"font-size": "9px", "width": "175px"},
                        ),
                    ], style={"width": "180px"}),
                ], style={"display": "flex", "align-items": "center",
                           "margin-bottom": "3px"}),
                # Riga 2: pipeline
                html.Div([
                    html.Span("Pipeline →",
                              style={"font-size": "8px", "color": "#1a5276",
                                     "font-weight": "bold", "white-space": "nowrap",
                                     "margin-right": "4px"}),
                    *pipe_steps,
                ], style={"display": "flex", "align-items": "center",
                           "flex-wrap": "wrap", "gap": "2px",
                           "background": "#eaf4fb",
                           "border-radius": "3px", "padding": "3px 5px"}),
            ], style={"padding": "5px 8px", "background": bg,
                       "border-bottom": "1px solid #eee"}))
        mn, mx, val, marks = _slider_params(df)
        return y_options, x_rows, mn, mx, val, marks

    @app.callback(
        Output("store-adl-extra", "data"),
        Output("adl-fred-status", "children"),
        Input("btn-adl-fred",     "n_clicks"),
        State("adl-fred-input",   "value"),
        State("api-key",          "value"),
        State("store-adl-extra",  "data"),
        prevent_initial_call=True,
    )
    def adl_add_fred(n, series_input, api_key, existing):
        if not series_input or not series_input.strip():
            return existing, "⚠ Inserisci almeno un ID serie"
        api_key = (api_key or FRED_API_KEY).strip()
        ids = [s.strip().upper()
               for s in series_input.replace(";", ",").split(",") if s.strip()]
        existing = existing or {}
        added, failed = [], []
        for sid in ids:
            try:
                s = fred_get(sid, api_key)
                if s is None or s.empty:
                    failed.append(sid); continue
                s.index = pd.to_datetime(s.index)
                existing[sid] = {
                    "dates":  s.index.strftime("%Y-%m-%d").tolist(),
                    "values": [float(v) if v == v else None for v in s.values],
                }
                added.append(sid)
            except Exception:
                failed.append(sid)
        parts = []
        if added:  parts.append(f"✅ {', '.join(added)}")
        if failed: parts.append(f"❌ {', '.join(failed)}")
        return existing, "  |  ".join(parts) or "—"

    @app.callback(
        Output("adl-slider-label", "children"),
        Input("adl-slider",        "value"),
    )
    def adl_slider_label(val):
        if not val or (val[1] - val[0]) < 86400:
            return ""
        s = pd.to_datetime(val[0], unit="s").strftime("%b %Y")
        e = pd.to_datetime(val[1], unit="s").strftime("%b %Y")
        return f"📅  {s}  →  {e}"

    @app.callback(
        Output("adl-geo-wrapper", "style"),
        Input("adl-source-type",  "value"),
    )
    def adl_toggle_geo(source_type):
        if source_type == "eur":
            return {"display": "block", "margin-bottom": "6px"}
        return {"display": "none", "margin-bottom": "6px"}

    # ── Clientside: segnala inizio caricamento sullo store (no allow_duplicate) ──
    app.clientside_callback(
        """
        function(n, source_type, geo) {
            if (!n) return window.dash_clientside.no_update;
            var src = source_type === "eur"
                ? "Eurostat \u2014 " + (geo || "EA20")
                : "FRED (USA)";
            return {"active": true, "src": src};
        }
        """,
        Output("store-adl-loading-state", "data"),
        Input("btn-adl-load",   "n_clicks"),
        State("adl-source-type","value"),
        State("adl-eur-geo",    "value"),
        prevent_initial_call=True,
    )

    # ── Reagisce allo store di loading: mostra/nasconde overlay e interval ──────
    @app.callback(
        Output("adl-loading-overlay", "style"),
        Output("adl-loading-title",   "children"),
        Output("adl-loading-source",  "children"),
        Output("adl-progress-tick",   "disabled"),
        Output("adl-progress-tick",   "n_intervals"),
        Input("store-adl-loading-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_loading_overlay(state):
        if state and state.get("active"):
            overlay_style = {
                "display": "flex", "position": "fixed",
                "top": "0", "left": "0",
                "width": "100%", "height": "100%",
                "background": "rgba(0,0,0,0.75)",
                "z-index": "9999",
                "align-items": "center", "justify-content": "center",
            }
            return overlay_style, "Caricamento dati in corso...", state.get("src",""), False, 0
        return {"display": "none"}, "", "", True, 0

    # ── Aggiorna barra di progresso ad ogni tick ──────────────────────────────
    @app.callback(
        Output("adl-progress-pct",    "children"),
        Output("adl-progress-bar",    "style"),
        Output("adl-progress-detail", "children"),
        Input("adl-progress-tick",    "n_intervals"),
        prevent_initial_call=True,
    )
    def tick_progress(n):
        pct = int(95 * (1 - math.exp(-n * 0.09)))
        pct = min(pct, 93)
        if pct < 20:
            detail = "Connessione ai server..."
        elif pct < 45:
            detail = "Download serie in corso..."
        elif pct < 70:
            detail = "Elaborazione dati temporali..."
        else:
            detail = "Quasi pronto..."
        bar_style = {
            "width": f"{pct}%", "height": "100%",
            "background": "linear-gradient(90deg,#1a5276,#2e86c1)",
            "border-radius": "6px", "transition": "width 0.3s ease",
        }
        return f"{pct}%", bar_style, detail

    # ── Download dati + nasconde overlay via store ────────────────────────────
    @app.callback(
        Output("store-adl-source",        "data"),
        Output("adl-source-status",       "children"),
        Output("store-adl-loading-state", "data",    allow_duplicate=True),
        Input("btn-adl-load",             "n_clicks"),
        State("adl-source-type",          "value"),
        State("adl-eur-geo",              "value"),
        State("api-key",                  "value"),
        prevent_initial_call=True,
    )
    def load_adl_source(n_clicks, source_type, geo, api_key):
        _done  = {"active": False}

        def _err(msg):
            return None, msg, _done

        if not _has_internet():
            return _err("⚠️  Nessuna connessione internet — verifica la rete e riprova")

        if source_type == "eur":
            geo = geo or "EA20"
            print(f"\n▶ ADL — Download Eurostat [{geo}]...")
            df = build_eurostat_dataframe(geo)
            if df.empty:
                return _err("❌ Download Eurostat fallito — server non raggiungibile")
            # Esportazioni nette nominali e reali
            exp_n = next((c for c in df.columns if "Esportazioni EUR Nom." in c), None)
            imp_n = next((c for c in df.columns if "Importazioni EUR Nom." in c), None)
            if exp_n and imp_n:
                df[NET_EXP_EUR_LABEL] = df[exp_n] - df[imp_n]
            exp_r = next((c for c in df.columns if "Esportazioni EUR Reali" in c), None)
            imp_r = next((c for c in df.columns if "Importazioni EUR Reali" in c), None)
            if exp_r and imp_r:
                df[NET_EXP_EUR_R_LABEL] = df[exp_r] - df[imp_r]
            # ── Brent crude (FRED — mensile) ──────────────────────────────────
            _ak = (api_key or FRED_API_KEY).strip()
            brent = fred_get("MCOILBRENTEU", _ak)
            if brent is not None:
                df["Brent Petrolio ($/barile)"] = to_monthly(brent, "M").reindex(
                    pd.date_range(df.index.min(), df.index.max(), freq="MS")
                ).ffill()
                print("  ✓ Brent aggiunto")
            # ── Indice azionario paese (Yahoo Finance) ────────────────────────
            eq_info = EUROSTAT_EQUITY.get(geo)
            if eq_info:
                eq_ticker, eq_name = eq_info
                eq_s = yfinance_monthly(eq_ticker, eq_name)
                if eq_s is not None:
                    df[eq_name] = eq_s.reindex(
                        pd.date_range(df.index.min(), df.index.max(), freq="MS")
                    ).ffill()
            label = EUROSTAT_GEO.get(geo, geo)
        else:
            api_key = (api_key or FRED_API_KEY).strip()
            print(f"\n▶ ADL — Download FRED USA ({len(ADL_USA_SERIES)} serie)...")
            df = build_dataframe(ADL_USA_SERIES, api_key)
            if df.empty:
                return _err("❌ Download FRED fallito — verifica API key o connessione")
            exp_col = next((c for c in df.columns if "Esportazioni USA" in c), None)
            imp_col = next((c for c in df.columns if "Importazioni USA" in c), None)
            if exp_col and imp_col:
                df[ADL_NET_EXP_LABEL] = df[exp_col] - df[imp_col]
            label = "USA (FRED)"

        d1  = df.index.min().strftime("%m/%Y")
        d2  = df.index.max().strftime("%m/%Y")
        msg = f"✅  {label}  |  {len(df.columns)} serie  |  {d1} → {d2}"
        return df.to_json(date_format="iso", orient="split"), msg, _done

    @app.callback(
        Output("adl-equation",        "children"),
        Output("adl-stats-table",     "children"),
        Output("adl-coef-table",      "children"),
        Output("chart-adl-fit",       "figure"),
        Output("chart-adl-irf",       "figure"),
        Output("chart-adl-resid",     "figure"),
        Output("chart-adl-qq",        "figure"),
        Output("adl-status",          "children"),
        Output("adl-irf-sigma-table", "children"),
        Input("btn-run-adl",       "n_clicks"),
        State("store-adl-source",  "data"),
        State("store-adl-extra",   "data"),
        State("adl-y-var",         "value"),
        State({"type": "adl-y-step",   "index": ALL}, "value"),
        State("adl-ar-lags",       "value"),
        State({"type": "adl-x-active", "index": ALL}, "value"),
        State({"type": "adl-x-active", "index": ALL}, "id"),
        State({"type": "adl-x-step",   "index": ALL}, "value"),
        State({"type": "adl-x-step",   "index": ALL}, "id"),
        State({"type": "adl-x-lags",   "index": ALL}, "value"),
        State("adl-cov-type",  "value"),
        State("adl-add-const", "value"),
        State("adl-slider",    "value"),
        State("adl-dummies",   "value"),
        prevent_initial_call=True,
    )
    def run_adl(n_clicks,
                source_data, extra_data,
                y_col, y_step_vals, ar_lags,
                x_active_vals, x_active_ids,
                x_step_vals, x_step_ids,
                x_lags_vals,
                cov_type, add_const, slider_val, dummy_vars):

        err8 = lambda msg: (msg, None, None,
                            empty_fig("Stima il modello"),
                            empty_fig(""), empty_fig(""), empty_fig(""),
                            f"❌ {msg}", None)

        if not y_col:
            return err8("Seleziona una variabile Y")
        if not source_data:
            return err8("Carica i dati prima (sezione ⓪ — Fonte dati)")

        df = pd.read_json(io.StringIO(source_data), orient="split")
        df.index = pd.to_datetime(df.index)
        if extra_data:
            for sid, meta in extra_data.items():
                idx  = pd.to_datetime(meta["dates"])
                vals = pd.array(meta["values"], dtype="Float64")
                df[sid] = pd.Series(vals, index=idx, name=sid)
        df = df.loc[:, ~df.columns.duplicated()]
        if df.empty:
            return err8("Nessun dato caricato")
        if y_col not in df.columns:
            return err8(f"Serie '{y_col}' non trovata")

        if slider_val and (slider_val[1] - slider_val[0]) > 86400:
            start = pd.to_datetime(slider_val[0], unit="s").normalize()
            end   = pd.to_datetime(slider_val[1], unit="s").normalize()
            df = df.loc[start:end]

        # ── Pipeline helper (stessa logica del tab Confronto) ─────────────────
        def _adl_pipeline(raw: pd.Series, step_list: list) -> tuple:
            from scipy.signal import savgol_filter as _sgf
            from statsmodels.tsa.filters.hp_filter import hpfilter as _hpf
            s = raw.ffill().dropna().copy()
            lbl_parts = []
            OP_LBL = {"log":"log","yoy":"YoY%","cumsum":"Σ","diff1":"Δ¹","diff2":"Δ²",
                      "x100":"×100","ma3":"MA3","ma6":"MA6","ma12":"MA12",
                      "ema3":"EMA3","ema6":"EMA6","ema12":"EMA12",
                      "sg":"S-G","hp":"HP","kalman":"Kalman"}
            for tr in step_list:
                if not tr or tr == "none":
                    continue
                if tr == "log":
                    s = np.log(s.clip(lower=1e-9))
                elif tr == "yoy":
                    s = ((s - s.shift(12)) / s.shift(12).abs()) * 100
                elif tr == "cumsum":
                    s = s.cumsum()
                elif tr == "diff1":
                    s = s.diff()
                elif tr == "diff2":
                    s = s.diff().diff()
                elif tr == "x100":
                    s = s * 100
                elif tr == "ma3":
                    s = s.rolling(3,  min_periods=1).mean()
                elif tr == "ma6":
                    s = s.rolling(6,  min_periods=1).mean()
                elif tr == "ma12":
                    s = s.rolling(12, min_periods=1).mean()
                elif tr == "ema3":
                    s = s.ewm(span=3,  adjust=False).mean()
                elif tr == "ema6":
                    s = s.ewm(span=6,  adjust=False).mean()
                elif tr == "ema12":
                    s = s.ewm(span=12, adjust=False).mean()
                elif tr == "sg":
                    clean = s.dropna()
                    wl = min(13, len(clean))
                    if wl % 2 == 0: wl -= 1
                    if wl >= 3:
                        s = pd.Series(_sgf(clean.values.astype(float),
                                          window_length=wl,
                                          polyorder=min(3, wl - 1)),
                                      index=clean.index)
                    else:
                        s = clean
                elif tr == "hp":
                    clean = s.dropna()
                    if len(clean) >= 8:
                        _, trend = _hpf(clean.values.astype(float), lamb=129600)
                        s = pd.Series(trend, index=clean.index)
                    else:
                        s = clean
                elif tr == "kalman":
                    y_k = s.values.astype(float)
                    q   = float(np.var(np.diff(y_k[~np.isnan(y_k)]))) if len(y_k) > 2 else 1.0
                    r   = float(np.var(y_k[~np.isnan(y_k)])) * 0.3 + 1e-9
                    x_e = np.full(len(y_k), np.nan)
                    x_c = y_k[~np.isnan(y_k)][0] if np.any(~np.isnan(y_k)) else 0.0
                    p_c = r
                    for t in range(len(y_k)):
                        p_c += q
                        if not np.isnan(y_k[t]):
                            k_g  = p_c / (p_c + r)
                            x_c  = x_c + k_g * (y_k[t] - x_c)
                            p_c  = (1 - k_g) * p_c
                        x_e[t] = x_c
                    s = pd.Series(x_e, index=s.index)
                lbl_parts.append(OP_LBL.get(tr, tr))
            lbl = " → ".join(lbl_parts) if lbl_parts else "Livelli"
            return s, lbl

        # ── Applica pipeline a Y ──────────────────────────────────────────────
        try:
            y_series, y_pipe_lbl = _adl_pipeline(df[y_col], y_step_vals or [])
            y_series = y_series.dropna()
            y_label  = f"{y_col} [{y_pipe_lbl}]"
        except Exception as _ey:
            return err8(f"Errore pipeline Y: {_ey}")

        # ── Mappa pipeline per X {col: [step0..3]} ────────────────────────────
        from collections import defaultdict as _dd_
        x_pipes = _dd_(list)
        for id_obj, val in zip(x_step_ids or [], x_step_vals or []):
            col_k = id_obj["index"]           # "FEDFUNDS___2"
            col_n, k = col_k.rsplit("___", 1)
            while len(x_pipes[col_n]) <= int(k):
                x_pipes[col_n].append("none")
            x_pipes[col_n][int(k)] = val

        # Costruisci X
        X_dict = {}
        for lag in sorted(ar_lags or []):
            X_dict[f"Y(t-{lag})"] = y_series.shift(lag)
        for av, id_d, lags in zip(x_active_vals, x_active_ids, x_lags_vals):
            if not av:
                continue
            col = id_d["index"]
            if col == y_col or col not in df.columns:
                continue
            try:
                x_base, x_lbl = _adl_pipeline(df[col], x_pipes.get(col, []))
            except Exception as _ex:
                return err8(f"Errore pipeline X '{col}': {_ex}")
            clbl = f"{col} [{x_lbl}]"
            for lag in sorted(lags or [0]):
                X_dict[f"{clbl}_L{lag}" if lag > 0 else clbl] = x_base.shift(lag)

        if not X_dict:
            return err8("Seleziona almeno una variabile X o un lag AR di Y")

        DUMMY_DEFS = {
            "dummy_covid":   ("2020-03-01", "2020-05-01",  "D_COVID"),
            "dummy_inflaz":  ("2021-11-01", "2022-12-01",  "D_Inflaz21"),
            "dummy_ucraina": ("2022-02-01", "2022-12-01",  "D_Ucraina"),
            "dummy_gfc":     ("2008-09-01", "2009-03-01",  "D_GFC"),
            "dummy_dotcom":  ("2000-03-01", "2002-10-01",  "D_Dotcom"),
        }
        DUMMY_COLORS = {
            "dummy_covid":   "rgba(214,39,40,0.10)",
            "dummy_inflaz":  "rgba(255,127,14,0.10)",
            "dummy_ucraina": "rgba(148,103,189,0.10)",
            "dummy_gfc":     "rgba(44,160,44,0.10)",
            "dummy_dotcom":  "rgba(140,86,75,0.10)",
        }
        for dv in (dummy_vars or []):
            if dv in DUMMY_DEFS:
                s0, s1, dcol = DUMMY_DEFS[dv]
                d_idx = y_series.index
                dummy = pd.Series(0.0, index=d_idx)
                dummy.loc[(d_idx >= s0) & (d_idx <= s1)] = 1.0
                X_dict[dcol] = dummy

        combined = pd.DataFrame({"__y__": y_series, **X_dict}).dropna()
        if len(combined) < len(X_dict) + 5:
            return err8(f"Osservazioni insufficienti ({len(combined)}) — "
                         "espandi il periodo o riduci i lag")

        y_fit = combined["__y__"]
        X_fit = combined[[c for c in combined.columns if c != "__y__"]]
        if "const" in (add_const or []):
            X_fit = sm.add_constant(X_fit)

        cov_type = cov_type or "HAC"
        try:
            ols = sm.OLS(y_fit, X_fit)
            if cov_type == "HAC":
                nlag = max(1, int(4 * (len(y_fit) / 100) ** (2/9)))
                model = ols.fit(cov_type="HAC", cov_kwds={"maxlags": nlag})
            elif cov_type == "HC3":
                model = ols.fit(cov_type="HC3")
            else:
                model = ols.fit()
        except Exception as e:
            return err8(f"Errore stima: {e}")

        n_obs    = int(model.nobs)
        n_params = int(model.df_model + 1)

        # Equazione
        terms = []
        for name, val in model.params.items():
            if name == "const":
                terms.append(f"  α = {val:+.4f}")
            else:
                sign = "+" if val >= 0 else "−"
                terms.append(f"  {sign} {abs(val):.4f} · {name}")
        eq_text = f"{y_label} =\n" + "\n".join(terms) + "  + ε"

        # Diagnostiche
        dw = float(sm.stats.stattools.durbin_watson(model.resid))
        jb_s, jb_p, jb_sk, jb_ku = sm.stats.stattools.jarque_bera(model.resid)
        try:
            bp_lm, bp_p, *_ = sm.stats.diagnostic.het_breuschpagan(
                model.resid, model.model.exog)
        except Exception:
            bp_lm, bp_p = float("nan"), float("nan")
        cov_lbl = {"nonrobust": "OLS classico",
                   "HC3": "HC3 eterosch.",
                   "HAC": "HAC Newey-West"}.get(cov_type, cov_type)

        stat_table = _make_stat_table([
            ["Statistica", "Valore", "Note"],
            ["N osservazioni", f"{n_obs}",                        ""],
            ["N parametri",    f"{n_params}",                     "incl. costante"],
            ["Std Error",      cov_lbl,                           ""],
            ["R²",             f"{model.rsquared:.6f}",           ""],
            ["R² adj.",        f"{model.rsquared_adj:.6f}",       ""],
            ["F-stat",         f"{model.fvalue:.4f}",
             f"p={model.f_pvalue:.4e} {_pstar(model.f_pvalue)}"],
            ["AIC",            f"{model.aic:.4f}",                ""],
            ["BIC",            f"{model.bic:.4f}",                ""],
            ["Durbin-Watson",  f"{dw:.4f}",                       "~2 = no autocorr."],
            ["Jarque-Bera",    f"{jb_s:.4f}",
             f"p={jb_p:.4e} {_pstar(jb_p)} | sk={jb_sk:.3f} ku={jb_ku:.3f}"],
            ["Breusch-Pagan",
             "n/d" if bp_lm != bp_lm else f"{bp_lm:.4f}",
             "richiede costante" if bp_lm != bp_lm else f"p={bp_p:.4e} {_pstar(bp_p)}"],
        ], "#1a3a5c")

        # VIF e tabella coefficienti
        conf = model.conf_int(alpha=0.05)
        x_cols_vif = [c for c in X_fit.columns if c != "const"]
        vif_dict = {}
        if len(x_cols_vif) > 1:
            for xc in x_cols_vif:
                other = [c for c in x_cols_vif if c != xc]
                try:
                    r2 = sm.OLS(X_fit[xc],
                                sm.add_constant(X_fit[other])).fit().rsquared
                    vif_dict[xc] = 1 / (1 - r2) if r2 < 1 else np.inf
                except Exception:
                    vif_dict[xc] = np.nan
        else:
            vif_dict = {c: np.nan for c in x_cols_vif}

        coef_rows = [["Variabile", "Coeff.", "Std Err", "t-stat",
                       "p-val", "Sig.", "IC95 inf", "IC95 sup", "VIF"]]
        for var in model.params.index:
            pv  = model.pvalues[var]
            vif = vif_dict.get(var, np.nan)
            vif_s = f"{vif:.2f}" if isinstance(vif, float) and not np.isnan(vif) else "—"
            coef_rows.append([var,
                               f"{model.params[var]:.6f}",
                               f"{model.bse[var]:.6f}",
                               f"{model.tvalues[var]:.4f}",
                               f"{pv:.4e}", _pstar(pv),
                               f"{conf.loc[var, 0]:.6f}",
                               f"{conf.loc[var, 1]:.6f}", vif_s])

        coef_table = html.Div([
            html.Div("Coefficienti",
                     style={"font-size": "11px", "font-weight": "bold",
                            "color": "#1a3a5c", "background": "#eaf4fb",
                            "padding": "5px 10px",
                            "border-radius": "4px 4px 0 0",
                            "border": "1px solid #aed6f1",
                            "border-bottom": "none", "margin-top": "10px"}),
            _make_stat_table(coef_rows, "#2e6da4"),
            html.Div("*** p<0.001  ** p<0.01  * p<0.05  · p<0.10",
                     style={"font-size": "10px", "color": "#777",
                            "font-style": "italic", "margin-top": "3px"}),
        ])

        # Fitted vs Actual
        fitted  = model.fittedvalues
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=y_fit.index, y=y_fit.values,
                                      name="Osservato",
                                      line=dict(color="#1f77b4", width=1.8)))
        fig_fit.add_trace(go.Scatter(x=fitted.index, y=fitted.values,
                                      name="Stimato",
                                      line=dict(color="#d62728", width=1.8, dash="dot")))
        fig_fit.add_hline(y=0, line_color="#aaa", line_dash="dot", line_width=1)

        # bande dummy sul grafico fitted
        for dv in (dummy_vars or []):
            if dv in DUMMY_DEFS:
                s0, s1, col = DUMMY_DEFS[dv]
                fig_fit.add_vrect(
                    x0=s0, x1=s1,
                    fillcolor=DUMMY_COLORS[dv],
                    line_width=0,
                    annotation_text=col,
                    annotation_position="top left",
                    annotation_font=dict(size=8, color="#555"),
                )       
        
        fig_fit.update_layout(
            title=dict(text=f"{y_label}  |  R²={model.rsquared:.4f}  "
                             f"R²adj={model.rsquared_adj:.4f}  "
                             f"DW={dw:.3f}  JB p={jb_p:.3f}",
                       font=dict(size=11), x=0.01),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)),
            margin=dict(t=50, b=30, l=55, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8")
        fig_fit.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
        fig_fit.update_yaxes(showgrid=True, gridcolor="#e8e8e8")

        # IRF dinamica — propagazione AR su H orizzonti
        H_IRF = 12
        # Trasformazioni che restituiscono variazioni decimali (es. 0.03 = 3%)
        _PCT_TRANSFORMS = {"yoy", "ldiff", "log_diff", "pct", "pct_change"}
        _pct_steps  = {"x100", "diff1", "diff2", "dlog", "yoy"}
        _pipe_has_pct = any(v in _pct_steps for v in (y_step_vals or []))
        irf_scale   = 100.0 if _pipe_has_pct else 1.0
        irf_y_label = f"Variazione Y [{y_pipe_lbl}]"
        irf_suffix  = "%" if irf_scale == 100.0 else ""

        # Coefficienti AR estratti dal modello
        ar_coefs = {}
        for var in model.params.index:
            if var.startswith("Y(t-"):
                try:
                    ar_coefs[int(var[4:-1])] = float(model.params[var])
                except ValueError:
                    pass

        # Nomi colonne dummy — escluse dall'IRF
        dummy_cols = {DUMMY_DEFS[dv][2] for dv in (dummy_vars or []) if dv in DUMMY_DEFS}

        # Raggruppa variabili esogene per base (esclude AR e dummy)
        x_groups = {}
        for var in model.params.index:
            if var == "const" or var.startswith("Y(t-") or var in dummy_cols:
                continue
            if "_L" in var:
                parts = var.rsplit("_L", 1)
                try:
                    base, lag = parts[0], int(parts[1])
                except ValueError:
                    base, lag = var, 0
            else:
                base, lag = var, 0
            x_groups.setdefault(base, []).append((lag, float(model.params[var])))

        n_g = len(x_groups)
        _sigma_tbl = None
        if n_g == 0:
            fig_irf = empty_fig("Nessuna variabile X (solo componente AR)")
        else:
            x_sigmas = {c: float(combined[c].std())
                        for c in combined.columns if c != "__y__"}

            # Recupera σ per ogni base (nell'unità di X dopo pipeline, senza *100)
            def _get_sigma(base):
                return next((x_sigmas.get(k, None)
                             for k in combined.columns
                             if k != "__y__" and (k == base or
                                k.startswith(base + "_L") or k == base + "_L0")), None)

            def _fmt_sigma(v):
                """Formato adattivo: 4 cifre sig., senza moltiplicare per 100."""
                if v is None: return "n/d"
                if abs(v) >= 100:  return f"{v:.1f}"
                if abs(v) >= 1:    return f"{v:.3f}"
                if abs(v) >= 0.01: return f"{v:.4f}"
                return f"{v:.2e}"

            cpr = min(3, n_g)
            nri = (n_g + cpr - 1) // cpr
            subplot_titles_irf = [f"{base}  [σ={_fmt_sigma(_get_sigma(base))}]"
                                   for base in x_groups]
            fig_irf = make_subplots(rows=nri, cols=cpr,
                                     subplot_titles=subplot_titles_irf,
                                     vertical_spacing=0.14, horizontal_spacing=0.08)

            # Tabella σ da mostrare sotto il grafico
            sigma_rows = [["Variabile X (dopo pipeline)", "σ (dev. std. campionaria)",
                           "Unità shock", "Interpretazione"]]
            for gi, (base, lc) in enumerate(x_groups.items()):
                ri = gi // cpr + 1; ci = gi % cpr + 1
                color = COLORS[gi % len(COLORS)]
                sigma = _get_sigma(base) or 1.0

                # Nota interpretativa: cosa rappresenta +1σ per questa X
                if "YoY" in base or "yoy" in base.lower():
                    unit_note = "punti percentuali YoY"
                    interp = f"+{_fmt_sigma(sigma)} pp → variazione tipica annua di X"
                elif "Δ¹" in base or "diff1" in base.lower():
                    unit_note = "variazione mensile (Δ)"
                    interp = f"+{_fmt_sigma(sigma)} → variazione mensile tipica di X"
                elif "log" in base.lower():
                    unit_note = "log-differenza"
                    interp = f"+{_fmt_sigma(sigma)} ≈ +{sigma*100:.2f}% variazione"
                else:
                    unit_note = "unità originali X"
                    interp = f"+{_fmt_sigma(sigma)} unità → shock tipico di X"
                sigma_rows.append([base, _fmt_sigma(sigma), unit_note, interp])

                # Effetto diretto di X a ogni orizzonte (coef × σ)
                direct = {lag: coef * sigma for lag, coef in lc}

                # Propagazione dinamica: irf[h] = direct[h] + Σ_j AR_j * irf[h-j]
                irf_vals = []
                for h in range(H_IRF):
                    d = direct.get(h, 0.0)
                    ar_part = sum(ar_coefs.get(j, 0.0) * irf_vals[h - j]
                                  for j in ar_coefs if 0 < j <= h)
                    irf_vals.append(d + ar_part)

                horizons   = list(range(H_IRF))
                irf_scaled = [v * irf_scale for v in irf_vals]
                cum_scaled = list(np.cumsum(irf_scaled))
                fmt = (lambda v: f"{v:+.2f}%") if irf_suffix else (lambda v: f"{v:+.4f}")

                fig_irf.add_trace(go.Bar(
                    x=horizons, y=irf_scaled, name=base,
                    marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in irf_scaled],
                    marker_line_width=0, showlegend=False,
                    text=[fmt(v) for v in irf_scaled],
                    textposition="outside", textfont=dict(size=8)),
                    row=ri, col=ci)
                fig_irf.add_trace(go.Scatter(
                    x=horizons, y=cum_scaled, mode="lines+markers",
                    name=f"Cum. {base}",
                    line=dict(color=color, width=2, dash="dot"),
                    marker=dict(size=5), showlegend=False),
                    row=ri, col=ci)
                fig_irf.add_hline(y=0, line_color="#aaa", line_dash="dot",
                                   line_width=1, row=ri, col=ci)

            fig_irf.update_layout(
                title=dict(text=f"IRF dinamica — risposta di Y a shock +1σ nei {H_IRF} mesi  "
                                "(barre = impatto mensile | linea tratteggiata = cumulato)",
                           font=dict(size=11), x=0.01),
                hovermode="x unified", margin=dict(t=55, b=35, l=50, r=20),
                paper_bgcolor="white", plot_bgcolor="#f8f8f8")
            fig_irf.update_xaxes(showgrid=True, gridcolor="#e8e8e8",
                                   tickmode="linear", tick0=0, dtick=1,
                                   title_text="Mesi dopo lo shock", title_font=dict(size=9))
            fig_irf.update_yaxes(showgrid=True, gridcolor="#e8e8e8",
                                   title_text=irf_y_label, title_font=dict(size=9),
                                   ticksuffix=irf_suffix)

            # Tabella σ da passare alla UI (usata nel div adl-irf-sigma-table)
            _sigma_tbl = _make_stat_table(sigma_rows, "#1a5276")

        # Residui
        resid = model.resid; std_r = resid.std()
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=resid.index, y=resid.values, mode="lines",
                                      name="Residui", line=dict(color="#2ca02c", width=1)))
        fig_res.add_hline(y=0, line_color="#555", line_dash="dot", line_width=1)
        fig_res.add_hline(y= 2*std_r, line_color="#ff7f0e", line_dash="dash",
                           line_width=1, annotation_text="+2σ")
        fig_res.add_hline(y=-2*std_r, line_color="#ff7f0e", line_dash="dash",
                           line_width=1, annotation_text="−2σ")
        fig_res.update_layout(title=dict(text="Residui (±2σ)", font=dict(size=11), x=0.01),
                               hovermode="x unified", margin=dict(t=40, b=30, l=55, r=20),
                               paper_bgcolor="white", plot_bgcolor="#f8f8f8")
        fig_res.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
        fig_res.update_yaxes(showgrid=True, gridcolor="#e8e8e8")

        # QQ plot
        (osm, osr), (slope, intercept, _) = sp_stats.probplot(resid.values)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                     marker=dict(size=4, color="#9467bd"),
                                     name="Quantili campione"))
        xl = np.array([min(osm), max(osm)])
        fig_qq.add_trace(go.Scatter(x=xl, y=slope*xl+intercept,
                                     mode="lines", line=dict(color="#d62728", width=1.5),
                                     name="Normale teorica"))
        fig_qq.update_layout(title=dict(text="Q-Q plot residui", font=dict(size=11), x=0.01),
                              xaxis_title="Quantili teorici",
                              yaxis_title="Quantili campione",
                              legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)),
                              margin=dict(t=40, b=35, l=55, r=20),
                              paper_bgcolor="white", plot_bgcolor="#f8f8f8")
        fig_qq.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
        fig_qq.update_yaxes(showgrid=True, gridcolor="#e8e8e8")

        status = (f"✅  ADL stimato — {n_obs} obs | {n_params} param | "
                  f"R²={model.rsquared:.4f} | F={model.fvalue:.2f} "
                  f"(p={model.f_pvalue:.2e}) | DW={dw:.3f} | JB p={jb_p:.3f}")

        return eq_text, stat_table, coef_table, fig_fit, fig_irf, fig_res, fig_qq, status, _sigma_tbl


    # =========================================================================
    # DSGE — New Keynesian
    # =========================================================================

    @app.callback(
        Output("dsge-equations",   "children"),
        Output("chart-dsge-irf",   "figure"),
        Output("chart-dsge-phase", "figure"),
        Output("chart-dsge-rates", "figure"),
        Output("dsge-status",      "children"),
        Input("btn-run-dsge",      "n_clicks"),
        State("dsge-sigma",         "value"),
        State("dsge-kappa",         "value"),
        State("dsge-beta",          "value"),
        State("dsge-phi-pi",        "value"),
        State("dsge-phi-y",         "value"),
        State("dsge-pi-star",       "value"),
        State("dsge-r-star",        "value"),
        State("dsge-demand-shock",  "value"),
        State("dsge-cost-push",     "value"),
        State("dsge-monetary-shock","value"),
        State("dsge-persistence",   "value"),
        State("dsge-periods",       "value"),
        prevent_initial_call=True,
    )
    def run_dsge(n_clicks,
                 sigma, kappa, beta, phi_pi, phi_y,
                 pi_star, r_star,
                 d_shock, u_shock_0, v_shock_0,
                 rho, T):
        """
        Simula il modello NK a 3 equazioni con aspettative adattive semplificate:

          IS:   x(t) = x(t-1) - (1/σ)·(i(t-1) - π(t-1) - r*) + d(t)
          NKPC: π(t) = π* + β·(π(t-1)−π*) + κ·x(t) + u(t)
          TR:   i(t) = r* + π* + φπ·(π(t)-π*) + φy·x(t) + v(t)

        dove d, u, v seguono AR(1) con persistenza ρ.
        """
        sigma  = float(sigma  or 1.0)
        kappa  = float(kappa  or 0.15)
        beta   = float(beta   or 0.99)
        phi_pi = float(phi_pi or 1.5)
        phi_y  = float(phi_y  or 0.5)
        pi_st  = float(pi_star or 2.0)
        r_st   = float(r_star  or 1.0)
        rho    = float(rho    or 0.5)
        T      = int(T        or 20)
        d_sh   = float(d_shock   or 0.0)
        u_sh   = float(u_shock_0 or 1.0)
        v_sh   = float(v_shock_0 or 0.0)

        u_path = [u_sh  * (rho ** t) for t in range(T)]
        v_path = [v_sh  * (rho ** t) for t in range(T)]
        d_path = [d_sh  * (rho ** t) for t in range(T)]

        x_path  = [d_path[0]]
        pi_path = [pi_st + u_path[0]]
        i_path  = []

        for t in range(1, T):
            x_prev  = x_path[-1]
            pi_prev = pi_path[-1]
            i_t = r_st + pi_st + phi_pi * (pi_prev - pi_st) + phi_y * x_prev + v_path[t-1]
            i_path.append(i_t)
            real_rate = i_t - pi_prev
            x_t  = x_prev - (1/sigma) * (real_rate - r_st) + d_path[t]
            pi_t = pi_st + beta * (pi_prev - pi_st) + kappa * x_t + u_path[t]
            x_path.append(float(x_t))
            pi_path.append(float(pi_t))

        i_path.append(r_st + pi_st + phi_pi*(pi_path[-1]-pi_st) + phi_y*x_path[-1])
        r_real = [i_path[t] - pi_path[t] for t in range(len(i_path))]
        periods = list(range(T))

        bk_ok = phi_pi > 1
        eq_text = (
            f"Modello New Keynesian — parametri\n"
            f"─────────────────────────────────────────────\n"
            f"IS:   x(t) = x(t-1) − (1/σ)·(i(t-1) − π(t-1) − r*) + d(t)\n"
            f"NKPC: π(t) = π* + β·(π(t-1)−π*) + κ·x(t) + u(t)\n"
            f"TR:   i(t) = r* + π* + φπ·(π(t-1)−π*) + φy·x(t-1) + v(t)\n\n"
            f"Parametri: σ={sigma}  κ={kappa}  β={beta}  φπ={phi_pi}  φy={phi_y}\n"
            f"Target: π*={pi_st}%  r*={r_st}%  |  Shock: ρ={rho}\n"
            f"Cond. Blanchard-Kahn (φπ > 1): {'✓ soddisfatta' if bk_ok else '✗ VIOLATA — sistema instabile'}"
        )

        # IRF
        fig_irf = make_subplots(rows=1, cols=3,
                                 subplot_titles=["Output gap (%)",
                                                  "Inflazione (%)",
                                                  "Tasso nominale (%)"],
                                 horizontal_spacing=0.08)
        for data, ci, color, name, eq_val in [
            (x_path,  1, "#1f77b4", "Output gap",     0),
            (pi_path, 2, "#d62728", "Inflazione",     pi_st),
            (i_path,  3, "#2ca02c", "Tasso nominale", r_st + pi_st),
        ]:
            fig_irf.add_trace(go.Scatter(
                x=periods[:len(data)], y=data,
                name=name, line=dict(color=color, width=2.5),
                hovertemplate=f"t=%{{x}}: %{{y:+.3f}}<extra>{name}</extra>"),
                row=1, col=ci)
            fig_irf.add_hline(y=eq_val, line_color="#aaa", line_dash="dot",
                               line_width=1, row=1, col=ci)

        fig_irf.update_layout(
            title=dict(text=f"IRF — shock: domanda={d_sh:+.1f}  "
                             f"costi={u_sh:+.1f}  monetario={v_sh:+.1f}  ρ={rho}",
                       font=dict(size=11), x=0.01),
            hovermode="x unified",
            margin=dict(t=60, b=35, l=50, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8")
        for c in [1, 2, 3]:
            fig_irf.update_xaxes(title_text="Periodi", showgrid=True,
                                  gridcolor="#e8e8e8", row=1, col=c)
            fig_irf.update_yaxes(showgrid=True, gridcolor="#e8e8e8", row=1, col=c)

        # Diagramma di fase
        n_colors = len(x_path)
        norm_t   = [t / max(n_colors - 1, 1) for t in range(n_colors)]
        phase_colors = [f"rgb({int(31+(196-31)*t)},{int(119+(39-119)*t)},{int(180+(40-180)*t)})"
                        for t in norm_t]
        fig_phase = go.Figure()
        for i in range(len(x_path) - 1):
            fig_phase.add_trace(go.Scatter(
                x=[x_path[i], x_path[i+1]],
                y=[pi_path[i], pi_path[i+1]],
                mode="lines",
                line=dict(color=phase_colors[i], width=2),
                showlegend=False,
                hovertemplate=f"t={i}: x={x_path[i]:+.3f}  π={pi_path[i]:+.3f}<extra></extra>"))
        fig_phase.add_trace(go.Scatter(
            x=[x_path[0]], y=[pi_path[0]], mode="markers",
            marker=dict(size=12, color="#2ca02c", symbol="circle",
                        line=dict(width=2, color="white")),
            name="Inizio"))
        fig_phase.add_trace(go.Scatter(
            x=[x_path[-1]], y=[pi_path[-1]], mode="markers",
            marker=dict(size=12, color="#d62728", symbol="square",
                        line=dict(width=2, color="white")),
            name="Fine"))
        fig_phase.add_hline(y=pi_st, line_color="#aaa", line_dash="dash", line_width=1,
                             annotation_text=f"π*={pi_st}%")
        fig_phase.add_vline(x=0, line_color="#aaa", line_dash="dash", line_width=1,
                             annotation_text="x*=0")
        fig_phase.update_layout(
            title=dict(text="Diagramma di fase — la traiettoria "
                             "converge all'equilibrio (x*=0, π*)",
                       font=dict(size=11), x=0.01),
            xaxis_title="Output gap (%)",
            yaxis_title="Inflazione (%)",
            legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)),
            margin=dict(t=50, b=40, l=60, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8")
        fig_phase.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
        fig_phase.update_yaxes(showgrid=True, gridcolor="#e8e8e8")

        # Tasso nominale vs reale
        fig_rates = go.Figure()
        fig_rates.add_trace(go.Scatter(x=periods, y=i_path,
                                        name="Tasso nominale",
                                        line=dict(color="#2ca02c", width=2)))
        fig_rates.add_trace(go.Scatter(x=periods, y=r_real,
                                        name="Tasso reale",
                                        line=dict(color="#ff7f0e", width=2, dash="dot")))
        fig_rates.add_hline(y=r_st + pi_st, line_color="#2ca02c", line_dash="dash",
                             line_width=1, annotation_text=f"i*={r_st+pi_st:.1f}%")
        fig_rates.add_hline(y=r_st, line_color="#ff7f0e", line_dash="dash",
                             line_width=1, annotation_text=f"r*={r_st:.1f}%")
        fig_rates.update_layout(
            title=dict(text="Tasso nominale vs reale nel tempo  "
                             "(equilibrio: i*=r*+π*, r*)",
                       font=dict(size=11), x=0.01),
            yaxis_title="%", hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)),
            margin=dict(t=50, b=35, l=55, r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8")
        fig_rates.update_xaxes(title_text="Periodi", showgrid=True, gridcolor="#e8e8e8")
        fig_rates.update_yaxes(showgrid=True, gridcolor="#e8e8e8")

        status = (f"✅  DSGE simulato — {T} periodi  |  "
                  f"BK: {'✓' if bk_ok else '✗'}  |  "
                  f"x finale={x_path[-1]:+.3f}%  π finale={pi_path[-1]:+.3f}%  "
                  f"i finale={i_path[-1]:+.3f}%")

        return eq_text, fig_irf, fig_phase, fig_rates, status


    # =========================================================================
    # PHILLIPS CURVE — stima
    # =========================================================================
    @app.callback(
        Output("store-phillips",    "data"),
        Output("pc-tab-content",    "children"),
        Output("pc-status",         "children"),
        Output("store-pc-kappa",    "data"),
        Output("store-pc-beta",     "data"),
        Output("store-pc-sigma",    "data"),
        Output("store-pc-phi-pi",   "data"),
        Output("store-pc-phi-y",    "data"),
        Input("btn-run-phillips",   "n_clicks"),
        Input("pc-result-tabs",     "value"),
        Input("pc-exp-methods",     "value"),
        Input("pc-gap-methods",     "value"),
        State("pc-source",          "value"),
        State("pc-freq",            "value"),
        State("pc-hp-lambda",       "value"),
        State("pc-ma-window",       "value"),
        State("pc-nairu-window",    "value"),
        State("pc-gap-mode",        "value"),
        State("pc-pi-star",         "value"),
        State("pc-exp-for-reg",     "value"),
        State("pc-gap-for-reg",     "value"),
        State("store-phillips",     "data"),
        State("api-key",            "value"),
        prevent_initial_call=True,
    )
    def run_phillips(n_clicks, active_tab,
                     exp_methods, gap_methods,
                     source, freq,
                     hp_lambda, ma_window,
                     nairu_window,
                     gap_mode, pi_star,
                     exp_for_reg, gap_for_reg,
                     stored, api_key):
        import statsmodels.api as sm_api
        import warnings
        import traceback

        try:
          return _run_phillips_inner(
            n_clicks, active_tab, source, freq, gap_methods, hp_lambda,
            exp_methods, ma_window, gap_mode, pi_star, exp_for_reg, gap_for_reg,
            stored, api_key, sm_api, nairu_window)
        except Exception as _e:
          tb = traceback.format_exc()
          print("=== PHILLIPS ERROR ===\n" + tb)
          err_div = html.Div([
              html.B("Errore: "), html.Span(str(_e)),
              html.Pre(tb, style={"font-size":"10px","overflow":"auto","max-height":"300px"})
          ], style={"color":"red","padding":"20px"})
          return no_update, err_div, f"❌ {_e}", no_update, no_update, no_update, no_update, no_update

    def _run_phillips_inner(n_clicks, active_tab,
                     source, freq,
                     gap_methods, hp_lambda,
                     exp_methods, ma_window,
                     gap_mode, pi_star,
                     exp_for_reg, gap_for_reg,
                     stored, api_key, sm_api, nairu_window=40):
        import warnings

        ctx = callback_context
        tid = ctx.triggered_id if hasattr(ctx, "triggered_id") else (
            ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        )
        no_content = html.Div("Clicca ▶ Carica & Stima per avviare l'analisi.",
                               style={"color": "#888", "padding": "40px",
                                      "text-align": "center", "font-size": "14px"})

        # ── solo cambio tab: re-render senza re-download ─────────────────────
        if tid != "btn-run-phillips":
            if not stored:
                return no_update, no_content, no_update, no_update, no_update, no_update, no_update, no_update
            try:
                df_stored = pd.read_json(stored) if isinstance(stored, str) else pd.DataFrame(stored)
                df_stored.index = pd.to_datetime(df_stored.index)
            except Exception:
                return no_update, no_content, no_update, no_update, no_update, no_update, no_update, no_update
            fig = _pc_build_content(df_stored, active_tab, gap_methods, exp_methods,
                                    gap_mode, pi_star, exp_for_reg, gap_for_reg,
                                    hp_lambda or 1600, ma_window or 4, freq or "Q",
                                    nairu_win=nairu_window or 40)
            return no_update, fig, no_update, no_update, no_update, no_update, no_update, no_update

        # ── download dati ─────────────────────────────────────────────────────
        _ak  = (api_key or FRED_API_KEY).strip()
        pi_star = float(pi_star or 2.0)
        lam  = int(hp_lambda or 1600)
        maw  = int(ma_window or 4)
        gap_methods  = gap_methods  or ["hp"]
        exp_methods  = exp_methods  or ["adaptive"]
        freq = freq or "Q"

        raw = {}

        if source == "usa":
            # CPI, Real GDP, Unemployment
            for sid, col in [("CPIAUCSL", "cpi"), ("GDPC1", "gdp"),
                               ("UNRATE", "unemp"), ("NROU", "nairu"),
                               ("NROUST", "nairu_st")]:
                s = fred_get(sid, _ak)
                if s is not None:
                    raw[col] = to_monthly(s, "Q" if sid in ("GDPC1","NROU","NROUST") else "M")

            # Breakeven TIPS (5Y)
            if "breakeven" in exp_methods:
                s = fred_get("T5YIE", _ak)
                if s is not None:
                    raw["breakeven"] = to_monthly(s, "M")

            # Survey Michigan 1Y ahead
            if "survey" in exp_methods:
                s = fred_get("MICH", _ak)
                if s is not None:
                    raw["survey"] = to_monthly(s, "M")

            # Fed Funds Rate (per IS curve e Taylor Rule)
            s_rate = fred_get("FEDFUNDS", _ak)
            if s_rate is not None:
                raw["rate"] = to_monthly(s_rate, "M")

            # Labor share (per GMM) — sempre scaricata per USA
            s_ls = fred_get("PRS85006173", _ak)   # Nonfarm business labor share
            if s_ls is not None:
                raw["laborshare"] = to_monthly(s_ls, "Q")
            s_ulc = fred_get("ULCNFB", _ak)        # Unit labor cost (proxy costi marginali)
            if s_ulc is not None:
                raw["ulc"] = to_monthly(s_ulc, "Q")

        else:  # EUR
            # HICP da Eurostat — prc_hicp_midx (definitivo) + ei_cphi_m (flash, più recente)
            hicp = eurostat_get("prc_hicp_midx", {"coicop": "CP00", "unit": "I15"}, "EA20")
            if hicp is None:
                hicp = eurostat_get("prc_hicp_midx", {"coicop": "CP00", "unit": "I15"}, "EA19")
            # Estendi con flash estimate (ei_cphi_m) per avere i mesi più recenti
            hicp_flash = eurostat_hicp_extended("EA20")
            if hicp_flash is not None and hicp is not None:
                # Riscala flash su base 2015=100 usando il periodo di sovrapposizione
                overlap = hicp.index.intersection(hicp_flash.index)
                if len(overlap) >= 6:
                    ratio = hicp.reindex(overlap).mean() / hicp_flash.reindex(overlap).mean()
                    hicp_flash_scaled = hicp_flash * ratio
                    # Estendi hicp con i mesi successivi dall'indice flash
                    new_idx = hicp_flash_scaled.index[hicp_flash_scaled.index > hicp.index.max()]
                    hicp = pd.concat([hicp, hicp_flash_scaled.reindex(new_idx)]).sort_index()
                    print(f"    ✓ HICP esteso con flash estimate fino a {hicp.index.max().strftime('%Y-%m')}")
            elif hicp_flash is not None and hicp is None:
                hicp = hicp_flash
            if hicp is not None:
                raw["cpi"] = to_monthly(hicp, "M")

            # Real GDP da Eurostat
            gdp_e = eurostat_get("namq_10_gdp",
                                  {"na_item": "B1GQ", "unit": "CLV15_MEUR", "s_adj": "SCA"},
                                  "EA20")
            if gdp_e is None:
                gdp_e = eurostat_get("namq_10_gdp",
                                      {"na_item": "B1GQ", "unit": "CLV15_MEUR", "s_adj": "SCA"},
                                      "EA19")
            if gdp_e is not None:
                raw["gdp"] = to_monthly(gdp_e, "Q")

            # Disoccupazione EUR (FRED)
            s = fred_get("LRHUTTTTEZM156S", _ak)
            if s is not None:
                raw["unemp"] = to_monthly(s, "M")

            # NAIRU EUR (OECD via FRED)
            s = fred_get("NAEXKP01EZQ656S", _ak)
            if s is not None:
                raw["nairu"] = to_monthly(s, "Q")

            # BCE Deposit Facility Rate (per IS curve e Taylor Rule)
            # ECBDFR = tasso depositi BCE; fallback: Euribor 3M come proxy
            s_rate = fred_get("ECBDFR", _ak)
            if s_rate is not None:
                raw["rate"] = to_monthly(s_rate, "M")
            else:
                s_rate = fred_get("IR3TIB01EZM156N", _ak)  # Euribor 3M
                if s_rate is not None:
                    raw["rate"] = to_monthly(s_rate, "M")

        # ── unifica frequenza ─────────────────────────────────────────────────
        if not raw:
            return None, no_content, "❌ Nessun dato scaricato.", None, None, None, None, None

        df = pd.DataFrame(raw).sort_index()

        # Ricampiona su frequenza selezionata
        if freq == "M":
            df = df.resample("MS").mean()
        elif freq == "Q":
            df = df.resample("QS").mean()
        else:
            df = df.resample("YS").mean()

        df = df.dropna(subset=["cpi"])

        if df.empty or len(df) < 12:
            return None, no_content, "❌ Dati insufficienti.", None, None, None, None, None

        # ── inflazione come diff log ──────────────────────────────────────────
        _annualize = {"M": 12, "Q": 4, "Y": 1}.get(freq, 4)
        df["pi"] = np.log(df["cpi"]).diff() * 100.0 * _annualize
        df = df.dropna(subset=["pi"])

        # ── output gap: HP filter ─────────────────────────────────────────────
        if "hp" in gap_methods and "gdp" in df.columns:
            gdp_clean = df["gdp"].dropna()
            if len(gdp_clean) >= 8:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cycle, trend = sm_api.tsa.filters.hpfilter(gdp_clean, lamb=lam)
                hp_gap = cycle / trend * 100.0
                df["gap_hp"] = hp_gap.reindex(df.index)

        # ── output gap: NAIRU gap ─────────────────────────────────────────────
        if "nairu" in gap_methods and "unemp" in df.columns:
            nairu_col = "nairu_st" if "nairu_st" in df.columns else "nairu"
            if nairu_col in df.columns:
                df["nairu_fred"] = df[nairu_col]
                df["gap_nairu"] = -(df["unemp"] - df[nairu_col])
            else:
                u_clean = df["unemp"].dropna()
                if len(u_clean) >= 8:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        u_cycle, u_trend = sm_api.tsa.filters.hpfilter(u_clean, lamb=lam)
                    df["gap_nairu"] = -u_cycle.reindex(df.index)

        # ── NAIRU HP filter (sulla disoccupazione) ────────────────────────────
        if "unemp" in df.columns:
            u_clean = df["unemp"].dropna()
            if len(u_clean) >= 8:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, u_trend_hp = sm_api.tsa.filters.hpfilter(u_clean, lamb=lam)
                df["nairu_hp"] = u_trend_hp.reindex(df.index)

        # ── NAIRU forma ridotta (curva di Phillips inversa) ───────────────────
        # Δπ = α + β·u + ε  →  NAIRU_t = -α/β  (costante nel tempo)
        # Per avere una stima time-varying: rolling window di 40 periodi
        if "pi" in df.columns and "unemp" in df.columns:
            dpi   = df["pi"].diff().dropna()
            unemp = df["unemp"].reindex(dpi.index).dropna()
            idx   = dpi.index.intersection(unemp.index)
            if len(idx) >= 20:
                dpi_a = dpi.reindex(idx).values
                u_a   = unemp.reindex(idx).values
                X_rf  = sm_api.add_constant(u_a)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ols_rf = sm_api.OLS(dpi_a, X_rf).fit()
                alpha_rf, beta_rf = ols_rf.params[0], ols_rf.params[1]
                if abs(beta_rf) > 1e-6:
                    nairu_rf_val = -alpha_rf / beta_rf
                    df["nairu_reduced"] = nairu_rf_val  # stima puntuale costante

                # Rolling NAIRU (finestra 40 periodi)
                win = min(int(nairu_window or 40), len(idx) // 2)
                nairu_roll = []
                dates_roll = []
                for i in range(win, len(idx)):
                    sl = slice(i - win, i)
                    X_r = sm_api.add_constant(u_a[sl])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            m = sm_api.OLS(dpi_a[sl], X_r).fit()
                            a0, b0 = m.params[0], m.params[1]
                            nairu_roll.append(-a0 / b0 if abs(b0) > 1e-6 else np.nan)
                        except Exception:
                            nairu_roll.append(np.nan)
                    dates_roll.append(idx[i])
                df["nairu_reduced_roll"] = pd.Series(nairu_roll,
                                                      index=dates_roll).reindex(df.index)

        # ── NAIRU Kalman Filter (State-Space) ─────────────────────────────────
        # Segnale:     Δπ_t = β·(u_t − nairu_t) + ε_t   ε ~ N(0,σ²_ε)
        # Transizione: nairu_t = nairu_{t-1} + η_t        η ~ N(0,σ²_η)
        if "pi" in df.columns and "unemp" in df.columns:
            try:
                from statsmodels.tsa.statespace.mlemodel import MLEModel
                import statsmodels.tsa.statespace.tools as ss_tools

                dpi_kf   = df["pi"].diff().dropna()
                unemp_kf = df["unemp"].reindex(dpi_kf.index).dropna()
                idx_kf   = dpi_kf.index.intersection(unemp_kf.index)

                if len(idx_kf) >= 30:
                    dpi_v  = dpi_kf.reindex(idx_kf).values.astype(float)
                    unemp_v = unemp_kf.reindex(idx_kf).values.astype(float)
                    n = len(dpi_v)

                    # Kalman manuale: random walk NAIRU con curva di Phillips
                    # Stato: x_t = nairu_t
                    # Osservazione: dpi_t = β*(u_t - x_t) + eps
                    # → dpi_t = -β*x_t + β*u_t + eps
                    # Con β fisso (stimato da OLS), filtriamo solo nairu_t

                    # Step 1: stima β da OLS sull'intero campione
                    u_gap_ols = unemp_v - unemp_v.mean()
                    X_b = sm_api.add_constant(u_gap_ols)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        beta_kf = sm_api.OLS(dpi_v, X_b).fit().params[1]

                    # Step 2: Kalman filter manuale
                    sig2_eps = np.var(dpi_v) * 0.9   # varianza osservazione
                    sig2_eta = np.var(dpi_v) * 0.01  # varianza transizione (quanto varia NAIRU)

                    x_filt  = np.zeros(n)
                    P_filt  = np.zeros(n)
                    x_t     = unemp_v.mean()   # inizializzazione
                    P_t     = 1.0

                    for t in range(n):
                        # Predizione
                        x_pred = x_t
                        P_pred = P_t + sig2_eta
                        # Innovazione
                        y_pred = -beta_kf * x_pred + beta_kf * unemp_v[t]
                        innov  = dpi_v[t] - y_pred
                        S      = beta_kf**2 * P_pred + sig2_eps
                        # Aggiornamento
                        K   = -beta_kf * P_pred / S
                        x_t = x_pred + K * innov
                        P_t = (1 - K * (-beta_kf)) * P_pred
                        x_filt[t] = x_t
                        P_filt[t] = P_t

                    nairu_kf_s = pd.Series(x_filt, index=idx_kf)
                    # banda di incertezza ±1σ
                    nairu_kf_hi = nairu_kf_s + np.sqrt(P_filt)
                    nairu_kf_lo = nairu_kf_s - np.sqrt(P_filt)

                    df["nairu_kalman"]    = nairu_kf_s.reindex(df.index)
                    df["nairu_kalman_hi"] = nairu_kf_hi.reindex(df.index)
                    df["nairu_kalman_lo"] = nairu_kf_lo.reindex(df.index)

            except Exception as _ek:
                print(f"  Kalman NAIRU: {_ek}")

        # ── output gap: trend lineare ─────────────────────────────────────────
        if "linear" in gap_methods and "gdp" in df.columns:
            gdp_nn = df["gdp"].dropna()
            if len(gdp_nn) >= 4:
                t = np.arange(len(gdp_nn))
                coef = np.polyfit(t, np.log(gdp_nn.values), 1)
                trend_lin = np.exp(np.polyval(coef, t))
                gap_lin   = (gdp_nn.values / trend_lin - 1) * 100.0
                df.loc[gdp_nn.index, "gap_linear"] = gap_lin

        # ── aspettative adattive ──────────────────────────────────────────────
        df["exp_adaptive"] = df["pi"].shift(1)

        # ── aspettative MA ────────────────────────────────────────────────────
        df["exp_ma"] = df["pi"].rolling(maw).mean().shift(1)

        # ── aspettative breakeven ─────────────────────────────────────────────
        if "breakeven" in raw:
            # resample alla stessa frequenza del df poi allinea per indice
            be = raw["breakeven"].resample({"M": "MS", "Q": "QS", "Y": "YS"}.get(freq, "QS")).mean()
            df["exp_breakeven"] = be.reindex(df.index).ffill().bfill()

        # ── aspettative survey Michigan ───────────────────────────────────────
        if "survey" in raw:
            sv = raw["survey"].resample({"M": "MS", "Q": "QS", "Y": "YS"}.get(freq, "QS")).mean()
            df["exp_survey"] = sv.reindex(df.index).ffill().bfill()

        # ── costi marginali reali (mc_t) per GMM ─────────────────────────────
        _rs = {"M": "MS", "Q": "QS", "Y": "YS"}.get(freq, "QS")
        for _col, _key in [("laborshare", "mc_ls"), ("ulc", "mc_ulc")]:
            if _col in raw:
                _s = raw[_col].resample(_rs).mean()
                _s_nn = _s.dropna()
                if len(_s_nn) >= 8:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, _tr = sm_api.tsa.filters.hpfilter(_s_nn, lamb=lam)
                    _gap = (np.log(_s_nn) - np.log(_tr)).reindex(df.index)
                    df[_key] = _gap

        # ── regressione OLS ───────────────────────────────────────────────────
        exp_col = f"exp_{exp_for_reg}"
        gap_col = f"gap_{gap_for_reg}"

        kappa_est = None
        beta_est  = None
        reg_result_div = html.Div("Seleziona i metodi e premi Stima.",
                                   style={"color": "#888", "padding": "20px"})

        if exp_col in df.columns and gap_col in df.columns:
            use_gap_mode = "gap" in (gap_mode or [])
            reg_df = df[["pi", exp_col, gap_col]].dropna().copy()

            if use_gap_mode:
                reg_df["pi"]      = reg_df["pi"]      - pi_star
                reg_df[exp_col]   = reg_df[exp_col]   - pi_star

            if len(reg_df) >= 10:
                try:
                    Y = reg_df["pi"].values
                    X = sm_api.add_constant(
                            reg_df[[exp_col, gap_col]].values)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ols = sm_api.OLS(Y, X).fit(cov_type="HC3")

                    alpha_e = float(ols.params[0])
                    beta_e  = float(ols.params[1])
                    kappa_e = float(ols.params[2])
                    r2      = float(ols.rsquared)
                    n_obs   = int(ols.nobs)

                    kappa_est = round(kappa_e, 4)
                    beta_est  = round(beta_e,  4)

                    pi_pred  = ols.fittedvalues
                    residual = ols.resid

                    # Tabella risultati
                    def _fmt(v, p):
                        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                        return f"{v:+.4f}{stars}"

                    pv = ols.pvalues
                    mode_lbl = f"Inflation Gap (π − {pi_star}%)" if use_gap_mode else "Livelli"
                    reg_result_div = html.Div([
                        html.H4(f"Risultati OLS — {mode_lbl}",
                                style={"font-size": "13px", "margin-bottom": "12px"}),
                        html.Table([
                            html.Thead(html.Tr([
                                html.Th("Parametro"), html.Th("Stima"),
                                html.Th("Std Err"), html.Th("p-value"), html.Th("")
                            ], style={"background": "#f0f0f0"})),
                            html.Tbody([
                                html.Tr([html.Td("α (costante)"),
                                         html.Td(f"{alpha_e:+.4f}"),
                                         html.Td(f"{ols.bse[0]:.4f}"),
                                         html.Td(f"{pv[0]:.4f}"),
                                         html.Td("≈ π*" if use_gap_mode else "inflazione strutturale")]),
                                html.Tr([html.Td("γ (persistenza aspettative)"),
                                         html.Td(f"{beta_e:+.4f}"),
                                         html.Td(f"{ols.bse[1]:.4f}"),
                                         html.Td(f"{pv[1]:.4f}"),
                                         html.Td("→ β nel DSGE")],
                                         style={"background": "#fff9e6"}),
                                html.Tr([html.Td("δ (pendenza Phillips = κ)"),
                                         html.Td(f"{kappa_e:+.4f}"),
                                         html.Td(f"{ols.bse[2]:.4f}"),
                                         html.Td(f"{pv[2]:.4f}"),
                                         html.Td("→ κ nel DSGE")],
                                         style={"background": "#e8f5e9"}),
                            ])
                        ], style={"width": "100%", "border-collapse": "collapse",
                                   "font-size": "12px",
                                   "border": "1px solid #ddd"}),

                        html.Div([
                            html.Span(f"R² = {r2:.3f}",
                                      style={"background": "#e8f0fe", "padding": "4px 10px",
                                             "border-radius": "12px", "margin-right": "10px",
                                             "font-size": "12px"}),
                            html.Span(f"N = {n_obs}",
                                      style={"background": "#f0f0f0", "padding": "4px 10px",
                                             "border-radius": "12px", "font-size": "12px"}),
                        ], style={"margin-top": "14px"}),

                        html.Hr(),

                        html.H4("Phillips Scatter + retta stimata",
                                style={"font-size": "13px", "margin-bottom": "8px"}),
                        dcc.Graph(figure=_pc_scatter_fig(reg_df, gap_col, exp_col,
                                                          pi_pred, kappa_e, beta_e,
                                                          pi_star, use_gap_mode, freq),
                                  style={"height": "340px"},
                                  config={"displayModeBar": False}),

                        html.Hr(),

                        html.H4("Residui nel tempo",
                                style={"font-size": "13px", "margin-bottom": "8px"}),
                        dcc.Graph(figure=_pc_residuals_fig(reg_df.index, residual),
                                  style={"height": "220px"},
                                  config={"displayModeBar": False}),

                        html.Hr(),
                        html.Div([
                            html.P([html.B("Interpretazione κ: "),
                                    f"Un aumento dell'output gap di 1pp sposta l'inflazione di {kappa_e:+.3f}pp. "
                                    f"{'Curva molto piatta — economia moderna.' if abs(kappa_e) < 0.1 else 'Curva moderata.' if abs(kappa_e) < 0.3 else 'Curva ripida.'}"],
                                   style={"font-size": "12px", "line-height": "1.6"}),
                            html.P([html.B("Interpretazione γ: "),
                                    f"L'inflazione passata spiega il {beta_e*100:.1f}% di quella attuale "
                                    f"({'alta inerzia' if beta_e > 0.7 else 'inerzia moderata' if beta_e > 0.4 else 'bassa inerzia'})."],
                                   style={"font-size": "12px", "line-height": "1.6"}),
                        ], style={"background": "#f8f8f8", "padding": "12px",
                                   "border-radius": "6px", "border-left": "4px solid #1f77b4"}),
                    ])
                except Exception as e:
                    reg_result_div = html.Div(f"Errore regressione: {e}",
                                               style={"color": "red", "padding": "20px"})

        # ── stima IS + Taylor Rule (pre-calcolo per store) ───────────────────
        sigma_est, phi_pi_est, phi_y_est = _pc_estimate_is_taylor(df, pi_star, freq)

        # ── serializza per store ──────────────────────────────────────────────
        store_data = df.to_json(date_format="iso")

        # ── build content per tab attivo ──────────────────────────────────────
        content = _pc_build_content(df, active_tab, gap_methods, exp_methods,
                                    gap_mode, pi_star, exp_for_reg, gap_for_reg,
                                    lam, maw, freq,
                                    reg_result_div=reg_result_div,
                                    nairu_win=int(nairu_window or 40))

        n_series = sum(1 for c in df.columns if not c.startswith(("cpi","gdp","unemp","nairu")))
        status_txt = (f"✅ {source.upper()} | freq={freq} | N={len(df)} | "
                      f"{'κ=' + str(kappa_est) if kappa_est else 'regressione n/d'}"
                      f"  γ={beta_est if beta_est else 'n/d'}")

        return store_data, content, status_txt, kappa_est, beta_est, sigma_est, phi_pi_est, phi_y_est


    def _pc_build_content(df, active_tab, gap_methods, exp_methods,
                           gap_mode, pi_star, exp_for_reg, gap_for_reg,
                           lam, maw, freq, reg_result_div=None, nairu_win=40):
        """Costruisce il contenuto del tab risultati in base al tab attivo."""
        pi_star = float(pi_star or 2.0)

        if active_tab == "pc-tab-prices":
            fig = make_subplots(rows=2, cols=1,
                                 subplot_titles=[
                                     "Inflazione  Δlog(P) annualizzata  (%)",
                                     "Inflation Gap  (π − π*)  (%)"],
                                 vertical_spacing=0.12)
            if "pi" in df.columns:
                # riga 1: inflazione periodale + YoY se CPI disponibile
                fig.add_trace(go.Scatter(x=df.index, y=df["pi"],
                                          name="Inflazione Δlog(P)",
                                          line=dict(color="#d62728", width=2)),
                              row=1, col=1)
                if "cpi" in df.columns:
                    periods_per_year = {"M": 12, "Q": 4, "Y": 1}.get(freq, 4)
                    pi_yoy = np.log(df["cpi"]).diff(periods_per_year) * 100.0
                    fig.add_trace(go.Scatter(x=df.index, y=pi_yoy,
                                              name="Inflazione YoY",
                                              line=dict(color="#1f77b4", width=1.8,
                                                         dash="dot")),
                                  row=1, col=1)
                fig.add_hline(y=0, line_color="#aaa", line_dash="dot",
                               line_width=1, row=1, col=1)

                # riga 2: inflation gap
                gap_inf = df["pi"] - pi_star
                fig.add_trace(go.Bar(x=df.index, y=gap_inf,
                                      name="Inflation Gap",
                                      marker_color=["#d62728" if v > 0 else "#1f77b4"
                                                     for v in gap_inf.fillna(0)]),
                              row=2, col=1)
                fig.add_hline(y=0, line_color="#2ca02c", line_dash="dash",
                               line_width=1.5,
                               annotation_text=f"π*={pi_star}%", row=2, col=1)

            fig.update_layout(hovermode="x unified",
                               margin=dict(t=50, b=30, l=55, r=20),
                               paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                               legend=dict(orientation="h", y=1.04, x=0,
                                           font=dict(size=10)))
            return dcc.Graph(figure=fig, style={"height": "calc(100vh - 190px)"},
                             config={"displayModeBar": False})

        elif active_tab == "pc-tab-gap":
            has_gdp = "gdp" in df.columns and df["gdp"].dropna().shape[0] >= 4
            rows = 2 if has_gdp else 1
            subtitles = ["Output Gap (%)"]
            if has_gdp:
                subtitles.append(f"PIL reale vs Trend HP  (λ={lam})")
            fig = make_subplots(rows=rows, cols=1,
                                 subplot_titles=subtitles,
                                 vertical_spacing=0.10,
                                 row_heights=[0.45, 0.55] if has_gdp else [1.0])

            # ── riga 1: output gap ────────────────────────────────────────────
            colors = {"gap_hp": "#1f77b4", "gap_nairu": "#d62728",
                      "gap_linear": "#2ca02c"}
            labels = {"gap_hp": f"HP Filter (λ={lam})",
                      "gap_nairu": "NAIRU Gap",
                      "gap_linear": "Trend lineare"}
            for col in ["gap_hp", "gap_nairu", "gap_linear"]:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[col],
                                              name=labels[col],
                                              line=dict(color=colors[col], width=2)),
                                  row=1, col=1)
            fig.add_hline(y=0, line_color="#aaa", line_dash="dash",
                           line_width=1, row=1, col=1)

            # ── riga 2: PIL reale + trend HP ──────────────────────────────────
            if has_gdp:
                import warnings as _w
                import statsmodels.api as _sm
                gdp_s = df["gdp"].dropna()
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    _, hp_trend = _sm.tsa.filters.hpfilter(gdp_s, lamb=lam)
                fig.add_trace(go.Scatter(x=gdp_s.index, y=gdp_s.values,
                                          name="PIL reale",
                                          line=dict(color="#ff7f0e", width=2)),
                              row=2, col=1)
                fig.add_trace(go.Scatter(x=gdp_s.index, y=hp_trend.values,
                                          name=f"Trend HP (λ={lam})",
                                          line=dict(color="#9467bd", width=2,
                                                     dash="dash")),
                              row=2, col=1)

            fig.update_layout(hovermode="x unified",
                               margin=dict(t=50, b=30, l=65, r=20),
                               paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                               legend=dict(orientation="h", y=1.04, x=0,
                                           font=dict(size=10)))
            fig.update_yaxes(title_text="%",    row=1, col=1)
            if has_gdp:
                fig.update_yaxes(title_text="Mld", row=2, col=1)
            return dcc.Graph(figure=fig, style={"height": "calc(100vh - 190px)"},
                             config={"displayModeBar": False})

        elif active_tab == "pc-tab-nairu":
            has_unemp = "unemp" in df.columns and df["unemp"].dropna().shape[0] >= 8

            def _nairu_panel(title, nairu_col, nairu_label, color, fill_color=None,
                              hi_col=None, lo_col=None, nairu_const=None, note=None):
                """Crea un grafico 2-righe: disoccupazione+NAIRU sopra, gap sotto."""
                f = make_subplots(rows=2, cols=1,
                                   subplot_titles=[
                                       f"{title}  —  disoccupazione vs NAIRU (%)",
                                       "Gap  (u − NAIRU)  [pp]",
                                   ],
                                   vertical_spacing=0.10,
                                   row_heights=[0.60, 0.40])
                # disoccupazione
                f.add_trace(go.Scatter(x=df.index, y=df["unemp"],
                                        name="Disoccupazione",
                                        line=dict(color="#333", width=2)),
                            row=1, col=1)
                # banda incertezza Kalman
                if hi_col and lo_col and hi_col in df.columns and lo_col in df.columns:
                    hi = df[hi_col].dropna(); lo = df[lo_col].dropna()
                    ib = hi.index.intersection(lo.index)
                    f.add_trace(go.Scatter(
                        x=list(ib) + list(ib[::-1]),
                        y=list(hi.reindex(ib)) + list(lo.reindex(ib)[::-1]),
                        fill="toself", fillcolor=fill_color or "rgba(44,160,44,0.15)",
                        line=dict(width=0), name="±1σ incertezza"),
                        row=1, col=1)
                # NAIRU time-varying
                if nairu_col and nairu_col in df.columns:
                    f.add_trace(go.Scatter(x=df.index, y=df[nairu_col],
                                            name=nairu_label,
                                            line=dict(color=color, width=2.2)),
                                row=1, col=1)
                # NAIRU costante (forma ridotta)
                if nairu_const is not None:
                    f.add_hline(y=nairu_const, line_color=color, line_dash="dash",
                                 line_width=1.8,
                                 annotation_text=f"NAIRU={nairu_const:.2f}%",
                                 row=1, col=1)
                # gap
                nairu_ref = df[nairu_col] if (nairu_col and nairu_col in df.columns) \
                            else pd.Series(nairu_const, index=df.index)
                gap = df["unemp"] - nairu_ref
                f.add_trace(go.Bar(x=df.index, y=gap,
                                    name="Gap u − NAIRU",
                                    marker_color=["#d62728" if v > 0 else "#1f77b4"
                                                   for v in gap.fillna(0)]),
                            row=2, col=1)
                f.add_hline(y=0, line_color="#aaa", line_dash="dash",
                             line_width=1, row=2, col=1)
                if note:
                    f.add_annotation(text=note, xref="paper", yref="paper",
                                      x=0.01, y=-0.06, showarrow=False,
                                      font=dict(size=9, color="#888"), align="left")
                f.update_layout(hovermode="x unified",
                                 margin=dict(t=45, b=40, l=60, r=15),
                                 paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                                 legend=dict(orientation="h", y=1.05, x=0,
                                             font=dict(size=9)))
                f.update_yaxes(title_text="%",  row=1, col=1)
                f.update_yaxes(title_text="pp", row=2, col=1)
                return f

            panels = []

            if has_unemp:
                H = "420px"

                # ── 1. NAIRU FRED (benchmark) ─────────────────────────────────
                if "nairu_fred" in df.columns:
                    panels.append(html.Div([
                        html.H4("① NAIRU CBO/OECD (fonte FRED)",
                                style={"font-size": "13px", "margin": "14px 0 4px",
                                       "color": "#9467bd"}),
                        html.P("Stima ufficiale del CBO (USA) o OECD (EUR). "
                               "Costruita con modelli strutturali complessi — usata come benchmark.",
                               style={"font-size": "11px", "color": "#666",
                                      "margin": "0 0 6px"}),
                        dcc.Graph(figure=_nairu_panel(
                            "NAIRU FRED", "nairu_fred", "NAIRU CBO/OECD", "#9467bd"),
                            style={"height": H}, config={"displayModeBar": False}),
                    ]))

                # ── 2. NAIRU HP Filter ────────────────────────────────────────
                if "nairu_hp" in df.columns:
                    panels.append(html.Div([
                        html.H4(f"② NAIRU HP Filter  (λ={lam})",
                                style={"font-size": "13px", "margin": "18px 0 4px",
                                       "color": "#1f77b4"}),
                        html.P(f"Trend di Hodrick-Prescott sulla serie della disoccupazione con λ={lam}. "
                               "Semplice ma sensibile alla scelta di λ e al problema del 'end-point bias'.",
                               style={"font-size": "11px", "color": "#666",
                                      "margin": "0 0 6px"}),
                        dcc.Graph(figure=_nairu_panel(
                            "HP Filter", "nairu_hp", f"NAIRU HP (λ={lam})", "#1f77b4"),
                            style={"height": H}, config={"displayModeBar": False}),
                    ]))

                # ── 3. Forma ridotta (curva di Phillips) ──────────────────────
                nr_const = None
                if "nairu_reduced" in df.columns:
                    s = df["nairu_reduced"].dropna()
                    nr_const = float(s.iloc[0]) if len(s) > 0 else None

                nairu_roll_col = "nairu_reduced_roll" if "nairu_reduced_roll" in df.columns else None
                if nr_const is not None or nairu_roll_col:
                    panels.append(html.Div([
                        html.H4(f"③ NAIRU Forma Ridotta  (Δπ = α + β·u)  —  finestra {nairu_win}p",
                                style={"font-size": "13px", "margin": "18px 0 4px",
                                       "color": "#d62728"}),
                        html.P([
                            "Stima econometrica: ",
                            html.B("NAIRU = −α/β"),
                            f" da regressione OLS su tutto il campione → {nr_const:.2f}% (linea tratteggiata). "
                            f"La curva rossa mostra la stima rolling su finestra mobile di {nairu_win} periodi "
                            "— cattura come il NAIRU cambia nel tempo. "
                            "Modifica lo slider 'Finestra NAIRU' e riclicca ▶ per ricalcolare.",
                        ], style={"font-size": "11px", "color": "#666", "margin": "0 0 6px"}),
                        dcc.Graph(figure=_nairu_panel(
                            "Forma Ridotta", nairu_roll_col,
                            f"NAIRU rolling ({nairu_win}p)", "#d62728",
                            nairu_const=nr_const,
                            note=f"Linea tratteg. = OLS full-sample  |  Curva = rolling {nairu_win} periodi"),
                            style={"height": H}, config={"displayModeBar": False}),
                    ]))

                # ── 4. Kalman Filter ──────────────────────────────────────────
                if "nairu_kalman" in df.columns:
                    panels.append(html.Div([
                        html.H4("④ NAIRU Kalman Filter  (State-Space)",
                                style={"font-size": "13px", "margin": "18px 0 4px",
                                       "color": "#2ca02c"}),
                        html.P("Il gold standard delle banche centrali. "
                               "Il NAIRU evolve come random walk; il filtro aggiorna la stima "
                               "ogni periodo usando la sorpresa inflazionistica. "
                               "La banda verde mostra l'incertezza ±1σ della stima.",
                               style={"font-size": "11px", "color": "#666",
                                      "margin": "0 0 6px"}),
                        dcc.Graph(figure=_nairu_panel(
                            "Kalman Filter", "nairu_kalman", "NAIRU Kalman", "#2ca02c",
                            fill_color="rgba(44,160,44,0.15)",
                            hi_col="nairu_kalman_hi", lo_col="nairu_kalman_lo",
                            note="Banda = ±1σ — quanto è incerta la stima del NAIRU in quel periodo"),
                            style={"height": H}, config={"displayModeBar": False}),
                    ]))

            if not panels:
                return html.Div("Nessun dato di disoccupazione disponibile. "
                                "Clicca ▶ Carica & Stima.",
                                style={"padding": "40px", "color": "#888",
                                       "text-align": "center"})

            return html.Div(panels, style={"padding": "0 10px 30px"})

        elif active_tab == "pc-tab-exp":
            fig = go.Figure()
            exp_cols = {"exp_adaptive": ("Adattive π_{t-1}", "#1f77b4"),
                        "exp_ma":       (f"MA({maw})",       "#ff7f0e"),
                        "exp_breakeven":("Breakeven TIPS",   "#2ca02c"),
                        "exp_survey":   ("Survey Michigan",  "#9467bd")}
            if "pi" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["pi"],
                                          name="Inflazione effettiva",
                                          line=dict(color="#d62728", width=2.5)))
            _exp_key_map = {"exp_adaptive": "adaptive", "exp_ma": "ma",
                            "exp_breakeven": "breakeven", "exp_survey": "survey"}
            _active_exp = set(exp_methods or [])
            for col, (lbl, clr) in exp_cols.items():
                if _exp_key_map.get(col) not in _active_exp:
                    continue
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[col],
                                              name=lbl,
                                              line=dict(color=clr, width=1.8,
                                                         dash="dot")))
            fig.add_hline(y=pi_star, line_color="#aaa", line_dash="dash",
                           line_width=1, annotation_text=f"π*={pi_star}%")
            fig.update_layout(title="Aspettative d'inflazione vs inflazione effettiva",
                               yaxis_title="%", hovermode="x unified",
                               margin=dict(t=50,b=30,l=55,r=20),
                               paper_bgcolor="white", plot_bgcolor="#f8f8f8")
            return dcc.Graph(figure=fig, style={"height": "calc(100vh - 190px)"},
                             config={"displayModeBar": False})

        elif active_tab == "pc-tab-scatter":
            exp_col = f"exp_{exp_for_reg}"
            gap_col = f"gap_{gap_for_reg}"
            if exp_col not in df.columns or gap_col not in df.columns:
                return html.Div("Seleziona output gap e aspettative validi per lo scatter.",
                                 style={"padding": "40px", "color": "#888"})
            use_gap = "gap" in (gap_mode or [])
            plot_df = df[[gap_col, "pi"]].dropna().copy()
            if use_gap:
                plot_df["pi"] = plot_df["pi"] - pi_star
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df[gap_col], y=plot_df["pi"],
                                      mode="markers",
                                      marker=dict(color="#1f77b4", size=6, opacity=0.7),
                                      name="Osservazioni"))
            if len(plot_df) >= 4:
                import warnings
                import statsmodels.api as sm_api
                X = sm_api.add_constant(plot_df[gap_col].values)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ols = sm_api.OLS(plot_df["pi"].values, X).fit()
                x_rng = np.linspace(plot_df[gap_col].min(), plot_df[gap_col].max(), 80)
                y_rng = ols.params[0] + ols.params[1] * x_rng
                fig.add_trace(go.Scatter(x=x_rng, y=y_rng, mode="lines",
                                          line=dict(color="#d62728", width=2),
                                          name=f"OLS (κ={ols.params[1]:+.3f})"))
            fig.add_hline(y=0, line_color="#aaa", line_dash="dot", line_width=1)
            fig.add_vline(x=0, line_color="#aaa", line_dash="dot", line_width=1)
            y_lbl = "Inflation Gap (π − π*)" if use_gap else "Inflazione (%)"
            _gap_lbl = {"hp": f"HP Filter (λ={lam})", "nairu": "NAIRU Gap",
                        "linear": "Trend lineare"}.get(gap_for_reg, gap_for_reg)
            _exp_lbl = {"adaptive": "π_{t-1}", "ma": f"MA({maw})",
                        "breakeven": "Breakeven TIPS", "survey": "Survey Michigan"}.get(exp_for_reg, exp_for_reg)
            _pi_lbl  = f"π_t − π*" if use_gap else "π_t"
            _eq_str  = f"π_t = α + γ·{_exp_lbl} + κ·ỹ_t   [ỹ = {_gap_lbl}]"
            fig.update_layout(
                title=dict(text=_eq_str, font=dict(size=12)),
                xaxis_title=f"Output Gap ỹ_t  —  {_gap_lbl}  (%)",
                yaxis_title=f"{_pi_lbl}  (%)",
                margin=dict(t=50,b=40,l=55,r=20),
                paper_bgcolor="white", plot_bgcolor="#f8f8f8")
            return dcc.Graph(figure=fig, style={"height": "calc(100vh - 190px)"},
                             config={"displayModeBar": False})

        elif active_tab == "pc-tab-reg":
            if reg_result_div is not None:
                return reg_result_div
            return html.Div("Clicca ▶ Carica & Stima per vedere i risultati.",
                             style={"padding": "40px", "color": "#888",
                                    "text-align": "center"})

        elif active_tab == "pc-tab-gmm":
            return _pc_gmm_tab(df, pi_star, lam, freq)

        elif active_tab == "pc-tab-is-taylor":
            return _pc_is_taylor_tab(df, pi_star, freq)

        return html.Div()


    def _pc_scatter_fig(reg_df, gap_col, exp_col, pi_pred, kappa, beta,
                         pi_star, use_gap_mode, freq):
        """Scatter π vs output gap con retta stimata e colorazione temporale."""
        import plotly.express as px
        n = len(reg_df)
        norm_t = np.linspace(0, 1, n)
        colors = [f"rgb({int(31+165*t)},{int(119+(-80)*t)},{int(180+(-140)*t)})"
                  for t in norm_t]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=reg_df[gap_col].values, y=reg_df["pi"].values,
            mode="markers",
            marker=dict(color=colors, size=6, opacity=0.8,
                        colorscale="RdBu", showscale=False),
            text=[str(d)[:10] for d in reg_df.index],
            hovertemplate="Gap: %{x:+.2f}%<br>π: %{y:+.2f}%<br>%{text}<extra></extra>",
            name="Osservazioni"))
        x_rng = np.linspace(reg_df[gap_col].min(), reg_df[gap_col].max(), 80)
        y_rng = (np.mean(reg_df["pi"].values - beta * reg_df[exp_col].values
                          - kappa * reg_df[gap_col].values)
                  + beta * np.mean(reg_df[exp_col].values) + kappa * x_rng)
        fig.add_trace(go.Scatter(
            x=x_rng, y=y_rng, mode="lines",
            line=dict(color="#d62728", width=2.5),
            name=f"Retta OLS  κ={kappa:+.3f}"))
        fig.add_hline(y=0, line_color="#aaa", line_dash="dot", line_width=1)
        fig.add_vline(x=0, line_color="#aaa", line_dash="dot", line_width=1)
        y_lbl = "Inflation Gap (π − π*)" if use_gap_mode else "Inflazione (%)"
        fig.update_layout(
            xaxis_title="Output Gap (%)", yaxis_title=y_lbl,
            margin=dict(t=20,b=40,l=55,r=20),
            paper_bgcolor="white", plot_bgcolor="#f8f8f8",
            legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)))
        return fig


    def _pc_residuals_fig(index, residual):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=index, y=residual,
                              marker_color=["#d62728" if r > 0 else "#1f77b4"
                                             for r in residual],
                              name="Residuo"))
        fig.add_hline(y=0, line_color="#aaa", line_dash="dash", line_width=1)
        fig.update_layout(yaxis_title="%", showlegend=False,
                           margin=dict(t=10,b=30,l=45,r=10),
                           paper_bgcolor="white", plot_bgcolor="#f8f8f8")
        return fig


    def _pc_gmm_tab(df, pi_star, lam, freq):
        """Stima GMM Galí-Gertler (1999): π_t = θ·π_{t-1} + γ·π_{t+1} + λ·mc_t + ε_t"""
        import warnings
        from statsmodels.sandbox.regression.gmm import LinearIVGMM
        import statsmodels.api as _sm

        # ── sceglie il miglior proxy mc_t disponibile ─────────────────────────
        mc_col = None
        for c in ["mc_ls", "mc_ulc"]:
            if c in df.columns and df[c].dropna().shape[0] >= 20:
                mc_col = c
                break

        mc_label = {"mc_ls": "Labor Share (HP-detrended)",
                    "mc_ulc": "Unit Labor Cost (HP-detrended)"}.get(mc_col, "n/d")

        # ── costruisce variabili ──────────────────────────────────────────────
        g = df.copy()
        g["pi_lag1"] = g["pi"].shift(1)
        g["pi_lag2"] = g["pi"].shift(2)
        g["pi_lag3"] = g["pi"].shift(3)
        g["pi_fwd"]  = g["pi"].shift(-1)
        if mc_col:
            g["mc_lag1"] = g[mc_col].shift(1)
            g["mc_lag2"] = g[mc_col].shift(2)

        cols_need = ["pi","pi_lag1","pi_lag2","pi_lag3","pi_fwd"]
        if mc_col:
            cols_need += [mc_col, "mc_lag1", "mc_lag2"]
        g = g[cols_need].dropna()

        if len(g) < 30:
            return html.Div("Dati insufficienti per GMM (< 30 osservazioni). "
                             "Prova con frequenza trimestrale.",
                             style={"padding":"40px","color":"#888","text-align":"center"})

        # ── matrice X (regressori, incluso π_{t+1} endogeno) ─────────────────
        if mc_col:
            X = np.column_stack([np.ones(len(g)),
                                  g["pi_lag1"].values,
                                  g["pi_fwd"].values,
                                  g[mc_col].values])
            Z = np.column_stack([np.ones(len(g)),
                                  g["pi_lag1"].values,
                                  g["pi_lag2"].values,
                                  g["pi_lag3"].values,
                                  g["mc_lag1"].values,
                                  g["mc_lag2"].values])
            param_names = ["α (costante)", "θ (backward)", "γ (forward)", "λ (mc_t)"]
        else:
            X = np.column_stack([np.ones(len(g)),
                                  g["pi_lag1"].values,
                                  g["pi_fwd"].values])
            Z = np.column_stack([np.ones(len(g)),
                                  g["pi_lag1"].values,
                                  g["pi_lag2"].values,
                                  g["pi_lag3"].values])
            param_names = ["α (costante)", "θ (backward)", "γ (forward)"]

        Y = g["pi"].values

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                start = [0.5] * X.shape[1]
                gmm_res = LinearIVGMM(Y, X, Z).fit(
                    start_params=start, maxiter=300, optim_method="bfgs")
        except Exception as e:
            return html.Div(f"Errore GMM: {e}",
                             style={"color":"red","padding":"20px"})

        theta = float(gmm_res.params[1])
        gamma = float(gmm_res.params[2])
        lam_mc = float(gmm_res.params[3]) if mc_col else None
        pv = gmm_res.pvalues
        se = gmm_res.bse

        def _stars(p):
            return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

        # ── tabella risultati ─────────────────────────────────────────────────
        rows = []
        for i, nm in enumerate(param_names):
            rows.append(html.Tr([
                html.Td(nm, style={"padding":"4px 8px"}),
                html.Td(f"{gmm_res.params[i]:+.4f}", style={"padding":"4px 8px","font-weight":"bold"}),
                html.Td(f"{se[i]:.4f}", style={"padding":"4px 8px","color":"#666"}),
                html.Td(f"{pv[i]:.4f}", style={"padding":"4px 8px"}),
                html.Td(_stars(pv[i]), style={"padding":"4px 8px","color":"#d62728"}),
            ], style={"background": "#fff9e6" if i==1 else "#e8f5e9" if i==2 else
                                    "#fce8e8" if i==3 else "white"}))

        # ── fitted vs actual ──────────────────────────────────────────────────
        fitted = gmm_res.fittedvalues
        fig_fit = make_subplots(rows=2, cols=1,
                                 subplot_titles=["π osservata vs π stimata (%)",
                                                  "Residui GMM (%)"],
                                 vertical_spacing=0.12, row_heights=[0.65, 0.35])
        fig_fit.add_trace(go.Scatter(x=g.index, y=Y,
                                      name="π osservata",
                                      line=dict(color="#333", width=1.8)),
                          row=1, col=1)
        fig_fit.add_trace(go.Scatter(x=g.index, y=fitted,
                                      name="π stimata GMM",
                                      line=dict(color="#1f77b4", width=2, dash="dot")),
                          row=1, col=1)
        fig_fit.add_hline(y=pi_star, line_color="#2ca02c", line_dash="dash",
                           line_width=1, annotation_text=f"π*={pi_star}%", row=1, col=1)
        resid = Y - fitted
        fig_fit.add_trace(go.Bar(x=g.index, y=resid,
                                  marker_color=["#d62728" if r>0 else "#1f77b4" for r in resid],
                                  name="Residui"),
                          row=2, col=1)
        fig_fit.add_hline(y=0, line_color="#aaa", line_dash="dash", line_width=1, row=2, col=1)
        fig_fit.update_layout(hovermode="x unified", margin=dict(t=45,b=30,l=55,r=15),
                               paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                               legend=dict(orientation="h", y=1.04, x=0, font=dict(size=9)))

        # ── interpretazione ───────────────────────────────────────────────────
        fwd_dom = gamma > theta
        interpret = html.Div([
            html.H4("Interpretazione", style={"font-size":"13px","margin":"16px 0 8px"}),
            html.Div([
                html.P([
                    html.B("Equazione stimata:  "),
                    f"π_t = {gmm_res.params[0]:+.3f}  +  {theta:+.3f}·π_{{t-1}}  "
                    f"+  {gamma:+.3f}·π_{{t+1}}",
                    (f"  +  {lam_mc:+.4f}·mc_t" if lam_mc else ""),
                ], style={"font-size":"12px","font-family":"monospace",
                           "background":"#f5f5f5","padding":"8px","border-radius":"4px"}),
                html.P([
                    html.B("θ (backward) = "), f"{theta:.3f}  →  ",
                    html.B("γ (forward) = "), f"{gamma:.3f}  →  ",
                    html.Span(
                        "l'economia è prevalentemente FORWARD-LOOKING " if fwd_dom
                        else "l'economia è prevalentemente BACKWARD-LOOKING (alta inerzia) ",
                        style={"color":"#2ca02c" if fwd_dom else "#d62728",
                               "font-weight":"bold"}),
                ], style={"font-size":"12px","margin-top":"8px"}),
                html.P([
                    html.B("θ + γ = "), f"{theta+gamma:.3f}  ",
                    html.Span(
                        "(< 1: sistema stabile, inflazione converge)" if theta+gamma < 1
                        else "(≥ 1: ATTENZIONE — inflazione esplosiva)",
                        style={"color": "#2ca02c" if theta+gamma < 1 else "#d62728"}),
                ], style={"font-size":"12px"}),
                html.P([
                    html.B("Implicazione per politica monetaria:  "),
                    ("Se γ > θ la BCE dovrebbe guardare principalmente alle aspettative future "
                     "— alzare i tassi oggi è giustificato solo se le aspettative restano "
                     "disancorate, non solo perché l'inflazione passata è alta."
                     if fwd_dom else
                     "Con alta inerzia (θ > γ) la disinflazione richiede più tempo e "
                     "costi reali maggiori — la BC deve mantenere i tassi elevati più a lungo."),
                ], style={"font-size":"12px","margin-top":"8px","line-height":"1.6",
                           "background":"#e8f0fe","padding":"10px","border-radius":"4px",
                           "border-left":"4px solid #1f77b4"}),
                html.P([
                    html.B("Costi marginali reali (λ): "),
                    (f"{lam_mc:+.4f} — " + (
                        "coefficiente piccolo: i costi reali (salari) influenzano "
                        "poco l'inflazione, coerente con la curva di Phillips piatta. "
                        "I prezzi dell'energia si trasmettono solo se i salari si adeguano."
                        if lam_mc is not None and abs(lam_mc) < 0.5
                        else "coefficiente significativo: i costi marginali reali guidano l'inflazione."
                    ) if lam_mc is not None else "Non disponibile (mancano dati Labor Share)."),
                ], style={"font-size":"12px","margin-top":"8px"}),
                html.P(f"N = {len(g)}  |  Strumenti Z: π_{{t-1}}, π_{{t-2}}, π_{{t-3}}"
                        + (", mc_{{t-1}}, mc_{{t-2}}" if mc_col else "") +
                       f"  |  mc_t = {mc_label}",
                       style={"font-size":"10px","color":"#888","margin-top":"12px"}),
            ], style={"background":"#fafafa","padding":"14px","border-radius":"6px",
                       "border":"1px solid #e0e0e0"}),
        ])

        return html.Div([
            html.H4("GMM Galí-Gertler (1999) — NKPC Ibrida",
                     style={"font-size":"14px","margin":"10px 0 4px",
                             "color":"#1f77b4","border-bottom":"2px solid #1f77b4",
                             "padding-bottom":"6px"}),
            html.P("π_t = α + θ·π_{t-1} + γ·E_t[π_{t+1}] + λ·mc_t + ε_t   "
                   "— π_{t+1} trattata come endogena, strumentata con valori ritardati",
                   style={"font-size":"11px","color":"#666","margin":"0 0 12px"}),

            html.Table([
                html.Thead(html.Tr([
                    html.Th("Parametro"), html.Th("Stima"), html.Th("Std Err"),
                    html.Th("p-value"), html.Th("")
                ], style={"background":"#f0f0f0","font-size":"12px"})),
                html.Tbody(rows),
            ], style={"width":"100%","border-collapse":"collapse",
                       "font-size":"12px","border":"1px solid #ddd",
                       "margin-bottom":"16px"}),

            interpret,

            html.Hr(style={"margin":"16px 0"}),
            dcc.Graph(figure=fig_fit, style={"height":"400px"},
                       config={"displayModeBar": False}),
        ], style={"padding":"10px 16px 30px"})


    def _pc_estimate_is_taylor(df, pi_star, freq):
        """Stima IS curve e Taylor Rule; restituisce (sigma, phi_pi, phi_y) o None."""
        import warnings
        import statsmodels.api as sm_api
        pi_star = float(pi_star or 2.0)

        sigma_est = None
        phi_pi_est = None
        phi_y_est  = None

        # ── scegli il miglior output gap disponibile ──────────────────────────
        gap_col = None
        for c in ["gap_hp", "gap_nairu", "gap_linear"]:
            if c in df.columns and df[c].dropna().shape[0] >= 20:
                gap_col = c
                break

        # ── scegli il miglior tasso nominale disponibile ─────────────────────
        rate_col = None
        for c in ["rate", "rate_short", "policy_rate"]:
            if c in df.columns and df[c].dropna().shape[0] >= 20:
                rate_col = c
                break

        if gap_col is None or rate_col is None:
            return sigma_est, phi_pi_est, phi_y_est

        try:
            # tasso reale ex-post
            df_w = df[[gap_col, rate_col, "pi"]].dropna().copy()
            df_w["r_real"] = df_w[rate_col] - df_w["pi"]
            df_w["gap_lag"] = df_w[gap_col].shift(1)
            df_w["r_real_lag"] = df_w["r_real"].shift(1)
            df_w["i_lag"] = df_w[rate_col].shift(1)
            df_w = df_w.dropna()

            if len(df_w) < 20:
                return sigma_est, phi_pi_est, phi_y_est

            # ── IS dinamica: gap_t = α + ρ·gap_{t-1} - (1/σ)·r_real_{t-1} ──
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_is = sm_api.add_constant(
                    np.column_stack([df_w["gap_lag"].values,
                                     df_w["r_real_lag"].values]))
                Y_is = df_w[gap_col].values
                ols_is = sm_api.OLS(Y_is, X_is).fit(cov_type="HC3")

            coef_r = float(ols_is.params[2])   # = -(1/σ)
            if abs(coef_r) > 1e-6:
                sigma_raw = -1.0 / coef_r
                # Accetta solo valori plausibili (0.1 – 10)
                sigma_est = round(float(np.clip(sigma_raw, 0.1, 10.0)), 3)

            # ── Taylor Rule con smoothing: i_t = ρ·i_{t-1} + (1-ρ)·[r*+π*+φπ(π-π*)+φy·gap] ──
            df_w["pi_gap"] = df_w["pi"] - pi_star
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_tr = sm_api.add_constant(
                    np.column_stack([df_w["i_lag"].values,
                                     df_w["pi_gap"].values,
                                     df_w[gap_col].values]))
                Y_tr = df_w[rate_col].values
                ols_tr = sm_api.OLS(Y_tr, X_tr).fit(cov_type="HC3")

            rho_tr   = float(ols_tr.params[1])    # smoothing
            phi_pi_raw = float(ols_tr.params[2])  # (1-ρ)·φπ
            phi_y_raw  = float(ols_tr.params[3])  # (1-ρ)·φy

            if 0.0 < rho_tr < 1.0:
                phi_pi_net = phi_pi_raw / (1 - rho_tr)
                phi_y_net  = phi_y_raw  / (1 - rho_tr)
                phi_pi_est = round(float(np.clip(phi_pi_net, 0.5, 4.0)), 3)
                phi_y_est  = round(float(np.clip(phi_y_net,  0.0, 2.0)), 3)

        except Exception as _e:
            print(f"  IS+Taylor pre-estimate: {_e}")

        return sigma_est, phi_pi_est, phi_y_est


    def _pc_is_taylor_tab(df, pi_star, freq):
        """Tab IS Curve + Taylor Rule con stima OLS e grafici fitted vs actual."""
        import warnings
        import statsmodels.api as sm_api
        pi_star = float(pi_star or 2.0)

        # ── scegli variabili ──────────────────────────────────────────────────
        gap_col = None
        for c in ["gap_hp", "gap_nairu", "gap_linear"]:
            if c in df.columns and df[c].dropna().shape[0] >= 20:
                gap_col = c
                break

        rate_col = None
        for c in ["rate", "rate_short", "policy_rate"]:
            if c in df.columns and df[c].dropna().shape[0] >= 20:
                rate_col = c
                break

        gap_label  = {"gap_hp": "HP Filter", "gap_nairu": "NAIRU Gap",
                      "gap_linear": "Trend Lineare"}.get(gap_col, gap_col)
        rate_label = {"rate": "Tasso Politica Monetaria",
                      "rate_short": "Tasso a Breve",
                      "policy_rate": "Policy Rate"}.get(rate_col, rate_col)

        if gap_col is None or rate_col is None:
            missing = []
            if gap_col is None: missing.append("output gap")
            if rate_col is None: missing.append("tasso d'interesse")
            return html.Div(
                f"⚠ Mancano: {', '.join(missing)}. Carica i dati prima con ▶ Carica & Stima.",
                style={"padding": "40px", "color": "#888", "text-align": "center"})

        try:
            df_w = df[[gap_col, rate_col, "pi"]].dropna().copy()
            df_w["r_real"]    = df_w[rate_col] - df_w["pi"]
            df_w["gap_lag"]   = df_w[gap_col].shift(1)
            df_w["r_real_lag"] = df_w["r_real"].shift(1)
            df_w["i_lag"]     = df_w[rate_col].shift(1)
            df_w["pi_gap"]    = df_w["pi"] - pi_star
            df_w = df_w.dropna()

            if len(df_w) < 20:
                return html.Div("Dati insufficienti (< 20 obs).",
                                 style={"padding": "40px", "color": "#888",
                                        "text-align": "center"})

            # ── IS dinamica ───────────────────────────────────────────────────
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_is = sm_api.add_constant(
                    np.column_stack([df_w["gap_lag"].values,
                                     df_w["r_real_lag"].values]))
                Y_is = df_w[gap_col].values
                ols_is = sm_api.OLS(Y_is, X_is).fit(cov_type="HC3")

            alpha_is = float(ols_is.params[0])
            rho_is   = float(ols_is.params[1])
            coef_r   = float(ols_is.params[2])
            sigma_raw = (-1.0 / coef_r) if abs(coef_r) > 1e-6 else None
            sigma_est = float(np.clip(sigma_raw, 0.1, 10.0)) if sigma_raw else None
            r2_is     = float(ols_is.rsquared)
            pv_is     = ols_is.pvalues
            se_is     = ols_is.bse

            # fitted IS
            fitted_is = ols_is.fittedvalues

            # ── Taylor Rule con smoothing ─────────────────────────────────────
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_tr = sm_api.add_constant(
                    np.column_stack([df_w["i_lag"].values,
                                     df_w["pi_gap"].values,
                                     df_w[gap_col].values]))
                Y_tr = df_w[rate_col].values
                ols_tr = sm_api.OLS(Y_tr, X_tr).fit(cov_type="HC3")

            alpha_tr   = float(ols_tr.params[0])
            rho_tr     = float(ols_tr.params[1])
            phi_pi_raw = float(ols_tr.params[2])
            phi_y_raw  = float(ols_tr.params[3])
            r2_tr      = float(ols_tr.rsquared)
            pv_tr      = ols_tr.pvalues
            se_tr      = ols_tr.bse

            phi_pi_net = phi_pi_raw / (1 - rho_tr) if 0 < rho_tr < 1 else phi_pi_raw
            phi_y_net  = phi_y_raw  / (1 - rho_tr) if 0 < rho_tr < 1 else phi_y_raw
            r_star_tr  = alpha_tr   / (1 - rho_tr) if 0 < rho_tr < 1 else alpha_tr

            fitted_tr = ols_tr.fittedvalues

        except Exception as e:
            return html.Div(f"Errore stima IS+Taylor: {e}",
                             style={"color": "red", "padding": "20px"})

        def _stars(p):
            return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

        # ── tabella IS ────────────────────────────────────────────────────────
        sigma_color = "#2ca02c" if (sigma_est and 0.5 <= sigma_est <= 3.0) else "#d62728"
        sigma_text  = (f"{sigma_est:.3f}" if sigma_est else "n/d — segno sbagliato")
        sigma_interp = (
            "Stima plausibile — sensibilità della domanda al tasso reale nella norma."
            if sigma_est and 0.5 <= sigma_est <= 3.0 else
            "⚠ Stima fuori range o segno errato. Probabile endogeneità."
        )

        # α IS: intercetta IS — in equilibrio r=r*, ỹ=0 → α≈0
        alpha_is_note = (
            "≈ 0: equilibrio stabile" if abs(alpha_is) < 0.3
            else f"{'domanda strutturalmente positiva' if alpha_is > 0 else 'domanda strutturalmente depressa'}"
        )
        # ρ IS: persistenza del ciclo economico
        rho_is_note = (
            f"{'alta' if rho_is > 0.8 else 'moderata' if rho_is > 0.5 else 'bassa'} inerzia del ciclo "
            f"({'molto lenta convergenza' if rho_is > 0.9 else 'convergenza lenta' if rho_is > 0.7 else 'convergenza rapida'})"
        )
        # −1/σ IS: sensibilità IS al tasso reale
        coef_r_note = (
            f"→ σ = {sigma_text} — "
            + ("elevata sensibilità al costo del denaro" if sigma_est and sigma_est < 0.8
               else "sensibilità normale (domanda moderatamente reattiva ai tassi)" if sigma_est and sigma_est <= 2.0
               else "bassa sensibilità — politica monetaria trasmessa con difficoltà" if sigma_est
               else "⚠ segno errato: endogeneità OLS")
        )

        is_rows = [
            html.Tr([html.Td("α (costante IS)"),
                     html.Td(f"{alpha_is:+.4f}", style={"font-weight": "bold"}),
                     html.Td(f"{se_is[0]:.4f}", style={"color": "#666"}),
                     html.Td(f"{pv_is[0]:.4f}"),
                     html.Td(_stars(pv_is[0]), style={"color": "#d62728"}),
                     html.Td(alpha_is_note, style={"color": "#555", "font-style": "italic"})]),
            html.Tr([html.Td("ρ (persistenza gap → DSGE interna)"),
                     html.Td(f"{rho_is:+.4f}", style={"font-weight": "bold"}),
                     html.Td(f"{se_is[1]:.4f}", style={"color": "#666"}),
                     html.Td(f"{pv_is[1]:.4f}"),
                     html.Td(_stars(pv_is[1]), style={"color": "#d62728"}),
                     html.Td(rho_is_note, style={"color": "#555", "font-style": "italic"})],
                    style={"background": "#fff9e6"}),
            html.Tr([html.Td("−1/σ (elasticità int. sostituzione → DSGE σ)"),
                     html.Td(f"{coef_r:+.4f}", style={"font-weight": "bold"}),
                     html.Td(f"{se_is[2]:.4f}", style={"color": "#666"}),
                     html.Td(f"{pv_is[2]:.4f}"),
                     html.Td(_stars(pv_is[2]), style={"color": "#d62728"}),
                     html.Td(coef_r_note, style={"color": sigma_color, "font-style": "italic",
                                                  "font-weight": "bold"})],
                    style={"background": "#e8f5e9"}),
        ]

        # ── note parametri Taylor Rule ────────────────────────────────────────
        r_star_note = (
            f"tasso naturale reale r* ≈ {r_star_tr - pi_star:.2f}% "
            f"(sottraendo π*={pi_star}%)"
        )
        rho_tr_note = (
            f"{'altissima' if rho_tr > 0.9 else 'alta' if rho_tr > 0.8 else 'moderata' if rho_tr > 0.6 else 'bassa'} "
            f"inerzia — la BC modifica i tassi {'gradualmente' if rho_tr > 0.7 else 'rapidamente'}"
        )
        phi_pi_note = (
            ("✓ > 1: rispetta principio Taylor — tasso reale sale quando π↑, "
             "inflazione stabilizzata" if phi_pi_net > 1.0
             else "⚠ < 1: viola principio Taylor — tasso reale scende quando π↑, "
             "inflazione potenzialmente esplosiva")
        )
        phi_y_note = (
            f"{'forte' if phi_y_net > 1.0 else 'moderata' if phi_y_net > 0.3 else 'debole'} "
            f"risposta al ciclo — "
            f"{'politica molto aggressiva sull output' if phi_y_net > 1.0 else 'Taylor standard (≈0.5)' if 0.4 <= phi_y_net <= 0.6 else 'priorità all inflazione su output'}"
        )

        # ── tabella Taylor Rule ───────────────────────────────────────────────
        tr_rows = [
            html.Tr([html.Td("α/(1−ρ) = r* + π* (tasso neutrale nominale)"),
                     html.Td(f"{r_star_tr:+.4f}", style={"font-weight": "bold"}),
                     html.Td(f"{se_tr[0]:.4f}", style={"color": "#666"}),
                     html.Td(f"{pv_tr[0]:.4f}"),
                     html.Td(_stars(pv_tr[0]), style={"color": "#d62728"}),
                     html.Td(r_star_note, style={"color": "#555", "font-style": "italic"})]),
            html.Tr([html.Td("ρ (interest rate smoothing → DSGE interna)"),
                     html.Td(f"{rho_tr:+.4f}", style={"font-weight": "bold"}),
                     html.Td(f"{se_tr[1]:.4f}", style={"color": "#666"}),
                     html.Td(f"{pv_tr[1]:.4f}"),
                     html.Td(_stars(pv_tr[1]), style={"color": "#d62728"}),
                     html.Td(rho_tr_note, style={"color": "#555", "font-style": "italic"})],
                    style={"background": "#e8f0fe"}),
            html.Tr([html.Td("φπ netto = φπ_raw/(1−ρ)  → DSGE φπ"),
                     html.Td(f"{phi_pi_net:+.4f}", style={"font-weight": "bold",
                              "color": "#2ca02c" if phi_pi_net > 1 else "#d62728"}),
                     html.Td(f"{se_tr[2]:.4f}", style={"color": "#666"}),
                     html.Td(f"{pv_tr[2]:.4f}"),
                     html.Td(_stars(pv_tr[2]), style={"color": "#d62728"}),
                     html.Td(phi_pi_note,
                             style={"color": "#2ca02c" if phi_pi_net > 1 else "#d62728",
                                    "font-style": "italic", "font-weight": "bold"})],
                    style={"background": "#fff9e6"}),
            html.Tr([html.Td("φy netto = φy_raw/(1−ρ)  → DSGE φy"),
                     html.Td(f"{phi_y_net:+.4f}", style={"font-weight": "bold"}),
                     html.Td(f"{se_tr[3]:.4f}", style={"color": "#666"}),
                     html.Td(f"{pv_tr[3]:.4f}"),
                     html.Td(_stars(pv_tr[3]), style={"color": "#d62728"}),
                     html.Td(phi_y_note, style={"color": "#555", "font-style": "italic"})],
                    style={"background": "#e8f5e9"}),
        ]

        taylor_rule_check = phi_pi_net > 1.0
        taylor_principle = html.Span(
            "✓ Principio di Taylor rispettato (φπ > 1)" if taylor_rule_check
            else "⚠ Principio di Taylor VIOLATO (φπ < 1): tassi reali non stabilizzano l'inflazione",
            style={"color": "#2ca02c" if taylor_rule_check else "#d62728",
                   "font-weight": "bold"})

        # ── grafici ───────────────────────────────────────────────────────────
        # IS: gap osservato vs stimato
        fig_is = make_subplots(rows=2, cols=1,
                               subplot_titles=["IS Curve: Output Gap osservato vs stimato (%)",
                                               "Residui IS (%)"],
                               vertical_spacing=0.14, row_heights=[0.65, 0.35])
        fig_is.add_trace(go.Scatter(x=df_w.index, y=Y_is,
                                    name="Gap osservato",
                                    line=dict(color="#333", width=1.8)), row=1, col=1)
        fig_is.add_trace(go.Scatter(x=df_w.index, y=fitted_is,
                                    name="Gap stimato (IS)",
                                    line=dict(color="#1f77b4", width=2, dash="dot")), row=1, col=1)
        fig_is.add_hline(y=0, line_color="#aaa", line_dash="dot", line_width=1, row=1, col=1)
        resid_is = Y_is - fitted_is
        fig_is.add_trace(go.Bar(x=df_w.index, y=resid_is,
                                marker_color=["#d62728" if r > 0 else "#1f77b4" for r in resid_is],
                                name="Residui"), row=2, col=1)
        fig_is.add_hline(y=0, line_color="#aaa", line_dash="dash", line_width=1, row=2, col=1)
        fig_is.update_layout(hovermode="x unified", margin=dict(t=40, b=20, l=55, r=15),
                              paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                              legend=dict(orientation="h", y=1.04, x=0, font=dict(size=9)))

        # Taylor Rule: tasso osservato vs stimato
        fig_tr = make_subplots(rows=2, cols=1,
                               subplot_titles=["Taylor Rule: tasso nominale osservato vs stimato (%)",
                                               "Residui Taylor Rule (%)"],
                               vertical_spacing=0.14, row_heights=[0.65, 0.35])
        fig_tr.add_trace(go.Scatter(x=df_w.index, y=Y_tr,
                                    name="Tasso osservato",
                                    line=dict(color="#333", width=1.8)), row=1, col=1)
        fig_tr.add_trace(go.Scatter(x=df_w.index, y=fitted_tr,
                                    name="Taylor Rule stimata",
                                    line=dict(color="#9467bd", width=2, dash="dot")), row=1, col=1)
        fig_tr.add_hline(y=pi_star, line_color="#2ca02c", line_dash="dash",
                         line_width=1, annotation_text=f"π*={pi_star}%", row=1, col=1)
        resid_tr = Y_tr - fitted_tr
        fig_tr.add_trace(go.Bar(x=df_w.index, y=resid_tr,
                                marker_color=["#d62728" if r > 0 else "#1f77b4" for r in resid_tr],
                                name="Residui"), row=2, col=1)
        fig_tr.add_hline(y=0, line_color="#aaa", line_dash="dash", line_width=1, row=2, col=1)
        fig_tr.update_layout(hovermode="x unified", margin=dict(t=40, b=20, l=55, r=15),
                              paper_bgcolor="white", plot_bgcolor="#f8f8f8",
                              legend=dict(orientation="h", y=1.04, x=0, font=dict(size=9)))

        tbl_style = {"width": "100%", "border-collapse": "collapse",
                     "font-size": "12px", "border": "1px solid #ddd",
                     "margin-bottom": "12px"}
        th_style  = {"background": "#f0f0f0", "font-size": "12px"}

        return html.Div([
            # ── IS Curve ──────────────────────────────────────────────────────
            html.H4("IS Curve Dinamica",
                    style={"font-size": "14px", "margin": "10px 0 4px",
                           "color": "#1f77b4", "border-bottom": "2px solid #1f77b4",
                           "padding-bottom": "6px"}),
            html.P(f"ỹ_t = α + ρ·ỹ_{{t-1}} − (1/σ)·r_{{t-1}}   "
                   f"[ỹ = {gap_label},  r = tasso reale ex-post]",
                   style={"font-size": "11px", "color": "#666", "margin": "0 0 8px"}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Parametro"), html.Th("Stima"), html.Th("Std Err"),
                    html.Th("p-value"), html.Th(""), html.Th("Nota")
                ], style=th_style)),
                html.Tbody(is_rows),
            ], style=tbl_style),
            html.Div([
                html.Span(f"R² = {r2_is:.3f}",
                          style={"background": "#e8f0fe", "padding": "4px 10px",
                                 "border-radius": "12px", "margin-right": "10px",
                                 "font-size": "12px"}),
                html.Span(f"N = {len(df_w)}",
                          style={"background": "#f0f0f0", "padding": "4px 10px",
                                 "border-radius": "12px", "font-size": "12px"}),
                html.Span(f"σ = {sigma_text}",
                          style={"background": "#f0f0f0", "padding": "4px 10px",
                                 "border-radius": "12px", "font-size": "12px",
                                 "margin-left": "10px", "color": sigma_color,
                                 "font-weight": "bold"}),
            ], style={"margin": "8px 0 12px"}),
            html.Div([
                html.P([html.B("α — Intercetta IS: "),
                        f"{alpha_is:+.4f}  →  {alpha_is_note}. "
                        "Teoricamente deve essere ≈ 0 in deviazioni dalla steady state. "
                        "Un valore significativamente diverso da zero segnala domanda autonoma "
                        "strutturale (es. stimoli fiscali permanenti o carenza strutturale)."],
                       style={"font-size": "12px", "line-height": "1.6"}),
                html.P([html.B("ρ — Persistenza del ciclo: "),
                        f"{rho_is:.4f}  →  {rho_is_note}. "
                        "Misura quanto il gap corrente dipende da quello passato. "
                        "Valori > 0.8 indicano cicli lenti (tipico per area euro), "
                        "valori < 0.5 indicano economia flessibile (tipico USA)."],
                       style={"font-size": "12px", "line-height": "1.6", "margin-top": "6px"}),
                html.P([html.B("−1/σ — Coefficiente tasso reale: "),
                        f"{coef_r:+.4f}  →  {coef_r_note}. "
                        "Questo coefficiente identifica σ (elasticità intertemporale di sostituzione). "
                        "Un valore negativo (corretto) significa che tassi reali più alti comprimono "
                        "la domanda. Un valore positivo o non significativo è sintomo di "
                        "endogeneità: la BC reagisce al gap alzando i tassi, creando correlazione "
                        "positiva spurio."],
                       style={"font-size": "12px", "line-height": "1.6", "margin-top": "6px"}),
                html.P([html.B("Nota metodologica OLS/endogeneità: "),
                        "L'OLS sulla IS soffre di endogeneità simultanea: la Banca Centrale alza "
                        "i tassi esattamente quando l'output gap è positivo (reverse causality). "
                        "Per una stima strutturale di σ affidabile si richiederebbe IV/2SLS con "
                        "strumenti esogeni (es. shock petroliferi, variazioni di liquidità globale, "
                        "dummy geopolitiche). Usa σ come ordine di grandezza, non come stima precisa."],
                       style={"font-size": "11px", "color": "#888",
                              "line-height": "1.5", "margin-top": "6px",
                              "border-top": "1px solid #ddd", "padding-top": "8px"}),
            ], style={"background": "#f0f6ff", "padding": "12px 14px",
                      "border-radius": "6px", "border-left": "4px solid #1f77b4",
                      "margin-bottom": "12px"}),
            dcc.Graph(figure=fig_is, style={"height": "350px"},
                      config={"displayModeBar": False}),

            html.Hr(style={"margin": "20px 0"}),

            # ── Taylor Rule ────────────────────────────────────────────────────
            html.H4("Taylor Rule con Interest Rate Smoothing",
                    style={"font-size": "14px", "margin": "10px 0 4px",
                           "color": "#9467bd", "border-bottom": "2px solid #9467bd",
                           "padding-bottom": "6px"}),
            html.P(f"i_t = ρ·i_{{t-1}} + (1−ρ)·[r* + π* + φπ·(π_t−π*) + φy·ỹ_t]   "
                   f"[ỹ = {gap_label},  i = {rate_label}]",
                   style={"font-size": "11px", "color": "#666", "margin": "0 0 8px"}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Parametro"), html.Th("Stima"), html.Th("Std Err"),
                    html.Th("p-value"), html.Th(""), html.Th("Nota")
                ], style=th_style)),
                html.Tbody(tr_rows),
            ], style=tbl_style),
            html.Div([
                html.Span(f"R² = {r2_tr:.3f}",
                          style={"background": "#e8f0fe", "padding": "4px 10px",
                                 "border-radius": "12px", "margin-right": "10px",
                                 "font-size": "12px"}),
                html.Span(f"N = {len(df_w)}",
                          style={"background": "#f0f0f0", "padding": "4px 10px",
                                 "border-radius": "12px", "font-size": "12px"}),
            ], style={"margin": "8px 0 12px"}),
            html.Div([
                html.P([html.B("Equazione stimata:  "),
                        html.Span(
                            f"i_t = {rho_tr:.3f}·i_{{t-1}} + {phi_pi_raw:.3f}·(π−π*) + "
                            f"{phi_y_raw:.3f}·ỹ_t + {alpha_tr:.3f}",
                            style={"font-family": "monospace", "font-size": "12px"})],
                       style={"font-size": "12px"}),
                html.P([html.B("Parametri netti (per DSGE):  "),
                        html.Span(
                            f"ρ = {rho_tr:.3f},   φπ = {phi_pi_net:.3f},   "
                            f"φy = {phi_y_net:.3f},   r*+π* ≈ {r_star_tr:.3f}%",
                            style={"font-family": "monospace"})],
                       style={"font-size": "12px", "margin-top": "6px"}),
                html.P(taylor_principle, style={"font-size": "12px", "margin-top": "8px"}),
                html.Hr(style={"margin": "10px 0"}),
                html.P([html.B("r* + π* (tasso nominale neutrale): "),
                        f"{r_star_tr:.4f}%  →  {r_star_note}. "
                        "Il tasso reale naturale r* è il tasso compatibile con output a potenziale "
                        "e inflazione all'obiettivo. Storicamente ~2% (USA) o ~0-1% (EUR post-crisi)."],
                       style={"font-size": "12px", "line-height": "1.6"}),
                html.P([html.B("ρ — Smoothing dei tassi: "),
                        f"{rho_tr:.4f}  →  {rho_tr_note}. "
                        "L'inerzia dei tassi riflette la preferenza della BC per cambiamenti "
                        "graduali ('gradualism'): evita segnali confusivi ai mercati e riduce "
                        "la volatilità finanziaria. Valori tipici: Fed ~0.85, BCE ~0.80."],
                       style={"font-size": "12px", "line-height": "1.6", "margin-top": "6px"}),
                html.P([html.B("φπ — Risposta all'inflation gap: "),
                        f"{phi_pi_net:.4f}  →  {phi_pi_note}. "
                        "Il Principio di Taylor richiede φπ > 1: solo così il tasso reale "
                        "(i − π) aumenta quando l'inflazione sale, raffreddando la domanda. "
                        f"Taylor (1993) originale: φπ = 1.5. Il tuo valore {phi_pi_net:.2f} "
                        f"{'è in linea con il benchmark' if 1.2 <= phi_pi_net <= 2.0 else 'si discosta dal benchmark (1.2–2.0)'}."],
                       style={"font-size": "12px", "line-height": "1.6", "margin-top": "6px"}),
                html.P([html.B("φy — Risposta all'output gap: "),
                        f"{phi_y_net:.4f}  →  {phi_y_note}. "
                        "Misura quanto la BC reagisce al ciclo economico oltre all'inflazione. "
                        "Taylor (1993): φy = 0.5. Valori > 1 indicano una BC molto attenta "
                        "alla stabilizzazione del ciclo; valori < 0.2 indicano priorità esclusiva "
                        "all'inflazione (stile Bundesbank)."],
                       style={"font-size": "12px", "line-height": "1.6", "margin-top": "6px"}),
            ], style={"background": "#f3eaff", "padding": "12px 14px",
                      "border-radius": "6px", "border-left": "4px solid #9467bd",
                      "margin-bottom": "12px"}),
            dcc.Graph(figure=fig_tr, style={"height": "350px"},
                      config={"displayModeBar": False}),

        ], style={"padding": "10px 16px 40px"})


    # ── Aggiorna HP lambda al cambio frequenza ────────────────────────────────
    @app.callback(
        Output("pc-hp-lambda", "value"),
        Input("pc-freq",       "value"),
    )
    def pc_update_lambda(freq):
        return {"M": 129600, "Q": 1600, "Y": 100}.get(freq, 1600)

    # ── Trasferimento κ e β al DSGE ──────────────────────────────────────────
    @app.callback(
        Output("dsge-kappa",        "value"),
        Output("dsge-beta",         "value"),
        Output("pc-to-dsge-status", "children"),
        Input("btn-pc-to-dsge",     "n_clicks"),
        State("store-pc-kappa",     "data"),
        State("store-pc-beta",      "data"),
        prevent_initial_call=True,
    )
    def pc_to_dsge(n_clicks, kappa, beta):
        if kappa is None or beta is None:
            return no_update, no_update, "⚠ Esegui prima la stima."
        k = max(0.01, min(0.5,  float(kappa)))
        b = max(0.90, min(0.999, float(beta)))
        return k, b, f"✓ κ={k:.4f}  β={b:.4f} → DSGE"

    # ── Trasferimento σ, φπ, φy al DSGE ──────────────────────────────────────
    @app.callback(
        Output("dsge-sigma",           "value"),
        Output("dsge-phi-pi",          "value"),
        Output("dsge-phi-y",           "value"),
        Output("pc-is-to-dsge-status", "children"),
        Input("btn-pc-is-to-dsge",     "n_clicks"),
        State("store-pc-sigma",        "data"),
        State("store-pc-phi-pi",       "data"),
        State("store-pc-phi-y",        "data"),
        prevent_initial_call=True,
    )
    def pc_is_to_dsge(n_clicks, sigma, phi_pi, phi_y):
        if sigma is None or phi_pi is None or phi_y is None:
            return no_update, no_update, no_update, "⚠ Esegui prima la stima IS+Taylor."
        s   = max(0.1,  min(3.0, float(sigma)))
        fpi = max(0.5,  min(4.0, float(phi_pi)))
        fy  = max(0.0,  min(2.0, float(phi_y)))
        return s, fpi, fy, f"✓ σ={s:.3f}  φπ={fpi:.3f}  φy={fy:.3f} → DSGE"


    # =========================================================================
    # VALUTAZIONE TITOLO AZIONARIO
    # =========================================================================

    @app.callback(
        Output("store-valuation",  "data"),
        Output("val-tab-content",  "children"),
        Output("val-fetch-status", "children"),
        Input("btn-run-valuation", "n_clicks"),
        Input("val-result-tabs",   "value"),
        State("val-ticker",        "value"),
        State("val-wacc",          "value"),
        State("val-g1",            "value"),
        State("val-g2",            "value"),
        State("val-gterm",         "value"),
        State("val-fcf-margin",    "value"),
        State("val-pe-sector",     "value"),
        State("val-ev-ebitda",     "value"),
        State("val-ke",            "value"),
        State("val-bond-yield",    "value"),
        State("store-valuation",   "data"),
        prevent_initial_call=True,
    )
    def run_valuation(n_clicks, active_tab,
                      ticker, wacc, g1, g2, gterm, fcf_margin,
                      pe_sector, ev_ebitda_mult, ke, bond_yield,
                      stored):
        import json, traceback
        import yfinance as yf

        ctx = callback_context
        tid = ctx.triggered_id if ctx.triggered_id else ""

        no_data_div = html.Div("Inserisci un ticker e clicca ▶ Carica & Valuta.",
                               style={"padding": "40px", "color": "#888",
                                      "text-align": "center", "font-size": "14px"})

        # ── parametri con defaults ────────────────────────────────────────────
        wacc       = float(wacc       or 9.0)   / 100
        g1         = float(g1         or 12.0)  / 100
        g2         = float(g2         or 6.0)   / 100
        gterm      = float(gterm      or 2.5)   / 100
        fcf_margin = float(fcf_margin or 15.0)  / 100
        pe_sector  = float(pe_sector  or 22.0)
        ev_mult    = float(ev_ebitda_mult or 12.0)
        ke         = float(ke         or 10.0)  / 100
        bond_yield = float(bond_yield or 4.5)   / 100

        # ── solo cambio tab: ri-renderizza senza re-fetch ─────────────────────
        if tid != "btn-run-valuation":
            if not stored:
                return no_update, no_data_div, no_update
            try:
                d = json.loads(stored) if isinstance(stored, str) else stored
                content = _val_build_content(d, active_tab, wacc, g1, g2, gterm,
                                             fcf_margin, pe_sector, ev_mult, ke, bond_yield)
                return no_update, content, no_update
            except Exception:
                return no_update, no_data_div, no_update

        # ── fetch yfinance ────────────────────────────────────────────────────
        if not ticker:
            return no_update, no_data_div, "⚠ Inserisci un ticker."

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

            # Aggiusta FCF margin: se yfinance ha FCF reale, usalo come riferimento
            if revenue > 0 and fcf_yf:
                fcf_margin_actual = fcf_yf / revenue
            else:
                fcf_margin_actual = fcf_margin

            d = {
                "ticker": ticker, "name": name, "sector": sector,
                "industry": industry, "currency": currency,
                "price": price, "market_cap": market_cap, "shares": shares,
                "eps_ttm": eps_ttm, "eps_fwd": eps_fwd,
                "revenue": revenue, "ebitda": ebitda, "fcf_yf": fcf_yf,
                "fcf_margin_actual": fcf_margin_actual,
                "total_debt": total_debt, "cash": cash, "net_debt": net_debt,
                "dividend": dividend, "beta": beta,
                "pe_trailing": pe_trailing, "pe_forward": pe_forward,
                "book_val": book_val, "revenue_growth": revenue_growth,
                "gross_margins": gross_margins, "gross_profits": gross_profits,
                "ebitda_margins": ebitda_margins, "operating_margins": operating_margins,
                "ps_trailing": ps_trailing, "ev": ev, "rd_expense": rd_expense,
            }

            content = _val_build_content(d, active_tab, wacc, g1, g2, gterm,
                                         fcf_margin, pe_sector, ev_mult, ke, bond_yield)
            status = f"✅ {name} ({ticker}) — {sector} | {currency} | prezzo: {price:.2f}"
            return json.dumps(d), content, status

        except Exception as e:
            tb = traceback.format_exc()
            print(f"=== VALUATION ERROR ===\n{tb}")
            return no_update, html.Div([
                html.B("Errore fetch: "), html.Span(str(e)),
            ], style={"color": "red", "padding": "20px"}), f"❌ {e}"


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
        """DCF a 2 fasi: 5 anni g1, 5 anni g2, poi terminal value Gordon."""
        if revenue <= 0 or shares <= 0:
            return None, [], []
        fcf0   = revenue * fcf_margin
        pv_sum = 0.0
        fcf_rows = []
        fcf_t = fcf0
        for yr in range(1, 6):
            fcf_t *= (1 + g1)
            pv = fcf_t / (1 + wacc) ** yr
            pv_sum += pv
            fcf_rows.append((yr, f"Fase 1 (g={g1*100:.1f}%)", fcf_t, pv))
        for yr in range(6, 11):
            fcf_t *= (1 + g2)
            pv = fcf_t / (1 + wacc) ** yr
            pv_sum += pv
            fcf_rows.append((yr, f"Fase 2 (g={g2*100:.1f}%)", fcf_t, pv))
        # Terminal value
        if wacc <= gterm:
            tv = 0
        else:
            tv = fcf_t * (1 + gterm) / (wacc - gterm)
        pv_tv = tv / (1 + wacc) ** 10
        pv_sum += pv_tv
        fair_price = pv_sum / shares
        return fair_price, fcf_rows, pv_tv


    def _val_ddm(dividend, ke, gterm):
        """Gordon Growth Model: P = D1 / (ke - g)."""
        if dividend <= 0 or ke <= gterm:
            return None
        d1 = dividend * (1 + gterm)
        return d1 / (ke - gterm)


    def _val_graham(eps, g_pct, bond_yield):
        """Formula di Graham aggiornata: P = EPS × (8.5 + 2g) × 4.4 / Y."""
        if eps <= 0 or bond_yield <= 0:
            return None
        return eps * (8.5 + 2 * g_pct) * 4.4 / (bond_yield * 100)


    def _val_build_content(d, active_tab, wacc, g1, g2, gterm,
                            fcf_margin, pe_sector, ev_mult, ke, bond_yield):
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
        rev_growth = d.get("revenue_growth", 0) or 0

        # usa margine FCF reale come default per il DCF
        fcf_m = fcf_margin_actual if fcf_margin_actual > 0 else fcf_margin

        # ── calcola tutti i modelli ───────────────────────────────────────────
        dcf_price, fcf_rows, pv_tv = _val_dcf(revenue, fcf_m, wacc, g1, g2, gterm, shares)
        ddm_price  = _val_ddm(dividend, ke, gterm)
        g_est_pct  = max(rev_growth * 100, g1 * 100 * 0.6)  # stima crescita EPS
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
                ("FCF",                _val_fmt_num(d.get("fcf_yf"), suffix=f" {currency}")),
                ("Margine FCF reale",  f"{fcf_margin_actual*100:.1f}%"),
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
                    html.Td(model, style=td_style),
                    html.Td(f"{fv:.2f} {currency}" if fv else "N/D",
                            style={**td_style, "font-weight": "bold"}),
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
            cols = ["#1f77b4" if yr <= 5 else "#ff7f0e" for yr in yrs]
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
                html.H4("DCF a 2 Fasi — Discounted Cash Flow",
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
                                    html.Td("Terminal", style={**td_style, "font-weight": "bold"}),
                                    html.Td(f"g={gterm*100:.2f}%", style=td_style),
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
                                    ("Crescita fase 1 (anni 1-5)", f"{g1*100:.1f}%"),
                                    ("Crescita fase 2 (anni 6-10)", f"{g2*100:.1f}%"),
                                    ("Crescita terminale", f"{gterm*100:.2f}%"),
                                    ("Margine FCF",     f"{fcf_m*100:.1f}%"),
                                    ("Revenue base",    _val_fmt_num(revenue, suffix=f" {currency}")),
                                    ("Azioni (shares)", _val_fmt_num(shares, 0)),
                                ]
                            ])
                        ], style=tbl_style),
                        html.Div([
                            html.P([html.B("Note: "),
                                    "Il DCF è molto sensibile a WACC e crescita terminale. "
                                    "Usa la tab Sensitività per vedere l'intervallo di fair value "
                                    "per diverse combinazioni di WACC e g."],
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
            if has_div and ke > gterm:
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
                    fig_ddm.add_vline(x=gterm * 100, line_color="#2ca02c",
                                       line_dash="dot", line_width=1.5,
                                       annotation_text=f"g={gterm*100:.1f}%")
                fig_ddm.update_layout(
                    title=dict(text="DDM Fair Value al variare della crescita terminale g",
                               font=dict(size=11)),
                    xaxis_title="g — Crescita terminale (%)",
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
                              f"Ke = {ke*100:.1f}%  |  g = {gterm*100:.2f}%",
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
                text=f"★ EPS attuale={eps_ttm:.2f},  g={g_est_pct:.0f}%  →  "
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
                            f"EPS = {eps_ttm:.2f}  ×  (8.5 + 2×{g_est_pct:.0f})  ×  "
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
                xaxis_title="Crescita terminale g",
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
                xaxis_title="Crescita terminale g",
                yaxis_title="WACC",
                margin=dict(t=45, b=40, l=65, r=20),
                paper_bgcolor="white")

            return html.Div([
                html.H4("Analisi di Sensitività DCF — WACC × Crescita Terminale",
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


register_new_tab_callbacks(app)


# =============================================================================
# CONFRONTO SERIE STORICHE — Callbacks
# =============================================================================

def register_compare_callbacks(app):

    # ── Popola dropdown dai dati caricati ────────────────────────────────────
    @app.callback(
        Output("csr-series-dropdown", "options"),
        Input("store-data",    "data"),
        Input("store-gdp",     "data"),
        Input("store-yields",  "data"),
        Input("store-shock",   "data"),
    )
    def csr_populate_dropdown(data_mon, data_gdp, data_yields, data_shock):
        all_cols = []
        for data in [data_mon, data_gdp, data_yields, data_shock]:
            if data:
                try:
                    df = pd.read_json(io.StringIO(data), orient="split")
                    all_cols.extend(df.columns.tolist())
                except Exception:
                    pass
        seen, opts = set(), []
        for c in all_cols:
            if c not in seen:
                seen.add(c)
                opts.append({"label": c, "value": c})
        return sorted(opts, key=lambda x: x["label"])

    # ── Aggiungi serie da FRED ────────────────────────────────────────────────
    @app.callback(
        Output("store-csr-extra",  "data"),
        Output("csr-fred-status",  "children"),
        Input("csr-fred-btn",      "n_clicks"),
        State("csr-fred-input",    "value"),
        State("api-key",           "value"),
        State("store-csr-extra",   "data"),
        prevent_initial_call=True,
    )
    def csr_add_fred(n, series_input, api_key, existing):
        if not series_input or not series_input.strip():
            return existing, "⚠ Inserisci almeno un ID serie"
        api_key = (api_key or FRED_API_KEY).strip()
        ids = [s.strip().upper()
               for s in series_input.replace(";", ",").split(",") if s.strip()]
        existing = existing or {}
        added, failed = [], []
        for sid in ids:
            try:
                s = fred_get(sid, api_key)
                if s is None or s.empty:
                    failed.append(sid)
                    continue
                s.index = pd.to_datetime(s.index)
                existing[sid] = {
                    "dates":  s.index.strftime("%Y-%m-%d").tolist(),
                    "values": [float(v) if v == v else None for v in s.values],
                    "label":  sid,
                }
                added.append(sid)
            except Exception:
                failed.append(sid)
        parts = []
        if added:  parts.append(f"✅ Aggiunte: {', '.join(added)}")
        if failed: parts.append(f"❌ Fallite: {', '.join(failed)}")
        return existing, "  |  ".join(parts) or "—"

    # ── Aggiorna slider temporale in base alle serie selezionate ─────────────
    @app.callback(
        Output("csr-date-slider", "min"),
        Output("csr-date-slider", "max"),
        Output("csr-date-slider", "value"),
        Output("csr-date-slider", "marks"),
        Output("csr-date-label",  "children"),
        Input("csr-series-dropdown", "value"),
        Input("store-csr-extra",     "data"),
        State("store-data",          "data"),
        State("store-gdp",           "data"),
        State("store-yields",        "data"),
        State("store-shock",         "data"),
    )
    def csr_update_slider(selected, extra, data_mon, data_gdp, data_yields, data_shock):
        frames = []
        for data in [data_mon, data_gdp, data_yields, data_shock]:
            if data:
                try:
                    df = pd.read_json(io.StringIO(data), orient="split")
                    df.index = pd.to_datetime(df.index)
                    df = df.resample("MS").last()   # normalizza tutto a mensile
                    frames.append(df)
                except Exception:
                    pass
        merged = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        if not merged.empty:
            merged = merged.loc[:, ~merged.columns.duplicated()]
        if extra:
            for sid, meta in extra.items():
                idx  = pd.to_datetime(meta["dates"])
                vals = pd.array(meta["values"], dtype="Float64")
                merged[sid] = pd.Series(vals, index=idx, name=sid)

        all_series = list(selected or []) + list((extra or {}).keys())
        avail = [s for s in all_series if s in merged.columns]

        if not avail or merged.empty:
            return 0, 1, [0, 1], {}, "— nessuna serie —"

        sub = merged[avail].dropna(how="all")
        if sub.empty:
            return 0, 1, [0, 1], {}, "— dati non disponibili —"

        t_min = sub.index.min()
        # t_max usa l'indice grezzo (include mesi NaN non ancora pubblicati da FRED)
        t_max = merged[avail].index.max()
        ts0   = int(t_min.timestamp())
        ts1   = int(t_max.timestamp())

        yr_range = pd.date_range(
            start=t_min.replace(month=1, day=1),
            end=t_max, freq="2YS",
        )
        marks = {int(y.timestamp()): {"label": str(y.year),
                                       "style": {"color": "#555", "font-size": "9px"}}
                 for y in yr_range}

        label = f"{t_min.strftime('%b %Y')}  →  {t_max.strftime('%b %Y')}"
        return ts0, ts1, [ts0, ts1], marks, label

    # ── Aggiorna label quando lo slider viene mosso ───────────────────────────
    @app.callback(
        Output("csr-date-label", "children", allow_duplicate=True),
        Input("csr-date-slider", "value"),
        prevent_initial_call=True,
    )
    def csr_slider_label(val):
        if not val or val[1] <= val[0]:
            return ""
        s = pd.to_datetime(val[0], unit="s").strftime("%b %Y")
        e = pd.to_datetime(val[1], unit="s").strftime("%b %Y")
        return f"{s}  →  {e}"

    # ── Genera pipeline di trasformazioni per serie (4 passi ordinati) ──────────
    @app.callback(
        Output("csr-transform-controls", "children"),
        Input("csr-series-dropdown",     "value"),
        Input("store-csr-extra",         "data"),
    )
    def csr_render_controls(selected, extra):
        all_series = list(selected or []) + list((extra or {}).keys())
        if not all_series:
            return html.Div(
                "— seleziona serie per configurare la pipeline —",
                style={"font-size": "10px", "color": "#666", "font-style": "italic"},
            )
        PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        STEP_OPTS = [
            {"label": "—",           "value": "none"},
            {"label": "log",         "value": "log"},
            {"label": "YoY %",       "value": "yoy"},
            {"label": "Σ cumsum",    "value": "cumsum"},
            {"label": "Δ¹",          "value": "diff1"},
            {"label": "Δ²",          "value": "diff2"},
            {"label": "× 100",       "value": "x100"},
            {"label": "MA 3m",       "value": "ma3"},
            {"label": "MA 6m",       "value": "ma6"},
            {"label": "MA 12m",      "value": "ma12"},
            {"label": "EMA 3m",      "value": "ema3"},
            {"label": "EMA 6m",      "value": "ema6"},
            {"label": "EMA 12m",     "value": "ema12"},
            {"label": "Savitzky-G",  "value": "sg"},
            {"label": "HP trend",    "value": "hp"},
            {"label": "Kalman",      "value": "kalman"},
        ]
        _dd_style = {
            "font-size": "10px", "width": "90px",
            "display": "inline-block", "vertical-align": "middle",
            "color": "#000",
        }
        rows = []
        for i, s in enumerate(all_series):
            color = PALETTE[i % len(PALETTE)]
            short = s[:24] + "…" if len(s) > 24 else s
            step_row = []
            for k in range(4):
                step_row.append(
                    html.Span([
                        html.Span(f"{'→' if k else '①②③④'[k]}",
                                  style={"font-size": "10px", "color": "#666",
                                         "margin": "0 2px", "vertical-align": "middle"}),
                        dcc.Dropdown(
                            id={"type": "csr-step", "index": f"{s}___{k}"},
                            options=STEP_OPTS,
                            value="none",
                            clearable=False,
                            style=_dd_style,
                        ),
                    ], style={"display": "inline-flex", "align-items": "center",
                               "margin-right": "2px"}),
                )
            rows.append(html.Div([
                html.Div([
                    html.Span("●", style={"color": color, "font-size": "13px",
                                          "margin-right": "5px",
                                          "vertical-align": "middle"}),
                    html.Span(short, style={"font-size": "10px", "font-weight": "bold",
                                            "color": "#1a1a1a",
                                            "vertical-align": "middle"}),
                ], style={"margin-bottom": "5px"}),
                html.Div(
                    html.Span("passo: ", style={"font-size": "9px", "color": "#555"}),
                    style={"margin-bottom": "3px"},
                ),
                html.Div(step_row,
                         style={"display": "flex", "flex-wrap": "wrap",
                                 "gap": "4px", "align-items": "center"}),
            ], style={
                "margin-bottom": "12px", "padding": "8px 8px 10px",
                "background": "#eaf4fb", "border-radius": "4px",
                "border-left": f"3px solid {color}",
            }))
        return rows

    # ── Aggiorna grafico con pipeline per serie ───────────────────────────────
    @app.callback(
        Output("csr-graph",                             "figure"),
        Input("csr-update-btn",                         "n_clicks"),
        Input({"type": "csr-step", "index": ALL},       "value"),
        State("csr-series-dropdown",                    "value"),
        State({"type": "csr-step", "index": ALL},       "id"),
        State("csr-layout-mode",                        "value"),
        State("csr-normalize",                          "value"),
        State("csr-date-slider",                        "value"),
        State("store-data",                             "data"),
        State("store-gdp",                              "data"),
        State("store-yields",                           "data"),
        State("store-shock",                            "data"),
        State("store-csr-extra",                        "data"),
        prevent_initial_call=True,
    )
    def csr_update_graph(n, step_vals, selected, step_ids,
                         layout_mode, normalize, date_range,
                         data_mon, data_gdp, data_yields, data_shock,
                         extra_data):
        _EMPTY_FIG = {
            "data": [],
            "layout": {
                "template": "plotly_white",
                "paper_bgcolor": "#ffffff",
                "plot_bgcolor":  "#ffffff",
                "annotations": [{"text": "Nessuna serie selezionata",
                                  "xref": "paper", "yref": "paper",
                                  "x": 0.5, "y": 0.5, "showarrow": False,
                                  "font": {"color": "#888", "size": 16}}],
            },
        }

        # — merge tutte le sorgenti dati ————————————————————————————————————
        frames = []
        for data in [data_mon, data_gdp, data_yields, data_shock]:
            if data:
                try:
                    df = pd.read_json(io.StringIO(data), orient="split")
                    df.index = pd.to_datetime(df.index)
                    df = df.resample("MS").last()   # normalizza tutto a mensile
                    frames.append(df)
                except Exception:
                    pass
        merged = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        if not merged.empty:
            merged = merged.loc[:, ~merged.columns.duplicated()]

        if extra_data:
            for sid, meta in extra_data.items():
                idx  = pd.to_datetime(meta["dates"])
                vals = pd.array(meta["values"], dtype="Float64")
                merged[sid] = pd.Series(vals, index=idx, name=sid)

        # — filtro temporale dallo slider ——————————————————————————————————
        if date_range and len(date_range) == 2 and date_range[1] > date_range[0]:
            d0 = pd.to_datetime(date_range[0], unit="s")
            d1 = pd.to_datetime(date_range[1], unit="s")
            merged = merged.loc[(merged.index >= d0) & (merged.index <= d1)]

        # — build pipeline {serie: {passo: trasformazione}} ————————————————
        from collections import defaultdict as _dd
        pipes = _dd(dict)
        for id_obj, val in zip(step_ids or [], step_vals or []):
            idx_str = id_obj["index"]            # es. "CPIAUCSL___2"
            ser, k  = idx_str.rsplit("___", 1)
            pipes[ser][int(k)] = val

        all_series = [s for s in
                      list(selected or []) + list((extra_data or {}).keys())
                      if s in merged.columns]

        if not all_series or merged.empty:
            return _EMPTY_FIG

        do_norm = "norm" in (normalize or [])
        PALETTE  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        OP_LABEL = {
            "log":    "log",   "yoy":    "YoY%",  "cumsum": "Σ",
            "diff1":  "Δ¹",    "diff2":  "Δ²",
            "x100":   "×100",
            "ma3":    "MA3",   "ma6":    "MA6",   "ma12":  "MA12",
            "ema3":   "EMA3",  "ema6":   "EMA6",  "ema12": "EMA12",
            "sg":     "S-G",   "hp":     "HP",    "kalman":"Kalman",
        }

        def _kalman_smooth(s: pd.Series) -> pd.Series:
            """Local-level Kalman smoother (forward pass only)."""
            y = s.values.astype(float)
            n = len(y)
            if n < 2:
                return s
            q = float(np.var(np.diff(y[~np.isnan(y)]))) if n > 2 else 1.0
            r = float(np.var(y[~np.isnan(y)])) * 0.3 + 1e-9
            x_est = np.full(n, np.nan)
            p = r
            x = y[~np.isnan(y)][0] if np.any(~np.isnan(y)) else 0.0
            for t in range(n):
                if np.isnan(y[t]):
                    x_est[t] = x
                    p += q
                    continue
                p += q
                k    = p / (p + r)
                x    = x + k * (y[t] - x)
                p    = (1 - k) * p
                x_est[t] = x
            return pd.Series(x_est, index=s.index)

        def _apply_pipeline(raw: pd.Series, serie: str):
            """Applica i passi in ordine e restituisce (serie_trasformata, label_pipeline)."""
            from scipy.signal import savgol_filter as _sgf
            from statsmodels.tsa.filters.hp_filter import hpfilter as _hpf
            s = raw.ffill().dropna().copy()
            steps = pipes.get(serie, {})
            lbl_parts = []
            for k in sorted(steps):
                tr = steps[k]
                if not tr or tr == "none":
                    continue
                if tr == "log":
                    s = np.log(s.clip(lower=1e-9))
                elif tr == "yoy":
                    s = ((s - s.shift(12)) / s.shift(12).abs()) * 100
                elif tr == "cumsum":
                    s = s.cumsum()
                elif tr == "diff1":
                    s = s.diff()
                elif tr == "diff2":
                    s = s.diff().diff()
                elif tr == "x100":
                    s = s * 100
                elif tr == "ma3":
                    s = s.rolling(3, min_periods=1).mean()
                elif tr == "ma6":
                    s = s.rolling(6, min_periods=1).mean()
                elif tr == "ma12":
                    s = s.rolling(12, min_periods=1).mean()
                elif tr == "ema3":
                    s = s.ewm(span=3,  adjust=False).mean()
                elif tr == "ema6":
                    s = s.ewm(span=6,  adjust=False).mean()
                elif tr == "ema12":
                    s = s.ewm(span=12, adjust=False).mean()
                elif tr == "sg":
                    clean = s.dropna()
                    wl = min(13, len(clean))
                    if wl % 2 == 0:
                        wl -= 1
                    if wl >= 3:
                        sg_vals = _sgf(clean.values.astype(float),
                                       window_length=wl, polyorder=min(3, wl - 1))
                        s = pd.Series(sg_vals, index=clean.index)
                    else:
                        s = clean
                elif tr == "hp":
                    clean = s.dropna()
                    if len(clean) >= 8:
                        _, trend = _hpf(clean.values.astype(float), lamb=129600)
                        s = pd.Series(trend, index=clean.index)
                    else:
                        s = clean
                elif tr == "kalman":
                    s = _kalman_smooth(s)
                lbl_parts.append(OP_LABEL[tr])
            if do_norm:
                clean = s.dropna()
                if len(clean) and clean.iloc[0] != 0:
                    s = (s / clean.iloc[0]) * 100
                    lbl_parts.append("=100")
            pipeline_lbl = " → ".join(lbl_parts) if lbl_parts else "Livelli"
            return s, pipeline_lbl

        n_s = len(all_series)

        # ── SUBPLOTS ──────────────────────────────────────────────────────────
        if layout_mode == "subplots":
            from plotly.subplots import make_subplots as _msp
            # pre-calcola le pipeline per i titoli
            precomp = {s: _apply_pipeline(merged[s].copy(), s) for s in all_series}
            subplot_titles = [f"{s}  [{precomp[s][1]}]" for s in all_series]
            fig = _msp(
                rows=n_s, cols=1,
                shared_xaxes=True,
                vertical_spacing=max(0.02, min(0.08, 0.6 / n_s)),
                subplot_titles=subplot_titles,
            )
            for i, s in enumerate(all_series):
                color   = PALETTE[i % len(PALETTE)]
                y, lbl  = precomp[s]
                fig.add_trace(
                    go.Scatter(
                        x=y.index, y=y.values,
                        name=s, mode="lines",
                        line={"color": color, "width": 1.6},
                        connectgaps=True,
                        hovertemplate=(
                            f"<b>{s}</b>  {lbl}<br>"
                            "%{x|%b %Y}<br>"
                            "%{y:.4g}<extra></extra>"
                        ),
                    ),
                    row=i + 1, col=1,
                )
                fig.update_yaxes(
                    title_text=lbl,
                    title_font={"size": 9, "color": "#8888aa"},
                    tickfont={"size": 9},
                    gridcolor="#e8e8e8",
                    row=i + 1, col=1,
                )
            fig.update_xaxes(gridcolor="#e8e8e8")
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                showlegend=False,
                margin={"l": 65, "r": 20, "t": 40, "b": 40},
                height=max(220 * n_s, 380),
            )
            for ann in fig.layout.annotations:
                ann.font.size  = 10
                ann.font.color = "#333333"

        # ── OVERLAY ───────────────────────────────────────────────────────────
        else:
            fig = go.Figure()
            for i, s in enumerate(all_series):
                color      = PALETTE[i % len(PALETTE)]
                y, lbl     = _apply_pipeline(merged[s].copy(), s)
                name_lbl   = f"{s}  [{lbl}]"
                fig.add_trace(go.Scatter(
                    x=y.index, y=y.values,
                    name=name_lbl, mode="lines",
                    line={"color": color, "width": 1.6},
                    connectgaps=True,
                    hovertemplate=(
                        f"<b>{name_lbl}</b><br>"
                        "%{x|%b %Y}<br>"
                        "%{y:.4g}<extra></extra>"
                    ),
                ))
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                autosize=True,
                legend={"font": {"size": 10}, "bgcolor": "rgba(240,244,250,0.9)",
                        "bordercolor": "#dee2e6", "borderwidth": 1},
                xaxis={"gridcolor": "#e8e8e8"},
                yaxis={"gridcolor": "#e8e8e8"},
                margin={"l": 65, "r": 20, "t": 20, "b": 40},
                hovermode="x unified",
            )

        return fig


register_compare_callbacks(app)


# =============================================================================
# MAIN
# =============================================================================
# Apertura embed da URL: /fred/?tab=tab2&embed=1 → apre quella tab (barra nascosta via CSS)
_FRED_TABS_VALID = {'tab1', 'tab2', 'tab3', 'tab5', 'tab-shock', 'tab-arima',
                    'tab-adl', 'tab-phillips', 'tab-dsge', 'tab-valuation',
                    'tab-compare', 'tab4'}


@app.callback(
    Output('main-tabs', 'value', allow_duplicate=True),
    Input('fred-tab-once', 'n_intervals'),
    State('fred-url', 'search'),
    prevent_initial_call=True,
)
def _fred_preselect_tab(_n, search):
    if not search:
        raise PreventUpdate
    import urllib.parse as _up
    wanted = (_up.parse_qs(search.lstrip('?')).get('tab', [''])[0] or '').strip()
    if wanted in _FRED_TABS_VALID:
        return wanted
    raise PreventUpdate


# Pre-carica la cache della curva tassi all'avvio (background, non blocca il boot)
def _prewarm_yields():
    import os as _os, time as _t, pickle as _pk
    _cache_f = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'yields_cache.pkl')
    try:
        if _os.path.exists(_cache_f) and (_t.time() - _os.path.getmtime(_cache_f) < 43200):
            return
        df_usa = build_daily_dataframe(YIELD_SERIES, FRED_API_KEY)
        df_eur = bce_get_yields_df()
        with open(_cache_f, 'wb') as _f:
            _pk.dump((df_usa, df_eur), _f)
        print("✓ [FRED] cache curva tassi pre-caricata", flush=True)
    except Exception as _e:
        print(f"⚠ [FRED] prewarm yields fallito: {_e}", flush=True)


import threading as _thr
_thr.Thread(target=_prewarm_yields, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True, port=8051)
