"""
Macro FED · BCE — Dashboard Analisi Monetaria
Navbar identica al sito portafoglio + Analisi Monetaria completa da FRED/BCE/Eurostat.
"""

import io
import json
import math
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context, no_update
from dash.exceptions import PreventUpdate

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
_EXTERNAL_STYLESHEETS = [
    'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=_EXTERNAL_STYLESHEETS,
    requests_pathname_prefix='/macro/',
    routes_pathname_prefix='/macro/',
)

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Macro FED · BCE</title>
{%favicon%}
{%css%}
<style>
  [data-tooltip] { position: relative; }
  [data-tooltip]::after {
    content: attr(data-tooltip);
    position: absolute; left: 100%; top: 50%;
    transform: translateY(-50%);
    background: #1a3a5c; color: #fff;
    padding: 4px 8px; border-radius: 4px;
    font-size: 11px; white-space: nowrap;
    z-index: 9999; pointer-events: none;
    opacity: 0; transition: opacity 0.15s; margin-left: 6px;
  }
  [data-tooltip]:hover::after { opacity: 1; }
  @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────────────────────
FRED_API_KEY = "65061ed1fa4c47d53b1d644e1cd858d3"

DEFAULT_SERIES = {
    "M2SL":    ("M2 Money Supply",    "M"),
    "M2V":     ("M2 Velocity",        "Q"),
    "CPIAUCSL":("CPI All Items",      "M"),
    "CPILFESL":("CPI Core",           "M"),
    "GDPC1":   ("Real GDP",           "Q"),
    "FEDFUNDS":("Fed Funds Rate",     "M"),
    "UNRATE":  ("Unemployment Rate",  "M"),
    "T10Y2Y":  ("Yield Curve 10Y-2Y","M"),
}

EUR_MONETARY_FRED = {
    "ECBDFR":          ("Fed Funds Rate",      "M"),
    "LRHUTTTTEZM156S": ("Unemployment Rate",   "M"),
    "IRLTLT01EZM156N": ("EUR 10Y Yield",       "M"),
    "IRT3TM01EZM156N": ("EUR 3M Yield",        "M"),
}

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

COLORS = [
    "#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd",
    "#8c564b","#e377c2","#17becf","#bcbd22","#7f7f7f",
]

# ─────────────────────────────────────────────────────────────────────────────
# Funzioni dati
# ─────────────────────────────────────────────────────────────────────────────

def fred_get(series_id: str, api_key: str, retries: int = 3) -> pd.Series | None:
    """Scarica una serie da FRED REST API (senza fredapi)."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}"
        f"&file_type=json&observation_start=1990-01-01"
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MacroDash/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
            obs = [(o["date"], float(o["value"]))
                   for o in data.get("observations", []) if o["value"] != "."]
            if not obs:
                return None
            dates, values = zip(*obs)
            s = pd.Series(list(values), index=pd.to_datetime(list(dates)))
            return s.dropna()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FRED [{series_id}]: {e}")
    return None


def to_monthly(s: pd.Series, freq: str) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    if freq in ("M",):
        s.index = s.index.to_period("M").to_timestamp()
        s = s[~s.index.duplicated(keep="last")]
    elif freq in ("Q",):
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
    """Scarica M2 Area Euro dalla BCE via SDMX REST API (mensile, miliardi EUR)."""
    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        "BSI/M.U2.Y.V.M20.X.1.U2.2300.Z01.E"
        "?format=csvdata&startPeriod=1997-01"
    )
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/csv"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
        lines = [ln for ln in raw.strip().split("\n") if ln.strip()]
        if len(lines) < 2:
            return None
        hdrs = [h.strip().strip('"') for h in lines[0].split(",")]
        ti = next((i for i, h in enumerate(hdrs) if h == "TIME_PERIOD"), -1)
        vi = next((i for i, h in enumerate(hdrs) if h == "OBS_VALUE"), -1)
        if ti < 0 or vi < 0:
            return None
        obs = {}
        for line in lines[1:]:
            cols = line.split(",")
            try:
                period = cols[ti].strip().strip('"')
                val    = float(cols[vi].strip().strip('"')) / 1000.0
                obs[period] = val
            except (ValueError, IndexError):
                continue
        if not obs:
            return None
        s = pd.Series(obs)
        s.index = pd.to_datetime(s.index)
        s = s.dropna().sort_index()
        print(f"  ✓ BCE M2: {len(s)} obs")
        return s
    except Exception as e:
        print(f"  ✗ BCE M2: {e}")
        return None


def eurostat_get(dataset: str, params: dict, geo: str) -> pd.Series | None:
    """Scarica una serie temporale dall'API JSON di Eurostat."""
    base = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    all_params = {**params, "geo": geo}
    qs   = "&".join(f"{k}={v}" for k, v in all_params.items())
    url  = f"{base}/{dataset}?{qs}&lang=en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MacroDash/1.0"})
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
            if v is None:
                continue
            sample = tcat
            if "-Q" in sample:
                yr, q = sample.split("-Q")
                m = (int(q) - 1) * 3 + 1
                result[f"{yr}-{m:02d}-01"] = float(v)
            elif len(sample) == 7 and sample[4] == "M":
                result[sample[:4] + "-" + sample[5:] + "-01"] = float(v)
            else:
                result[sample + "-01"] = float(v)
        if not result:
            return None
        s = pd.Series(result)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:
        print(f"  Eurostat parse [{dataset}/{geo}]: {e}")
        return None


def build_dataframe(series_dict: dict, api_key: str) -> pd.DataFrame:
    """Scarica serie FRED e costruisce DataFrame mensile."""
    frames = {}
    def _fetch(sid, label, freq):
        raw = fred_get(sid, api_key)
        if raw is not None:
            s = to_monthly(raw, freq)
            print(f"  ✓ {sid}: {len(s)} obs")
            return label, s
        return label, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_fetch, sid, lbl, freq) for sid, (lbl, freq) in series_dict.items()]
        for fut in as_completed(futs):
            lbl, s = fut.result()
            if s is not None:
                frames[lbl] = s

    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def build_monetary_eur_df(geo: str, api_key: str) -> pd.DataFrame:
    """Costruisce il DataFrame monetario per l'Area Euro."""
    frames = {}

    def _run_bce():
        m2 = bce_get_m2()
        return "M2 Money Supply", to_monthly(m2, "M") if m2 is not None else None

    def _run_fred(sid, label, freq):
        s = fred_get(sid, api_key)
        return label, to_monthly(s, freq) if s is not None else None

    def _run_eurostat(dataset, params, label):
        raw = eurostat_get(dataset, params, geo)
        return label, to_monthly(raw, "M") if raw is not None else None

    def _run_eurostat_q(dataset, params, label):
        raw = eurostat_get(dataset, params, geo)
        return label, to_monthly(raw, "Q") if raw is not None else None

    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [
            ex.submit(_run_bce),
            ex.submit(_run_eurostat, "prc_hicp_midx", {"coicop": "CP00",            "unit": "I15"}, "CPI All Items"),
            ex.submit(_run_eurostat, "prc_hicp_midx", {"coicop": "TOT_X_NRG_FOOD", "unit": "I15"}, "CPI Core"),
            ex.submit(_run_eurostat_q, "namq_10_gdp", {"na_item": "B1GQ", "unit": "CLV15_MEUR", "s_adj": "SCA"}, "Real GDP"),
            ex.submit(_run_eurostat_q, "namq_10_gdp", {"na_item": "B1GQ", "unit": "CP_MEUR",    "s_adj": "SCA"}, "__GDP_NOM__"),
        ]
        futs += [ex.submit(_run_fred, sid, lbl, freq) for sid, (lbl, freq) in EUR_MONETARY_FRED.items()]
        for fut in as_completed(futs):
            lbl, s = fut.result()
            if s is not None:
                frames[lbl] = s

    # Yield Curve = 10Y − 3M
    if "EUR 10Y Yield" in frames and "EUR 3M Yield" in frames:
        frames["Yield Curve 10Y-2Y"] = frames.pop("EUR 10Y Yield") - frames.pop("EUR 3M Yield")
        print("    ✓ Yield Curve EUR = 10Y − 3M")
    else:
        frames.pop("EUR 10Y Yield", None)
        frames.pop("EUR 3M Yield",  None)

    # M2 Velocity = PIL Nominale / M2
    pil_n = frames.pop("__GDP_NOM__", None)
    m2_s  = frames.get("M2 Money Supply")
    if pil_n is not None and m2_s is not None:
        idx = m2_s.dropna().index.intersection(pil_n.dropna().index)
        if len(idx) >= 8:
            vel = pil_n.reindex(idx) / 1000.0 / m2_s.reindex(idx)
            vel.name = "M2 Velocity"
            frames["M2 Velocity"] = vel
            print(f"    ✓ M2 Velocity ({len(vel)} obs)")

    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Trasformazioni
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Figure Plotly
# ─────────────────────────────────────────────────────────────────────────────

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
        hovermode="closest",
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


def make_mvpq_chart(df: pd.DataFrame,
                    start: pd.Timestamp, end: pd.Timestamp,
                    mvpq_show: list = None,
                    show_mv: bool = True,
                    show_pq: bool = True) -> go.Figure:
    col_m2 = next((c for c in df.columns if "M2 Money" in c or c == "M2 Velocity"[:0] or "M2 " in c), None)
    col_m2 = next((c for c in df.columns if "M2 Money" in c), None)
    col_v  = next((c for c in df.columns if "Velocity" in c), None)
    col_p  = next((c for c in df.columns if "CPI All" in c), None)
    col_q  = next((c for c in df.columns if "GDP" in c or "PIL" in c), None)

    missing = [n for n, c in [("M2", col_m2), ("Velocity", col_v),
                               ("CPI", col_p), ("GDP", col_q)] if c is None]
    if missing:
        return empty_fig(f"Mancano serie per MV=PQ: {', '.join(missing)}")

    common = (df[col_m2].dropna().index
              .intersection(df[col_v].dropna().index)
              .intersection(df[col_p].dropna().index)
              .intersection(df[col_q].dropna().index))
    if len(common) < 24:
        return empty_fig("Dati insufficienti per MV=PQ (< 24 mesi comuni)")

    m2, v, p, q = (df[c].reindex(common) for c in [col_m2, col_v, col_p, col_q])
    mv_raw = m2 * v
    pq_raw = p  * q / 100

    mv_full    = mv_raw / mv_raw.iloc[0] * 100
    pq_full    = pq_raw / pq_raw.iloc[0] * 100
    mv_yoy_full = yoy(mv_full)
    pq_yoy_full = yoy(pq_full)

    mv_yoy = mv_yoy_full.loc[start:end].copy()
    pq_yoy = pq_yoy_full.loc[start:end].copy()
    if mv_yoy.empty or pq_yoy.empty:
        return empty_fig("Nessun dato nel range selezionato")

    gap = mv_yoy - pq_yoy
    mv_raw_sl  = mv_raw.loc[start:end].dropna()
    pq_raw_sl  = pq_raw.loc[start:end].dropna()

    def _cumprod_from_zero(s):
        r = s.pct_change().fillna(0)
        return ((1 + r).cumprod() - 1) * 100

    mv = _cumprod_from_zero(mv_raw_sl)
    pq = _cumprod_from_zero(pq_raw_sl)

    show_yoy = "yoy" in (mvpq_show or ["yoy", "cum"])
    show_cum = "cum" in (mvpq_show or ["yoy", "cum"])

    if not show_mv and not show_pq:
        return empty_fig("Seleziona almeno una serie MV=PQ (M·V o P·Q)")
    if not show_yoy and not show_cum:
        return empty_fig("Seleziona almeno YoY o CumProd per il grafico MV=PQ")

    mv_yoy_mean = float(mv_yoy.dropna().mean())
    pq_yoy_mean = float(pq_yoy.dropna().mean())
    mv_cum_mean = float(mv.dropna().mean())
    pq_cum_mean = float(pq.dropna().mean())

    layout_base = dict(
        hovermode="closest", autosize=True, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=9),
                    bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=50, b=35, l=60, r=110),
        paper_bgcolor="white", plot_bgcolor="#f8f8f8",
    )

    def _hline_annot(fig, y, color, text, row=None):
        kw = dict(row=row, col=1) if row else {}
        fig.add_hline(y=y, line_color=color, line_dash="dashdot", line_width=1.2,
                      annotation_text=text, annotation_position="right",
                      annotation_font=dict(size=9, color=color), **kw)

    fig = go.Figure()

    if show_yoy:
        if show_mv:
            fig.add_trace(go.Scatter(x=mv_yoy.index, y=mv_yoy.values, name="M·V YoY",
                line=dict(color="#1f77b4", width=2.5),
                hovertemplate="M·V YoY: %{y:.2f}%<extra></extra>"))
        if show_pq:
            fig.add_trace(go.Scatter(x=pq_yoy.index, y=pq_yoy.values, name="P·Q YoY",
                line=dict(color="#d62728", width=2.5),
                hovertemplate="P·Q YoY: %{y:.2f}%<extra></extra>"))
        if show_mv and show_pq:
            fig.add_trace(go.Bar(x=gap.clip(lower=0).index, y=gap.clip(lower=0).values,
                name="Gap+ (MV>PQ)", marker_color="rgba(44,160,44,0.45)"))
            fig.add_trace(go.Bar(x=gap.clip(upper=0).index, y=gap.clip(upper=0).values,
                name="Gap− (PQ>MV)", marker_color="rgba(214,39,40,0.35)"))
        _hline_annot(fig, 0, "#555", "")
        if show_mv:
            _hline_annot(fig, mv_yoy_mean, "#1f77b4", f"μ M·V={mv_yoy_mean:.2f}%")
        if show_pq:
            _hline_annot(fig, pq_yoy_mean, "#d62728", f"μ P·Q={pq_yoy_mean:.2f}%")

    if show_cum:
        if show_mv:
            fig.add_trace(go.Scatter(x=mv.index, y=mv.values, name="M·V Cum%",
                line=dict(color="#1f77b4", width=2, dash="dash"),
                hovertemplate="M·V cum: %{y:.2f}%<extra></extra>"))
        if show_pq:
            fig.add_trace(go.Scatter(x=pq.index, y=pq.values, name="P·Q Cum%",
                line=dict(color="#d62728", width=2, dash="dash"),
                hovertemplate="P·Q cum: %{y:.2f}%<extra></extra>"))
        if show_mv:
            _hline_annot(fig, mv_cum_mean, "#1f77b4", f"μ M·V={mv_cum_mean:.2f}%")
        if show_pq:
            _hline_annot(fig, pq_cum_mean, "#d62728", f"μ P·Q={pq_cum_mean:.2f}%")

    y_label = "Δ% YoY" if show_yoy and not show_cum else "Crescita % cum." if show_cum and not show_yoy else "Δ% YoY / Cum%"
    title   = "MV = PQ — Δ% YoY" if show_yoy and not show_cum else "MV = PQ — Cumulata %" if show_cum and not show_yoy else "MV = PQ — YoY + Cumulata %"
    fig.update_yaxes(title_text=y_label, showgrid=True, gridcolor="#e8e8e8")
    fig.update_layout(
        title=dict(text=title, font=dict(size=11), x=0.01),
        barmode="overlay",
        **layout_base)

    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
    return fig


def _extract_mvpq_components(df: pd.DataFrame, suffix: str):
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
    m2, v, p, q = (df[col_m2].reindex(common), df[col_v].reindex(common),
                   df[col_p].reindex(common), df[col_q].reindex(common))
    return {"mv_raw": m2 * v, "pq_raw": p * q / 100, "common": common}, []


def make_mvpq_both_chart(df: pd.DataFrame, start, end, mvpq_show: list, series_show: list):
    mv_usa_c, pq_usa_c = "#1f77b4", "#d62728"
    mv_eur_c, pq_eur_c = "#2ca02c", "#ff7f0e"

    comps_usa, miss_usa = _extract_mvpq_components(df, "🇺🇸")
    comps_eur, miss_eur = _extract_mvpq_components(df, "🇪🇺")

    if comps_usa is None and comps_eur is None:
        return empty_fig(f"Dati MV=PQ non disponibili — USA: {miss_usa} | EUR: {miss_eur}")

    show_yoy = "yoy" in (mvpq_show or [])
    show_cum = "cum" in (mvpq_show or [])
    series_show = series_show or ["mv_usa", "pq_usa", "mv_eur", "pq_eur"]

    def _yoy(s):
        return ((s - s.shift(12)) / s.shift(12).abs()) * 100

    def _cum(s):
        r = _yoy(s.loc[start:end]) / 100
        return ((1 + r).cumprod() - 1) * 100

    fig = go.Figure()
    added = 0

    for comps, flag, mv_c, pq_c, mv_key, pq_key in [
        (comps_usa, "🇺🇸", mv_usa_c, pq_usa_c, "mv_usa", "pq_usa"),
        (comps_eur, "🇪🇺", mv_eur_c, pq_eur_c, "mv_eur", "pq_eur"),
    ]:
        if comps is None:
            continue
        mv_raw = comps["mv_raw"]
        pq_raw = comps["pq_raw"]

        if show_yoy:
            mv_yoy = _yoy(mv_raw).loc[start:end].dropna()
            pq_yoy = _yoy(pq_raw).loc[start:end].dropna()
            if mv_key in series_show and not mv_yoy.empty:
                fig.add_trace(go.Scatter(x=mv_yoy.index, y=mv_yoy.values,
                    name=f"M·V YoY {flag}", line=dict(color=mv_c, width=2.5),
                    hovertemplate=f"M·V YoY {flag}: %{{y:.2f}}%<extra></extra>"))
                added += 1
            if pq_key in series_show and not pq_yoy.empty:
                fig.add_trace(go.Scatter(x=pq_yoy.index, y=pq_yoy.values,
                    name=f"P·Q YoY {flag}", line=dict(color=pq_c, width=2.5, dash="dot"),
                    hovertemplate=f"P·Q YoY {flag}: %{{y:.2f}}%<extra></extra>"))
                added += 1

        if show_cum:
            mv_cum = _cum(mv_raw).dropna()
            pq_cum = _cum(pq_raw).dropna()
            if mv_key in series_show and not mv_cum.empty:
                fig.add_trace(go.Scatter(x=mv_cum.index, y=mv_cum.values,
                    name=f"M·V Cum {flag}", line=dict(color=mv_c, width=2, dash="dash"),
                    hovertemplate=f"M·V Cum {flag}: %{{y:.2f}}%<extra></extra>"))
                added += 1
            if pq_key in series_show and not pq_cum.empty:
                fig.add_trace(go.Scatter(x=pq_cum.index, y=pq_cum.values,
                    name=f"P·Q Cum {flag}", line=dict(color=pq_c, width=2, dash="dashdot"),
                    hovertemplate=f"P·Q Cum {flag}: %{{y:.2f}}%<extra></extra>"))
                added += 1

    if added == 0:
        return empty_fig("Seleziona almeno una serie MV=PQ nel pannello controlli")

    _hline_annot(fig, 0, "#555", "")
    layout_base = dict(
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        legend=dict(orientation="h", y=-0.28, font=dict(size=9)),
        margin=dict(l=50, r=20, t=40, b=70),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8",
                     title_text="Δ% YoY / Crescita % cumulata")
    fig.update_layout(
        title=dict(text="MV = PQ — Confronto USA 🇺🇸 vs Europa 🇪🇺",
                   font=dict(size=11), x=0.01),
        **layout_base,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slider_params(df: pd.DataFrame, step_years: int = 5):
    mn = int(df.index.min().timestamp())
    mx = int(df.index.max().timestamp())
    marks = {
        int(pd.Timestamp(yr, 1, 1).timestamp()): str(yr)
        for yr in range(df.index.min().year, df.index.max().year + 1, step_years)
    }
    return mn, mx, [mn, mx], marks


def _navbar():
    """Navbar identica al sito portafoglio."""
    link_style = {
        "fontSize": "0.82rem", "fontWeight": "600",
        "color": "#6b7a99", "letterSpacing": "0.04em",
        "textTransform": "uppercase", "textDecoration": "none",
        "transition": "color 0.2s", "fontFamily": "Inter, sans-serif",
    }
    return html.Nav([
        # Brand
        html.A([
            html.Span("A·C", style={
                "fontFamily": "'Playfair Display', serif",
                "fontSize": "1.1rem", "color": "#1a3a6b",
                "fontWeight": "700", "marginRight": "10px",
            }),
            html.Span("FinecoBank", style={
                "fontFamily": "Inter, sans-serif",
                "fontSize": "0.62rem", "fontWeight": "700",
                "letterSpacing": "0.1em", "textTransform": "uppercase",
                "color": "#f37021",
                "background": "rgba(243,112,33,0.1)",
                "border": "1px solid rgba(243,112,33,0.3)",
                "padding": "3px 8px", "borderRadius": "4px",
            }),
        ], href="https://andreacappelletti.app", target="_blank",
           style={"textDecoration": "none", "display": "flex", "alignItems": "center"}),

        # Link navigazione
        html.Ul([
            html.Li(html.A("Home",         href="https://andreacappelletti.app",             target="_blank", style=link_style)),
            html.Li(html.A("Chi Sono",     href="https://andreacappelletti.app#chi-sono",    target="_blank", style=link_style)),
            html.Li(html.A("Esperienza",   href="https://andreacappelletti.app#esperienza",  target="_blank", style=link_style)),
            html.Li(html.A("Strumenti",    href="https://andreacappelletti.app#dashboard",   target="_blank", style=link_style)),
            html.Li(html.A("Prenota Call", href="https://andreacappelletti.app#prenota",     target="_blank", style=link_style)),
            html.Li(html.A("Contatti",     href="https://andreacappelletti.app#contatti",    target="_blank", style=link_style)),
        ], style={"display": "flex", "gap": "2rem", "listStyle": "none",
                  "margin": "0", "padding": "0", "alignItems": "center"}),

        # CTA
        html.A([
            html.I(className="fa-regular fa-calendar", style={"marginRight": "7px"}),
            "Prenota call",
        ], href="https://andreacappelletti.app#prenota", target="_blank", style={
            "padding": "9px 20px",
            "background": "#1a3a6b", "color": "white",
            "borderRadius": "7px", "fontSize": "0.8rem", "fontWeight": "700",
            "letterSpacing": "0.04em", "textTransform": "uppercase",
            "textDecoration": "none", "display": "inline-flex",
            "alignItems": "center", "fontFamily": "Inter, sans-serif",
        }),
    ], style={
        "position": "fixed", "top": "0", "left": "0", "right": "0",
        "zIndex": "1000",
        "display": "flex", "alignItems": "center",
        "justifyContent": "space-between",
        "padding": "0 5%", "height": "64px",
        "background": "rgba(255,255,255,0.96)",
        "backdropFilter": "blur(14px)",
        "borderBottom": "1px solid #e2e8f0",
        "boxShadow": "0 1px 8px rgba(26,58,107,0.07)",
        "fontFamily": "Inter, sans-serif",
    })


def _sidebar():
    return html.Div([
        html.Div(html.B("Serie attive", style={"font-size": "11px"}),
                 style={"padding-bottom": "6px", "margin-bottom": "8px",
                        "border-bottom": "2px solid #ccc"}),
        html.Div(id="mon-series-checklist",
                 children="— carica i dati per vedere le serie —",
                 style={"font-size": "10px", "color": "#aaa",
                        "font-style": "italic"}),
        html.Hr(style={"margin": "10px 0"}),
        html.Div("Seleziona / Deseleziona", style={
            "font-size": "9px", "font-weight": "600",
            "color": "#6b7a99", "margin-bottom": "5px",
            "text-transform": "uppercase", "letter-spacing": "0.05em",
        }),
        html.Div([
            html.Button("✔ Seleziona", id="sel-all", n_clicks=0,
                        style={"font-size": "9px", "padding": "3px 8px",
                               "margin-right": "4px", "cursor": "pointer",
                               "background": "#e8f5e9", "border": "1px solid #a5d6a7",
                               "border-radius": "4px", "color": "#1b5e20"}),
            html.Button("✘ Deseleziona", id="sel-none", n_clicks=0,
                        style={"font-size": "9px", "padding": "3px 8px",
                               "cursor": "pointer",
                               "background": "#fce4ec", "border": "1px solid #f48fb1",
                               "border-radius": "4px", "color": "#880e4f"}),
        ], style={"display": "flex"}),
    ], style={"padding": "12px", "overflow-y": "auto"})


def _controls_bar():
    return html.Div([
        # Fonte
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
                    options=[{"label": v, "value": k} for k, v in EUROSTAT_GEO.items()],
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

        # Vista
        html.Div([
            html.Label("Vista:", style={"font-size": "11px", "font-weight": "bold",
                                        "margin-right": "8px", "white-space": "nowrap"}),
            dcc.Checklist(
                id="view-mode",
                options=[
                    {"label": " Assoluta", "value": "abs"},
                    {"label": " Δ% YoY",  "value": "yoy"},
                ],
                value=["yoy"], inline=True,
                style={"font-size": "11px"},
                inputStyle={"margin-right": "3px"},
                labelStyle={"margin-right": "14px"},
            ),
        ], style={"display": "flex", "align-items": "center",
                  "background": "#fff8e1", "border": "1px solid #ffe082",
                  "border-radius": "4px", "padding": "5px 12px",
                  "margin-right": "14px"}),

        # MV=PQ
        html.Div([
            html.Label("MV=PQ:", style={"font-size": "11px", "font-weight": "bold",
                                         "margin-right": "8px", "white-space": "nowrap"}),
            dcc.Checklist(
                id="mvpq-show",
                options=[
                    {"label": " YoY",     "value": "yoy"},
                    {"label": " CumProd", "value": "cum"},
                ],
                value=["yoy", "cum"], inline=True,
                style={"font-size": "11px"},
                inputStyle={"margin-right": "3px"},
                labelStyle={"margin-right": "10px"},
            ),
        ], style={"display": "flex", "align-items": "center",
                  "background": "#e8f5e9", "border": "1px solid #a5d6a7",
                  "border-radius": "4px", "padding": "5px 12px",
                  "margin-right": "14px"}),

        # Serie MV=PQ
        html.Div([
            html.Label("MV=PQ serie:", style={"font-size": "11px", "font-weight": "bold",
                                              "margin-right": "8px", "white-space": "nowrap"}),
            dcc.Checklist(
                id="mvpq-series-show",
                options=[
                    {"label": " M·V 🇺🇸", "value": "mv_usa"},
                    {"label": " P·Q 🇺🇸", "value": "pq_usa"},
                ],
                value=["mv_usa", "pq_usa"],
                inline=True,
                style={"font-size": "11px"},
                inputStyle={"margin-right": "3px"},
                labelStyle={"margin-right": "10px"},
            ),
        ], id="mvpq-series-wrapper",
           style={"display": "flex", "align-items": "center",
                  "background": "#fce4ec", "border": "1px solid #f48fb1",
                  "border-radius": "4px", "padding": "5px 12px",
                  "margin-right": "14px"}),

        # Bottone
        html.Button(
            "🔄  Carica dati",
            id="btn-aggiorna", n_clicks=0,
            style={
                "background": "#1a3a6b", "color": "white",
                "border": "none", "padding": "8px 22px",
                "border-radius": "5px", "cursor": "pointer",
                "font-size": "13px", "font-weight": "bold",
                "letter-spacing": "0.5px",
                "box-shadow": "0 2px 4px rgba(0,0,0,0.2)",
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
            id="date-slider", min=0, max=1, value=[0, 1],
            marks={}, step=86400 * 30,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(id="slider-label",
                 style={"font-size": "10px", "color": "#666",
                        "text-align": "center", "margin-top": "2px"}),
    ], style={"padding": "8px 28px 2px"})


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div([
    # ── Stores ───────────────────────────────────────────────────────────────
    dcc.Loading(
        id="global-loading",
        type="circle",
        fullscreen=True,
        color="#1a3a6b",
        overlay_style={"background": "rgba(255,255,255,0.75)", "zIndex": 9999},
        children=dcc.Store(id="store-data"),
    ),
    dcc.Store(id="store-mon-source-type", data="usa"),

    # ── Navbar ───────────────────────────────────────────────────────────────
    _navbar(),

    # ── Contenuto (margine top 64px per navbar fissa) ─────────────────────────
    html.Div([

        # ── Intestazione pagina ───────────────────────────────────────────────
        html.Div([
            html.H1("Macro FED · BCE", style={
                "margin": "0",
                "font-size": "1.6rem",
                "font-weight": "700",
                "color": "#1a3a6b",
                "font-family": "'Playfair Display', serif",
                "letter-spacing": "0.02em",
            }),
            html.P("Analisi monetaria USA & Area Euro — M2, Velocità, CPI, PIL, MV=PQ", style={
                "margin": "2px 0 0 0",
                "font-size": "0.78rem",
                "color": "#6b7a99",
                "font-family": "Inter, sans-serif",
            }),
        ], style={
            "padding": "14px 20px 12px",
            "border-bottom": "2px solid #e2e8f0",
            "background": "linear-gradient(90deg, #f0f4fb 0%, #ffffff 100%)",
            "margin-bottom": "0",
        }),

        # ── Barra controlli ───────────────────────────────────────────────────
        _controls_bar(),

        # ── Corpo: sidebar + grafici ──────────────────────────────────────────
        html.Div([
            # Sidebar
            html.Div(
                _sidebar(),
                style={
                    "width": "210px", "min-width": "200px",
                    "border-right": "1px solid #ddd",
                    "height": "calc(100vh - 186px)",
                    "overflow-y": "auto",
                    "background": "#fafafa",
                }
            ),

            # Area grafici
            html.Div([
                _slider_area(),

                # Grafico Assoluto
                html.Div([
                    dcc.Loading(type="circle", color="#1a3a6b", children=[
                        dcc.Graph(
                            id="chart-main",
                            figure=empty_fig("Clicca  🔄 Carica dati  per scaricare le serie"),
                            style={"height": "42vh", "width": "100%"},
                            config={"responsive": True, "scrollZoom": True,
                                    "displayModeBar": True, "displaylogo": False},
                        ),
                    ]),
                ], id="main-abs-wrapper"),

                # Grafico YoY
                html.Div([
                    dcc.Loading(type="circle", color="#1a3a6b", children=[
                        dcc.Graph(
                            id="chart-main-yoy",
                            figure=empty_fig("Clicca  🔄 Carica dati  per scaricare le serie"),
                            style={"height": "42vh", "width": "100%"},
                            config={"responsive": True, "scrollZoom": True,
                                    "displayModeBar": True, "displaylogo": False},
                        ),
                    ]),
                ], id="main-yoy-wrapper", style={"display": "none"}),

                # Banda + Grafico MV=PQ (nascosti se nessun checkbox selezionato)
                html.Div([
                    html.Div([
                        html.B("MV = PQ — Teoria Quantitativa della Moneta",
                               style={"font-size": "11px", "color": "#1a5276"}),
                        html.Span("  M·V = Moneta × Velocità  |  P·Q = CPI × PIL Reale",
                                  style={"font-size": "10px", "color": "#666", "margin-left": "10px"}),
                    ], style={"padding": "5px 16px",
                              "background": "#eaf4fb",
                              "border-top": "1px solid #aed6f1",
                              "border-bottom": "1px solid #aed6f1"}),
                    dcc.Loading(type="circle", color="#1a3a6b", children=[
                        dcc.Graph(
                            id="chart-mvpq",
                            figure=empty_fig("Clicca  🔄 Carica dati  per scaricare le serie"),
                            style={"height": "43vh", "width": "100%"},
                            config={"responsive": True, "scrollZoom": True,
                                    "displayModeBar": True, "displaylogo": False},
                        ),
                    ]),
                ], id="mvpq-wrapper"),

            ], style={"flex": "1", "min-width": "0",
                      "overflow-y": "auto",
                      "height": "calc(100vh - 186px)"}),

        ], style={"display": "flex"}),

    ], style={"margin-top": "64px", "font-family": "Inter, sans-serif"}),

], style={"font-family": "Inter, sans-serif"})


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("mon-geo-wrapper", "style"),
    Input("mon-source-type", "value"),
)
def toggle_geo(source_type):
    base = {"margin-left": "6px"}
    return base if source_type in ("eur", "both") else {**base, "display": "none"}


@app.callback(
    Output("mvpq-series-wrapper",  "style"),
    Output("mvpq-series-show",     "options"),
    Output("mvpq-series-show",     "value"),
    Input("mon-source-type", "value"),
)
def toggle_series_check(source_type):
    base = {
        "display": "flex", "align-items": "center",
        "background": "#fce4ec", "border": "1px solid #f48fb1",
        "border-radius": "4px", "padding": "5px 12px",
        "margin-right": "14px",
    }
    if source_type == "eur":
        opts  = [{"label": " M·V 🇪🇺", "value": "mv_eur"},
                 {"label": " P·Q 🇪🇺", "value": "pq_eur"}]
        vals  = ["mv_eur", "pq_eur"]
    elif source_type == "both":
        opts  = [{"label": " M·V 🇺🇸", "value": "mv_usa"},
                 {"label": " P·Q 🇺🇸", "value": "pq_usa"},
                 {"label": " M·V 🇪🇺", "value": "mv_eur"},
                 {"label": " P·Q 🇪🇺", "value": "pq_eur"}]
        vals  = ["mv_usa", "pq_usa", "mv_eur", "pq_eur"]
    else:  # usa
        opts  = [{"label": " M·V 🇺🇸", "value": "mv_usa"},
                 {"label": " P·Q 🇺🇸", "value": "pq_usa"}]
        vals  = ["mv_usa", "pq_usa"]
    return base, opts, vals


@app.callback(
    Output("store-data",            "data"),
    Output("status-msg",            "children"),
    Output("date-slider",           "min"),
    Output("date-slider",           "max"),
    Output("date-slider",           "value"),
    Output("date-slider",           "marks"),
    Output("store-mon-source-type", "data"),
    Input("btn-aggiorna",           "n_clicks"),
    State("mon-source-type",        "value"),
    State("mon-eur-geo",            "value"),
    prevent_initial_call=True,
)
def aggiorna(n_clicks, source_type, eur_geo):
    if not n_clicks:
        raise PreventUpdate

    api_key = FRED_API_KEY

    if source_type == "both":
        geo = eur_geo or "EA20"
        print(f"\n▶ Download confronto USA + EUR [{geo}]...")
        df_usa = build_dataframe(DEFAULT_SERIES, api_key)
        df_eur = build_monetary_eur_df(geo, api_key)
        if df_usa.empty and df_eur.empty:
            return None, "❌ Nessun dato — controlla la connessione", 0, 1, [0, 1], {}, source_type
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
    else:
        print("\n▶ Download monetario USA (FRED)...")
        df = build_dataframe(DEFAULT_SERIES, api_key)
        source_lbl = "USA (FRED)"

    if df.empty:
        return None, "❌ Nessun dato — controlla la connessione", 0, 1, [0, 1], {}, source_type

    d1  = df.index.min().strftime("%m/%Y")
    d2  = df.index.max().strftime("%m/%Y")
    msg = f"✅  {source_lbl}  |  {len(df.columns)} serie  |  {len(df)} obs  ({d1} → {d2})"
    return df.to_json(date_format="iso", orient="split"), msg, *_slider_params(df), source_type


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
    Input("store-data",            "data"),
    State("store-mon-source-type", "data"),
    prevent_initial_call=False,
)
def populate_checklist(data, source_type):
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
    Output("chart-main",        "figure"),
    Output("chart-main-yoy",    "figure"),
    Output("chart-mvpq",        "figure"),
    Output("main-abs-wrapper",  "style"),
    Output("main-yoy-wrapper",  "style"),
    Output("mvpq-wrapper",      "style"),
    Input("store-data",         "data"),
    Input("date-slider",        "value"),
    Input({"type": "series-check", "index": ALL}, "value"),
    Input("view-mode",          "value"),
    Input("mvpq-show",          "value"),
    Input("mvpq-series-show",   "value"),
    State("store-mon-source-type", "data"),
    prevent_initial_call=False,
)
def update_charts(data, slider_val, checks, view_mode, mvpq_show, mvpq_series_show, source_type):
    view_mode   = view_mode or []
    show_abs    = "abs" in view_mode
    show_yoy    = "yoy" in view_mode
    show_mvpq   = bool(mvpq_show)

    abs_style   = {} if show_abs  else {"display": "none"}
    yoy_style   = {} if show_yoy  else {"display": "none"}
    mvpq_style  = {} if show_mvpq else {"display": "none"}

    if not data:
        f = empty_fig("Clicca  🔄 Carica dati  per scaricare le serie")
        return f, f, f, abs_style, yoy_style, mvpq_style

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

    # Suffisso titolo
    suffix_map = {"eur": " — Area Euro (Eurostat)", "both": " — USA 🇺🇸 vs Europa 🇪🇺", "usa": " — USA (FRED)"}
    title_suffix = suffix_map.get(source_type or "usa", " — USA (FRED)")

    empty_sel = empty_fig("Seleziona almeno una serie nel pannello di sinistra")
    empty_rng = empty_fig("Nessun dato nel range selezionato")

    # Grafico Assoluto
    if not avail:
        fig_abs = empty_sel
    else:
        df_abs = transform_df(df[avail], "abs", start, end)
        fig_abs = empty_rng if df_abs.empty else make_line_chart(
            df_abs,
            "Serie Monetarie — Valori Assoluti" + title_suffix,
            "Valore", zero_line=False,
        )

    # Grafico YoY
    if not avail:
        fig_yoy = empty_sel
    else:
        df_yoy = transform_df(df[avail], "yoy", start, end)
        fig_yoy = empty_rng if df_yoy.empty else make_line_chart(
            df_yoy,
            "Serie Monetarie — Δ% Anno su Anno" + title_suffix,
            "Δ% YoY", zero_line=True,
        )

    # Grafico MV=PQ
    if source_type == "both":
        fig_mvpq = make_mvpq_both_chart(df, start, end, mvpq_show, mvpq_series_show)
    else:
        suffix   = "eur" if source_type == "eur" else "usa"
        defaults = [f"mv_{suffix}", f"pq_{suffix}"]
        checked  = mvpq_series_show or defaults
        show_mv  = f"mv_{suffix}" in checked
        show_pq  = f"pq_{suffix}" in checked
        fig_mvpq = make_mvpq_chart(df, start, end, mvpq_show, show_mv=show_mv, show_pq=show_pq)

    return fig_abs, fig_yoy, fig_mvpq, abs_style, yoy_style, mvpq_style


# ─────────────────────────────────────────────────────────────────────────────
# Redirect root → /macro/   (utile in locale)
# ─────────────────────────────────────────────────────────────────────────────
from flask import redirect as _redirect

@app.server.route('/')
def _root_redirect():
    return _redirect('/macro/')

@app.server.route('/health')
def _health():
    return 'OK', 200

# ─────────────────────────────────────────────────────────────────────────────
# Esposizione server
# ─────────────────────────────────────────────────────────────────────────────
server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8052)
