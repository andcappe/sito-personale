"""
Valutazione di un titolo azionario — modulo condiviso.

Sei modelli su dati fondamentali Yahoo Finance (yfinance): DCF, DDM, formula di
Graham, P/E relativo, EV/EBITDA e heatmap di sensitività, più una scheda
SaaS & Growth.

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
import plotly.graph_objects as go                       # noqa: F401  (usato nei callback)
from dash import html, dcc, Input, Output, State, callback_context, no_update


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
              "overflow-y": "auto", "height": "calc(100vh - 250px)",
              "min-height": "520px"})

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
