"""DCF Valuation Studio — a Streamlit app for FINAN 623 (vibe-coding project).

Run locally:  streamlit run app.py
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# PART 1 - FINANCE LOGIC (pure Python: no Streamlit calls, so it can be tested)
# =============================================================================


@dataclass
class Assumptions:
    base_revenue: float = 1_000.0      # last fiscal year revenue ($M)
    growth_start: float = 0.12         # year-1 revenue growth
    growth_end: float = 0.04           # final-year growth (linear fade)
    years: int = 5
    ebit_margin: float = 0.20
    tax_rate: float = 0.21
    da_pct: float = 0.04               # D&A as % of revenue
    capex_pct: float = 0.05            # CapEx as % of revenue
    nwc_pct: float = 0.10              # NWC as % of the *change* in revenue
    wacc: float = 0.09
    terminal_method: str = "Perpetuity growth"  # or "Exit multiple"
    terminal_growth: float = 0.025
    exit_multiple: float = 12.0        # EV / EBITDA
    net_debt: float = 200.0            # debt - cash ($M)
    shares: float = 100.0              # diluted shares (M)
    mid_year: bool = True


def capm_wacc(rf: float, beta: float, erp: float, pre_tax_kd: float,
              tax_rate: float, debt_weight: float) -> tuple[float, float]:
    """Return (cost of equity, WACC)."""
    ke = rf + beta * erp
    wacc = (1 - debt_weight) * ke + debt_weight * pre_tax_kd * (1 - tax_rate)
    return ke, wacc


def growth_path(a: Assumptions) -> np.ndarray:
    if a.years == 1:
        return np.array([a.growth_start])
    return np.linspace(a.growth_start, a.growth_end, a.years)


def projection(a: Assumptions) -> pd.DataFrame:
    """Unlevered free cash flow projection."""
    g = growth_path(a)
    revenue = a.base_revenue * np.cumprod(1 + g)
    prev_rev = np.concatenate([[a.base_revenue], revenue[:-1]])
    ebit = revenue * a.ebit_margin
    nopat = ebit * (1 - a.tax_rate)
    da = revenue * a.da_pct
    capex = revenue * a.capex_pct
    d_nwc = (revenue - prev_rev) * a.nwc_pct
    fcf = nopat + da - capex - d_nwc

    t = np.arange(1, a.years + 1) - (0.5 if a.mid_year else 0.0)
    df_factor = 1 / (1 + a.wacc) ** t

    return pd.DataFrame({
        "Year": np.arange(1, a.years + 1),
        "Revenue growth": g,
        "Revenue": revenue,
        "EBIT": ebit,
        "NOPAT": nopat,
        "D&A": da,
        "CapEx": capex,
        "Change in NWC": d_nwc,
        "Unlevered FCF": fcf,
        "Discount factor": df_factor,
        "PV of FCF": fcf * df_factor,
    })


def value(a: Assumptions) -> dict:
    """Run the full DCF and return the valuation bridge."""
    if a.terminal_method == "Perpetuity growth" and a.wacc <= a.terminal_growth:
        raise ValueError("WACC must be greater than the terminal growth rate.")

    proj = projection(a)
    last = proj.iloc[-1]
    if a.terminal_method == "Perpetuity growth":
        tv = last["Unlevered FCF"] * (1 + a.terminal_growth) / (a.wacc - a.terminal_growth)
    else:
        tv = (last["EBIT"] + last["D&A"]) * a.exit_multiple

    # Terminal value is discounted from the end of the final year.
    pv_tv = tv / (1 + a.wacc) ** a.years
    pv_fcf = proj["PV of FCF"].sum()
    ev = pv_fcf + pv_tv
    equity = ev - a.net_debt
    price = equity / a.shares

    # Cross-checks between the two terminal methods
    terminal_ebitda = last["EBIT"] + last["D&A"]
    implied_multiple = tv / terminal_ebitda if terminal_ebitda else np.nan
    implied_g = (tv * a.wacc - last["Unlevered FCF"]) / (tv + last["Unlevered FCF"])

    return {
        "projection": proj,
        "terminal_value": tv,
        "pv_terminal": pv_tv,
        "pv_fcf": pv_fcf,
        "enterprise_value": ev,
        "equity_value": equity,
        "price": price,
        "tv_share_of_ev": pv_tv / ev if ev else np.nan,
        "implied_exit_multiple": implied_multiple,
        "implied_perpetual_growth": implied_g,
    }


def sensitivity(a: Assumptions, waccs: list[float], seconds: list[float]) -> pd.DataFrame:
    """Implied share price grid: WACC (rows) x terminal growth or exit multiple (columns)."""
    field = "terminal_growth" if a.terminal_method == "Perpetuity growth" else "exit_multiple"
    out = pd.DataFrame(index=waccs, columns=seconds, dtype=float)
    for w in waccs:
        for s in seconds:
            try:
                out.loc[w, s] = value(replace(a, wacc=w, **{field: s}))["price"]
            except ValueError:
                out.loc[w, s] = np.nan
    return out


def monte_carlo(a: Assumptions, n: int = 5_000, seed: int = 42,
                sd_growth: float = 0.03, sd_margin: float = 0.03,
                sd_wacc: float = 0.01, sd_tg: float = 0.005) -> np.ndarray:
    """Vectorised simulation of implied share price under normally distributed shocks."""
    rng = np.random.default_rng(seed)
    gs = a.growth_start + rng.normal(0, sd_growth, n)
    ge = a.growth_end + rng.normal(0, sd_growth / 2, n)
    m = np.clip(a.ebit_margin + rng.normal(0, sd_margin, n), -0.5, 0.9)
    w = np.clip(a.wacc + rng.normal(0, sd_wacc, n), 0.02, None)
    tg = a.terminal_growth + rng.normal(0, sd_tg, n)
    tg = np.minimum(tg, w - 0.005)  # keep the Gordon model defined

    yrs = a.years
    frac = np.linspace(0, 1, yrs) if yrs > 1 else np.zeros(1)
    g = gs[:, None] + (ge - gs)[:, None] * frac[None, :]
    rev = a.base_revenue * np.cumprod(1 + g, axis=1)
    prev = np.concatenate([np.full((n, 1), a.base_revenue), rev[:, :-1]], axis=1)
    ebit = rev * m[:, None]
    da = rev * a.da_pct
    fcf = ebit * (1 - a.tax_rate) + da - rev * a.capex_pct - (rev - prev) * a.nwc_pct

    t = np.arange(1, yrs + 1) - (0.5 if a.mid_year else 0.0)
    disc = 1 / (1 + w[:, None]) ** t[None, :]
    pv_fcf = (fcf * disc).sum(axis=1)

    if a.terminal_method == "Perpetuity growth":
        tv = fcf[:, -1] * (1 + tg) / (w - tg)
    else:
        mult = a.exit_multiple + rng.normal(0, 1.5, n)
        tv = (ebit[:, -1] + da[:, -1]) * mult
    pv_tv = tv / (1 + w) ** yrs
    return (pv_fcf + pv_tv - a.net_debt) / a.shares


# =============================================================================
# PART 2 - STREAMLIT USER INTERFACE
# =============================================================================


st.set_page_config(page_title="DCF Valuation Studio", page_icon="📈", layout="wide")


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def money(x, d=1):
    return f"${x:,.{d}f}"


# ---------------------------------------------------------------- live company data (Yahoo Finance)
# Statement row names Yahoo uses. Several names are tried because they vary by company.
ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "ebit": ["Operating Income", "EBIT"],
    "tax_rate": ["Tax Rate For Calcs"],
    "da": ["Depreciation And Amortization", "Reconciled Depreciation",
           "Depreciation Amortization Depletion"],
    "capex": ["Capital Expenditure"],
    "debt": ["Total Debt"],
    "cash": ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
    "shares": ["Ordinary Shares Number", "Share Issued"],
}


def latest(df, names):
    """Most recent non-missing value of the first matching row in a statement (or None)."""
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").dropna()
            if not s.empty:
                return float(s.sort_index().iloc[-1])
    return None


def series(df, names):
    """A statement row as an oldest-to-newest series (or an empty series)."""
    if df is None or getattr(df, "empty", True):
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            return pd.to_numeric(df.loc[n], errors="coerce").dropna().sort_index()
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_symbol(query: str) -> str:
    """Turn a ticker or company name into a ticker symbol using Yahoo's search."""
    import yfinance as yf
    q = query.strip()
    try:
        quotes = yf.Search(q, max_results=8).quotes
        equities = [x for x in quotes if x.get("quoteType") == "EQUITY"]
        exact = [x for x in equities if x.get("symbol", "").upper() == q.upper()]
        if exact:
            return exact[0]["symbol"]
        if equities:
            return equities[0]["symbol"]
    except Exception:
        pass
    return q.upper()  # fall back to treating the text as a ticker


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company(symbol: str) -> dict:
    """Download profile, statements and price history for one ticker (cached for 1 hour)."""
    import yfinance as yf
    t = yf.Ticker(symbol)
    info = t.info or {}
    data = {
        "symbol": symbol,
        "info": info,
        "income": t.income_stmt,
        "balance": t.balance_sheet,
        "cashflow": t.cashflow,
        "history": t.history(period="5y")[["Close"]],
    }
    try:  # 10-year US Treasury yield (^TNX is quoted in percent, e.g. 4.3)
        tnx = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()
        data["rf"] = float(tnx.iloc[-1]) / 100 if not tnx.empty else None
    except Exception:
        data["rf"] = None
    return data


def clamp(x, lo, hi):
    return float(min(max(x, lo), hi))


def model_inputs(d: dict) -> tuple[dict, list]:
    """Translate raw company data into DCF inputs. Returns (values, notes on where each came from)."""
    info, inc, bal, cf = d["info"], d["income"], d["balance"], d["cashflow"]
    vals, notes = {}, []

    def note(field, value, source):
        notes.append({"Input": field, "Value": value, "Source": source})

    name = info.get("longName") or info.get("shortName") or d["symbol"]
    vals["company"] = f"{name} ({d['symbol']})"

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price and not d["history"].empty:
        price = float(d["history"]["Close"].iloc[-1])
    if price:
        vals["current_price"] = float(price)
        note("Current share price", money(price, 2), "Latest market price")

    rev = latest(inc, ROWS["revenue"])
    if rev and rev > 0:
        vals["base_revenue"] = max(rev / 1e6, 1.0)
        note("Last-year revenue", money(rev / 1e6, 0) + "M", "Income statement, latest fiscal year")

        revs = series(inc, ROWS["revenue"])
        revs = revs[revs > 0]
        if len(revs) >= 2:
            n = len(revs) - 1
            cagr = (revs.iloc[-1] / revs.iloc[0]) ** (1 / n) - 1
            vals["g_start"] = clamp(cagr * 100, -20, 50)
            vals["g_end"] = clamp(min(cagr * 100, 4.0), -20, 50)
            note("Year-1 revenue growth", pct(cagr), f"{n}-year historical revenue CAGR")
            note("Final-year revenue growth", pct(vals["g_end"] / 100),
                 "Lower of historical CAGR and 4% (assumed fade)")

        ebit = latest(inc, ROWS["ebit"])
        if ebit is not None:
            vals["margin"] = clamp(ebit / rev * 100, -50, 90)
            note("EBIT margin", pct(ebit / rev), "Operating income ÷ revenue")
        da = latest(cf, ROWS["da"])
        if da is None:
            da = latest(inc, ["Reconciled Depreciation"])
        if da is not None:
            vals["da"] = clamp(abs(da) / rev * 100, 0, 50)
            note("D&A % of revenue", pct(abs(da) / rev), "Cash flow statement")
        capex = latest(cf, ROWS["capex"])
        if capex is not None:
            vals["capex"] = clamp(abs(capex) / rev * 100, 0, 50)
            note("CapEx % of revenue", pct(abs(capex) / rev), "Cash flow statement")

    tr = latest(inc, ROWS["tax_rate"])
    if tr is not None and 0 <= tr <= 0.6:
        vals["tax"] = tr * 100
        note("Tax rate", pct(tr), "Effective tax rate (income statement)")

    beta = info.get("beta")
    if beta is not None and 0 < beta < 4:
        vals["beta"] = float(beta)
        note("Equity beta", f"{beta:.2f}", "Yahoo Finance (5-year monthly)")

    if d.get("rf"):
        vals["rf"] = clamp(d["rf"] * 100, 0, 15)
        note("Risk-free rate", pct(d["rf"], 2), "10-year US Treasury yield (^TNX)")

    debt = info.get("totalDebt") or latest(bal, ROWS["debt"]) or 0.0
    cash = info.get("totalCash") or latest(bal, ROWS["cash"]) or 0.0
    vals["net_debt"] = (debt - cash) / 1e6
    note("Net debt", money((debt - cash) / 1e6, 0) + "M",
         f"Total debt {money(debt / 1e6, 0)}M − cash {money(cash / 1e6, 0)}M")

    shares = info.get("sharesOutstanding") or latest(bal, ROWS["shares"])
    if shares:
        vals["shares"] = max(shares / 1e6, 0.01)
        note("Shares outstanding", f"{shares / 1e6:,.1f}M", "Yahoo Finance")

    interest = latest(inc, ["Interest Expense", "Interest Expense Non Operating"])
    if interest and debt and debt > 0:
        kd_est = abs(interest) / debt
        if 0.01 <= kd_est <= 0.20:
            vals["kd"] = kd_est * 100
            note("Pre-tax cost of debt", pct(kd_est), "Interest expense ÷ total debt")

    mcap = info.get("marketCap") or (price * shares if price and shares else None)
    if mcap and debt is not None:
        vals["wd"] = clamp(debt / (debt + mcap) * 100, 0, 95)
        note("Debt / (Debt + Equity)", pct(debt / (debt + mcap)), "Market values: total debt and market cap")

    return vals, notes


# Default inputs (the "Example Co." case). Widgets read their values from session state,
# which lets the "Load company" button overwrite them.
DEFAULTS = {
    "company": "Example Co.", "current_price": 25.0, "base_revenue": 1000.0, "years": 5,
    "g_start": 12.0, "g_end": 4.0, "margin": 20.0, "tax": 21.0, "da": 4.0, "capex": 5.0,
    "nwc": 10.0, "rf": 4.3, "beta": 1.1, "erp": 5.5, "kd": 6.0, "wd": 25.0, "wacc_direct": 9.0,
    "tg": 2.5, "exit_mult": 12.0, "net_debt": 200.0, "shares": 100.0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("company_data", None)


def load_company():
    """Button callback: look up the company and copy its numbers into the inputs."""
    query = st.session_state.get("query", "").strip()
    st.session_state["load_msg"] = None
    if not query:
        st.session_state["load_msg"] = ("warning", "Type a ticker or company name first.")
        return
    try:
        symbol = resolve_symbol(query)
        d = fetch_company(symbol)
        vals, notes = model_inputs(d)
        if "base_revenue" not in vals:
            st.session_state["load_msg"] = (
                "error", f"Couldn't find financial statements for '{query}'. Check the ticker and try again.")
            return
        for k, v in vals.items():
            st.session_state[k] = v
        st.session_state["company_data"] = d
        st.session_state["load_notes"] = notes
        st.session_state["load_msg"] = ("success", f"Loaded {vals['company']}.")
    except Exception as e:  # network problems, rate limits, unknown tickers
        st.session_state["load_msg"] = (
            "error", f"Couldn't load data for '{query}' ({type(e).__name__}). Yahoo Finance may be "
                     "busy, so wait a minute and try again, or enter the inputs by hand.")


def reset_inputs():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["company_data"] = None
    st.session_state["load_msg"] = None


# ---------------------------------------------------------------- sidebar inputs
st.sidebar.title("Assumptions")
with st.sidebar.container(border=True):
    st.text_input("Look up a company", key="query", placeholder="Ticker or name, e.g. AAPL or Microsoft")
    b1, b2 = st.columns(2)
    b1.button("Load company", on_click=load_company, type="primary", use_container_width=True)
    b2.button("Reset", on_click=reset_inputs, use_container_width=True)
    msg = st.session_state.get("load_msg")
    if msg:
        getattr(st, msg[0])(msg[1])
    st.caption("Data from Yahoo Finance. Check it before relying on it, and adjust any input below.")

company = st.sidebar.text_input("Company name", key="company")
current_price = st.sidebar.number_input("Current share price ($)", 0.0, step=0.50, key="current_price")

with st.sidebar.expander("Revenue & operations", expanded=True):
    base_revenue = st.number_input("Last-year revenue ($M)", 1.0, step=50.0, key="base_revenue")
    years = st.slider("Projection years", 3, 10, key="years")
    g_start = st.number_input("Year-1 revenue growth (%)", -50.0, 100.0, step=0.5, key="g_start") / 100
    g_end = st.number_input("Final-year revenue growth (%)", -50.0, 100.0, step=0.5, key="g_end") / 100
    margin = st.number_input("EBIT margin (%)", -50.0, 90.0, step=0.5, key="margin") / 100
    tax = st.number_input("Tax rate (%)", 0.0, 60.0, step=0.5, key="tax") / 100

with st.sidebar.expander("Reinvestment"):
    da = st.number_input("D&A (% of revenue)", 0.0, 50.0, step=0.5, key="da") / 100
    capex = st.number_input("CapEx (% of revenue)", 0.0, 50.0, step=0.5, key="capex") / 100
    nwc = st.number_input("Net working capital (% of revenue change)", -50.0, 100.0, step=1.0, key="nwc") / 100

with st.sidebar.expander("Discount rate (WACC)", expanded=True):
    use_capm = st.toggle("Build WACC with CAPM", value=True)
    if use_capm:
        rf = st.number_input("Risk-free rate (%)", 0.0, 15.0, step=0.1, key="rf") / 100
        beta = st.number_input("Equity beta", 0.0, 4.0, step=0.05, key="beta")
        erp = st.number_input("Equity risk premium (%)", 0.0, 15.0, step=0.1, key="erp") / 100
        kd = st.number_input("Pre-tax cost of debt (%)", 0.0, 20.0, step=0.1, key="kd") / 100
        wd = st.number_input("Debt / (Debt + Equity) (%)", 0.0, 95.0, step=1.0, key="wd") / 100
        ke, wacc = capm_wacc(rf, beta, erp, kd, tax, wd)
        st.caption(f"Cost of equity **{pct(ke, 2)}** · WACC **{pct(wacc, 2)}**")
    else:
        wacc = st.number_input("WACC (%)", 1.0, 30.0, step=0.1, key="wacc_direct") / 100

with st.sidebar.expander("Terminal value", expanded=True):
    method = st.radio("Method", ["Perpetuity growth", "Exit multiple"], horizontal=True)
    tg = st.number_input("Terminal growth (%)", -5.0, 10.0, step=0.1, key="tg") / 100
    exit_mult = st.number_input("Exit EV/EBITDA multiple (x)", 0.0, 50.0, step=0.5, key="exit_mult")
    mid_year = st.checkbox("Mid-year discounting convention", value=True)

with st.sidebar.expander("Equity bridge"):
    net_debt = st.number_input("Net debt ($M, debt minus cash)", step=25.0, key="net_debt")
    shares = st.number_input("Diluted shares outstanding (M)", 0.01, step=5.0, key="shares")

a = Assumptions(
    base_revenue=base_revenue, growth_start=g_start, growth_end=g_end, years=years,
    ebit_margin=margin, tax_rate=tax, da_pct=da, capex_pct=capex, nwc_pct=nwc,
    wacc=wacc, terminal_method=method, terminal_growth=tg, exit_multiple=exit_mult,
    net_debt=net_debt, shares=shares, mid_year=mid_year,
)

# ---------------------------------------------------------------- header
st.title("📈 DCF Valuation Studio")
st.write(
    f"Estimate the intrinsic value per share of **{company}** using an unlevered "
    "free-cash-flow DCF, then stress-test it with sensitivity tables and a Monte Carlo "
    "simulation. Look up a real company or type your own assumptions in the sidebar. "
    "Everything updates instantly."
)

try:
    r = value(a)
except ValueError as e:
    st.error(f"⚠️ {e} Lower terminal growth or raise WACC in the sidebar.")
    st.stop()

price = r["price"]
upside = price / current_price - 1 if current_price > 0 else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Implied share price", money(price, 2),
          None if np.isnan(upside) else f"{upside * 100:+.1f}% vs. market")
c2.metric("Enterprise value", money(r["enterprise_value"], 0) + "M")
c3.metric("Equity value", money(r["equity_value"], 0) + "M")
c4.metric("WACC", pct(wacc, 2))

# Only give a verdict when both the market price and the implied value are positive
if not np.isnan(upside) and price > 0:
    if upside > 0.15:
        st.success(f"The model suggests {company} is **undervalued** by {upside:.0%} at these assumptions.")
    elif upside < -0.15:
        st.warning(f"The model suggests {company} is **overvalued** by {-upside:.0%} at these assumptions.")
    else:
        st.info(f"The model suggests {company} is **fairly valued** (within ±15% of the market price).")

# --- Edge-case warnings: flag results that are mathematically valid but suspicious
if r["projection"]["Unlevered FCF"].iloc[-1] <= 0 and method == "Perpetuity growth":
    st.warning("⚠️ Final-year free cash flow is negative, so the perpetuity terminal value is negative. "
               "Try the exit-multiple method or revisit margins/CapEx.")
if r["equity_value"] < 0:
    st.error("⚠️ Net debt is larger than enterprise value, so equity value is negative. "
             "At these assumptions the shares would be worth nothing.")
if g_end > 0 and method == "Perpetuity growth" and tg > g_end:
    st.caption("Note: terminal growth is above the final projection-year growth. Usually it should fade down, not up.")
if tg > 0.04 and method == "Perpetuity growth":
    st.caption("Note: terminal growth above ~4% implies the firm eventually outgrows the whole economy.")

if r["tv_share_of_ev"] > 0.8:
    st.caption(f"⚠️ Terminal value is {pct(r['tv_share_of_ev'], 0)} of EV. The valuation leans heavily on the long-run assumptions.")

tab1, tab_co, tab2, tab3, tab4 = st.tabs(
    ["Valuation", "Company data", "Sensitivity", "Monte Carlo", "Methodology"])

# ---------------------------------------------------------------- tab 1: valuation
with tab1:
    proj = r["projection"]
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Free cash flow projection ($M)")
        rows = ["Revenue growth", "Revenue", "EBIT", "NOPAT", "D&A", "CapEx",
                "Change in NWC", "Unlevered FCF", "Discount factor", "PV of FCF"]
        table = proj.set_index("Year")[rows].T
        table.columns = [f"Year {y}" for y in table.columns]
        shown = table.copy().astype(object)
        for row in rows:
            if row == "Revenue growth":
                shown.loc[row] = [pct(v) for v in table.loc[row]]
            elif row == "Discount factor":
                shown.loc[row] = [f"{v:.3f}" for v in table.loc[row]]
            else:
                shown.loc[row] = [f"{v:,.1f}" for v in table.loc[row]]
        st.dataframe(shown)

        fig = go.Figure()
        fig.add_bar(x=proj["Year"], y=proj["Unlevered FCF"], name="Unlevered FCF")
        fig.add_bar(x=proj["Year"], y=proj["PV of FCF"], name="PV of FCF")
        fig.add_scatter(x=proj["Year"], y=proj["Revenue"], name="Revenue",
                        yaxis="y2", mode="lines+markers")
        fig.update_layout(
            title="Cash flows and revenue", barmode="group", height=380,
            xaxis_title="Year", yaxis_title="$M (FCF)",
            yaxis2=dict(title="$M (Revenue)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig)

    with right:
        st.subheader("Value bridge ($M)")
        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total", "relative", "total"],
            x=["PV of FCFs", "PV of terminal value", "Enterprise value", "Less net debt", "Equity value"],
            y=[r["pv_fcf"], r["pv_terminal"], 0, -net_debt, 0],
            text=[f"{v:,.0f}" for v in [r["pv_fcf"], r["pv_terminal"], r["enterprise_value"],
                                       -net_debt, r["equity_value"]]],
            textposition="outside",
        ))
        wf.update_layout(height=380, margin=dict(t=20, b=10), showlegend=False)
        st.plotly_chart(wf)

        st.subheader("Sanity checks")
        st.markdown(
            f"- Terminal value: **{money(r['terminal_value'], 0)}M** "
            f"({pct(r['tv_share_of_ev'], 0)} of EV)\n"
            f"- Implied exit EV/EBITDA: **{r['implied_exit_multiple']:.1f}x**\n"
            f"- Implied perpetual growth: **{pct(r['implied_perpetual_growth'], 2)}**\n"
            f"- Equity value ÷ {shares:,.1f}M shares = **{money(price, 2)}** per share"
        )

    st.download_button(
        "⬇️ Download projection (CSV)",
        proj.to_csv(index=False).encode(),
        file_name=f"{company.replace(' ', '_')}_dcf_projection.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------- tab: company data
with tab_co:
    d = st.session_state.get("company_data")
    if d is None:
        st.info("Type a ticker (e.g. **AAPL**) or a company name (e.g. **Microsoft**) in the sidebar "
                "and click **Load company**. This tab will show the company's profile, key statistics, "
                "price history and financial statements, and the DCF inputs will fill in automatically.")
    else:
        info = d["info"]
        st.subheader(info.get("longName") or d["symbol"])
        facts = [info.get("sector"), info.get("industry"), info.get("country")]
        st.caption(" · ".join(str(f) for f in facts if f) + (f" · {info['website']}" if info.get("website") else ""))

        if info.get("sector") == "Financial Services":
            st.warning("⚠️ Banks and insurers don't fit a free-cash-flow DCF well, because debt is part of "
                       "their operations. Treat this valuation with extra caution.")
        fin_cur, px_cur = info.get("financialCurrency"), info.get("currency")
        if fin_cur and px_cur and fin_cur != px_cur:
            st.warning(f"⚠️ Financial statements are reported in **{fin_cur}** but the share price is in "
                       f"**{px_cur}**. Convert revenue and net debt to {px_cur} before trusting the value per share.")

        if info.get("longBusinessSummary"):
            with st.expander("Business description"):
                st.write(info["longBusinessSummary"])

        def fmt_big(x):
            if not x:
                return "n/a"
            for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
                if abs(x) >= div:
                    return f"${x / div:,.2f}{suf}"
            return f"${x:,.0f}"

        def fmt_x(x):
            return f"{x:.1f}x" if isinstance(x, (int, float)) and x > 0 else "n/a"

        px = info.get("currentPrice") or info.get("regularMarketPrice")
        div_rate = info.get("dividendRate")
        k = st.columns(4)
        k[0].metric("Share price", money(px, 2) if px else "n/a")
        k[1].metric("Market cap", fmt_big(info.get("marketCap")))
        k[2].metric("Enterprise value", fmt_big(info.get("enterpriseValue")))
        k[3].metric("Beta", f"{info['beta']:.2f}" if info.get("beta") else "n/a")
        k = st.columns(4)
        k[0].metric("Trailing P/E", fmt_x(info.get("trailingPE")))
        k[1].metric("Forward P/E", fmt_x(info.get("forwardPE")))
        k[2].metric("EV / EBITDA", fmt_x(info.get("enterpriseToEbitda")))
        k[3].metric("Dividend yield", pct(div_rate / px, 2) if div_rate and px else "n/a")
        lo52, hi52 = info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh")
        if lo52 and hi52:
            st.caption(f"52-week range: {money(lo52, 2)} – {money(hi52, 2)}")

        hist = d["history"]
        if not hist.empty:
            period = st.radio("Price history", ["1Y", "3Y", "5Y"], index=2, horizontal=True)
            days = {"1Y": 365, "3Y": 3 * 365, "5Y": 5 * 365}[period]
            h = hist[hist.index >= hist.index.max() - pd.Timedelta(days=days)]
            pfig = go.Figure(go.Scatter(x=h.index, y=h["Close"], mode="lines", name="Close"))
            pfig.update_layout(height=320, yaxis_title=f"Price ({px_cur or 'USD'})",
                               margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(pfig)

        st.subheader("Financial statements")
        st.caption(f"Annual, in millions of {fin_cur or 'USD'}. Source: Yahoo Finance.")
        stmt_name = st.radio("Statement", ["Income statement", "Balance sheet", "Cash flow"],
                             horizontal=True, label_visibility="collapsed")
        stmt = {"Income statement": d["income"], "Balance sheet": d["balance"],
                "Cash flow": d["cashflow"]}[stmt_name]
        key_rows = {
            "Income statement": ["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Expense",
                                 "Operating Income", "EBITDA", "Interest Expense", "Pretax Income",
                                 "Tax Provision", "Net Income", "Diluted EPS"],
            "Balance sheet": ["Cash Cash Equivalents And Short Term Investments", "Current Assets",
                              "Total Assets", "Current Liabilities", "Total Debt",
                              "Total Liabilities Net Minority Interest", "Stockholders Equity",
                              "Ordinary Shares Number"],
            "Cash flow": ["Operating Cash Flow", "Depreciation And Amortization", "Capital Expenditure",
                          "Free Cash Flow", "Repurchase Of Capital Stock", "Cash Dividends Paid"],
        }[stmt_name]
        if stmt is None or stmt.empty:
            st.write("Not available for this company.")
        else:
            show_all = st.checkbox("Show all line items")
            rows = list(stmt.index) if show_all else [r for r in key_rows if r in stmt.index]
            tbl = stmt.loc[rows].apply(pd.to_numeric, errors="coerce")
            tbl = tbl[sorted(tbl.columns)]  # oldest -> newest, left to right
            tbl = tbl.dropna(axis=1, how="all")
            tbl.columns = [c.strftime("FY %Y") if hasattr(c, "strftime") else str(c) for c in tbl.columns]
            per_share = {"Diluted EPS", "Basic EPS", "Tax Rate For Calcs"}
            shown = tbl.astype(object)
            for r_ in tbl.index:
                vals_ = tbl.loc[r_]
                if r_ in per_share:
                    shown.loc[r_] = ["" if pd.isna(v) else f"{v:,.2f}" for v in vals_]
                else:
                    shown.loc[r_] = ["" if pd.isna(v) else f"{v / 1e6:,.0f}" for v in vals_]
            st.dataframe(shown)

        notes = st.session_state.get("load_notes") or []
        if notes:
            st.subheader("What was loaded into the model")
            st.dataframe(pd.DataFrame(notes), hide_index=True)
            st.caption("Projection years, revenue fade, NWC, equity risk premium and terminal "
                       "assumptions are not pulled from data (cost of debt is filled only when interest expense is reported). They are your judgment calls, so review them in the sidebar.")

# ---------------------------------------------------------------- tab 2: sensitivity
with tab2:
    st.subheader("Implied share price sensitivity")
    step = st.select_slider("WACC step", options=[0.25, 0.5, 1.0], value=0.5,
                            format_func=lambda v: f"{v}%")
    # Keep full precision so the center cell equals the base case exactly
    waccs = [wacc + k * step / 100 for k in range(-2, 3)]
    if method == "Perpetuity growth":
        seconds = [tg + k * 0.0025 for k in range(-2, 3)]
        col_labels = [pct(s, 2) for s in seconds]
        col_title = "Terminal growth"
    else:
        seconds = [exit_mult + k * 1.0 for k in range(-2, 3)]
        col_labels = [f"{s:.1f}x" for s in seconds]
        col_title = "Exit multiple"

    grid = sensitivity(a, waccs, seconds)
    hm = go.Figure(go.Heatmap(
        z=grid.values, x=col_labels, y=[pct(w, 2) for w in waccs],
        colorscale="RdYlGn", zmid=current_price if current_price > 0 else None,
        text=[[("n/a" if np.isnan(v) else f"${v:,.2f}") for v in row] for row in grid.values],
        texttemplate="%{text}", hovertemplate="WACC %{y}<br>" + col_title + " %{x}<br>%{text}<extra></extra>",
        colorbar=dict(title="$/share"),
    ))
    # type="category" stops Plotly from converting labels like "8.95%" into plain numbers
    hm.update_layout(xaxis_title=col_title, yaxis_title="WACC", height=430,
                     xaxis=dict(type="category"),
                     yaxis=dict(type="category", autorange="reversed"), margin=dict(t=20))
    st.plotly_chart(hm)
    st.caption("Green cells are above the current market price and red cells are below it. "
               "The center cell is your base case.")

    st.subheader("Which assumption matters most?")
    shocks = {
        "Year-1 growth ±3 pts": ("growth_start", 0.03),
        "EBIT margin ±3 pts": ("ebit_margin", 0.03),
        "WACC ±1 pt": ("wacc", 0.01),
        "Terminal growth ±0.5 pt": ("terminal_growth", 0.005),
        "Exit multiple ±2x": ("exit_multiple", 2.0),
        "CapEx % ±1 pt": ("capex_pct", 0.01),
    }
    if method == "Perpetuity growth":
        shocks.pop("Exit multiple ±2x")
    else:
        shocks.pop("Terminal growth ±0.5 pt")
    rows = []
    for label, (field, d) in shocks.items():
        try:
            lo = value(replace(a, **{field: getattr(a, field) - d}))["price"]
            hi = value(replace(a, **{field: getattr(a, field) + d}))["price"]
            rows.append((label, lo, hi))
        except ValueError:
            pass
    rows.sort(key=lambda t: abs(t[2] - t[1]))
    tor = go.Figure()
    tor.add_bar(y=[t[0] for t in rows], x=[t[1] - price for t in rows], base=price,
                orientation="h", name="Input lowered")
    tor.add_bar(y=[t[0] for t in rows], x=[t[2] - price for t in rows], base=price,
                orientation="h", name="Input raised")
    tor.add_vline(x=price, line_dash="dash")
    tor.update_layout(barmode="overlay", height=360, xaxis_title="Implied share price ($)",
                      legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
    st.plotly_chart(tor)

# ---------------------------------------------------------------- tab 3: monte carlo
with tab3:
    st.subheader("Monte Carlo simulation")
    st.write("Randomly shocks the key inputs thousands of times to show a *range* of plausible "
             "values instead of a single point estimate.")
    m1, m2, m3, m4 = st.columns(4)
    sd_g = m1.number_input("Growth std. dev. (pts)", 0.0, 20.0, 3.0, 0.5) / 100
    sd_m = m2.number_input("Margin std. dev. (pts)", 0.0, 20.0, 3.0, 0.5) / 100
    sd_w = m3.number_input("WACC std. dev. (pts)", 0.0, 5.0, 1.0, 0.25) / 100
    n = m4.select_slider("Simulations", options=[1_000, 5_000, 10_000, 25_000], value=10_000)

    sims = monte_carlo(a, n=n, sd_growth=sd_g, sd_margin=sd_m, sd_wacc=sd_w)
    sims = sims[np.isfinite(sims)]
    p5, p50, p95 = np.percentile(sims, [5, 50, 95])
    prob_up = (sims > current_price).mean() if current_price > 0 else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Median value", money(p50, 2))
    k2.metric("5th percentile", money(p5, 2))
    k3.metric("95th percentile", money(p95, 2))
    k4.metric("Chance undervalued", "n/a" if np.isnan(prob_up) else pct(prob_up, 0))

    lo_clip, hi_clip = np.percentile(sims, [0.5, 99.5])
    hist = go.Figure(go.Histogram(x=sims[(sims >= lo_clip) & (sims <= hi_clip)], nbinsx=60))
    if current_price > 0:
        hist.add_vline(x=current_price, line_dash="dash", line_color="red",
                       annotation_text="Market price", annotation_position="top left")
    hist.add_vline(x=price, line_dash="dot", annotation_text="Base case",
                   annotation_position="top right")
    hist.update_layout(xaxis_title="Implied share price ($)", yaxis_title="Simulations",
                       height=400, margin=dict(t=30), showlegend=False)
    st.plotly_chart(hist)

# ---------------------------------------------------------------- tab 4: methodology
with tab4:
    st.subheader("How the model works")
    st.markdown(r"""
**1. Project unlevered free cash flow (FCF)** for each year *t*:

$$FCF_t = EBIT_t(1-T) + D\&A_t - CapEx_t - \Delta NWC_t$$

Revenue growth fades linearly from the Year-1 rate to the final-year rate.

**2. Discount rate (WACC).** Cost of equity comes from CAPM:

$$k_e = r_f + \beta \times ERP \qquad WACC = \tfrac{E}{V}k_e + \tfrac{D}{V}k_d(1-T)$$

**3. Terminal value** at the end of the projection, using either
- Gordon growth: $TV = \dfrac{FCF_N(1+g)}{WACC - g}$, or
- Exit multiple: $TV = EBITDA_N \times \text{multiple}$

**4. Enterprise value** $= \sum \dfrac{FCF_t}{(1+WACC)^t} + \dfrac{TV}{(1+WACC)^N}$
(with the mid-year convention, cash flows are discounted at $t-0.5$).

**5. Equity value** $=$ EV $-$ net debt, and **value per share** $=$ equity ÷ diluted shares.

**Limitations:** A DCF can only be as good as its inputs. Small changes in WACC and
terminal assumptions move the answer a lot, which is why the Sensitivity and Monte Carlo
tabs are included. This app is an educational tool, not investment advice.
""")
    # --- Built-in self-check: run the model on inputs simple enough to verify by hand
    st.subheader("✅ Built-in self-check (known answers)")
    st.write("These tests run the same functions the app uses on inputs simple enough to check "
             "with a calculator. They should all pass.")
    flat = Assumptions(base_revenue=1000, growth_start=0, growth_end=0, years=5,
                       ebit_margin=0.10, tax_rate=0.0, da_pct=0, capex_pct=0, nwc_pct=0,
                       wacc=0.10, terminal_growth=0.0, net_debt=0, shares=10, mid_year=False)
    # Flat FCF of $100 forever at 10% = perpetuity worth 100 / 0.10 = $1,000 -> $100/share
    _, test_wacc = capm_wacc(0.04, 1.0, 0.06, 0.05, 0.20, 0.5)
    checks = [
        ("Zero-growth perpetuity: FCF 100, WACC 10%, g 0% → EV = 100 / 0.10 = 1,000",
         1000.0, value(flat)["enterprise_value"]),
        ("…and value per share with 10 shares, no debt = $100", 100.0, value(flat)["price"]),
        ("Gordon TV: FCF 100, g 2%, WACC 10% → 100 × 1.02 / 0.08 = 1,275",
         1275.0, value(replace(flat, terminal_growth=0.02))["terminal_value"]),
        ("CAPM: 4% + 1.0 × 6% = 10% cost of equity; 50/50 with 5% debt, 20% tax → WACC 7%",
         0.07, test_wacc),
        ("Exit multiple: EBITDA 100 × 8x = TV 800", 800.0,
         value(replace(flat, terminal_method="Exit multiple", exit_multiple=8))["terminal_value"]),
    ]
    for label, expected, actual in checks:
        ok = abs(expected - actual) < 1e-6
        st.write(f"{'✅' if ok else '❌'} {label}  (app result: {actual:,.4f})")

    st.caption("Built for FINAN 623 using AI-assisted coding (Claude).")
