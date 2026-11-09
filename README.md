# 📈 DCF Valuation Studio

**Live app:** https://dcf-valuation-app-9yddu6ndlqf9avvzwq4mny.streamlit.app/

A Streamlit web app that estimates a company's **intrinsic value per share** with a
discounted cash flow (DCF) model and then stress-tests that estimate. Built for
FINAN 623 (Financial Modeling) using AI-assisted coding ("vibe coding").

## The finance problem it solves
A DCF in a spreadsheet produces one number, and it is hard to see how fragile that
number is. This app makes the model interactive. Every assumption is a slider or input,
results update instantly, and three tools show how much the answer depends on those
assumptions: a sensitivity table, a tornado chart and a Monte Carlo simulation.

## Live company data
Type a ticker (e.g. `AAPL`) or a company name (e.g. `Microsoft`) in the sidebar and click
**Load company**. The app pulls data from Yahoo Finance (via the `yfinance` library) and:
- fills in the DCF inputs: price, revenue, historical growth, EBIT margin, tax rate, D&A, CapEx,
  beta, net debt, shares, debt weight, cost of debt, and the 10-year Treasury yield as the risk-free rate
- shows a **Company data** tab with the profile, key stats (market cap, P/E, EV/EBITDA, dividend yield),
  a 1–5 year price chart, the income statement, balance sheet and cash flow statement, and a table
  showing exactly which inputs were loaded and where each came from

Forecast judgment calls (growth fade, NWC, equity risk premium, terminal assumptions) stay manual.
If Yahoo Finance is unavailable, the app shows a message and you can still enter inputs by hand.

## Features
| Tab | What it does |
|---|---|
| **Company data** | Profile, key stats, price chart, financial statements, and the source of every loaded input |
| **Valuation** | 5–10 year unlevered FCF projection, value bridge waterfall (PV of FCF + PV of TV − net debt = equity), implied price vs. market price, CSV download |
| **Sensitivity** | WACC × terminal growth (or exit multiple) heat map; tornado chart ranking which input moves value most |
| **Monte Carlo** | 1,000–25,000 simulations shocking growth, margin, WACC and terminal growth; shows 5th/50th/95th percentile values and the probability the stock is undervalued |
| **Methodology** | Formulas used, limitations, and a built-in self-check that runs the model on hand-verifiable inputs |

Other details:
- WACC can be entered directly or built with **CAPM**: kₑ = r_f + β × ERP
- Terminal value by **Gordon growth** or **exit EV/EBITDA multiple**, with cross-checks (implied multiple / implied growth)
- **Mid-year discounting** option
- Input validation and warnings (WACC ≤ g, negative equity, negative terminal FCF, aggressive terminal growth, TV > 80% of EV)

## Key formulas
- FCF = EBIT × (1 − tax) + D&A − CapEx − ΔNWC
- WACC = E/V × kₑ + D/V × k_d × (1 − tax)
- Gordon TV = FCF_N × (1 + g) / (WACC − g); Exit TV = EBITDA_N × multiple
- EV = Σ FCF_t / (1 + WACC)^t + TV / (1 + WACC)^N
- Value per share = (EV − net debt) / diluted shares

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py`: the app. Part 1 holds the finance logic and Part 2 the Streamlit interface.
- `requirements.txt`: Python packages Streamlit Cloud installs.
- `TESTING.md`: the AI prompt used, verification checklist and documented test results.

## Disclaimer
Educational tool only, not investment advice. Outputs are only as good as the inputs.
