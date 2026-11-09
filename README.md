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

## Features
| Tab | What it does |
|---|---|
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
