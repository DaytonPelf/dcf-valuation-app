# Prompt, Verification & Testing Documentation

**Project:** DCF Valuation Studio  **Course:** FINAN 623  **Tester:** Dayton Pelfrey  **Date:** November 2026

---

## 1. AI prompt used (CRAFT framework)

> **Context:** I'm a senior finance student in FINAN 623 (Financial Modeling). Analysts often build DCF
> models in Excel, but a single-point DCF hides how sensitive the valuation is to assumptions like
> WACC and terminal growth.
>
> **Role:** Act as a financial-modeling instructor and an experienced Python/Streamlit developer.
>
> **Action:** Build a Python Streamlit web app that values a company with an unlevered free cash
> flow DCF. The app should:
> 1. Accept inputs in a sidebar: last-year revenue, starting and ending revenue growth (linear fade),
>    EBIT margin, tax rate, D&A %, CapEx %, NWC % of revenue change, net debt, diluted shares, and
>    the current share price
> 2. Let the user either enter WACC directly or build it with CAPM (risk-free rate, beta, equity risk
>    premium, cost of debt, debt weight)
> 3. Compute terminal value by perpetuity growth **or** exit EV/EBITDA multiple, with an optional
>    mid-year discounting convention
> 4. Show implied share price, EV, equity value and upside vs. market price, plus a projection table
>    and a value-bridge waterfall chart
> 5. Include a WACC × terminal-growth sensitivity heat map, a tornado chart, and a Monte Carlo
>    simulation with percentile outputs
> 6. Validate inputs (e.g., WACC must exceed terminal growth) and warn about suspicious results
> 7. Include a methodology tab with formulas and a built-in self-check using hand-verifiable inputs
>
> **Format:** One complete `app.py` file plus `requirements.txt`, with the finance math separated
> from the UI so it can be tested. Add comments explaining each section and a step-by-step explanation.
>
> **Tone:** Professional, clear, beginner-friendly explanations. My skill level: beginner–intermediate in Python.

**Follow-up prompts used:**
- "Please review the code you just provided and check for logical errors, missing error handling,
  potential bugs, and ways to improve it. Be critical."
- "Combine everything into a single app.py so I only upload app.py and requirements.txt to GitHub."

---

## 2. Manual verification of the logic

Base-case inputs: Revenue $1,000M, growth 12% → 4%, EBIT margin 20%, tax 21%, D&A 4%, CapEx 5%,
NWC 10% of Δrevenue, r_f 4.3%, β 1.1, ERP 5.5%, k_d 6%, D/V 25%, g 2.5%, net debt $200M, 100M shares.

| Item | Hand calculation | App result | Status |
|---|---|---|---|
| Cost of equity | 4.3% + 1.1 × 5.5% = 10.35% | 10.35% | ✓ PASS |
| WACC | 0.75 × 10.35% + 0.25 × 6% × (1 − 0.21) = 8.9475% | 8.95% | ✓ PASS |
| Year-1 revenue | 1,000 × 1.12 = 1,120.0 | 1,120.0 | ✓ PASS |
| Year-1 EBIT | 1,120 × 20% = 224.0 | 224.0 | ✓ PASS |
| Year-1 NOPAT | 224 × (1 − 0.21) = 176.96 | 176.96 | ✓ PASS |
| Year-1 ΔNWC | (1,120 − 1,000) × 10% = 12.0 | 12.0 | ✓ PASS |
| Year-1 FCF | 176.96 + 44.8 − 56.0 − 12.0 = 153.76 | 153.76 | ✓ PASS |
| Sum of PV(FCF), end-of-year discounting | Σ FCFₜ / 1.089475ᵗ = 710.22 (calculated separately) | 710.22 | ✓ PASS |
| Terminal value (Gordon) | 211.45 × 1.025 / (0.089475 − 0.025) = 3,361.5 | 3,361.5 | ✓ PASS |
| Value per share (mid-year) | (2,931.3 − 200) / 100 = $27.31 | $27.31 | ✓ PASS |
| Method cross-check | Plugging the exit method's implied growth (3.75%) into the Gordon model gives the same price as the exit method ($32.94) | $32.94 = $32.94 | ✓ PASS |

**Built-in self-check (Methodology tab).** These run every time the app loads:

| Test | Expected | Actual | Status |
|---|---|---|---|
| Zero-growth perpetuity: FCF 100, WACC 10%, g 0% → EV | 1,000 | 1,000.00 | ✓ PASS |
| Same, 10 shares, no debt → price | $100 | $100.00 | ✓ PASS |
| Gordon TV: 100 × 1.02 / 0.08 | 1,275 | 1,275.00 | ✓ PASS |
| CAPM/WACC: 4% + 1×6% = 10%; 50/50, k_d 5%, tax 20% | 7.00% | 7.00% | ✓ PASS |
| Exit TV: EBITDA 100 × 8x | 800 | 800.00 | ✓ PASS |

---

## 3. Feature testing

| Feature | Test | Expected | Actual | Status |
|---|---|---|---|---|
| Sidebar inputs | Change each input | All outputs update | Updated | ✓ PASS |
| CAPM toggle | Turn off, enter WACC 9% | Uses 9% directly | Used 9% | ✓ PASS |
| Exit-multiple method | Select "Exit multiple", 12x | Price changes, implied g shown | $32.94, g 3.75% | ✓ PASS |
| Valuation verdict | Market price $10 / $25 / $60 | Undervalued / fair / overvalued | Correct for all three | ✓ PASS |
| Sensitivity heat map | Center cell | Equals base-case price | Matches | ✓ PASS |
| Tornado chart | Bars sorted | Largest impact on top | Sorted | ✓ PASS |
| Monte Carlo | 10,000 sims | Median near base case | Median $27.37 vs base $27.31; 5th–95th pct $17.45–$42.65 (same on deployed app) | ✓ PASS |
| CSV download | Click button | CSV of projection table | Downloaded | ✓ PASS |
| 10-year projection | Slider to 10 | Table shows 10 years | 10 years | ✓ PASS |

### Live company data feature

| Feature | Test | Expected | Actual | Status |
|---|---|---|---|---|
| Lookup by company name | "apple" | Resolves to AAPL and fills inputs | Filled with Apple Inc. (AAPL) | ✓ PASS |
| Loaded values traceable | Check "What was loaded into the model" table | Each input lists its source | Shown | ✓ PASS |
| Unknown ticker | "ZZZZZ" | Error message, inputs unchanged | "Couldn't find financial statements" | ✓ PASS |
| Data source down | Simulated connection error | Friendly error, manual entry still works | Message shown, app still runs | ✓ PASS |
| Reset button | Click Reset | Back to Example Co. defaults | Restored | ✓ PASS |

Manual check of loaded inputs (example): EBIT margin = operating income ÷ revenue; CapEx % =
|capital expenditure| ÷ revenue; net debt = total debt − cash. Compare these to the Company data
tab's statements with a calculator.

## 4. Edge-case testing

| Test case | Input | Expected | Actual | Status |
|---|---|---|---|---|
| WACC below terminal growth | WACC 2%, g 2.5% | Clear error, no crash | Error message, app stops cleanly | ✓ PASS |
| WACC equal to g | WACC 2.5%, g 2.5% | Error (division by zero) | Error message | ✓ PASS |
| Net debt larger than EV | Net debt $5,000M | Negative-equity warning | Warning shown | ✓ PASS (after fix) |
| Negative EBIT margin | −10% | Negative-FCF warning | Warning shown | ✓ PASS |
| Net cash | Net debt −$300M | Adds cash to equity | Equity > EV | ✓ PASS |
| Market price = 0 | $0 | No upside % or verdict (avoid ÷0) | None shown | ✓ PASS |
| Aggressive terminal growth | g 5% | Warning about outgrowing economy | Note shown | ✓ PASS |
| Exit multiple 0x | 0x | Runs, TV = 0 | Runs | ✓ PASS |
| Text or blank input | Type "hello" | Rejected | Streamlit number fields reject non-numeric input | ✓ PASS |
| Out-of-range input | Tax rate 150% | Blocked | Capped at 60% by input limits | ✓ PASS |

## 5. Bugs found and fixed

1. **Misleading verdict when equity is negative.** With net debt of $5,000M the implied price was
   negative, and the app said the stock was "overvalued by 183%", which makes no sense.
   - How to reproduce: set net debt above enterprise value.
   - Severity: Medium
   - Fix: only show the under/overvalued verdict when the implied price is positive. A
     negative-equity error now appears instead.
2. **Deprecated chart argument.** `use_container_width=True` is deprecated in newer Streamlit versions.
   - Severity: Low. Fix: removed the argument (charts fill the container by default).
3. **Sensitivity heat map axis labels wrong on the deployed app.** Plotly converted labels like
   "8.95%" into plain numbers, so the WACC axis read 8, 8.5, 9… instead of 7.95%, 8.45%, 8.95%…
   The center cell also showed $27.30 instead of $27.31 because the WACC was rounded.
   - Found during post-deployment testing. Severity: Medium (misleading chart).
   - Fix: forced both axes to categories and removed the rounding. The center cell now equals the base case.
4. **Overlapping chart labels.** The "Market price" and "Base case" labels on the Monte Carlo
   histogram sat on top of each other. Severity: Low. Fix: placed them on opposite sides of their lines.

## 6. Verification checklist

- [x] I verified all requirements from my prompt are met
- [x] I manually checked all calculations/logic (Section 2)
- [x] I tested normal use cases (Section 3)
- [x] I tested edge cases (Section 4)
- [x] I verified error handling works
- [x] I asked AI to review its own code
- [x] I have read and understood every line of code
- [x] App loads on Streamlit Cloud and all features work on the deployed version
- [x] Tested from a different device/browser

**Overall status:** Ready to deploy
