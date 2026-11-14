# VolGAN-X v3.2 — NIFTY Options Trade Decision Matrix (RESEARCH ONLY)

## Why the old `ABORT` was always printed
The v3.1 training command printed one **global** policy status: experimental model checkpoints, no validated trading thresholds, and live execution permanently disabled. It did not explain whether a **particular** call/put pair had a positive or negative estimated historical pricing discrepancy. This version separates those topics.

## New output: one row per eligible CE/PE pair
The matrix screens **NIFTY index call/put pairs at the same strike and expiry**. These are *not individual stocks* and *not actual executable trade orders*. The proxy uses historical daily closing marks rather than bid/ask quotes. Every row includes:

- Entry date, expiry, strike, spot, call and put closes, lot size, days to expiry, and daily rank
- **Residual points** = signed leave-one-strike-out put/call parity-intercept discrepancy (not a VG implied-volatility residual)
- **Gross discrepancy (₹)** = absolute residual points × lot size
- **Modeled cost (₹)** = conservative paired-option round-trip friction estimate (not observed spread)
- **Estimated proxy net edge (₹)** = gross discrepancy − modeled cost
- `research_selection`, `decision_reason`, `live_decision`, `live_reason`, and after-the-fact exit marker

The original 5 VolGAN thresholds (0.70, 2.50, ₹250, 3, ₹25,000) are **unvalidated defaults** and are **not used** to approve trades. The only ranking/screening comparison is **modeled proxy net edge > ₹0**, a descriptive filter, not a trading rule.

## Decision vocabulary

| Value | Meaning |
|---|---|
| `SELECTED_FOR_REVIEW` | Best estimated positive after-cost parity discrepancy on the date; historical research review **only**. |
| `REJECTED_BY_SCREEN` | Modeled gross discrepancy is insufficient to cover modeled cost. |
| `NOT_SELECTED_RANK` | Positive proxy edge, but a better-ranked candidate existed on the same date. |
| `BLOCKED_LIVE` | Every row remains non-executable, regardless of research selection. |
| `NOT_VALIDATED` / `NOT_AVAILABLE` | Per-contract ML/capital-risk threshold evidence is missing; no number is guessed. |

## Gate-by-gate matrix

- **G0:** Historical pair/price/liquidity/moneyness/expiry input filters (passed to reach this table)
- **G1:** Estimated parity proxy versus modeled costs (positive/negative; *descriptive*)
- **G2:** CNN-LSTM persistence (`THETA_Y`): contract-specific value/threshold not validated
- **G3:** VAE anomaly (`TAU_HEALTHY`): contract-specific value/threshold not validated
- **G4:** Original VolGAN minimum edge: not evaluated using the unrelated parity proxy
- **G5:** Validated contract-specific GAN 95% CVaR: unavailable
- **G6:** Risk mandate (`KAPPA` and budget): not validated
- **G7:** Live trading: permanently `BLOCKED_LIVE`

## Files produced in `research_outputs/`

- `trade_decision_matrix.xlsx` — multi-sheet, Excel-friendly audit workbook (included in this ZIP)
- `trade_decision_matrix.html` — open in a browser to read ranked rows and a gate example
- `trade_daily_summary.csv` — one top-ranked pair per trading date (13 rows)
- `trade_decision_matrix.csv` — top 5 pairs per date by default (65 rows)
- `trade_decision_matrix_all.csv` — all eligible historical CE/PE pairs (867 rows on bundled inputs)
- `trade_gate_breakdown.csv` — all eight gate reasons for each displayed top-N candidate
- `trade_decision_summary.json` — counts, status definitions, default-value notes, and audit conventions
- Original v3.1 loss curves/research metrics/checkpoints remain included.

## Windows / VS Code
Extract the ZIP *completely*, with the `data` directory beside the Python scripts. Run:

```powershell
python -m pip install -r requirements_volgan.txt
python volgan_combined.py --mode matrix --output-dir research_outputs --top-n 5 --explain-date 2024-01-08
python volgan_combined.py --mode train --output-dir research_outputs --top-n 5
python -m pytest -q test_volgan_combined.py test_historical_threshold_research.py test_volgan_model_research.py test_trade_decision_matrix.py
```

`--mode matrix` is the fastest, **does not retrain models**, and makes the individual candidate explanation report. Use `--explain-date YYYY-MM-DD` to print G0–G7 observed values and reasons for a chosen daily top candidate in your terminal. `--mode train` also prints the matrix after training. Excel workbook is a pre-generated research snapshot in the ZIP; regenerating the CSV/HTML doesn't edit the XLSX automatically.

## Methodology cautions

1. Only 13 snapshot dates are available, with long gaps. Some historical exits are unavailable; outcome P&L is always **post-hoc** and is never used to decide entry ranking. 
2. Input liquidity filtering is not bid/ask or volume-at-price evidence. Modeled transaction costs and hedging are approximations; `net edge` is **not expected profit**.
3. The all-candidates list includes both call and put legs. A single row is a **put-call parity pair**, not a standalone option/stock buy or sell signal.
4. The 13 snapshots cannot independently validate persistence, anomaly, WGAN crash-tail probabilities, CVaR, historical risk mandates, or profitable executable arbitrage. The live trading gate remains blocked in code.
5. A historical winner is **not** a future recommendation. Do not connect this research code to a broker or trade automatically.
