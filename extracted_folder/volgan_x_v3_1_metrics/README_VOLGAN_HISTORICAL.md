# VolGAN-X v2.2: Historical Threshold Research (NON-EXECUTABLE)

**Status:** RESEARCH ONLY. All trade decisions are `ABORT`. This software contains **no broker API or order submission** and must not be used as live trade authorization.

## Overview

Two different analyses are intentionally separated:

1. **Synthetic VolGAN-X demo** (`volgan_combined.py --mode demo`): original five-block pipeline, synthetic IV surfaces, randomly initialized persistence CNN-LSTM, VAE and GAN. The model scores are **untrained**. The fail-closed gate always aborts.
2. **Historical CSV research** (`volgan_combined.py --mode historical` or `historical_threshold_research.py`): actual supplied daily NIFTY option **closing-price** snapshots. A separately labeled, explainable **put-call parity-discrepancy proxy** is tested; it is **not VolGAN predictive evidence** or a tradable arbitrage signal. Its results do not populate or validate the original strategy's risk thresholds.

## Input CSVs

The **fixed ZIP includes all 13 uploaded `nifty_options_YYYY-MM-DD.csv` files in the `data/` subfolder**. Extract the entire ZIP, including the `data/` directory. By default the loader discovers `data/` next to `historical_threshold_research.py`, regardless of the working directory. You may override the location using `--data-dir` pointing at the CSV folder. Required columns:

`TRADE_DATE, INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, CLOSE, SETTLE_PR, CONTRACTS, OPEN_INT, UNDERLYING_CLOSE`.

The run examined **13 files / 19,678 rows** on 13 separate dates from January 1 to June 4, 2024. Source columns include no bid/ask, intraday execution quotes, futures hedge prices, interest-rate series, dividends, margin or validated machine-learning outputs. These prevent a realistic executable backtest.

## Setup & run

Recommended environment: Python 3.10+ with compatible PyTorch wheels. Install:

```bash
python -m pip install -r requirements_volgan.txt
```

Run the CSV research:

```bash
python volgan_combined.py --mode historical --output-dir .
```

Run synthetic demonstration (never an approval):

```bash
python volgan_combined.py --mode demo --date 2024-01-15 --window 10 --regime normal
```

Run tests:

```bash
python -m pytest -q test_volgan_combined.py test_historical_threshold_research.py
```

If you have *only* the historical research module, it can run independently of PyTorch:

```bash
python historical_threshold_research.py --output-dir .
```


### Windows / VS Code input-path troubleshooting

In VS Code, open the extracted package folder and run:

```powershell
python volgan_combined.py --mode historical --output-dir .
```

The full extracted package must contain `volgan_combined.py`, `historical_threshold_research.py`, and a `data` folder containing the 13 `nifty_options_*.csv` files. You may start the script from any working directory because CSV discovery uses the script's location. If you keep the CSVs elsewhere, point `--data-dir` to their actual **folder**, for example:

```powershell
python volgan_combined.py --mode historical --data-dir "C:\Users\hpc\Downloads\my_nifty_csvs" --output-dir .
```

The special value `--data-dir .` means the *terminal's current working directory*, not the script folder. Missing files now raise a descriptive error showing searched paths and the corrective command. The input data was not changed, and the safety rule **never approve live trades** remains intact.

## Outputs

* `historical_threshold_report.json`: reproducible structured audit, source-date checks, candidate cutoff comparison, risk blockers and status.
* `historical_threshold_comparison.csv`: one line per proposed threshold, separate training vs. holdout trade counts / paper P&L.
* `historical_daily_candidates.csv`: selected entry-day diagnostics and next-day marks with cost estimates and missing-exit flags.

**The uploaded files produced the following chronological comparison:**

| Net research-edge cutoff (INR) | Training observed trades | Training proxy net P&L (INR) | Holdout observed trades | Holdout proxy net P&L (INR) |
|---:|---:|---:|---:|---:|
| 0 | 2 | -607.71 | 1 | 584.03 |
| 250 | 1 | -872.45 | 1 | 584.03 |
| 500 | 0 | 0.00 | 1 | 584.03 |
| 1000 | 0 | 0.00 | 0 | 0.00 |
| 2000 | 0 | 0.00 | 0 | 0.00 |

The 10 eligible consecutive-source-day comparisons were divided into **six training dates** (Jan 1–8) and **four holdout dates** (Jan 9–12); the corresponding next-day closing marks may fall on later days. The Jan 15 -> Feb 1 gap is 17 calendar days and Feb 1 -> Jun 4 is 124 days, so neither was treated as an overnight trade. The audit selected 10 candidates for eligible transitions; only **8** had the required qualifying next-session exit prices. All P&L is a hypothetical single-lot, synthetic spot-hedged mark-to-market, *not* observed execution profit.

There was **no qualifying training-selected threshold**; the research module explicitly reports `provisional_research_min_edge_inr: null`. **This is a result, not a software error.** The smallest cutoff that yielded at least 3 observed training trades could not be found. A train-only median modeled round-trip friction of **₹988.98** and p90 of **₹1,057.59** are reported as *cost-model diagnostics only*, not as candidate thresholds.

## How the diagnostic works

- Screen each entry day to `NIFTY / OPTIDX`, observed call + put at the same expiry and strike; each leg must meet at least 10 contracts, 100 open interest, and INR 1 closing premium. Expiries 3–45 calendar days from the current date and strikes within ±2.5% of the underlying close are eligible.
- For each expiry and strike, compute `C - P + exp(-r*tau)*K`. If same-expiry quotes were synchronous, this *should* be roughly constant across strikes by put-call parity. Estimate a robust **leave-one-out median** from the other strikes. The magnitude of a deviation, times lot size, minus an illustrative 2-option roundtrip cost is the *research-edge diagnostic*.
- Choose **at most one candidate each entry day** by maximum same-day diagnostic. Selection never uses next-day prices.
- Evaluate the **same contract** on the next supplied date *only when the gap is 1–4 calendar days and expiry is still in the future*. Both option legs must have a qualifying close on that next date; otherwise report **NO_VALID_EXIT** and leave P&L as missing (never zero).
- Paper P&L estimates a parity-pair position hedged by the **observed NIFTY underlying close change**: `side * lot * [change(C-P) - change(spot)] - modeled roundtrip options/hedge costs`. This spot hedge is a **non-executable approximation**; carry/dividends are simplified. It does not describe real hedge executions.
- Compare a fixed, disclosed threshold grid `[0, 250, 500, 1000, 2000]` in a chronological **60% training / 40% holdout** evaluation. Threshold choice, if enough training observations existed, would use training results only. The holdout is never used to choose the cutoff.

**Model risk warning:** The diagnostic does not calibrate a true VolGAN IV residual nor estimate persistence `y_hat`, VAE anomaly z-score, actual forward/funding costs, actual 95% CVaR, or margin. No reliable `THETA_Y`, `TAU_HEALTHY`, `MIN_EDGE_INR`, `KAPPA`, or `RISK_BUDGET_INR` can be inferred for the VolGAN decision engine from these files. They are explicitly `null` in the output report.

## Safeguards and corrections

1. **Hard fail-closed approval:** `python_risk_gate` cannot return `EXECUTE` under any inputs, including a user-supplied `models_validated=True`. Without validation, it returns `G0_MODEL`; even with that flag, it returns `G0_LIVE_DISABLED`.
2. **No fabricated learned outputs:** The historic CSV branch never instantiates untrained CNN-LSTM/VAE/GAN outputs or interprets them as probabilities, anomalies, or tail-risk estimates.
3. **Honest calibration:** Original numerical tolerances and original proposed risk gate settings remain **research defaults only** in the demo. The historical branch does not overwrite them or claim new 'optimal' settings.
4. **Correct 2024 lot-size logic:** The historical route uses 50 for January and February 2024, and 25 for June 4, 2024, following NSE circular NSE/FAOP/61415 dated Apr 2, 2024 (https://nsearchives.nseindia.com/content/circulars/FAOP61415.pdf). The ambiguous transition period raises an error without contract-master information rather than guessing a lot.
5. **Data integrity and liquidity:** Validates required schema, filename/trade-date consistency, duplicate contract keys, liquidity, and price/expiry/moneyness windows.
6. **No time leakage:** Selects candidates based only on entry-day observations; uses chronological split; holdout kept separate; does not bridge 17- or 124-day CSV gaps.
7. **Missing exits not hidden:** Candidate trades without qualifying next-date marks remain `NO_VALID_EXIT` and `NaN`, not paper profits or zeros. These introduce possible selection bias and block authorization.
8. **Explicit transaction friction:** Shows per-pair estimated spreads and taxes/fees in INR, plus a placeholder hedge friction. **Historical exchange levies and actual broker charges were not confirmed**; these are scenario assumptions.
9. **Auditable outputs and tests:** Report + comparison + daily candidate files; tests for source integrity, lot transitions, no lookahead, gap handling, missing exits, and inability to override the no-live rule.

## Additional data required for real model calibration

At minimum, a substantially longer **continuous historical date series**, bid/ask and tradability measures, contract-master lot sizes, dividends/rates/futures or forward series, calibrated VG parameter histories, model training/validation datasets with out-of-sample labels, and a verified fee/hedging/slippage model. The number of required trading dates for statistical reliability depends on the sampling approach, regime diversity, and desired confidence level; the module's 20/10-date minimum is a **basic procedural blocker**, not evidence of statistical sufficiency.

The `RISK_BUDGET_INR` and `KAPPA` settings must additionally be justified by capital, margin and independent risk governance; they cannot be inferred from options closes alone.
