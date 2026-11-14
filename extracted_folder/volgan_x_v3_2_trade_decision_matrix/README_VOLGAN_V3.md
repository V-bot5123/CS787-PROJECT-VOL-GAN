# VolGAN-X v3.1 — exploratory model fitting and historical audits

**Safety:** RESEARCH ONLY. All outputs are `ABORT` and `trade_authorization = DISABLED`. The package contains no broker connector or order submission. Neither a command-line option nor a checkpoint flag enables execution.

## What changed from v2.2

1. **Local NIFTY data import.** `import_nse_bhavcopy.py` converts already-downloaded official NSE derivatives bhavcopy CSV / single-CSV ZIP archives (legacy and post-July-2024 UDiFF) to `data/nifty_options_YYYY-MM-DD.csv`. It preserves UDiFF board lot quantity, checks dates, duplicates and underlying close, and refuses to overwrite files. Older bhavcopies require a separate index-close CSV (see below). It does not download or manufacture missing historical data.
2. **Historical IV surfaces.** Derive one-day implied-volatility surfaces from NIFTY options closing prices via Black-76; the forward is estimated from same-day liquid call/put parity pairs (not verified exchange futures). Map actual expiries into nearest *toleranced* target tenor; never mark an imputed cell as an actual quote. `surface_coverage.csv` records observed cells by day.
3. **Experimental VAE training.** Fit the original `VolatilityVAE` only on chronological training snapshots. Compute reconstruction error using observed cells, training-only normalization, train median/MAD/p95 and later-date holdout reconstruction errors. Training p95 is **NOT** a valid `TAU_HEALTHY` trading threshold.
4. **Experimental CNN-LSTM persistence training.** Fit a small variant of the original `CellPersistenceNet`. For each cell, target is whether a local strike-median IV residual >= 0.005 in absolute value persists with the same sign at the **next observed session**. **This is an observable proxy target, NOT a VolGAN/VG-calibrated residual and NOT a trade return.** Only overlapping observed cells contribute to the loss, and labels never cross gaps > 4 calendar days. Predictions are evaluated on separate chronological validation and final test entries, with far too few dates for a reliable estimate. Brier skill is reported against a simple constant probability fitted on training label prevalence.
5. **Experimental conditional WGAN-GP.** The original `VolGANGenerator` and `VolGANCritic` are fitted with gradient penalty on adjacent spot and *partly observed* IV changes. The 5-day and 21-day realized-volatility inputs use only contiguous historical observations; absent windows are encoded with a `-1` sentinel and an explicit 21-day-validity flag (9 total conditioning inputs). Missing IV shifts are zero-filled for dimensionality but *not* represented as observed; the critic is not mask-aware, making the resulting generator unsuitable for risk estimates. A two-point holdout 5% spot-quantile diagnostic is reported, **not a valid CVaR**.
6. **Expanding-origin walk-forward parity-proxy test.** At each later date choose a cost-adjusted minimum-edge cutoff only from observed earlier cases whose exit is known by entry time. If fewer than 5 observed qualifying training trades exist for a cutoff, it is not selected. Missing exits are not treated as zero returns. This is the standalone call/put parity-discrepancy *proxy* from v2.2, **not** a strategy-level ML backtest.
7. **Research-only model checkpoints, audit reports and new tests.** VAE, classifier and GAN generator weights are saved with `EXPERIMENTAL_UNVALIDATED` metadata. `torch.load(..., weights_only=True)` is supported; nothing grants permission to trade.

## Current user-provided data and experiment

- 13 CSV snapshots, 19,678 rows, from 2024-01-01 to 2024-06-04.
- 11 dates form the continuous Jan 1–15 span; Feb 1 and Jun 4 are isolated. No 1–3-year extension was retrieved for this run.
- 1,952 **mapped observed** IV cells of the 4,160 possible cells (46.92%); unobserved cells are explicitly masked.
- 3-session lookback; 4 supervised training, 2 validation, 2 test windows. Only 7 IV snapshots are used to fit the VAE.
- Persistence final two-window test Brier score **0.22377** versus a constant training-prevalence baseline **0.05964**; Brier skill is negative, so the research model underperforms the baseline.
- GAN trained on 6 overlapping observed-market transitions, with mean paired IV-cell coverage 44.69%; 21-session realized vol is not available in training; two final held-out spot outcomes are far too few for assessing 95% tail risk.
- No threshold qualifies for the expanding-origin parity-proxy backtest using the stated minimum evidence rule. Model trading thresholds remain `null`/`NOT_VALIDATED`.
- Some model features rely on estimated forwards and temporally nonsynchronous daily closing marks. No intraday quotes, actual spread, verified futures hedge, margin, dividend/funding series, market-impact data or crisis sample were supplied.

**Important distinction:** The original model's VG-calibrated residual is *not* the persistence label in this research module. Do not interpret the trained classifier as validated for the original options strategy. Its use would require genuine VG calibration with fit QC and price/IV conventions on each date, plus much more data.

## Run on Windows / VS Code

Extract the full ZIP, including the `data/` subfolder, and open the extracted folder in VS Code. Install dependencies:

```powershell
python -m pip install -r requirements_volgan.txt
```

Run all three experimental model stages plus expanding-origin research on the supplied CSVs:

```powershell
python volgan_combined.py --mode train --output-dir research_outputs
```

Run previous independent parity-proxy historical audit:

```powershell
python volgan_combined.py --mode historical --output-dir research_outputs
```

Run the tests:

```powershell
python -m pytest -q test_volgan_combined.py test_historical_threshold_research.py test_volgan_model_research.py
```

Adjust compute steps for research only:

```powershell
python volgan_combined.py --mode train --vae-epochs 30 --persistence-epochs 30 --gan-steps 50 --output-dir research_outputs
```

**More epochs cannot compensate for insufficient independent history.** Training on the original 13 dates will remain nonvalidated.

## Add real historical data from official NSE archives

Official NSE derivatives reports: https://www.nseindia.com/all-reports-derivatives

NSE's legacy F&O bhavcopy was replaced by **F&O-UDiFF Common Bhavcopy Final (ZIP)** on July 8, 2024. This package only imports local data already obtained through permitted NSE mechanisms; it **does not automatically download** archives. Verify your access terms and retain the raw files.

Place the official daily raw ZIP/CSV files (each containing a single date) into `official_downloads/`. For legacy files lacking a NIFTY index-close field, supply an independently verified `nifty_daily_closes.csv` with exactly:

```csv
TRADE_DATE,UNDERLYING_CLOSE
2024-01-01,21888.90
```

The line above illustrates the file format and corresponds to the value already present in the user-supplied January 1 CSV; it is not a newly obtained official source. Supply actual daily closes for every date being imported.

Convert your archive files:

```powershell
python import_nse_bhavcopy.py --input-dir official_downloads --output-dir new_normalized --underlying-csv nifty_daily_closes.csv
```

For UDiFF with valid `UndrlygPric` and `NewBrdLotQty` and no external daily index close, you may omit `--underlying-csv`:

```powershell
python import_nse_bhavcopy.py --input-dir official_udiff --output-dir new_normalized
```

Review the normalized output before moving it into `data/`. The importer rejects missing underlying levels, nonpositive volumes/prices, duplicate contract keys, and missing **post-transition** lot sizes; it does not silently guess. `NewBrdLotQty` is expected on UDiFF input. If your NSE file schema differs, explicitly map and test it rather than silently changing columns.

Then run:

```powershell
python volgan_combined.py --mode train --data-dir data --output-dir research_outputs
```

For credible evaluation, collect **at least 1–3 years of consecutive, provenance-checked options and underlying data** covering ordinary and stressed regimes, with bid/ask and independent funding/hedge prices when possible. Revisit model architecture, coverage thresholds, leakage testing, multiseed convergence, walk-forward refits, margin and realistic costs. Current code remains research-only even after data expansion.

## Output files

| File | Contents |
|---|---|
| `research_outputs/model_training_report.json` | Training diagnostics, coverage, train/validation/test metrics, risk blockers, all trading thresholds null |
| `research_outputs/metrics_summary.csv` | VAE NMSE/MSE/RMSE and classifier BCE/Brier by chronological split |
| `research_outputs/vae_epoch_metrics.csv` | VAE train/validation losses and NMSE by epoch |
| `research_outputs/persistence_epoch_metrics.csv` | CNN-LSTM train/validation BCE/Brier by epoch |
| `research_outputs/gan_step_metrics.csv` | GAN generator/critic/gradient penalty loss per step |
| `research_outputs/surface_coverage.csv` | Actual observed 8x40 IV-grid cells per source date |
| `research_outputs/persistence_holdout_metrics.csv` | Masked supervised test diagnostics by date |
| `research_outputs/walkforward_threshold_proxy.csv` | Strictly later chronological proxy cutoff choices and outcomes |
| `research_outputs/experimental_checkpoints/*.pt` | Research-only weights with `EXPERIMENTAL_UNVALIDATED` metadata |
| `research_outputs/historical_threshold_report.json` | Original independent v2.2 parity-proxy research report |

## Readiness gates, deliberately unmet

Research feasibility checks currently require at least 120 contiguous supervised training windows, 40 independent holdout windows, 150 valid GAN transitions, observed 21-session history, and an executable-quotes data source. These are **minimum planning guardrails, not validation criteria and not live risk-gate thresholds**. No amount of passing them alone ever enables live order execution in this package.

**NEVER treat `THETA_Y=0.70`, `TAU_HEALTHY=2.50`, `MIN_EDGE_INR=250`, `KAPPA=3`, or `RISK_BUDGET_INR=25000` from the old demo as historically verified.** Original demonstration values remain only for the isolated synthetic `--mode demo` run. Real-data model reports keep all trading threshold values null.

---

## v3.1: Train / validation / test metrics (research only)

Run:

```powershell
python volgan_combined.py --mode train --output-dir research_outputs
python plot_research_metrics.py --output-dir research_outputs
```

The 8 labelable persistence windows are now split **chronologically into 4 training,
2 validation, and 2 test windows**. The VAE is fitted on the initial 7 snapshots;
its validation/test snapshots (2 each) occur later. These are descriptive
experiments, not reliable model validation: adjacent windows share history and
individual option cells are highly correlated. The 2-window test is never
used for optimization or early stopping; fixed epoch counts are used instead.

- `metrics_summary.csv` reports **VAE train/validation/test standardized-IV MSE,
  RMSE, and NMSE**, and **CNN-LSTM train/validation/test unweighted BCE and
  Brier scores**. The VAE NMSE is the sum of squared reconstruction errors on
  *actually observed cells* divided by the squared errors of the constant zero
  predictor **in training-standardized space**. The reference is the fixed
  training-only per-cell mean surface. **NMSE around 1 means no improvement
  over that reference; smaller is better.** The standardized-IV MSE units are
  not raw implied-volatility-squared units. Unobserved cells are excluded.
- `vae_epoch_metrics.csv` contains optimization loss, masked stochastic
  reconstruction loss, KL divergence, deterministic train/validation MSE and
  NMSE every epoch. `vae_epoch_metrics.png` plots the NMSE curves.
- `persistence_epoch_metrics.csv` contains optimization-weighted BCE plus
  *unweighted* train/validation BCE and Brier every epoch;
  `persistence_epoch_metrics.png` plots comparable BCE curves.
- `gan_step_metrics.csv` contains per-step critic, generator, and gradient
  penalty objectives. `gan_step_metrics.png` plots them. **WGAN losses are
  not an NMSE, validation loss, or proof of tail distribution quality.**
  The GAN has only **six training transitions** under this stricter split.
- `model_training_report.json` contains the same metrics plus exploratory final
  two-window test diagnostics. The final test is evaluated once after the fit;
  no test trajectory is produced because that would invite test-set tuning.

**Known limitations:** Individual cells on the same date are not independent
samples; this is not a powered test. Test accuracy is particularly misleading
with rare persistence events, so compare Brier to the constant training-label
prevalence baseline. No finance trading threshold is validated, and
`trade_authorization` remains `DISABLED` with decision `ABORT`.
