# VOL-Gan: VolGAN-X Quantitative Research & Implied Volatility Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Asset](https://img.shields.io/badge/Asset-NSE%20NIFTY%2050%20Options-green.svg)](https://www.nseindia.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **Research Paper:** *VolGAN-X: A Multi-Model Framework for NIFTY Implied-Volatility Surface Analysis*  
> **Course:** CS 787 — Advanced Machine Learning / Financial Engineering  
> **Authors:** Shresta, Vaibhav Balani, Aditi

---

## 📌 Executive Overview

**VOL-Gan (VolGAN-X)** is an institutional-grade quantitative derivative engine designed to detect, classify, stress-test, and execute arbitrage and relative-value opportunities across the National Stock Exchange of India (**NSE NIFTY 50**) options market.

Addressing the critical quantitative problem of **"Execution and Microstructure Traps"** (where apparent pricing dislocations evaporate upon execution or incur catastrophic tail losses), this project implements a 5-tier neuro-symbolic pipeline combining physical no-arbitrage calculus, deep spatial-temporal representation learning, generative adversarial market simulation, and fail-closed deterministic risk gates.

---

## 📂 Repository Contents

| File / Package | Size | Description |
| :--- | :---: | :--- |
| 📄 **`VolGAN_X_Research_Paper_7_Pages.pdf`** | `543 KB` | **The 7-Page Academic Research Paper:** Complete formal methodology, mathematical formulation, empirical results, and evaluation. |
| 🗜️ **`NIFTY_Options_Dataset.zip`** | `12.6 MB` | **Multi-Year NSE Bhavcopy Dataset:** 510 trading sessions (2 June 2022 – 28 June 2024) containing 863,354 real option contracts. |
| 📦 **`volgan_x_v3_1_metrics_complete.zip`** | `2.20 MB` | **Core ML Models & Training Outputs:** Neural network checkpoints (`persistence_research.pt`, `vae_research.pt`, `wgan_generator_research.pt`), epoch loss logs, and PNG metric plots. |
| 📦 **`volgan_x_v3_2_trade_decision_matrix_complete.zip`** | `2.41 MB` | **8-Gate Decision Matrix & Audit Engine:** Standalone pipeline code (`trade_decision_matrix.py`), interactive HTML dashboards, and full contract audit reports. |

---

## 🏗️ The 5-Block Architectural Pipeline

```
Raw NSE Bhavcopy (510 Days, 863k Rows)
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ BLOCK 1: Pricing & Inversion Engine                    │
│ • Carr-Madan Damped FFT (Variance Gamma Measure)       │
│ • Dual Black-76 Inversion (Newton-Raphson + Brent)     │
│ • Outputs: 8x40 Volatility Surface X, Residuals R, Vega│
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ BLOCK 2: Spatial-Temporal Persistence Net              │
│ • 8 Laplacian Convexity Filters (Butterfly & Calendar) │
│ • 10-Day LSTM Memory + Bahdanau Additive Attention     │
│ • Outputs: Calibrated Persistence Probability ŷ ∈ [0,1]│
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ BLOCK 3: β-VAE Regime Engine                           │
│ • Non-linear Surface Compression into 5D Latent Mean μ │
│ • Robust Median Absolute Deviation (MAD) Z-Scores      │
│ • Outputs: 8D VolGAN Conditioning Vector c_t           │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ BLOCK 4: Conditional VolGAN Generative Simulator       │
│ • WGAN-GP with Durrleman & Monotonicity Penalties      │
│ • 10,000 Arbitrage-Free Synthetic Future Shocks        │
│ • Non-linear Black-76 Repricing & 95% CVaR Tail Loss   │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ BLOCK 5: Fail-Closed Risk Gates (G0 – G7)              │
│ • Full Indian Statutory Tax Arithmetic (STT, GST, SEBI)│
│ • Strict Deterministic Decision Matrix                 │
│ • Output: EXECUTE (Alpha) vs. ABORT (Trap)             │
└────────────────────────────────────────────────────────┘
```

---

## 🛡️ The 8-Gate Trade Audit Matrix

Every potential trade must survive eight consecutive mathematical and risk hurdles:

1. **Gate G0 (Data & Liquidity QC):** Both legs liquid (`Close >= ₹1`, `Volume >= 10`, `OI >= 100`, `DTE 3-45 days`).
2. **Gate G1 (Gross vs Friction Screen):** Estimated parity discrepancy strictly exceeds modeled transaction costs.
3. **Gate G2 (Persistence Hurdle):** Spatio-temporal persistence probability $\hat{y} \ge \tau_{\text{hurdle}}$ ($0.55 - 0.70$).
4. **Gate G3 (Regime Stability):** Robust VAE anomaly z-score within healthy regime boundaries ($\tau \le 2.50$).
5. **Gate G4 (Minimum Net Edge):** Net executable edge exceeds minimum threshold after all exchange friction.
6. **Gate G5 (Tail Risk / CVaR):** Position repriced across 10,000 GAN scenarios to guarantee acceptable 95% CVaR.
7. **Gate G6 (Capital Limit / Mandate):** Total tail loss remains strictly within authorized desk risk budget.
8. **Gate G7 (Live Execution Policy):** Immutable fail-closed research authorization safeguard (`BLOCKED_LIVE`).

---

## 📊 Dataset Statistics

* **Exchange:** National Stock Exchange of India (NSE)
* **Instrument:** NIFTY 50 Index Options (`OPTIDX`)
* **Time Span:** June 2, 2022 to June 28, 2024 (~510 trading sessions)
* **Total Options Records:** 863,354 rows
* **Liquid Filtered Contracts:** 340,691 rows meeting volume and open interest hurdles
* **Regimes Covered:** Rate hike cycles, steady bull trends, Union Budget volatility spikes, and election shocks.

---

## 📜 Citation & Credits

```bibtex
@article{volganx2026,
  title={VolGAN-X: A Multi-Model Framework for NIFTY Implied-Volatility Surface Analysis},
  author={Shresta and Balani, Vaibhav and Aditi},
  journal={CS 787 Research Report},
  year={2026},
  month={November}
}
```
