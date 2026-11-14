# VOL-Gan: VolGAN-X Quantitative Research & Implied Volatility Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Asset](https://img.shields.io/badge/Asset-NSE%20NIFTY%2050%20Options-green.svg)](https://www.nseindia.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **Research Paper:** *VolGAN-X: A Multi-Model Framework for NIFTY Implied-Volatility Surface Analysis*  
> **Course:** CS 787 — Advanced Machine Learning / Financial Engineering  
> **Authors:** Aditi, Vaibhav Balani

---

## 📌 Research Context & Theoretical Motivation

In quantitative derivatives pricing, the **implied volatility (IV) surface** $\sigma_{\text{implied}}(K, T)$ encapsulates cross-sectional market expectations of asset price variance across strike prices $K$ and tenors $T$. Traditional financial engineering approaches, such as parametric surface smoothing (e.g., SVI, SABR) or structural jump-diffusion models, frequently struggle to capture the non-stationary dynamics and asymmetric volatility smiles characteristic of emerging market indices like the **NSE NIFTY 50**.

Conversely, unconstrained end-to-end deep learning architectures face three critical institutional failure modes:
1. **High-Dimensional Surface Sparsity:** Observed market surfaces are partially masked due to illiquidity in deep out-of-the-money (OTM) or long-tenor contracts.
2. **Microstructure Noise vs. Persistent Discrepancies:** Transient order-book friction can easily be misclassified as exploitable pricing anomalies.
3. **Severe Tail Risk:** Heavy-tailed return distributions and non-Gaussian market crashes can rapidly invalidate unhedged statistical arbitrage positions.

### Multi-Model Division of Labour

Rather than treating a single black-box neural network as both the trade signal and its risk estimator, **VolGAN-X** enforces an institutional division of labour across three decoupled neural architectures and an independent empirical Put–Call Parity benchmark:

* **Unsupervised Latent Representation (Variational Autoencoder):** Encodes standardized $8 \times 40$ masked implied-volatility surfaces into a 5-dimensional Gaussian latent state $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. The VAE provides non-linear dimensionality reduction and tracks market regime shifts via reconstruction divergence.
* **Temporal Persistence Modelling (CNN–LSTM Network):** Evaluates a three-session sequential window of multi-channel surface features via spatial convolutions and additive temporal attention pooling. It explicitly estimates the conditional probability that a local strike-median volatility anomaly persists at session $t+1$, filtering transient microstructure noise.
* **Non-Parametric Stress Testing (Conditional WGAN-GP):** A Wasserstein Generative Adversarial Network with gradient penalty, conditioned on latent regime vectors and historical realized volatility windows. It simulates joint underlying price returns and surface shifts across 10,000 synthetic stress scenarios to compute empirical Value-at-Risk (VaR) and Expected Shortfall (CVaR).
* **Empirical Arbitrage Benchmark (Put–Call Parity Screen):** Formulates an independent, model-free pricing benchmark evaluating leave-one-strike-out synthetic forward deviations, net of Indian exchange-specific statutory transaction costs (Securities Transaction Tax, stamp duty, exchange turnover charges, and modeled execution slippage).

### The 8-Gate Research Decision Ledger

Candidate option structures are subjected to an **8-Gate Sequential Elimination Ledger** ($G_0$ to $G_7$), ensuring that signals are rejected unless liquidity thresholds, persistence probability hurdles, regime stability criteria, tail-risk coverage, and portfolio capital mandates are strictly satisfied. Live execution remains permanently governed by an immutable research policy (`BLOCKED_LIVE`).

---

## 📂 Repository Contents

| File / Package | Size | Description |
| :--- | :---: | :--- |
| 📄 **`VolGAN_X_Research_Paper_7_Pages.pdf`** | `521 KB` | **The 7-Page Academic Research Paper:** Complete formal methodology, mathematical formulation, empirical results, and evaluation. |
| 📁 **`extracted_folder/`** | `~6 MB` | **Uncompressed Codebases & Outputs:** Direct browsable source code (`trade_decision_matrix.py`, etc.), PyTorch checkpoints, and visual reports. |
| 🗜️ **`NIFTY_Options_Dataset.zip`** | `12.6 MB` | **Multi-Year NSE Bhavcopy Dataset:** 510 trading sessions (2 June 2022 – 28 June 2024) containing 863,354 real option contracts. |
| 📦 **`volgan_x_v3_1_metrics_complete.zip`** | `2.20 MB` | **Core ML Models & Training Outputs:** Neural network checkpoints (`persistence_research.pt`, `vae_research.pt`, `wgan_generator_research.pt`), epoch loss logs, and PNG metric plots. |
| 📦 **`volgan_x_v3_2_trade_decision_matrix_complete.zip`** | `2.41 MB` | **8-Gate Decision Matrix & Audit Engine:** Standalone pipeline code (`trade_decision_matrix.py`), interactive HTML dashboards, and full contract audit reports. |

---

## 🏗️ Multi-Model Research Architecture

The framework operates on a standardized, masked $8 \times 40$ Implied Volatility surface derived via Black-76 inversion, evaluated across four coordinated components:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │              DAILY NSE NIFTY OPTION RECORDS             │
                  │             (510 Dated CSVs, 863,354 Records)           │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                                               ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │               MASKED 8 × 40 IV SURFACE                  │
                  │   (8 Expiry Tenors × 40 Standardized Moneyness Strikes) │
                  └───────┬─────────────────┬─────────────────┬─────────────┘
                          │                 │                 │
           ┌──────────────┴──────────┐      │      ┌──────────┴──────────────┐
           ▼                         ▼      │      ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  VARIATIONAL         │  │  CNN–LSTM            │  │  CONDITIONAL         │  │  PUT–CALL PARITY     │
│  AUTOENCODER (VAE)   │  │  PERSISTENCE NET     │  │  WGAN-GP             │  │  BENCHMARK           │
│                      │  │                      │  │                      │  │                      │
│ • 5D Latent State μ  │  │ • 3-Day Lookback     │  │ • Generator + Critic │  │ • Independent Price  │
│ • Mask-Weighted NMSE │  │ • 2-Layer Conv2D     │  │ • 10,000 Joint Shock │  │   Discrepancy Screen │
│ • Surface Curvature  │  │ • 16-Unit LSTM       │  │   Scenarios (ΔS, ΔIV)│  │ • Round-Trip Cost    │
│   & Regime Encoding  │  │ • Attention Pooling  │  │ • Tail Risk (VaR/CVaR│  │   Coverage Ratio      │
└──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
           │                         │                         │                         │
           └─────────────────────────┼─────────────────────────┴─────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   8-GATE RESEARCH DECISION LEDGER (G0 – G7)                            │
│                                                                                        │
│ • G0: Data & Liquidity Input Filter (Close >= ₹1, Vol >= 10, OI >= 100, DTE 3–45)      │
│ • G1: Parity-Proxy Screen Net of Modeled Round-Trip Transaction Friction (> ₹0)        │
│ • G2: Calibrated CNN-LSTM Persistence Probability Hurdle (θ_y = 0.70)                  │
│ • G3: Robust VAE Anomaly / Market Regime Stability Check (τ_healthy = 2.50)            │
│ • G4: Minimum Model Edge Hurdle (MIN_EDGE_INR = ₹250)                                  │
│ • G5: Calibrated Contract-Level 95% CVaR Tail Risk Availability                        │
│ • G6: Risk Mandate & Portfolio Capital Limits (κ = 3.0, Budget = ₹25,000)              │
│ • G7: Live Authorization Policy (IMMUTABLE RESEARCH POLICY: BLOCKED_LIVE)              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Component Breakdown

### 1. Variational Autoencoder (VAE)
* **Architecture:** 320-dimensional surface flattened through 128 and 64 hidden units to a 5-dimensional Gaussian latent bottleneck state $\mu$, reconstructed through a symmetric decoder.
* **Loss Function:** Mask-weighted MSE on observed liquid cells plus Kullback-Leibler divergence regularization ($\beta = 0.005$):
  $$\mathcal{L}_{\text{VAE}} = \frac{\sum_{i,j} \mathcal{M}_{i,j} (X_{i,j} - \hat{X}_{i,j})^2}{\sum_{i,j} \mathcal{M}_{i,j}} + \beta D_{\text{KL}}(q(z|X) \parallel \mathcal{N}(0, I))$$
* **Role:** Unsupervised dimensionality reduction, global regime representation, and surface reconstruction error.

### 2. CNN–LSTM Persistence Network
* **Architecture:** Three-day sequence of 5-channel surface feature maps processed through two Conv2D layers, a 16-unit temporal LSTM, and additive attention pooling.
* **Target:** Evaluates whether a local strike-median volatility residual ($|R| \ge 0.005$) persists with unchanged sign at the next observed trading session:
  $$y_{t,i,j} = \mathbf{1}\{|R_{t,i,j}| \ge 0.005, \; |R_{t+1,i,j}| \ge 0.005, \; R_{t,i,j} R_{t+1,i,j} > 0\}$$
* **Role:** Disentangles structural, persistent mispricings from transient microstructure order-book noise.

### 3. Conditional WGAN-GP
* **Architecture:** Generative Adversarial Network with Wasserstein distance and gradient penalty. Conditioned on VAE latent coordinates and historical realized volatility windows (5-day and 21-day).
* **Output:** Joint simulation of 1 underlying index return ($\Delta S$) and 320 log-implied volatility changes ($\Delta \log IV$).
* **Role:** Non-parametric downside stress-testing and empirical Value-at-Risk (VaR) / Expected Shortfall evaluation.

### 4. Put–Call Parity Benchmark
* **Mechanism:** Independent price-based discrepancy screening across matched strikes and expiries using observed closing marks:
  $$\hat{F}_{t,T} = \text{median}_K \left( K + e^{rT} (C_{t,K,T} - P_{t,K,T}) \right), \quad r = 0.065$$
* **Role:** Empirical ground-truth baseline that directly accounts for statutory Indian transaction costs, slippage, and STT.

---

## 📊 Dataset Statistics

* **Exchange:** National Stock Exchange of India (NSE)
* **Instrument:** NIFTY 50 Index Options (`OPTIDX`)
* **Time Span:** June 2, 2022 to June 28, 2024 (~510 trading sessions)
* **Total Options Records:** 863,354 rows
* **Liquid Filtered Contracts:** 340,691 rows meeting volume ($\ge 10$) and open interest ($\ge 100$) hurdles.
* **Regimes Covered:** Central bank rate hike cycles, bull market runs, Union Budget volatility spikes, and Lok Sabha election shocks.

---

## 📜 Citation & Credits

```bibtex
@article{volganx2026,
  title={VolGAN-X: A Multi-Model Framework for NIFTY Implied-Volatility Surface Analysis},
  author={Aditi and Balani, Vaibhav},
  journal={CS 787 Research Report},
  year={2026},
  month={November}
}
```
