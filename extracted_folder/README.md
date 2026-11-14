# Extracted Research Codebases & Outputs

This directory contains the uncompressed, browsable source code, neural network model checkpoints, and evaluation reports from the VolGAN-X research packages.

---

## 📁 Directory Structure

```text
extracted_folder/
├── volgan_x_v3_2_trade_decision_matrix/    # Full 8-Gate Trade Audit & Decision Engine
│   ├── trade_decision_matrix.py           # Core 8-gate contract-level decision engine
│   ├── volgan_combined.py                 # Integrated VolGAN model architecture
│   ├── volgan_model_research.py           # VAE, CNN-LSTM, and WGAN-GP training logic
│   ├── historical_threshold_research.py   # Historical Put-Call Parity screening
│   ├── build_decision_workbook.py         # Multi-tab Excel audit workbook builder
│   ├── test_*.py                          # Full pytest test suite
│   ├── requirements_volgan.txt            # Python dependencies
│   ├── data/                              # Sample 13-session NIFTY options Bhavcopies
│   └── research_outputs/                  # Audit outputs, HTML report, checkpoints & plots
│       ├── trade_decision_matrix.html     # Interactive browser audit table
│       ├── trade_decision_matrix.xlsx     # Formatted Excel decision matrix
│       ├── trade_gate_breakdown.csv       # Per-gate audit log for every contract
│       ├── experimental_checkpoints/      # PyTorch .pt model weights (VAE, GAN, LSTM)
│       └── *.png                          # Loss curves and GAN step metrics
│
└── volgan_x_v3_1_metrics/                  # Exploratory Model Fitting & Baseline Run
    ├── volgan_model_research.py           # Model research and training scripts
    ├── plot_research_metrics.py           # Evaluation plotting routines
    ├── research_outputs/                  # Epoch metrics, Brier scores, and training logs
    └── data/                              # Sample training dataset
```

---

## 🚀 Quick Execution

To inspect or run the 8-Gate Trade Decision Matrix:

```bash
cd volgan_x_v3_2_trade_decision_matrix
pip install -r requirements_volgan.txt
python trade_decision_matrix.py
```

To run the unit test suite:

```bash
pytest
```
