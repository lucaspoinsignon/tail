# Environment
Requires Python >= 3.9 with numpy, scipy, matplotlib, and torch.
The tested setup is the conda environment in `environment.yml`:

````bash
conda env create -f environment.yml
conda activate tails

## Baselines-only package

Standalone training and evaluation of three published generative baselines for
multivariate return series -- **no tailfm model code included**:

| Baseline | Reference | Implementation |
|---|---|---|
| `baselines/timevae.py` | Desai et al. 2021, *TimeVAE* | Faithful PyTorch port of the TF2/Keras reference (same architecture, loss, training schedule; custom-seasonality layer omitted as in the reference default config). |
| `baselines/timegan.py` | Yoon et al., NeurIPS 2019, *TimeGAN* | Faithful PyTorch port of the TF1 reference (which cannot run on Python >= 3.8): five GRU networks, identical loss formulas/weights, three-phase schedule. |
| `baselines/tailgan.py` | Cont, Cucuringu, Xu, Zhang 2022, *Tail-GAN* | Adaptation of the PyTorch reference: NeuralSort + projected (VaR, ES) discriminator, `S_quant` score, strategy PnL engine; portfolio matrix and thresholds regenerated in memory from the training data. |

`evaluation/` contains the assessment stack (windowing, Hill / tail-dependence /
marginal VaR-CVaR diagnostics, GPD-refined portfolio risk with bootstrap CIs,
Kupiec backtest). `dataio.py` loads CSV (with or without header / Date column)
or .npy return matrices; use `--prices` for price series (log-returns taken).

Requirements: numpy, scipy, matplotlib, torch.

## Run

```bash
# 1) Train all three baselines at reference-grade budgets, generate 50k
#    windows each, print diagnostics + comparison table, save both figures
python run_baselines.py --data returns.csv --n 24 --gen 50000 \
    --timegan-iters 50000 \
    --tailgan-alphas 0.05,0.01,0.005 --tailgan-epochs 10000 \
    --timevae-recon-wt 100 --timevae-latent 32

# 2) Multi-portfolio backtest on the saved samples (seconds, no retraining)
python backtest_portfolios.py --data returns.csv --n 24 --gen-dir baseline_out
```

Outputs in `baseline_out/`: `gen_<model>.npy`, `baseline_diagnostics.png`
(lower-tail QQ, tail-dependence curves, portfolio loss survival) and
`empirical_distributions.png` (per-feature log-scale densities and one-step
loss survival curves, real vs generated).

Useful flags: `--quick` (two-minute CPU smoke test), `--reuse` (skip training
and re-evaluate saved `gen_<model>.npy`), `--models timevae,timegan,tailgan`
(subset), `--seed`, `--device`, `--weights`, `--horizon`. Reference training
budgets and known failure modes of each baseline are documented in the module
docstrings under `baselines/`.
