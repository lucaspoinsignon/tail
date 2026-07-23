# 1) Train the three baselines at the fair budgets, generate 50k windows each,
#    print all diagnostics + the comparison table, save both figures
python run_baselines.py --data returns.csv --n 24 --gen 50000 \
    --timegan-iters 50000 \
    --tailgan-alphas 0.05,0.01,0.005 --tailgan-epochs 10000 \
    --timevae-recon-wt 100 --timevae-latent 32

# 2) Multi-portfolio backtest on the saved samples
python backtest_portfolios.py --data returns.csv --n 24 --gen-dir baseline_out
