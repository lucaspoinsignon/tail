"""Why is TimeVAE collapsing? Separates 'prior hole' from 'incompressible data'.

Usage:
    python diagnostic_timevae.py --data timeseries.csv --prices --n 24 \
        --recon-wt 100 --latent 32 --epochs 300
"""
import argparse

import numpy as np
import torch

from baselines.common import MinMax01Scaler, default_device, set_seed
from baselines.timevae import TimeVAE, train_timevae
from dataio import load_returns
from evaluation import make_windows

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--prices", action="store_true")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--recon-wt", type=float, default=100.0)
ap.add_argument("--latent", type=int, default=32)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = default_device()
set_seed(a.seed)

r = load_returns(a.data, a.prices)
real = make_windows(r[: int(0.8 * len(r))], a.n, 1)
scaler = MinMax01Scaler()
w01 = scaler.fit_transform(real)
print(f"data {r.shape} -> {real.shape[0]} windows of {a.n}x{real.shape[2]}\n")

model = TimeVAE(seq_len=a.n, feat_dim=real.shape[2], latent_dim=a.latent,
                reconstruction_wt=a.recon_wt)
train_timevae(model, w01, max_epochs=a.epochs, batch_size=a.batch,
              device=dev, verbose=True)
model.eval()

x = torch.tensor(w01[:2000], dtype=torch.float32, device=dev)
with torch.no_grad():
    mu, log_var, z_post = model.encode(x)
    x_post = model.decode(z_post)
    x_hat = model.decode(mu)
    kl_per_dim = (-0.5 * (1 + log_var - mu ** 2 - log_var.exp())).mean(0)
gen01 = model.get_prior_samples(2000, device=dev)
spread = float(gen01.std(axis=0).max())

s_real = float(x.std())
s_post = float(x_post.std())
s_prior = float(gen01.std())
xn, xh = x.cpu().numpy(), x_hat.cpu().numpy()
r2_feat = np.array([1.0 - ((xn[:, :, j] - xh[:, :, j]) ** 2).mean()
                    / xn[:, :, j].var() for j in range(xn.shape[2])])
r2 = float(r2_feat.mean())
r2_pooled = 1.0 - float(((x - x_hat) ** 2).mean()) / float(x.var())
active = int((kl_per_dim > 0.01).sum())

print("\n" + "=" * 58)
print(f"  std of real windows            {s_real:.5f}")
print(f"  std decoded from posterior     {s_post:.5f}   ({s_post/s_real:6.1%} of real)")
print(f"  std decoded from prior N(0,I)  {s_prior:.5f}   ({s_prior/s_real:6.1%} of real)")
print(f"  reconstruction R^2 per feature {r2:6.3f}   (pooled {r2_pooled:.3f}, misleading)")
print(f"  spread across prior samples    {spread:.2e}")
print(f"  active latent dims (KL>0.01)   {active} / {a.latent}")
print("=" * 58)

if active == 0:
    print("\nDIAGNOSIS: complete posterior collapse.")
    print("  No latent dimension carries information, so the decoder ignores z and")
    print("  emits one constant window. Every generated sample is identical -> zero")
    print("  dispersion, flat VaR, lambda_L = 0. Raising --recon-wt or --latent will")
    print("  not help: using the latent costs KL but barely reduces reconstruction")
    print("  error on near-white-noise windows, so the optimiser zeroes it out.")
    print("  Report this as the result; it is a property of the model on returns.")
elif s_post / s_real > 0.7 and s_prior / s_real < 0.4:
    print("\nDIAGNOSIS: prior hole.")
    print("  The decoder is fine but the aggregate posterior has drifted off N(0,I),")
    print("  so prior draws land where it was never trained. Lower --recon-wt")
    print("  (try 3, 10, 30) or fit a learned prior. Higher recon-wt makes it worse.")
elif s_post / s_real < 0.7:
    print("\nDIAGNOSIS: incompressibility (the structural case).")
    print("  Even decoding from the *posterior* under-disperses, so the decoder")
    print("  itself is mean-like: E[x|z] carries little of Var(x). No setting of")
    print("  --recon-wt or --latent fixes this; the missing variance has no channel")
    print("  to the output because the decoder is deterministic.")
else:
    print("\nDIAGNOSIS: dispersion looks healthy - collapse is elsewhere in the run.")

print("\n  Per-feature R^2 near 0 confirms return windows are near-incompressible;")
print("  on the TimeVAE paper's smooth datasets this number is close to 1.")
print("  The pooled R^2 is inflated by cross-feature level offsets from min-max")
print("  scaling - a constant decoder scores well on it. Quote the per-feature one.")
