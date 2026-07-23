"""TimeGAN baseline -- faithful PyTorch port of the reference implementation.

Reference: J. Yoon, D. Jarrett, M. van der Schaar, "Time-series Generative
Adversarial Networks", NeurIPS 2019.  Reference code:
TimeGAN-master/{timegan.py, utils.py} (TensorFlow 1.x, tf.contrib).

The reference relies on tf.contrib and therefore only runs on TF 1.15 /
Python <= 3.7; it cannot be executed in a modern environment.  This module is
a line-by-line port to PyTorch preserving:

  * Architecture: five networks (embedder, recovery, generator, supervisor,
    discriminator), each a stacked GRU (num_layers; supervisor num_layers-1)
    followed by a per-time-step fully-connected layer.  Sigmoid output for
    embedder/recovery/generator/supervisor; linear logit for the
    discriminator -- exactly as in the tf.contrib.layers.fully_connected
    calls.
  * Losses (identical formulas and weights):
      E_loss0  = 10 * sqrt(MSE(X, X_tilde))
      E_loss   = E_loss0 + 0.1 * G_loss_S
      G_loss_S = MSE(H[:,1:,:], H_hat_supervise[:,:-1,:])
      G_loss_U = BCE(Y_fake, 1); G_loss_U_e = BCE(Y_fake_e, 1)
      G_loss_V = mean|std_batch(X_hat) - std_batch(X)|
                 + mean|mean_batch(X_hat) - mean_batch(X)|      (biased var)
      G_loss   = G_loss_U + gamma * G_loss_U_e + 100*sqrt(G_loss_S) + 100*G_loss_V
      D_loss   = BCE(Y_real,1) + BCE(Y_fake,0) + gamma * BCE(Y_fake_e,0),
                 with the discriminator updated only when D_loss > 0.15.
  * Training schedule: (1) embedding pre-training, (2) supervised-only
    pre-training, (3) joint training with two generator(+embedder) updates
    per discriminator check, all with Adam at the TF1 default lr=1e-3.
  * Data handling: per-feature min-max scaling to [0,1] over all windows and
    time (timegan.py::MinMaxScaler), uniform [0,1] noise
    (utils.py::random_generator), z_dim = feature dim, gamma = 1.
  * Defaults from main_timegan.py: hidden_dim=24, num_layer=3, batch_size=128,
    module='gru', iterations=50000 (per phase; reduce for CPU runs).

Deviations: sequences are fixed-length windows here, so the variable-length
machinery (extract_time / sequence_length masking) is dropped; the 'lstm' and
'lstmLN' cell options are omitted ('gru' is the reference default).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MinMax01Scaler, default_device, set_seed


class _RNNBlock(nn.Module):
    """Stacked GRU + per-step FC, matching dynamic_rnn + fully_connected."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, sigmoid: bool):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.fc(out)
        return torch.sigmoid(out) if self.sigmoid else out


class TimeGAN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 24, num_layers: int = 3):
        super().__init__()
        z_dim = dim
        self.embedder = _RNNBlock(dim, hidden_dim, num_layers, hidden_dim, True)
        self.recovery = _RNNBlock(hidden_dim, hidden_dim, num_layers, dim, True)
        self.generator = _RNNBlock(z_dim, hidden_dim, num_layers, hidden_dim, True)
        self.supervisor = _RNNBlock(hidden_dim, hidden_dim, max(num_layers - 1, 1),
                                    hidden_dim, True)
        self.discriminator = _RNNBlock(hidden_dim, hidden_dim, num_layers, 1, False)
        self.dim, self.z_dim = dim, z_dim


def _moment_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """G_loss_V: first/second batch moments, biased variance as tf.nn.moments."""
    v1 = torch.mean(torch.abs(
        torch.sqrt(x_hat.var(dim=0, unbiased=False) + 1e-6)
        - torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)))
    v2 = torch.mean(torch.abs(x_hat.mean(dim=0) - x.mean(dim=0)))
    return v1 + v2


def train_timegan(net: TimeGAN, windows01: np.ndarray, iterations: int = 50_000,
                  batch_size: int = 128, device: str = "cpu", gamma: float = 1.0,
                  verbose: bool = True, log_every: int = 1000) -> TimeGAN:
    data = torch.tensor(windows01, dtype=torch.float32, device=device)
    N, T, dim = data.shape
    net.to(device).train()

    e_r = list(net.embedder.parameters()) + list(net.recovery.parameters())
    g_s = list(net.generator.parameters()) + list(net.supervisor.parameters())
    opt_E0 = torch.optim.Adam(e_r, lr=1e-3)
    opt_E = torch.optim.Adam(e_r, lr=1e-3)
    opt_GS = torch.optim.Adam(g_s, lr=1e-3)
    opt_G = torch.optim.Adam(g_s, lr=1e-3)
    opt_D = torch.optim.Adam(net.discriminator.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    def batch() -> torch.Tensor:
        idx = torch.randint(0, N, (batch_size,), device=device)
        return data[idx]

    def noise(n: int) -> torch.Tensor:
        return torch.rand(n, T, net.z_dim, device=device)   # U[0,1] as reference

    # ------------------------------------------- 1. embedding network training
    if verbose:
        print("  [timegan] phase 1/3: embedding")
    for itt in range(iterations):
        X = batch()
        H = net.embedder(X)
        X_tilde = net.recovery(H)
        E_loss_T0 = F.mse_loss(X_tilde, X)
        E_loss0 = 10 * torch.sqrt(E_loss_T0)
        opt_E0.zero_grad(); E_loss0.backward(); opt_E0.step()
        if verbose and itt % log_every == 0:
            print(f"    step {itt}/{iterations}, e_loss: "
                  f"{np.sqrt(E_loss_T0.item()):.4f}")

    # ------------------------------------------------- 2. supervised loss only
    if verbose:
        print("  [timegan] phase 2/3: supervised")
    for itt in range(iterations):
        X = batch()
        H = net.embedder(X)
        H_hat_supervise = net.supervisor(H)
        G_loss_S = F.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])
        opt_GS.zero_grad(); G_loss_S.backward(); opt_GS.step()
        if verbose and itt % log_every == 0:
            print(f"    step {itt}/{iterations}, s_loss: "
                  f"{np.sqrt(G_loss_S.item()):.4f}")

    # ---------------------------------------------------------- 3. joint phase
    if verbose:
        print("  [timegan] phase 3/3: joint")
    step_d_loss = torch.tensor(0.0)
    for itt in range(iterations):
        for _ in range(2):                                   # G twice per D
            X, Z = batch(), noise(batch_size)
            # generator + supervisor update
            H = net.embedder(X)
            E_hat = net.generator(Z)
            H_hat = net.supervisor(E_hat)
            H_hat_supervise = net.supervisor(H)
            X_hat = net.recovery(H_hat)
            Y_fake = net.discriminator(H_hat)
            Y_fake_e = net.discriminator(E_hat)
            G_loss_U = bce(Y_fake, torch.ones_like(Y_fake))
            G_loss_U_e = bce(Y_fake_e, torch.ones_like(Y_fake_e))
            G_loss_S = F.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])
            G_loss_V = _moment_loss(X_hat, X)
            G_loss = (G_loss_U + gamma * G_loss_U_e
                      + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V)
            opt_G.zero_grad(); G_loss.backward(); opt_G.step()
            # embedder update
            H = net.embedder(X)
            X_tilde = net.recovery(H)
            H_hat_supervise = net.supervisor(H)
            E_loss_T0 = F.mse_loss(X_tilde, X)
            G_loss_S_e = F.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])
            E_loss = 10 * torch.sqrt(E_loss_T0) + 0.1 * G_loss_S_e
            opt_E.zero_grad(); E_loss.backward(); opt_E.step()

        # discriminator update (only when it underperforms, as reference)
        X, Z = batch(), noise(batch_size)
        with torch.no_grad():
            H = net.embedder(X)
            E_hat = net.generator(Z)
            H_hat = net.supervisor(E_hat)
        Y_real = net.discriminator(H)
        Y_fake = net.discriminator(H_hat)
        Y_fake_e = net.discriminator(E_hat)
        D_loss = (bce(Y_real, torch.ones_like(Y_real))
                  + bce(Y_fake, torch.zeros_like(Y_fake))
                  + gamma * bce(Y_fake_e, torch.zeros_like(Y_fake_e)))
        if D_loss.item() > 0.15:
            opt_D.zero_grad(); D_loss.backward(); opt_D.step()
            step_d_loss = D_loss.detach()
        if verbose and itt % log_every == 0:
            print(f"    step {itt}/{iterations}, d: {step_d_loss.item():.4f}, "
                  f"g_u: {G_loss_U.item():.4f}, "
                  f"g_s: {np.sqrt(G_loss_S.item()):.4f}, "
                  f"g_v: {G_loss_V.item():.4f}")
    return net


@torch.no_grad()
def generate_timegan(net: TimeGAN, num: int, seq_len: int, device: str,
                     batch: int = 2048) -> np.ndarray:
    net.eval()
    out = []
    for s in range(0, num, batch):
        Z = torch.rand(min(batch, num - s), seq_len, net.z_dim, device=device)
        X_hat = net.recovery(net.supervisor(net.generator(Z)))
        out.append(X_hat.cpu().numpy())
    return np.concatenate(out, axis=0)


def fit_and_generate(train_windows: np.ndarray, num_gen: int, seed: int = 0,
                     device: str | None = None, hidden_dim: int = 24,
                     num_layers: int = 3, iterations: int = 50_000,
                     batch_size: int = 128, verbose: bool = True) -> np.ndarray:
    """(N, n, f) real windows -> (num_gen, n, f) TimeGAN samples."""
    device = default_device(device)
    set_seed(seed)
    scaler = MinMax01Scaler()
    w01 = scaler.fit_transform(train_windows)
    net = TimeGAN(dim=train_windows.shape[2], hidden_dim=hidden_dim,
                  num_layers=num_layers)
    train_timegan(net, w01, iterations=iterations, batch_size=batch_size,
                  device=device, verbose=verbose)
    gen01 = generate_timegan(net, num_gen, seq_len=train_windows.shape[1],
                             device=device)
    return scaler.inverse_transform(gen01)
