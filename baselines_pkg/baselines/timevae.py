"""TimeVAE baseline -- faithful PyTorch port of the reference implementation.

Reference: A. Desai, C. Freeman, Z. Wang, I. Beaver,
"TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation",
2021.  Reference code: timeVAE-main/src/vae/{vae_base.py, timevae.py}
(TensorFlow 2.16 / Keras).

Port notes (kept 1:1 with the reference unless stated):
  * Encoder: Conv1D(filters=hidden_layer_sizes, kernel 3, stride 2, padding
    "same", ReLU) stack -> Flatten -> Dense heads (z_mean, z_log_var);
    reparameterization sampling as in vae_base.Sampling.  PyTorch Conv1d with
    padding=1 reproduces Keras "same" output length ceil(L/2) for stride 2.
  * Decoder: level branch (Dense f ReLU -> Dense f, broadcast over time)
    + optional trend branch (TrendLayer: polynomial basis (t/T)^p)
    + residual branch (Dense -> reshape -> Conv1DTranspose stack -> Flatten
    -> Dense(seq_len*feat_dim)), summed.  ConvTranspose1d(padding=1,
    output_padding=1) reproduces Keras "same" length doubling.
  * Loss (sums, not means, exactly as _get_reconstruction_loss / train_step):
        recon = sum (X - Xr)^2  +  sum (mean_feat X - mean_feat Xr)^2
        kl    = sum -0.5 (1 + log s^2 - mu^2 - s^2)
        total = reconstruction_wt * recon + kl
  * Optimizer Adam(lr=1e-3) (Keras default), batch_size 16, EarlyStopping on
    the epoch-mean training loss (min_delta 1e-2, patience 50) and
    ReduceLROnPlateau (factor 0.5, patience 30), as in fit_on_data.
  * Weights: Xavier-uniform / zero bias to match Keras glorot_uniform defaults.
  * Data are min-max scaled to [0, 1] per feature over all windows and time
    steps (src/data_utils.py::MinMaxScaler), inverted after generation.
  * Defaults from src/config/hyperparameters.yaml: latent_dim=8,
    hidden_layer_sizes=[50,100,200], reconstruction_wt=3.0, trend_poly=0,
    use_residual_conn=True.
  * Deviation: the custom-seasonality layer (SeasonalLayer) is omitted -- it
    is disabled in the reference default config (custom_seas: null) and is
    meaningless for daily-return windows.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .common import MinMax01Scaler, default_device, set_seed


def _init_keras_like(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class TrendLayer(nn.Module):
    """Port of timevae.py::TrendLayer: polynomial trend (t/T)^(p+1), p<P."""

    def __init__(self, latent_dim: int, feat_dim: int, trend_poly: int, seq_len: int):
        super().__init__()
        self.feat_dim, self.trend_poly, self.seq_len = feat_dim, trend_poly, seq_len
        self.trend_dense1 = nn.Linear(latent_dim, feat_dim * trend_poly)
        self.trend_dense2 = nn.Linear(feat_dim * trend_poly, feat_dim * trend_poly)
        lin_space = torch.arange(0, float(seq_len), 1.0) / seq_len
        poly_space = torch.stack([lin_space ** float(p + 1)
                                  for p in range(trend_poly)], dim=0)   # (P, T)
        self.register_buffer("poly_space", poly_space)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        params = self.trend_dense2(torch.relu(self.trend_dense1(z)))
        params = params.view(-1, self.feat_dim, self.trend_poly)        # (B, D, P)
        trend = torch.matmul(params, self.poly_space)                   # (B, D, T)
        return trend.permute(0, 2, 1)                                   # (B, T, D)


class TimeVAE(nn.Module):
    model_name = "TimeVAE"

    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int = 8,
                 hidden_layer_sizes=(50, 100, 200), reconstruction_wt: float = 3.0,
                 trend_poly: int = 0, use_residual_conn: bool = True):
        super().__init__()
        self.seq_len, self.feat_dim, self.latent_dim = seq_len, feat_dim, latent_dim
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.reconstruction_wt = reconstruction_wt
        self.trend_poly, self.use_residual_conn = trend_poly, use_residual_conn

        # ------------------------------------------------------------ encoder
        convs, in_ch, L = [], feat_dim, seq_len
        for num_filters in self.hidden_layer_sizes:
            convs += [nn.Conv1d(in_ch, num_filters, kernel_size=3, stride=2,
                                padding=1), nn.ReLU()]
            in_ch, L = num_filters, math.ceil(L / 2)
        self.enc_convs = nn.Sequential(*convs)
        self.enc_len = L                                  # length after convs
        self.encoder_last_dense_dim = self.hidden_layer_sizes[-1] * L
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)

        # ------------------------------------------------------------ decoder
        self.level1 = nn.Linear(latent_dim, feat_dim)
        self.level2 = nn.Linear(feat_dim, feat_dim)
        self.trend = (TrendLayer(latent_dim, feat_dim, trend_poly, seq_len)
                      if trend_poly and trend_poly > 0 else None)
        if use_residual_conn:
            self.dec_dense = nn.Linear(latent_dim, self.encoder_last_dense_dim)
            deconvs, L_dec = [], self.enc_len
            in_ch = self.hidden_layer_sizes[-1]
            for num_filters in reversed(self.hidden_layer_sizes[:-1]):
                deconvs += [nn.ConvTranspose1d(in_ch, num_filters, kernel_size=3,
                                               stride=2, padding=1,
                                               output_padding=1), nn.ReLU()]
                in_ch, L_dec = num_filters, 2 * L_dec
            deconvs += [nn.ConvTranspose1d(in_ch, feat_dim, kernel_size=3,
                                           stride=2, padding=1,
                                           output_padding=1), nn.ReLU()]
            L_dec *= 2
            self.dec_deconvs = nn.Sequential(*deconvs)
            self.dec_final = nn.Linear(feat_dim * L_dec, seq_len * feat_dim)
        _init_keras_like(self)

    # ------------------------------------------------------------------ parts
    def encode(self, x: torch.Tensor):
        """x: (B, T, D) in [0,1] -> (z_mean, z_log_var, z)."""
        h = self.enc_convs(x.permute(0, 2, 1)).flatten(1)
        mu, log_var = self.z_mean(h), self.z_log_var(h)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return mu, log_var, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        level = self.level2(torch.relu(self.level1(z))).unsqueeze(1)     # (B,1,D)
        out = level.expand(-1, self.seq_len, -1)
        if self.trend is not None:
            out = out + self.trend(z)
        if self.use_residual_conn:
            h = torch.relu(self.dec_dense(z))
            h = h.view(-1, self.enc_len, self.hidden_layer_sizes[-1])
            h = self.dec_deconvs(h.permute(0, 2, 1))                     # (B,D,L)
            h = h.permute(0, 2, 1).flatten(1)
            out = out + self.dec_final(h).view(-1, self.seq_len, self.feat_dim)
        return out

    # ------------------------------------------------------------------- loss
    def loss(self, x: torch.Tensor):
        mu, log_var, z = self.encode(x)
        xr = self.decode(z)
        recon = torch.sum((x - xr) ** 2)
        recon = recon + torch.sum((x.mean(dim=2) - xr.mean(dim=2)) ** 2)
        kl = torch.sum(-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)))
        return self.reconstruction_wt * recon + kl, recon, kl

    @torch.no_grad()
    def get_prior_samples(self, num_samples: int, device: str,
                          batch: int = 4096) -> np.ndarray:
        self.eval()
        out = []
        for s in range(0, num_samples, batch):
            zb = torch.randn(min(batch, num_samples - s), self.latent_dim,
                             device=device)
            out.append(self.decode(zb).cpu().numpy())
        return np.concatenate(out, axis=0)


def train_timevae(model: TimeVAE, windows01: np.ndarray, max_epochs: int = 1000,
                  batch_size: int = 16, device: str = "cpu",
                  verbose: bool = True) -> TimeVAE:
    """Port of vae_base.fit_on_data: Adam(1e-3), EarlyStopping(total_loss,
    min_delta=1e-2, patience=50), ReduceLROnPlateau(factor=0.5, patience=30)."""
    x = torch.tensor(windows01, dtype=torch.float32, device=device)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       factor=0.5, patience=30)
    best, wait, patience, min_delta = float("inf"), 0, 50, 1e-2
    N = x.shape[0]
    for epoch in range(max_epochs):
        perm = torch.randperm(N, device=device)
        losses = []
        for s in range(0, N, batch_size):
            xb = x[perm[s:s + batch_size]]
            total, _, _ = model.loss(xb)
            opt.zero_grad(); total.backward(); opt.step()
            losses.append(total.item())
        epoch_loss = float(np.mean(losses))
        sched.step(epoch_loss)
        if verbose and epoch % 20 == 0:
            print(f"  [timevae] epoch {epoch:4d}  loss {epoch_loss:.2f}")
        if epoch_loss < best - min_delta:
            best, wait = epoch_loss, 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  [timevae] early stop at epoch {epoch}")
                break
    return model


def fit_and_generate(train_windows: np.ndarray, num_gen: int, seed: int = 0,
                     device: str | None = None, latent_dim: int = 8,
                     hidden_layer_sizes=(50, 100, 200),
                     reconstruction_wt: float = 3.0, trend_poly: int = 0,
                     max_epochs: int = 1000, batch_size: int = 16,
                     verbose: bool = True) -> np.ndarray:
    """(N, n, f) real windows -> (num_gen, n, f) TimeVAE samples."""
    device = default_device(device)
    set_seed(seed)
    scaler = MinMax01Scaler()
    w01 = scaler.fit_transform(train_windows)
    model = TimeVAE(seq_len=train_windows.shape[1], feat_dim=train_windows.shape[2],
                    latent_dim=latent_dim, hidden_layer_sizes=hidden_layer_sizes,
                    reconstruction_wt=reconstruction_wt, trend_poly=trend_poly)
    train_timevae(model, w01, max_epochs=max_epochs, batch_size=batch_size,
                  device=device, verbose=verbose)
    gen01 = model.get_prior_samples(num_gen, device=device)
    return scaler.inverse_transform(gen01)
