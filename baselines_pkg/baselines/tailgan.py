"""Tail-GAN baseline -- adapted from the reference PyTorch implementation.

Reference: R. Cont, M. Cucuringu, R. Xu, C. Zhang, "Tail-GAN: Learning to
Simulate Tail Risk Scenarios", 2022.  Reference code:
Tail-GAN-main/{TailGAN.py, Transform.py, util.py, gen_thresholds.py}.

Kept 1:1 with the reference:
  * Generator: MLP z -> 128 -> 256 -> 512 -> 1024 -> (n_assets * n_steps)
    with BatchNorm+LeakyReLU blocks and a hard clamp to [-1, 1]
    (TailGAN.py::Generator).
  * Discriminator: strategy PnLs -> differentiable sort
    (util.py::deterministic_NeuralSort, temperature temp) -> MLP
    (batch_size -> 256 -> 128 -> 2*|alphas|) producing per-strategy
    (VaR_alpha, ES_alpha) estimates, projected onto the constraint set
    {W v >= e} (Discriminator.project_op).
  * Loss: joint (VaR, ES) Fissler-Ziegel-type score S_quant
    (or S_stats), with loss_D = score(real) - score(fake) and
    loss_G = score(fake vs real PnL), exactly as in Train_Single.
  * Strategies (Transform.py): buy-and-hold on each asset, buy-and-hold on
    static portfolios, mean-reversion and trend-following with the reference
    z-score constructions (prices from Inc2Price with P_0 = 1, z = dev/0.01),
    thresholds at the [31, 69] percentiles of the training z-scores
    (gen_thresholds.py logic, computed in memory).
  * Hyperparameters: latent_dim=1000, temp=0.01, Adam(b1=0.5, b2=0.999),
    lr_G=1e-6, lr_D=1e-7, Cap=10, WH=10, W=10, alphas=(0.05,), noise 't5'
    (Student-t_5), n_critic_D = n_critic_G = 1.

Adaptations required to run on user CSV windows (documented deviations):
  * A scenario is one window: (N, n, f) windows are transposed to the
    reference layout (N, n_assets=f, n_steps=n); returns are scaled per
    feature to [-1, 1] by the training max-abs (the reference clamp assumes
    this range) and unscaled after generation.  The clamp implies Tail-GAN
    cannot exceed the training-set extreme.
  * The reference loads a precomputed random static-portfolio matrix
    (Static_Port_Transform/TransMat_IS.npy, not shipped); we regenerate it in
    memory with a fixed seed: n_trans long-short weight vectors w ~ U(-1,1)^f
    normalized to ||w||_1 = 1 ('LShort').
  * The reference trains numNN=10 GANs and screens by final loss
    (Screen_Ensemble); we train one by default (numNN configurable upstream).
  * Batches are dropped to a fixed batch_size (the discriminator's first
    linear layer has in_features = batch_size).
  * D-step uses gen.detach() instead of backward(retain_graph=True) --
    identical gradients w.r.t. discriminator parameters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SymmetricMaxScaler, default_device, set_seed


# --------------------------------------------------------------------- util.py
def deterministic_neural_sort(s: torch.Tensor, tau: float) -> torch.Tensor:
    """s: (batch, n, 1) -> soft permutation matrices (batch, n, n)."""
    n = s.size(1)
    one = torch.ones((n, 1), dtype=s.dtype, device=s.device)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, one.t()))
    scaling = (n + 1 - 2 * (torch.arange(n, device=s.device) + 1)).to(s.dtype)
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    return torch.softmax(P_max / tau, dim=-1)


# ---------------------------------------------------------------- Transform.py
def inc2price(data: torch.Tensor) -> torch.Tensor:
    """(N, f, n) increments -> (N, f, n+1) prices with P_0 = 1 (Inc2Price)."""
    p0 = torch.ones(data.shape[0], data.shape[1], 1, dtype=data.dtype,
                    device=data.device)
    return torch.cumsum(torch.cat((p0, data), dim=2), dim=2)


def moving_average(values: torch.Tensor, WH: int) -> torch.Tensor:
    """Transform.py::movingaverage with a fixed uniform kernel (per channel)."""
    N, C, L = values.shape
    kernel = torch.full((1, 1, WH), 1.0 / WH, dtype=values.dtype,
                        device=values.device)
    out = F.conv1d(values.reshape(N * C, 1, L), kernel).reshape(N, C, L - WH + 1)
    all_output = values.clone()
    all_output[:, :, WH - 1:] = out
    return all_output


def buy_hold(prices: torch.Tensor, Cap: float) -> torch.Tensor:
    money = prices * Cap
    return money[:, :, -1] - money[:, :, 0]


def _positions_from_signals(cross: torch.Tensor, enter: torch.Tensor,
                            sign: float, n_cols: int) -> torch.Tensor:
    """Shared body of Position_MR / Position_TF for one side (long or short).

    cross: (rows, L) zero/trend-crossing events (True at t=0), enter: (rows, L)
    threshold-crossing events.  Returns cumulative position in {0, sign}.
    """
    sig = -1 * cross.int() + 1 * enter.int() + 1 * (cross & enter).int()
    flat = sig.flatten()
    index_l = torch.arange(flat.numel(), device=flat.device)
    nz, idx_nz = flat[flat != 0], index_l[flat != 0]
    open_t = torch.cat((torch.zeros(1, dtype=torch.bool, device=flat.device),
                        (nz[:-1] < 0) & (nz[1:] > 0)))
    open_ts = idx_nz[open_t]
    close_t = torch.cat((torch.zeros(1, dtype=torch.bool, device=flat.device),
                         (nz[:-1] > 0) & (nz[1:] < 0)))
    close_ts = idx_nz[close_t]
    pos_open = torch.zeros(flat.numel(), device=flat.device)
    pos_open[open_ts[open_ts % n_cols != 0]] = sign
    pos_close = torch.zeros(flat.numel(), device=flat.device)
    pos_close[close_ts[close_ts % n_cols != 0]] = -sign
    rows = cross.shape[0]
    return (torch.cumsum(pos_open.reshape(rows, -1), dim=1)
            + torch.cumsum(pos_close.reshape(rows, -1), dim=1))


def _position(zscores: torch.Tensor, Cap: float, LR: float, SR: float,
              ST: torch.Tensor, LT: torch.Tensor,
              short_on_up: bool) -> torch.Tensor:
    """Position_MR (short_on_up=True) / Position_TF (short_on_up=False)."""
    L = zscores.shape[-1]
    rows = zscores.shape[0]
    cross = (((zscores[:, :-1] < 0) & (zscores[:, 1:] >= 0))
             | ((zscores[:, :-1] > 0) & (zscores[:, 1:] <= 0)))
    cross = torch.cat((torch.ones(rows, 1, dtype=torch.bool,
                                  device=zscores.device), cross), dim=1)
    up = (zscores[:, :-1] < ST) & (zscores[:, 1:] >= ST)      # cross ST upward
    down = (zscores[:, :-1] > LT) & (zscores[:, 1:] <= LT)    # cross LT downward
    pad = torch.zeros(rows, 1, dtype=torch.bool, device=zscores.device)
    up, down = torch.cat((pad, up), 1), torch.cat((pad, down), 1)
    if short_on_up:                       # mean-reversion: short high, long low
        short_sig, long_sig = up, down
    else:                                 # trend-following: short down, long up
        # reference TF uses ST for the downward and LT for the upward crossing
        down_tf = (zscores[:, :-1] > ST) & (zscores[:, 1:] <= ST)
        up_tf = (zscores[:, :-1] < LT) & (zscores[:, 1:] >= LT)
        short_sig = torch.cat((pad, down_tf), 1)
        long_sig = torch.cat((pad, up_tf), 1)
    short_pos = _positions_from_signals(cross, short_sig, -1.0, L)
    long_pos = _positions_from_signals(cross, long_sig, 1.0, L)
    position = Cap * SR * short_pos + Cap * LR * long_pos
    position[:, -1] = 0
    return position


def mean_rev(prices: torch.Tensor, Cap: float, WH: int, LR: float, SR: float,
             ST: torch.Tensor, LT: torch.Tensor) -> torch.Tensor:
    """Transform.py::MeanRev (static MA over the first WH+1 prices)."""
    N, C, L = prices.shape
    flat = prices.reshape(N * C, L)
    ma = torch.mean(prices[:, :, :WH + 1], dim=2).reshape(N * C, 1)
    z = (flat - ma) / 0.01
    ST_t = ST.repeat(N).reshape(-1, 1)
    LT_t = LT.repeat(N).reshape(-1, 1)
    pos = _position(z, Cap, LR, SR, ST_t, LT_t, short_on_up=True)
    pnl = pos[:, :-1] * (flat[:, 1:] - flat[:, :-1])
    return pnl.reshape(N, C, -1).sum(dim=2)


def trend_follow(prices: torch.Tensor, Cap: float, WH: int, LR: float,
                 SR: float, ST: torch.Tensor, LT: torch.Tensor) -> torch.Tensor:
    """Transform.py::TrendFollow (MA(WH) - MA(2 WH) crossover)."""
    N, C, L = prices.shape
    flat = prices.reshape(N * C, L)
    ma1 = moving_average(prices, WH).reshape(N * C, L)
    ma2 = moving_average(prices, WH * 2).reshape(N * C, L)
    z = (ma1 - ma2) / 0.01
    ST_t = ST.repeat(N).reshape(-1, 1)
    LT_t = LT.repeat(N).reshape(-1, 1)
    pos = _position(z, Cap, LR, SR, ST_t, LT_t, short_on_up=False)
    pnl = pos[:, :-1] * (flat[:, 1:] - flat[:, :-1])
    return pnl.reshape(N, C, -1).sum(dim=2)


# ---------------------------------------------------- gen_thresholds.py logic
def compute_thresholds(train_R: torch.Tensor, strategy: str,
                       percentile_l=(31, 69), WH: int = 10) -> torch.Tensor:
    """Percentiles of the training z-score distribution (pooled over assets,
    as in the reference loop) -> (n_assets, len(percentile_l))."""
    with torch.no_grad():
        prices = inc2price(train_R)
        N, C, L = prices.shape
        flat = prices.reshape(N * C, L)
        if strategy == "MR":
            ma = torch.mean(prices[:, :, :WH + 1], dim=2).reshape(N * C, 1)
            z = (flat - ma) / 0.01
        else:  # TF
            ma1 = moving_average(prices, WH).reshape(N * C, L)
            ma2 = moving_average(prices, WH * 2).reshape(N * C, L)
            z = (ma1 - ma2) / 0.01
        z_np = z.cpu().numpy()
        thr = np.array([np.percentile(z_np, p) for p in percentile_l])
        return torch.tensor(np.tile(thr, (C, 1)), dtype=train_R.dtype,
                            device=train_R.device)


def make_static_portfolios(f: int, n_trans: int, seed: int,
                           device: str) -> torch.Tensor:
    """Long-short random portfolio weights (f, n_trans), ||w||_1 = 1.

    In-memory replacement for the reference's precomputed TransMat_IS.npy
    ('LShort' variant), which is not shipped with the repository.
    """
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, size=(f, n_trans))
    w = w / np.abs(w).sum(axis=0, keepdims=True)
    return torch.tensor(w, dtype=torch.float32, device=device)


def static_port(prices: torch.Tensor, trans_mat: torch.Tensor) -> torch.Tensor:
    """Transform.py::StaticPort with the in-memory transition matrix."""
    swap = prices.permute(0, 2, 1)                       # (N, L, f)
    port = torch.matmul(swap, trans_mat)                 # (N, L, n_trans)
    return port.permute(0, 2, 1)                         # (N, n_trans, L)


# ----------------------------------------------------------------- TailGAN.py
class Generator(nn.Module):
    def __init__(self, latent_dim: int, R_shape: tuple[int, int]):
        super().__init__()
        self.R_shape = R_shape

        def block(i, o, normalize=True):
            layers = [nn.Linear(i, o)]
            if normalize:
                layers.append(nn.BatchNorm1d(o, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256), *block(256, 512), *block(512, 1024),
            nn.Linear(1024, int(np.prod(R_shape))),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = torch.clamp(self.model(z), min=-1, max=1)
        return out.view(out.shape[0], *self.R_shape)


class PnLEngine:
    """Compute_PNL of TailGAN.py with in-memory portfolios and thresholds."""

    def __init__(self, train_R: torch.Tensor, n_trans: int = 50, Cap: float = 10,
                 WH: int = 10, ratios=(1.0, 1.0), thresholds_pct=((31, 69),),
                 strategies=("Port", "MR", "TF"), seed: int = 0,
                 device: str = "cpu"):
        self.Cap, self.WH, self.LR, self.SR = Cap, WH, ratios[0], ratios[1]
        self.strategies, self.thresholds_pct = strategies, thresholds_pct
        f = train_R.shape[1]
        self.trans_mat = make_static_portfolios(f, n_trans, seed, device)
        self.thr = {("MR", p): compute_thresholds(train_R, "MR", p, WH)
                    for p in thresholds_pct}
        self.thr.update({("TF", p): compute_thresholds(train_R, "TF", p, WH)
                         for p in thresholds_pct})

    def __call__(self, R: torch.Tensor) -> torch.Tensor:
        prices = inc2price(R)
        pnl_l = [buy_hold(prices, self.Cap)]
        for strat in self.strategies:
            if strat == "Port":
                pnl_l.append(buy_hold(static_port(prices, self.trans_mat),
                                      self.Cap))
            elif strat == "MR":
                for p in self.thresholds_pct:
                    thr = self.thr[("MR", p)]
                    pnl_l.append(mean_rev(prices, self.Cap, self.WH, self.LR,
                                          self.SR, ST=thr[:, -1], LT=thr[:, -2]))
            elif strat == "TF":
                for p in self.thresholds_pct:
                    thr = self.thr[("TF", p)]
                    pnl_l.append(trend_follow(prices, self.Cap, self.WH,
                                              self.LR, self.SR,
                                              ST=thr[:, 0], LT=thr[:, 1]))
        return torch.cat(pnl_l, dim=1)                    # (batch, n_strats)


class Discriminator(nn.Module):
    def __init__(self, pnl_engine: PnLEngine, batch_size: int, alphas=(0.05,),
                 W: float = 10.0, temp: float = 0.01, project: bool = True):
        super().__init__()
        self.pnl_engine, self.alphas = pnl_engine, alphas
        self.W, self.temp, self.project = W, temp, project
        self.model = nn.Sequential(
            nn.Linear(batch_size, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 2 * len(alphas)),
        )

    def project_op(self, validity: torch.Tensor) -> torch.Tensor:
        validity = validity.clone()
        for i, alpha in enumerate(self.alphas):
            v = validity[:, 2 * i].clone()
            e = validity[:, 2 * i + 1].clone()
            ind = torch.sign(torch.as_tensor(0.5 - alpha,
                                             device=validity.device))
            hit = (self.W * v >= e).float()
            miss = 1.0 - hit
            validity[:, 2 * i] = ind * (miss * v
                                        + hit * (v + self.W * e) / (1 + self.W ** 2))
            validity[:, 2 * i + 1] = ind * (miss * e
                                            + hit * self.W * (v + self.W * e)
                                            / (1 + self.W ** 2))
        return validity

    def forward(self, R: torch.Tensor):
        PNL = self.pnl_engine(R)                          # (batch, S)
        PNL_t = PNL.T                                     # (S, batch)
        s = PNL_t.reshape(*PNL_t.shape, 1)
        perm = deterministic_neural_sort(s, self.temp)
        PNL_sort = torch.bmm(perm, s)
        validity = self.model(PNL_sort.reshape(*PNL_t.shape))
        if self.project:
            validity = self.project_op(validity)
        return PNL, validity


# ---------------------------------------------------------- score functions
def _G1_quant(v, W):
    return -W * v ** 2 / 2


def S_quant(v, e, X, alpha, W=10.0):
    """TailGAN.py::S_quant -- joint (VaR, ES) strictly consistent score."""
    if alpha < 0.5:
        rt = (((X <= v).float() - alpha) * (_G1_quant(v, W) - _G1_quant(X, W))
              + 1.0 / alpha * (alpha * e) * (X <= v).float() * (v - X)
              + (alpha * e) * (e - v) - alpha * e ** 2 / 2)
    else:
        a_inv = 1 - alpha
        rt = (((X >= v).float() - a_inv) * (_G1_quant(v, W) - _G1_quant(X, W))
              + 1.0 / a_inv * (a_inv * (-e)) * (X >= v).float() * (X - v)
              + (a_inv * (-e)) * (v - e) - a_inv * e ** 2 / 2)
    return torch.mean(rt)


class Score(nn.Module):
    def __init__(self, alphas=(0.05,), W: float = 10.0):
        super().__init__()
        self.alphas, self.W = alphas, W

    def forward(self, validity: torch.Tensor, PNL: torch.Tensor) -> torch.Tensor:
        loss = 0
        for i, alpha in enumerate(self.alphas):
            v = validity[:, [2 * i]]
            e = validity[:, [2 * i + 1]]
            loss = loss + S_quant(v, e, PNL.T, alpha, self.W)
        return loss


# ------------------------------------------------------------------ training
def train_tailgan(train_R: torch.Tensor, n_epochs: int = 3000,
                  batch_size: int = 1000, latent_dim: int = 1000,
                  lr_G: float = 1e-6, lr_D: float = 1e-7, b1: float = 0.5,
                  b2: float = 0.999, alphas=(0.05,), W: float = 10.0,
                  temp: float = 0.01, n_trans: int = 50, Cap: float = 10,
                  WH: int = 10, noise_df: float | None = 5.0, seed: int = 0,
                  device: str = "cpu", verbose: bool = True,
                  log_every: int = 100) -> Generator:
    """Port of TailGAN.py::Train_Single on in-memory scenarios (N, f, n)."""
    torch.manual_seed(seed)
    N, f, n = train_R.shape
    batch_size = min(batch_size, N)
    engine = PnLEngine(train_R, n_trans=n_trans, Cap=Cap, WH=WH,
                      seed=seed, device=device)
    generator = Generator(latent_dim, (f, n)).to(device)
    discriminator = Discriminator(engine, batch_size, alphas=alphas, W=W,
                                  temp=temp).to(device)
    criterion = Score(alphas=alphas, W=W)
    opt_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(b1, b2))

    def noise(m: int) -> torch.Tensor:
        if noise_df is not None:                          # 't5' reference default
            z = np.random.standard_t(noise_df, (m, latent_dim))
        else:
            z = np.random.normal(0, 1, (m, latent_dim))
        return torch.tensor(z, dtype=torch.float32, device=device)

    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        ep_D, ep_G = [], []
        for s in range(0, N - batch_size + 1, batch_size):   # drop_last
            real_R = train_R[perm[s:s + batch_size]]
            gen_R = generator(noise(real_R.shape[0]))

            # ------------------------------------------- train discriminator
            opt_D.zero_grad()
            PNL, PNL_validity = discriminator(real_R)
            _, gen_validity = discriminator(gen_R.detach())
            loss_D = criterion(PNL_validity, PNL) - criterion(gen_validity, PNL)
            loss_D.backward()
            opt_D.step()
            ep_D.append(loss_D.item())

            # ----------------------------------------------- train generator
            opt_G.zero_grad()
            _, gen_validity = discriminator(gen_R)
            loss_G = criterion(gen_validity, PNL.detach())
            loss_G.backward()
            opt_G.step()
            ep_G.append(loss_G.item())
        if verbose and epoch % log_every == 0:
            print(f"  [tailgan] epoch {epoch}/{n_epochs}  "
                  f"D {np.mean(ep_D):.4f}  G {np.mean(ep_G):.4f}")
    return generator


@torch.no_grad()
def generate_tailgan(generator: Generator, num: int, latent_dim: int,
                     noise_df: float | None, device: str,
                     batch: int = 4096) -> torch.Tensor:
    generator.eval()
    out = []
    for s in range(0, num, batch):
        m = min(batch, num - s)
        if noise_df is not None:
            z = np.random.standard_t(noise_df, (m, latent_dim))
        else:
            z = np.random.normal(0, 1, (m, latent_dim))
        z = torch.tensor(z, dtype=torch.float32, device=device)
        out.append(generator(z).cpu())
    return torch.cat(out, dim=0)


def fit_and_generate(train_windows: np.ndarray, num_gen: int, seed: int = 0,
                     device: str | None = None, n_epochs: int = 3000,
                     batch_size: int = 1000, latent_dim: int = 1000,
                     lr_G: float = 1e-6, lr_D: float = 1e-7, alphas=(0.05,),
                     n_trans: int = 50, WH: int = 10,
                     noise_df: float | None = 5.0,
                     verbose: bool = True) -> np.ndarray:
    """(N, n, f) real windows -> (num_gen, n, f) Tail-GAN samples."""
    device = default_device(device)
    set_seed(seed)
    assert train_windows.shape[1] > 2 * WH, \
        f"window length n={train_windows.shape[1]} must exceed 2*WH={2*WH}"
    scaler = SymmetricMaxScaler()
    w = scaler.fit_transform(train_windows)                # [-1, 1]
    train_R = torch.tensor(w.transpose(0, 2, 1), dtype=torch.float32,
                           device=device)                  # (N, f, n)
    gen = train_tailgan(train_R, n_epochs=n_epochs, batch_size=batch_size,
                        latent_dim=latent_dim, lr_G=lr_G, lr_D=lr_D,
                        alphas=alphas, n_trans=n_trans, WH=WH,
                        noise_df=noise_df, seed=seed, device=device,
                        verbose=verbose)
    gen_R = generate_tailgan(gen, num_gen, latent_dim, noise_df, device)
    gen_w = gen_R.numpy().transpose(0, 2, 1)               # (num, n, f)
    return scaler.inverse_transform(gen_w)
