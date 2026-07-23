"""Shared utilities for the baselines: scalers matching each reference
implementation's convention, and small torch helpers."""

from __future__ import annotations

import numpy as np
import torch


def default_device(device: str | None = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class MinMax01Scaler:
    """Per-feature min-max scaling to [0, 1] over ALL windows and time steps.

    This is the convention shared (up to the placement of the 1e-7) by the
    TimeGAN reference (timegan.py::MinMaxScaler, reduces over axes (0, 1)) and
    TimeVAE (src/data_utils.py::MinMaxScaler with 3-d input).  Both reference
    models end in sigmoid / are trained on [0, 1] data, so this scaler is part
    of the model, not a preprocessing choice.
    """

    def fit(self, windows: np.ndarray) -> "MinMax01Scaler":
        self.min_ = windows.min(axis=(0, 1))                    # (f,)
        self.max_ = (windows - self.min_).max(axis=(0, 1))      # (f,) range
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return (windows - self.min_) / (self.max_ + 1e-7)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)

    def inverse_transform(self, windows01: np.ndarray) -> np.ndarray:
        # Same order as the TimeGAN reference renormalization:
        #   generated_data = generated_data * max_val; ... + min_val
        return windows01 * self.max_ + self.min_


class SymmetricMaxScaler:
    """Per-feature scaling to [-1, 1] by the max absolute value.

    The Tail-GAN generator hard-clamps its output to [-1, 1]
    (TailGAN.py::Generator.forward, torch.clamp(img, -1, 1)); the reference
    experiments use simulated returns already living in that range.  For real
    returns we therefore map each feature into [-1, 1] before training and
    invert afterwards.  NOTE the clamp means Tail-GAN cannot generate a return
    more extreme than the training maximum -- a property of the reference
    implementation worth remembering when reading its tail diagnostics.
    """

    def fit(self, windows: np.ndarray) -> "SymmetricMaxScaler":
        self.scale_ = np.abs(windows).max(axis=(0, 1)) + 1e-12  # (f,)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return windows / self.scale_

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)

    def inverse_transform(self, windows_pm1: np.ndarray) -> np.ndarray:
        return windows_pm1 * self.scale_
