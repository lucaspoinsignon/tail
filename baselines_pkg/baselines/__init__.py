"""Baseline generative models for comparison against tailfm.

All baselines consume/produce windows of shape (N, n, f) -- identical to the
tailfm pipeline -- so they plug into the same evaluation code
(tailfm.evaluate.print_report, tailfm.risk.estimate_risk, kupiec_test).

    timevae   PyTorch port of TimeVAE      (Desai et al. 2021; ref. impl. TF2/Keras)
    timegan   PyTorch port of TimeGAN      (Yoon et al., NeurIPS 2019; ref. impl. TF1)
    tailgan   adaptation of Tail-GAN       (Cont, Cucuringu, Xu, Zhang 2022; ref. impl. PyTorch)

Each module exposes

    fit_and_generate(train_windows, num_gen, seed=0, device=None, **hparams)
        -> np.ndarray of shape (num_gen, n, f)

See each module's docstring for the exact relationship to the reference code
and the (few) deliberate deviations.
"""

from .timevae import fit_and_generate as timevae_fit_and_generate
from .timegan import fit_and_generate as timegan_fit_and_generate
from .tailgan import fit_and_generate as tailgan_fit_and_generate

BASELINES = {
    "timevae": timevae_fit_and_generate,
    "timegan": timegan_fit_and_generate,
    "tailgan": tailgan_fit_and_generate,
}
