"""Shape and behaviour checks that need no data and no GPU.

These are the invariants that broke in practice, so each one is a regression guard rather
than a formality.
"""
import numpy as np
import torch

from flowrr import config, readout
from flowrr.model import PhysNet2D, n_params
from flowrr.losses import band_psd, info_nce


def test_model_shapes_and_size():
    m = PhysNet2D(out_ch=1)
    assert n_params(m) == 290817, "architecture drifted from CalibrationPhys Fig. 4"
    for M in (64, 32, 16):                    # ROI axis is pooled then averaged away
        assert m(torch.randn(2, 1, M, 300)).shape == (2, 300)


def test_band_psd_normalised_and_band_limited():
    fs, T = config.FS_TRAIN, 300
    t = torch.arange(T) / fs
    x = torch.stack([torch.sin(2 * np.pi * f * t) for f in (0.15, 0.25, 0.40)])
    p = band_psd(x, fs)
    assert torch.allclose(p.sum(1), torch.ones(3), atol=1e-4)
    # the band holds 9 independent bins at 20 s: df = 1/20 s = 0.05 Hz over 0.1-0.5 Hz
    assert p.shape[1] == 9
    assert p.argmax(1).tolist() == [1, 3, 6]


def test_info_nce_prefers_matching_spectra():
    fs, T = config.FS_TRAIN, 300
    t = torch.arange(T) / fs
    a = torch.stack([torch.sin(2 * np.pi * f * t) for f in (0.15, 0.25, 0.35, 0.45)])
    same = info_nce(band_psd(a, fs), band_psd(a + 0.01 * torch.randn_like(a), fs))
    shuffled = info_nce(band_psd(a, fs), band_psd(a.flip(0), fs))
    assert same < shuffled


def test_readout_recovers_a_known_rate():
    fs, T = config.FS_TRAIN, 900
    t = np.arange(T) / fs
    for bpm in (9.0, 15.0, 24.0):
        w = np.sin(2 * np.pi * (bpm / 60.0) * t)
        got = readout.all_readouts(w, 300, fs)
        for k, v in got.items():
            assert abs(v - bpm) < 1.0, f"{k} gave {v:.2f} for {bpm}"
