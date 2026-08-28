"""Waveform -> one respiratory rate in bpm.

Four readouts are provided because they fail differently, not because the choice is arbitrary:

  psd_sum   sum the band PSDs of every sliding window, then take one argmax. Default. It can
            return the band edge verbatim when a model has collapsed, which is a useful tell.
  mean      argmax each window, then average the rates. Structurally cannot emit the band
            edge, so it hides collapse instead of showing it.
  median    as `mean`, but robust to a few bad windows.
  snr_mean  as `mean`, weighted by each window's in-band peak-to-median ratio.
"""
import numpy as np

from . import config


def all_readouts(wave, win, fs, step_s=1.0, nfft=4096):
    """{name: bpm} for one session waveform."""
    wave = np.asarray(wave, float)
    win = int(min(win, len(wave)))
    step = max(1, int(round(fs * step_s)))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (f >= config.LO) & (f <= config.HI)
    fb = f[band]

    acc, peaks, snrs = np.zeros(band.sum()), [], []
    for a in range(0, max(1, len(wave) - win + 1), step):
        w = wave[a:a + win]
        if len(w) < win:
            break
        p = np.abs(np.fft.rfft(w - w.mean(), n=nfft)) ** 2
        pb = p[band]
        acc += pb
        peaks.append(fb[pb.argmax()] * 60.0)
        snrs.append(pb.max() / (np.median(pb) + 1e-20))
    if not peaks:
        return {k: float("nan") for k in ("psd_sum", "mean", "median", "snr_mean")}
    peaks = np.asarray(peaks); snrs = np.asarray(snrs)
    return dict(psd_sum=float(fb[acc.argmax()] * 60.0),
                mean=float(peaks.mean()),
                median=float(np.median(peaks)),
                snr_mean=float((peaks * snrs).sum() / (snrs.sum() + 1e-20)))
