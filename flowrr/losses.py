"""Band-limited PSD and the InfoNCE objective.

`band_psd` is where the whole method's supervision lives, and it has one property worth
stating plainly because it is easy to miss: the slice is normalised to sum 1 WITHIN the band,
so any energy the model puts outside 0.1-0.5 Hz is invisible to the loss. That is deliberate
(it makes the objective scale-free and rate-focused) but it also means the objective cannot
tell a working model from one whose output has left the band entirely. Checkpoint selection on
the validation InfoNCE inherits that blind spot. If you add a term, this is the reason.
"""
import torch

from . import config


def band_psd(wave, fs, lo=None, hi=None, nfft=None):
    """(B, T) waveform -> (B, K) power spectrum inside [lo, hi], normalised to sum 1.

    nfft zero-pads the transform. The extra bins are interpolated, not independent: the
    resolution that matters is set by the clip's DURATION (Rayleigh, df = 1/seconds).
    """
    lo = config.LO if lo is None else lo
    hi = config.HI if hi is None else hi
    w = wave - wave.mean(dim=1, keepdim=True)
    n = int(nfft) if nfft else w.shape[1]
    spec = torch.fft.rfft(w, n=n, dim=1)
    p = spec.real ** 2 + spec.imag ** 2
    f = torch.fft.rfftfreq(n, d=1.0 / fs).to(w.device)
    m = (f >= lo) & (f <= hi)
    pb = p[:, m]
    return pb / (pb.sum(dim=1, keepdim=True) + 1e-12)


def info_nce(pa, pb, tau=None):
    """Symmetric InfoNCE over band PSDs, with intra- and cross-view negatives.

    The similarity is negative mean squared error between two band PSDs, which is what
    CalibrationPhys uses; it is not cosine similarity, and the two are not interchangeable
    here because the PSDs are already normalised to sum 1.
    """
    tau = config.TAU if tau is None else tau
    B = pa.shape[0]

    def d(x, y):
        return -((x[:, None, :] - y[None, :, :]) ** 2).mean(dim=2)

    eye = torch.eye(B, dtype=torch.bool, device=pa.device)
    loss = 0
    for dxx, dxy in ((d(pa, pa), d(pa, pb)), (d(pb, pb), d(pb, pa))):
        pos = dxy.diagonal() / tau
        denom = torch.logsumexp(
            torch.cat([(dxx / tau).masked_fill(eye, -1e9),
                       (dxy / tau).masked_fill(eye, -1e9)], dim=1), dim=1)
        loss = loss - (pos - denom).mean()
    return loss / 2
