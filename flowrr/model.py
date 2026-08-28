"""PhysNet2D — the 2D-CNN of CalibrationPhys (Akamatsu et al., J-BHI 2024), Fig. 4.

The input is a spatio-temporal map (B, 1, M, T): M ROI blocks on one axis, time on the other.
The ROI axis is pooled away progressively and then collapsed by a mean, so the same weights
accept any M >= 16; the time axis is halved once in the middle and restored by upsampling, so
the output waveform has the input's length.

The paper's own reason for a 2D CNN over a 3D one is cost: a 3D network over raw frames is too
heavy for a phone, while a 2D network over a precomputed ST-map is not. That trade is what
makes the ST-map front end (see flowrr/preprocess) part of the method rather than a detail.

  x     (B, 1, M, T)
  wave  (B, T)                    out_ch=1, the default
  feat  (B, feat_dim, T)          feat_dim set, for frame-wise features
"""
import torch.nn as nn


def _blk(cin, cout, k=(3, 3)):
    p = (k[0] // 2, k[1] // 2)
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=p), nn.BatchNorm2d(cout), nn.ELU())


class PhysNet2D(nn.Module):
    def __init__(self, out_ch=1, feat_dim=None):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, (1, 5), padding=(0, 2)), nn.BatchNorm2d(32), nn.ELU())
        self.s1 = nn.Sequential(nn.AvgPool2d((2, 1)), _blk(32, 64), _blk(64, 64))
        self.s2 = nn.Sequential(nn.AvgPool2d((2, 1)), _blk(64, 64), _blk(64, 64))
        self.mid = nn.Sequential(nn.AvgPool2d((2, 2)), _blk(64, 64), _blk(64, 64))
        self.s3 = nn.Sequential(nn.AvgPool2d((2, 1)), _blk(64, 64), _blk(64, 64))
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode="bilinear", align_corners=False),
            _blk(64, 64, (3, 1)))
        self.head = nn.Conv2d(64, feat_dim or out_ch, 1)
        self.feat_dim = feat_dim

    def forward(self, x):                      # (B, 1, M, T)
        T = x.shape[-1]
        h = self.up(self.s3(self.mid(self.s2(self.s1(self.stem(x))))))
        if h.shape[-1] != T:                   # odd T after halve+upsample
            h = nn.functional.interpolate(h, size=(h.shape[-2], T),
                                          mode="bilinear", align_corners=False)
        h = self.head(h.mean(dim=2, keepdim=True)).squeeze(2)      # (B, d, T)
        return h if self.feat_dim else h.squeeze(1)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
