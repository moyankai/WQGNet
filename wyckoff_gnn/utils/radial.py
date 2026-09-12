"""Radial feature extractor: Bessel basis + polynomial cutoff + MLP."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PolynomialCutoff(nn.Module):
    """Polynomial cutoff (DimeNet style, p=6)."""
    def __init__(self, p: float = 6.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: normalized distance r/r_max in [0, 1]
        p = self.p
        out = 1.0
        out -= ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p)
        out += (p * (p + 2.0) * torch.pow(x, p + 1.0))
        out -= (p * (p + 1.0) / 2.0) * torch.pow(x, p + 2.0)
        return out * (x < 1.0)


def bessel_basis(
    r: torch.Tensor,
    n_bessel: int,
    r_max: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Bessel basis: sinc(n*x) * n = sin(pi*n*x) / (pi*x) with polynomial cutoff."""
    x = r / r_max
    n = torch.arange(1, n_bessel + 1, device=r.device, dtype=r.dtype)
    x_safe = x.clamp(min=eps).unsqueeze(-1)
    # sinc(n*x) * n where sinc(z) = sin(pi*z)/(pi*z)
    bessel = torch.sin(math.pi * x_safe * n) / (math.pi * x_safe) * n
    cutoff = PolynomialCutoff(p=6.0)(x).unsqueeze(-1)
    return bessel * cutoff


class RadialFeat(nn.Module):
    """Bessel basis + MLP for radial feature extraction. Forward-weight initialization."""

    def __init__(
        self,
        r_max: float,
        n_bessel: int,
        n_out: int,
        mlp_hidden: list[int],
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.r_max = r_max
        self.n_bessel = n_bessel
        self.eps = eps
        dims = [n_bessel] + list(mlp_hidden) + [n_out]
        self.mlp = nn.Sequential()
        for i, (h_in, h_out) in enumerate(zip(dims, dims[1:])):
            linear = nn.Linear(h_in, h_out, bias=False)
            gain = 1.0 if i == 0 else math.sqrt(2.0)
            nn.init.uniform_(linear.weight, -math.sqrt(3), math.sqrt(3))
            linear.weight.data.mul_(gain / math.sqrt(h_in))
            self.mlp.append(linear)
            if i < len(dims) - 2:
                self.mlp.append(nn.SiLU())

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        basis = bessel_basis(r, self.n_bessel, self.r_max, self.eps)
        return self.mlp(basis)
