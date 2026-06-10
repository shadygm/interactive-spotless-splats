from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RunningStats:
    bin_size: int = 10000
    robust_percentile: float = 0.7
    lower_bound: float = 0.5
    upper_bound: float = 0.9
    decay: float = 0.95
    hist_err: torch.Tensor = field(init=False)
    avg_err: float = 1.0
    lower_err: float = 0.0
    upper_err: float = 1.0

    def __post_init__(self):
        self.hist_err = torch.zeros(self.bin_size, dtype=torch.float32)

    def update(self, err_histogram: torch.Tensor) -> None:
        err_histogram = err_histogram.detach().to(device=self.hist_err.device, dtype=self.hist_err.dtype)
        if err_histogram.numel() != self.hist_err.numel():
            self.hist_err = torch.zeros_like(err_histogram)

        self.hist_err.mul_(self.decay).add_(err_histogram)
        total = torch.sum(self.hist_err).clamp_min(1e-12)
        cdf = torch.cumsum(self.hist_err, dim=0)
        bins = torch.linspace(0.0, 1.0, self.hist_err.numel() + 1, device=self.hist_err.device)

        self.avg_err = self._percentile(bins, cdf, total * self.robust_percentile)
        self.lower_err = self._percentile(bins, cdf, total * self.lower_bound)
        self.upper_err = self._percentile(bins, cdf, total * self.upper_bound)

    def _percentile(self, bins: torch.Tensor, cdf: torch.Tensor, target: torch.Tensor) -> float:
        idx = torch.where(cdf >= target)[0]
        if idx.numel() == 0:
            return float(bins[-1].item())
        return float(bins[idx[0]].item())

