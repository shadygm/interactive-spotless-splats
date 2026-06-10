from __future__ import annotations

import torch


def get_positional_encodings(
    height: int,
    width: int,
    num_frequencies: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return [H, W, 4 * num_frequencies] sine/cosine encodings."""
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    if height > 1:
        y = y / (height - 1)
    if width > 1:
        x = x / (width - 1)

    frequencies = (2.0 ** torch.arange(num_frequencies, device=device, dtype=torch.float32)) * torch.pi
    y_enc = torch.cat(
        [torch.sin(frequencies * y[..., None]), torch.cos(frequencies * y[..., None])],
        dim=-1,
    )
    x_enc = torch.cat(
        [torch.sin(frequencies * x[..., None]), torch.cos(frequencies * x[..., None])],
        dim=-1,
    )
    return torch.cat([y_enc, x_enc], dim=-1)


def get_positional_encodings_from_coords(
    x: torch.Tensor,
    y: torch.Tensor,
    height: int,
    width: int,
    num_frequencies: int,
) -> torch.Tensor:
    """Return positional encodings for arbitrary pixel coordinates.

    Args:
        x: Pixel x coordinates in [0, width - 1].
        y: Pixel y coordinates in [0, height - 1].
    """
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    if height > 1:
        y = y / (height - 1)
    if width > 1:
        x = x / (width - 1)

    frequencies = (2.0 ** torch.arange(num_frequencies, device=x.device, dtype=torch.float32)) * torch.pi
    y_enc = torch.cat(
        [torch.sin(frequencies * y[..., None]), torch.cos(frequencies * y[..., None])],
        dim=-1,
    )
    x_enc = torch.cat(
        [torch.sin(frequencies * x[..., None]), torch.cos(frequencies * x[..., None])],
        dim=-1,
    )
    return torch.cat([y_enc, x_enc], dim=-1)
