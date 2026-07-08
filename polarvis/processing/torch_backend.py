
# Builtin

# External
import torch
import numpy as np
from numpy.typing import NDArray

# Internal
from ..utils.array_ops import raw_to_metapixel_channels_torch, raw_to_metapixel_channels_batch_torch
from ..processing.calibration import Calibration



def calibrated_resolve_polarization(image_arr: NDArray, cal: Calibration, device: torch.device) -> NDArray:
    """
    Use a calibration file to resolve the polarization state of an image.
    Return an image array shaped (H//2, W//2, 3) with channels:
    Channel 0 - Intensity, scaled [0, 1] according to calibration bit depth
    Channel 1 - DOLP, [0, 1]
    Channel 2 - Polarization angle, [0, pi]
    """

    # Move everything to torch
    image = torch.as_tensor(image_arr, dtype=torch.float32, device=device)  # (H, W)
    dark = torch.as_tensor(cal.dark_frame, dtype=torch.float32, device=device)  # (H, w)
    flat = torch.as_tensor(cal.flat_field, dtype=torch.float32, device=device)  # (H, W)
    S = torch.as_tensor(cal.stokes_reconstruction, dtype=torch.float32, device=device)  # (H//2, W//2, 3, 4)

    # Dark current correction
    image = image - dark

    # Flat field normalization
    image = image / flat

    # Reshape
    metapx = raw_to_metapixel_channels_torch(image)  # (H//2, W//2, 4)

    # Stokes reconstruction
    # S:      (H//2, W//2, 3, 4)
    # metapx: (H//2, W//2, 4)
    # output: (H//2, W//2, 3)

    stokes = torch.einsum(
        'hwab, hwb -> hwa',
        S,
        metapx
    )

    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]

    I = s0 / (2**cal.bit_depth - 1)

    DoLP = torch.sqrt(s1**2 + s2**2) / (s0 + 1e-8)
    DoLP = torch.clamp(DoLP, 0, 1)

    AoP = 0.5 * torch.atan2(s2, s1)
    AoP = torch.remainder(AoP, torch.pi)

    output = torch.stack((I, DoLP, AoP), dim=-1)

    # Return a numpy array
    return output.detach().cpu().numpy()


def batch_resolve_polarization(images: NDArray, cal: Calibration, device: torch.device) -> NDArray:
    """
    Process a batch of images simultaneously.
    Input: (N, H, W)
    Output: (N, H//2, W//2, 3)
    """

    # Move everything to torch
    batch = torch.as_tensor(images, dtype=torch.float32, device=device)  # (N, H, W)
    dark = torch.as_tensor(cal.dark_frame, dtype=torch.float32, device=device)  # (H, w)
    flat = torch.as_tensor(cal.flat_field, dtype=torch.float32, device=device)  # (H, W)
    S = torch.as_tensor(cal.stokes_reconstruction, dtype=torch.float32, device=device)  # (H//2, W//2, 3, 4)

    # Broadcast over batch dimension
    batch = batch - dark
    batch = batch / flat

    metapx = raw_to_metapixel_channels_batch_torch(batch)

    # Stokes reconstruction
    # S:      (H//2, W//2, 3, 4)
    # metapx: (N, H//2, W//2, 4)
    # output: (N, H//2, W//2, 3)

    stokes = torch.einsum(
        'hwab, nhwb -> nhwa',
        S,
        metapx
    )


    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]

    I = s0 / (2**cal.bit_depth - 1)

    DoLP = torch.sqrt(s1**2 + s2**2) / (s0 + 1e-8)
    DoLP = torch.clamp(DoLP, 0, 1)

    AoP = 0.5 * torch.atan2(s2, s1)
    AoP = torch.remainder(AoP, torch.pi)

    output = torch.stack((I, DoLP, AoP), dim=-1)

    # Return a numpy array
    return output.detach().cpu().numpy()
