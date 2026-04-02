"""
low-level torch functions that process image arrays 
"""

from core.utils import *
import torch

import numpy as np
from numpy.typing import NDArray
from processing.calibration import Calibration


# =============================================
# CHATGPT CODE FOR SOLVING THE PROBLEM ANALYTICALLY
# =============================================
PHIS = torch.tensor([
    torch.pi / 2,   # channel 0 → top-left
    torch.pi / 4,   # channel 1 → top-right
    -torch.pi / 4,  # channel 2 → bottom-left
    0.0             # channel 3 → bottom-right
])

# TODO pseudo-inverse computation is completely independent from your image case,
# move its computation to pipeline initialisation
# also make sure this thing can handle batches, as for small cases cpu is far faster

def analytic_resolve_metapixels(image_arr_metapx, device="cpu", eps=1e-12):  # TODO deprecated format
    """
    image_arr_metapx: numpy or torch array of shape (H, W, 4) with the 4 measured intensities per metapixel
    Returns: numpy array (H, W, 3) with [I_unpol, I_pol, theta] per metapixel (theta in radians, in [0, pi))
    """
    # ensure torch tensor on device
    if not torch.is_tensor(image_arr_metapx):
        y = torch.tensor(image_arr_metapx, dtype=torch.float32, device=device)
    else:
        y = image_arr_metapx.to(device, dtype=torch.float32)
    H, W, n_ch = y.shape
    assert n_ch == 4, "expected 4 intensities per metapixel"

    # build design matrix X (4 x 3): cols = [1, cos(2phi), sin(2phi)]
    phis = PHIS.to(device)
    cos2phi = torch.cos(2.0 * phis)   # (4,)
    sin2phi = torch.sin(2.0 * phis)   # (4,)
    X = torch.stack([torch.ones_like(cos2phi), cos2phi, sin2phi], dim=1)  # (4,3)

    # compute pseudo-inverse once: pinv = (X^T X)^{-1} X^T  -> shape (3,4)
    XtX = X.t() @ X                    # (3,3)
    # small regularizer for numerical stability
    reg = eps * torch.eye(3, device=device)
    XtX_inv = torch.linalg.inv(XtX + reg)  # (3,3)
    pinv = XtX_inv @ X.t()                  # (3,4)

    # reshape y to (4, H*W)
    y_flat = y.view(-1, 4).t()  # (4, H*W)

    # solve for params: params = pinv @ y_flat -> (3, H*W)
    params = pinv @ y_flat

    A = params[0, :].view(H, W)   # 0.5 * (I_unpol + I_pol)
    C = params[1, :].view(H, W)   # B cos(2theta)
    D = params[2, :].view(H, W)   # -B sin(2theta)

    # B = sqrt(C^2 + D^2)
    B = torch.sqrt(C * C + D * D + eps)

    # I_pol = 2 * B
    I_pol = 2.0 * B

    # I_unpol = 2A - I_pol
    I_unpol = 2.0 * A - I_pol

    # compute 2*theta = atan2(-D, C)  -> theta = 0.5 * atan2(-D, C)
    two_theta = torch.atan2(-D, C)   # in (-pi, pi]
    theta = 0.5 * two_theta

    # normalize theta into [0, pi)
    theta = (theta % torch.pi)

    # clamp physically (optional): I_pol and I_unpol >= 0
    I_pol = torch.clamp(I_pol, min=0.0)
    I_unpol = torch.clamp(I_unpol, min=0.0)

    result = torch.stack([I_unpol, I_pol, theta], dim=-1)  # (H, W, 3)
    return result.cpu().numpy()


def analytic_resolve_metapixels_for_testing(img_arr: NDArray, **kwargs) -> NDArray:
    """
    Wrapper for the analytic_resolve_metapixels that ensures the function signature is
    as expected by the testing module
    """
    img_arr_metapx = raw_to_metapixel_channels(img_arr)
    return analytic_resolve_metapixels(img_arr_metapx, **kwargs)

# =============================================
# SLIDING SAMPLER WINDOW IMPLEMENTATION
# =============================================

def make_phi_map(H, W, device="cpu"):
    """
    Returns phi_map of shape (H, W)
    Angles are assigned per pixel and repeat spatially.
    """
    # TODO inject callibration offsets into here
    phi_tile = torch.tensor(  # explicitly construct using the channel definitions from PHIS
        [[PHIS[0],  PHIS[1]],
         [PHIS[2], PHIS[3]]],
        device=device
    )  # shape (2, 2)

    reps_y = (H + 1) // 2
    reps_x = (W + 1) // 2

    phi_map = phi_tile.repeat(reps_y, reps_x)
    return phi_map[:H, :W]


def sliding_window_polarization(  # TODO deprecated format
    image_arr,
    phi_map,
    k,
    device="cpu",
    eps=1e-12
):
    """
    image_arr : (H, W) numpy or torch
    phi_map   : (H, W) torch tensor
    k         : window size (k x k)

    Returns:
        (H-k+1, W-k+1, 3) tensor with
        [I_unpol, I_pol, theta]
    """

    # --- input handling ---
    if not torch.is_tensor(image_arr):
        image = torch.tensor(image_arr, dtype=torch.float32, device=device)
    else:
        image = image_arr.to(device, dtype=torch.float32)

    phi_map = phi_map.to(device)

    H, W = image.shape
    assert phi_map.shape == (H, W)
    assert k >= 2

    out_H = H - k + 1
    out_W = W - k + 1
    N = k * k

    # --- extract sliding windows ---
    # image_windows: (out_H, out_W, k, k)
    image_windows = image.unfold(0, k, 1).unfold(1, k, 1)
    phi_windows   = phi_map.unfold(0, k, 1).unfold(1, k, 1)

    # flatten window pixels
    y = image_windows.reshape(out_H, out_W, N)      # intensities
    phi = phi_windows.reshape(out_H, out_W, N)      # angles

    # --- build design matrix ---
    cos2 = torch.cos(2.0 * phi)
    sin2 = torch.sin(2.0 * phi)

    # X shape: (out_H, out_W, N, 3)
    X = torch.stack(
        [
            torch.ones_like(cos2),
            cos2,
            sin2
        ],
        dim=-1
    )

    # --- solve least squares in batch ---
    # Compute normal equations
    Xt = X.transpose(-1, -2)               # (..., 3, N)
    XtX = Xt @ X                           # (..., 3, 3)

    # regularization for stability
    eye = torch.eye(3, device=device)
    XtX = XtX + eps * eye

    XtX_inv = torch.linalg.inv(XtX)        # (..., 3, 3)
    Xty = Xt @ y.unsqueeze(-1)             # (..., 3, 1)

    params = (XtX_inv @ Xty).squeeze(-1)   # (..., 3)

    A = params[..., 0]
    C = params[..., 1]
    D = params[..., 2]

    # --- recover physical quantities ---
    B = torch.sqrt(C * C + D * D + eps)

    I_pol = 2.0 * B
    I_unpol = 2.0 * A - I_pol

    theta = 0.5 * torch.atan2(D, C)
    theta = theta % torch.pi

    I_pol = torch.clamp(I_pol, min=0.0)
    I_unpol = torch.clamp(I_unpol, min=0.0)

    return torch.stack([I_unpol, I_pol, theta], dim=-1)


def resolve_sliding_polarization(image_arr, k, device):
    H, W = image_arr.shape
    phi_map = make_phi_map(H, W, device=device)

    result = sliding_window_polarization(
        image_arr=image_arr,
        phi_map=phi_map,
        k=k,
        device=device
    )

    return result.cpu().numpy()


# =============================================
# CALIBRATED IMPLEMENTATION
# =============================================

def calibrated_resolve_polarization(image_arr: NDArray, cal: Calibration) -> NDArray:
    """
    Use a calibration file to resolve the polarization state of an image.
    Return an image array with three channels:
    Channel 0 - Intensity, scaled [0, 1] according to calibration bit depth
    Channel 1 - DOLP, [0, 1]
    Channel 2 - Polarization angle, [0, pi]
    """
    dark = cal.dark_frame.astype(np.float64)  # (H, W)
    flat = cal.flat_field.astype(np.float64)  # (H, W)
    S = cal.stokes_reconstruction.astype(np.float64) # (H//2, W//2, 3, 4)
    image_arr = image_arr.copy().astype(np.float64)

    # Dark current correction
    image_arr -= dark

    # Flat field normalization
    image_arr /= flat

    # Stokes parameter reconstruction
    metapx = raw_to_metapixel_channels(image_arr)  # (H//2, W//2, 4)
    stokes = np.einsum('hwab, hwb -> hwa', S, metapx)  # shape (H//2, W//2, 3)

    # Convert to output parameters
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]

    I_tot = s0
    I_scaled = I_tot / (2**cal.bit_depth - 1)

    DOLP = np.sqrt(s1**2 + s2**2) / (s0 + 1e-8)
    DOLP = np.clip(DOLP, 0, 1)

    theta = 0.5 * np.atan2(s2, s1)
    theta = np.mod(theta, np.pi)

    return np.stack((I_scaled, DOLP, theta), axis=-1)

