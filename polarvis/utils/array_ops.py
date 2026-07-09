"""
Utility functions for array operations
"""

# External Imports
import numpy as np
from numpy.typing import NDArray
import torch


def raw_to_metapixels(image_arr: NDArray) -> NDArray:
    """
    Convert an image array into an array of 2x2 metapixels
    """
    H, W = image_arr.shape
    metapx_arr = image_arr.reshape(H // 2, 2, W // 2, 2).swapaxes(1, 2)
    return metapx_arr


def raw_to_metapixel_channels(image_arr: NDArray) -> NDArray: 
    """
    Convert raw image to (H//2, W//2, 4) metapixels
    Channel order:
      * 0 = (0,0) top-left
      * 1 = (0,1) top-right
      * 2 = (1,0) bottom-left
      * 3 = (1,1) bottom-right
    """
    H, W = image_arr.shape
    assert H % 2 == 0 and W % 2 == 0

    out = np.empty((H // 2, W // 2, 4), dtype=image_arr.dtype)

    out[..., 0] = image_arr[0::2, 0::2]
    out[..., 1] = image_arr[0::2, 1::2]
    out[..., 2] = image_arr[1::2, 0::2]
    out[..., 3] = image_arr[1::2, 1::2]

    return out

def raw_to_metapixel_channels_torch(image_arr: torch.Tensor) -> torch.Tensor: 
    """
    Convert a raw torch tensor to (H//2, W//2, 4) metapixels
    Channel order:
      * 0 = (0,0) top-left
      * 1 = (0,1) top-right
      * 2 = (1,0) bottom-left
      * 3 = (1,1) bottom-right
    """

    return torch.stack(
        (
            image_arr[0::2, 0::2],
            image_arr[0::2, 1::2],
            image_arr[1::2, 0::2],
            image_arr[1::2, 1::2],
        ),
        dim=-1
    )

def raw_to_metapixel_channels_batch_torch(images: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch tensor into metapixel channels
    Input: (N,H,W)
    Output: (N,H//2,W//2,4)
    """

    return torch.stack(
        (
            images[:,0::2,0::2],
            images[:,0::2,1::2],
            images[:,1::2,0::2],
            images[:,1::2,1::2],
        ),
        dim=-1
    )

def raw_to_metapixels_3d(image_arr: NDArray) -> NDArray:
    """
    Convert a stack of image arrays into a stack of arrays of 2x2 metapixels
    """
    H, W, N = image_arr.shape
    metapx_arr = image_arr.reshape(H // 2, 2, W // 2, 2, N).swapaxes(1, 2)
    return metapx_arr


def metapixels_to_raw(metapx_arr: NDArray) -> NDArray:
    """
    Convert an array of 2x2 metapixels back into the original image array
    """
    H, W, _, _ = metapx_arr.shape
    image_arr = metapx_arr.swapaxes(1, 2).reshape(H * 2, W * 2)
    return image_arr


def metapixels_to_raw_3d(metapx_arr: NDArray) -> NDArray:
    """
    Convert a stack of arrays of 2x2 metapixels back into the original image array stack
    """
    H, W, _, _, N = metapx_arr.shape
    image_arr = metapx_arr.swapaxes(1, 2).reshape(H * 2, W * 2, N)
    return image_arr


def metapixels_to_pixel_list(metapx_arr: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Convert an array of 2x2 metapixels into 4 arrays, one for each of the corners
    """
    H, W, _, _ = metapx_arr.shape
    flat = metapx_arr.reshape(-1, 2, 2)

    top_left = flat[:, 0, 0]
    top_right = flat[:, 0, 1]
    bottom_left = flat[:, 1, 0]
    bottom_right = flat[:, 1, 1]

    return top_left, top_right, bottom_left, bottom_right


def pixel_list_to_metapixels(top_left: NDArray, top_right: NDArray, bottom_left: NDArray, bottom_right: NDArray, H: int, W: int):
    """
    Convert arrays for all the metapixel corners into a an array of 2x2 metapixels
    """
    N = H * W

    flat = np.zeros((N, 2, 2))
    flat[:, 0, 0] = top_left
    flat[:, 0, 1] = top_right
    flat[:, 1, 0] = bottom_left
    flat[:, 1, 1] = bottom_right

    metapx_arr = flat.reshape(H, W, 2, 2)

    return metapx_arr


def simulate_image(data: NDArray) -> NDArray:
    """
    Simulate an expected image given the data about polarized and unpolarized intensity and angle
    Note that this is NOT the exact definition of Malus' law, as in the camera sensor there is a polarized
    filter in front of every pixel, which mandatorily halves the intensity of incoming unpolarized flux
    """

    N = data.shape[0]
    assert data.shape == (N, N, 3), "data array must be of shape (data_size, data_size, 3)"

    I_unpol = data[..., 0]
    I_pol   = data[..., 1]
    theta   = data[..., 2]

    phi = np.array([np.pi/2, np.pi/4, -np.pi/4, 0.0])
    out = np.zeros((2*N, 2*N), dtype=np.float32)

    cos_sq = np.cos(phi.reshape(1, 1, 4) - theta[..., None]) ** 2
    I = 0.5 * I_unpol[..., None] + I_pol[..., None] * cos_sq

    out[0::2, 0::2] = I[..., 0]  # top-left
    out[0::2, 1::2] = I[..., 1]  # top-right
    out[1::2, 0::2] = I[..., 2]  # bottom-left
    out[1::2, 1::2] = I[..., 3]  # bottom-right

    return out

