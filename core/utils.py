"""
Bunch of helper functions for all kinds of things,
primarily array manipulations
"""

# External Imports
import numpy as np
from numpy.typing import NDArray
from torch import cuda

def raw_to_metapixels(image_arr: NDArray) -> NDArray:
    """
    Converts an image array into an array of 2x2 metapixels
    """
    H, W = image_arr.shape
    metapx_arr = image_arr.reshape(H // 2, 2, W // 2, 2).swapaxes(1, 2)
    return metapx_arr

def raw_to_metapixel_list(image_arr: NDArray) -> NDArray:
    """
    Converts an image array into an array of 4-length lists for metapixels
    """
    H, W = image_arr.shape
    metapx_list_arr = (
        image_arr.reshape(H // 2, 2, W // 2, 2)
        .swapaxes(1, 2)
        .reshape(H // 2, W // 2, 4)
    )
    return metapx_list_arr

def raw_to_metapixels_3d(image_arr: NDArray) -> NDArray:
    """
    Converts a stack of image arrays into a stack of arrays of 2x2 metapixels
    """
    H, W, N = image_arr.shape
    metapx_arr = image_arr.reshape(H // 2, 2, W // 2, 2, N).swapaxes(1, 2)
    return metapx_arr


def metapixels_to_raw(metapx_arr: NDArray) -> NDArray:
    """
    Converts an array of 2x2 metapixels back into the original image array
    """
    H, W, _, _ = metapx_arr.shape
    image_arr = metapx_arr.swapaxes(1, 2).reshape(H * 2, W * 2)
    return image_arr


def metapixels_to_raw_3d(metapx_arr: NDArray) -> NDArray:
    """
    Converts a stack of arrays of 2x2 metapixels back into the original image array stack
    """
    H, W, _, _, N = metapx_arr.shape
    image_arr = metapx_arr.swapaxes(1, 2).reshape(H * 2, W * 2, N)
    return image_arr


def metapixels_to_pixel_list(metapx_arr: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Converts an array of 2x2 metapixels into 4 arrays, one for each of the corners
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
    Converts arrays for all the metapixel corners into a an array of 2x2 metapixels
    """
    N = H * W

    flat = np.zeros((N, 2, 2))
    flat[:, 0, 0] = top_left
    flat[:, 0, 1] = top_right
    flat[:, 1, 0] = bottom_left
    flat[:, 1, 1] = bottom_right

    metapx_arr = flat.reshape(H, W, 2, 2)

    return metapx_arr


def describe(arr: NDArray) -> tuple[float, float, float, float, float]:
    """
    Computes the minimum, maximum, Q1, Q2 and Q3 of an array
    """
    minimum = np.min(arr)
    Q1 = np.percentile(arr, 25)
    Q2 = np.median(arr)
    Q3 = np.percentile(arr, 75)
    maximum = np.max(arr)

    return minimum, Q1, Q2, Q3, maximum


def simulate_image(data: NDArray) -> NDArray:
    """
    Simulates an expected image given the data about polarized and unpolarized intensity and angle
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

    out[0::2, 0::2] = I[..., 0]
    out[0::2, 1::2] = I[..., 1]
    out[1::2, 0::2] = I[..., 2]
    out[1::2, 1::2] = I[..., 3]

    return out

def cuda_check() -> None:
    availability = cuda.is_available()
    name = cuda.get_device_name(0)

    print(f"CUDA avaialable: {availability}")
    print(f"Used device: {name}")