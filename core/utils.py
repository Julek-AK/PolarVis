"""
Bunch of helper functions for all kinds of things,
primarily array manipulations
"""

# External Imports
import numpy as np
import torch


def raw_to_metapixels(image_arr):
    """
    Converts an image array into an array of 2x2 metapixels
    """
    H, W = image_arr.shape
    metapx_arr = image_arr.reshape(H // 2, 2, W // 2, 2).swapaxes(1, 2)
    return metapx_arr


def raw_to_metapixels_3d(image_arr):
    """
    Converts a stack of image arrays into a stack of arrays of 2x2 metapixels
    """
    H, W, N = image_arr.shape
    metapx_arr = image_arr.reshape(H // 2, 2, W // 2, 2, N).swapaxes(1, 2)
    return metapx_arr


def metapixels_to_raw(metapx_arr):
    """
    Converts an array of 2x2 metapixels back into the original image array
    """
    H, W, _, _ = metapx_arr.shape
    image_arr = metapx_arr.swapaxes(1, 2).reshape(H * 2, W * 2)
    return image_arr


def metapixels_to_raw_3d(metapx_arr):
    """
    Converts a stack of arrays of 2x2 metapixels back into the original image array stack
    """
    H, W, _, _, N = metapx_arr.shape
    image_arr = metapx_arr.swapaxes(1, 2).reshape(H * 2, W * 2, N)
    return image_arr


def metapixels_to_pixel_list(metapx_arr):
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


def pixel_list_to_metapixels(top_left, top_right, bottom_left, bottom_right, H, W):
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


def describe(arr):
    """
    Computes the minimum, maximum, Q1, Q2 and Q3 of an array
    """
    minimum = np.min(arr)
    Q1 = np.percentile(arr, 25)
    Q2 = np.median(arr)
    Q3 = np.percentile(arr, 75)
    maximum = np.max(arr)

    return minimum, Q1, Q2, Q3, maximum


def cuda_check():
    availability = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0)

    print(f"CUDA avaialable: {availability}")
    print(f"Used device: {name}")