"""
Here we keep various functions used for interesting data display options
things like pure intensity display, DoLP filters, polarization angle filters etc.
currently all operates under uint8
all of these take in arrays of processed image data (i.e. one where each pixel has I_unpol, I_pol, theta_pol
"""

# Builtins
from PIL import Image
from dataclasses import dataclass

# External Libraries
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt

# Internal Support
from ..utils.array_ops import *
from ..utils.color_ops import hsv_to_rgb_vec



@dataclass
class VisualisationResult:
    image: Image.Image
    cmap: str | None = None
    label: str | None = None


def cmap_to_arr(img_arr: NDArray, cmap: str) -> NDArray:
    """
    creates an image-ready array from a 0-1 scaled array
    """
    colormap = plt.get_cmap(cmap)
    colored = colormap(img_arr)
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)

    return colored_uint8


def pure_intensity(img_data: NDArray, cmap: str ='gray', raw=False) -> VisualisationResult | NDArray:
    """
    Creates a pure colormap of image intensity
    """
    intensity = img_data[..., 0] / 2
    arr = cmap_to_arr(intensity, cmap)
    
    if raw:
        return arr
    
    image = Image.fromarray(arr)
    return VisualisationResult(
        image=image,
        cmap=cmap,
        label="Intensity",
    )


def pure_DoLP(img_data: NDArray, cmap: str = 'viridis', raw=False) -> VisualisationResult | NDArray:
    """
    Creates a pure colormap of image degree of linear polarization
    """
    dolp = img_data[..., 1]
    arr = cmap_to_arr(dolp, cmap)

    if raw:
        return arr
    
    image = Image.fromarray(arr)
    return VisualisationResult(
        image=image,
        cmap=cmap,
        label="DoLP"
    )


def pure_AoP(img_data: NDArray, cmap: str = 'hsv', raw=False) -> VisualisationResult | NDArray:
    """
    Creates a pure colormap of image polarization angle, from 0 to pi
    """
    angle = np.mod(img_data[..., 2], np.pi)
    angle_norm = angle / np.pi  # Normalize to [0, 1]
    arr = cmap_to_arr(angle_norm, cmap)

    if raw:
        return arr
    
    image = Image.fromarray(arr)
    return VisualisationResult(
        image=image,
        cmap=cmap,
        label="AoP"
    )


def polarimetric_colormap(img_data: NDArray, cmap: str = 'hsv', raw=False) -> VisualisationResult | NDArray:
    """
    Creates an image with total intensity as brightness, DoLP as saturation and polarization angle as hue
    """
    if cmap != 'hsv':
        raise NotImplementedError("[Visualisation] Polarimetric visualisation techniques other than standard HSV are not implemented yet.")

    hsv = np.ones_like(img_data)

    hsv[..., 2] = img_data[..., 0] / 2  # Intensity

    hsv[..., 1] = img_data[..., 1]  # DOLP

    angle = np.mod(img_data[..., 2], np.pi)  # Theta
    hsv[..., 0] = angle / np.pi

    rgb = hsv_to_rgb_vec(hsv)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    if raw:
        return rgb_uint8
    
    image = Image.fromarray(rgb_uint8)
    return VisualisationResult(
        image=image,
        cmap=cmap,
        label="AoP DoLP Intensity"
    )


def polar_data(img_data: NDArray, cmap: str = 'hsv', raw=False) -> VisualisationResult | NDArray:
    """
    Displays exclusively polarization data, neglecting original image brightness
    """
    if cmap != 'hsv':
        raise NotImplementedError("[Visualisation] Polarimetric visualisation techniques other than standard HSV are not implemented yet.")

    hsv = np.ones_like(img_data)  # Brightness override

    hsv[..., 1] = img_data[..., 1]  # DOLP

    angle = np.mod(img_data[..., 2], np.pi)  # Theta
    hsv[..., 0] = angle / np.pi

    rgb = hsv_to_rgb_vec(hsv)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    if raw:
        return rgb_uint8

    image = Image.fromarray(rgb_uint8)
    return VisualisationResult(
        image=image,
        cmap=cmap,
        label="AoP DoLP"
    )
