"""
Here we keep various functions used for interesting data display options
things like pure intensity display, DoLP filters, polarization angle filters etc.
currently all operates under uint8
all of these take in arrays of processed image data (i.e. one where each pixel has I_unpol, I_pol, theta_pol
"""

# Builtins
from PIL import Image
import colorsys

# External Libraries
import numpy as np
import torch
from matplotlib import pyplot as plt

# Internal Support
from core.utils import *

# Convenient color conversions
hsv_to_rgb_vec = np.vectorize(colorsys.hsv_to_rgb)


def cmap_to_img(img_arr, cmap):
    """
    creates a colormap image from a 0-1 scaled array
    """
    colormap = plt.get_cmap(cmap)
    colored = colormap(img_arr)
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(colored_uint8)

    return img


def normalize_arr(img_arr):
    """
    scales the array from 0-max to within 0-1
    """
    arr_norm = img_arr / np.max(img_arr)
    arr_norm = np.clip(arr_norm, 0, 1)

    return arr_norm


def pure_intensity(img_data, cmap='magma'):
    """
    Creates a pure colormap of image intensity
    """
    intensity = np.maximum(img_data[..., 0] + img_data[..., 1], 1E-8)
    intensity_norm = normalize_arr(intensity)

    image = cmap_to_img(intensity_norm, cmap)
    return image


def pure_DoLP(img_data, cmap='viridis'):
    """
    Creates a pure colormap of image degree of linear polarization
    """
    intensity = np.maximum(img_data[..., 0] + img_data[..., 1], 1E-8)
    dolp = img_data[..., 1] / intensity
    dolp_norm = normalize_arr(dolp)  # Should be already normalized, but better make sure

    image = cmap_to_img(dolp_norm, cmap)
    return image


def pure_theta(img_data, cmap='hsv'):
    """
    Creates a pure colormap of image polarization angle, from 0 to pi
    """
    angle = np.mod(img_data[..., 2], np.pi)
    angle_norm = angle / np.pi

    image = cmap_to_img(angle_norm, cmap)
    return image


def tinted_theta(img_data, hue=0):
    """
    Creates a pure intensity image, adds color depending on polarization angle
    """
    intensity = np.maximum(img_data[..., 0] + img_data[..., 1], 1E-8)
    brightness = normalize_arr(intensity)

    angle = np.mod(img_data[..., 2], np.pi)
    saturation = angle / np.pi

    r, g, b = hsv_to_rgb_vec(hue, saturation, brightness)
    rgb = np.stack([r, g, b], axis=-1)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(rgb_uint8)

    return image


def tinted_DoLP(img_data, hue=0):
    """
    Creates a pure intensity image, adds color depending on degree of linear polarization
    """
    intensity = np.maximum(img_data[..., 0] + img_data[..., 1], 1E-8)
    brightness = normalize_arr(intensity)

    dolp = img_data[..., 1] / intensity
    saturation = normalize_arr(dolp)  # Should be already normalized, but better make sure

    r, g, b = hsv_to_rgb_vec(hue, saturation, brightness)
    rgb = np.stack([r, g, b], axis=-1)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(rgb_uint8)

    return image


def polarimetric_colormap(img_data):
    """
    Creates an image with total intensity as brightness, DoLP as saturation and polarization angle as hue
    """
    intensity = np.maximum(img_data[..., 0] + img_data[..., 1], 1E-8)
    brightness = normalize_arr(intensity)

    dolp = img_data[..., 1] / intensity
    saturation = normalize_arr(dolp)  # Should be already normalized, but better make sure

    angle = np.mod(img_data[..., 2], np.pi)
    hue = angle / np.pi

    r, g, b = hsv_to_rgb_vec(hue, saturation, brightness)
    rgb = np.stack([r, g, b], axis=-1)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(rgb_uint8)

    return image


def polar_data(img_data):
    """
    Displays exclusively polarization data, neglecting original image brightness
    """
    intensity = np.maximum(img_data[..., 0] + img_data[..., 1], 1E-8)
    brightness = np.ones_like(intensity)  # Brightness override

    dolp = img_data[..., 1] / intensity
    saturation = normalize_arr(dolp)  # Should be already normalized, but better make sure

    angle = np.mod(img_data[..., 2], np.pi)
    hue = angle / np.pi

    r, g, b = hsv_to_rgb_vec(hue, saturation, brightness)
    rgb = np.stack([r, g, b], axis=-1)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(rgb_uint8)

    return image