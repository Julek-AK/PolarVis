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


def pure_intensity(img_data, cmap='gist_gray'):
    """
    Creates a pure colormap of image intensity
    """
    intensity = img_data[..., 0] / 2
    image = cmap_to_img(intensity, cmap)
    return image


def pure_DoLP(img_data, cmap='viridis'):
    """
    Creates a pure colormap of image degree of linear polarization
    """
    dolp = img_data[..., 1]
    image = cmap_to_img(dolp, cmap)
    return image


def pure_theta(img_data, cmap='hsv'):
    """
    Creates a pure colormap of image polarization angle, from 0 to pi
    """
    angle = np.mod(img_data[..., 2], np.pi)
    angle_norm = angle / np.pi  # Normalize to [0, 1]

    image = cmap_to_img(angle_norm, cmap)
    return image


def tinted_theta(img_data, hue=0):
    """
    Creates a pure intensity image, adds color depending on polarization angle
    """
    brightness = img_data[..., 0] / 2 # Intensity

    angle = np.mod(img_data[..., 2], np.pi)  # Theta
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
    brightness = img_data[..., 0] / 2 # Intensity

    saturation = img_data[..., 1]  # DOLP

    r, g, b = hsv_to_rgb_vec(hue, saturation, brightness)
    rgb = np.stack([r, g, b], axis=-1)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(rgb_uint8)

    return image


def polarimetric_colormap(img_data):
    """
    Creates an image with total intensity as brightness, DoLP as saturation and polarization angle as hue
    """
    brightness = img_data[..., 0] / 2 # Intensity

    saturation = img_data[..., 1]  # DOLP

    angle = np.mod(img_data[..., 2], np.pi)  # Theta
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
    brightness = np.ones_like(img_data[..., 0])  # Brightness override

    saturation = img_data[..., 1]  # DOLP

    angle = np.mod(img_data[..., 2], np.pi)  # Theta
    hue = angle / np.pi

    r, g, b = hsv_to_rgb_vec(hue, saturation, brightness)
    rgb = np.stack([r, g, b], axis=-1)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(rgb_uint8)

    return image