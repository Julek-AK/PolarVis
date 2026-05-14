
# Builtins
from pathlib import Path
from PIL import Image

# External Libraries
from numpy.typing import NDArray

# Internal Support
from processing import visualisation as vis


VISUALISATION_FUNCS = {
    "Pure Intensity": vis.pure_intensity,
    "Pure DoLP": vis.pure_DoLP,
    "Pure AoP": vis.pure_theta,
    # "Tinted Theta": vis.tinted_theta,  # Currently deprecated since this visualisation is awful to look at
    # "Tinted DoLP": vis.tinted_DoLP,  # Currently deprecated since this visualisation is awful to look at
    "Full Polarimetric Colormap": vis.polarimetric_colormap,
    "Polar Only": vis.polar_data 
}


def list_visualisations() -> list:
    return list(VISUALISATION_FUNCS.keys())


def generate_visualisation(name: str, img_data: NDArray, **kwargs) -> Image.Image:
    func = VISUALISATION_FUNCS.get(name)
    if func is None:
        raise ValueError(f"[Visualisation] Unknown visualisation type: {name}")
    return func(img_data, **kwargs)

