
# Builtins
from dataclasses import dataclass
from typing import Callable, Optional

# External Libraries
from numpy.typing import NDArray

# Internal Support
from ..processing import image_visualisation as vis
from ..processing import image_legend as legend
from ..app.config.settings import settings


@dataclass
class VisualisationDefinition:
    name: str
    generator: Callable[..., vis.VisualisationResult]
    legend_renderer: Optional[Callable] = None
    colormap : Optional[str] = None


VISUALISATIONS: dict[str, VisualisationDefinition] = {
    'Pure Intensity': VisualisationDefinition(
        name="Pure Intensity",
        generator=vis.pure_intensity,
        legend_renderer=legend.scalar_legend,
        colormap=settings.get('visualization.colormaps.intensity'),
    ),

    'Pure DoLP': VisualisationDefinition(
        name="Pure DoLP",
        generator=vis.pure_DoLP,
        legend_renderer=legend.scalar_legend,
        colormap=settings.get('visualization.colormaps.dolp'),
    ),

    'Pure AoP': VisualisationDefinition(
        name="Pure AoP",
        generator=vis.pure_AoP,
        legend_renderer=legend.angle_legend,
        colormap=settings.get('visualization.colormaps.aop'),
    ),

    'Full Polarimetric Colormap': VisualisationDefinition(
        name="Full Polarimetric Colormap",
        generator=vis.polarimetric_colormap,
        legend_renderer=legend.polarimetric_legend,
    ),

    'Polar Only': VisualisationDefinition(
        name="Polar Only",
        generator=vis.polar_data,
        legend_renderer=legend.polar_only_legend,
    ),
}


def list_visualisations() -> list[str]:
    """Returns all available visualisation names."""
    return list(VISUALISATIONS.keys())


def get_visualisation(name: str) -> VisualisationDefinition:
    """Returns the visualisation definition corresponding
    to the requested UI/display name."""
    vis_def = VISUALISATIONS.get(name)

    if vis_def is None:
        raise ValueError(f"[Visualisation] Unknown visualisation type: {name}")

    return vis_def


def generate_visualisation(
    name: str,
    img_data: NDArray,
    **kwargs
) -> vis.VisualisationResult:
    
    vis_def = get_visualisation(name)
    
    # Update colormap from settings
    if vis_def.name in {"Pure Intensity", "Pure DoLP", "Pure AoP"}:
        kwargs['cmap'] = vis_def.colormap

    return vis_def.generator(img_data, **kwargs)


def generate_legend(
    name: str,
    image,
    result: vis.VisualisationResult,
    **kwargs
):
    
    vis_def = get_visualisation(name)

    if vis_def.legend_renderer is None:
        return image

    return vis_def.legend_renderer(
        image=image.copy(),
        result=result,
        **kwargs
    )
