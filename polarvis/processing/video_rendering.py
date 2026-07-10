
# Builtin
from typing import Dict

# External
from numpy.typing import NDArray

# Internal
from ..processing.image_visualisation import *


class VisualisationRenderer:

    def render(self, result: NDArray, modes: list, cmaps: Dict[str, str]) -> Dict[str, NDArray]:

        frames = {}

        if 'intensity' in modes:
            frames['intensity'] = self.render_intensity(result, cmaps['intensity'])

        if 'dolp' in modes:
            frames['dolp'] = self.render_dolp(result, cmaps['dolp'])

        if 'aop' in modes:
            frames['aop'] = self.render_aop(result, cmaps['aop'])

        if 'polarimetric' in modes:
            frames['polarimetric'] = self.render_polarimetric(result, cmaps['aop'])

        if 'polar' in modes:
            frames['polar'] = self.render_polar(result, cmaps['aop'])

        return frames
    
    def render_intensity(self, img_data: NDArray, cmap: str) -> NDArray:
        return pure_intensity(img_data, cmap, raw=True)
    
    def render_dolp(self, img_data: NDArray, cmap: str):
        return pure_DoLP(img_data, cmap, raw=True)

    def render_aop(self, img_data: NDArray, cmap: str):
        return pure_AoP(img_data, cmap, raw=True)
    
    def render_polarimetric(self, img_data: NDArray, cmap: str):
        return polarimetric_colormap(img_data, cmap, raw=True)

    def render_polar(self, img_data: NDArray, cmap: str):
        return polar_data(img_data, cmap, raw=True)