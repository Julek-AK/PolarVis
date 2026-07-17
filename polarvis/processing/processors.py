
# Builtin
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path

# External
import numpy as np
from numpy.typing import NDArray


# Internal
from ..processing.torch_backend import (
    calibrated_resolve_polarization,
    batch_resolve_polarization,
)
from ..processing.video_rendering import VisualisationRenderer
from ..processing.calibration import Calibration

from ..io.video import VideoReader
from ..io.video import VideoWriter


    
class SingleProcessor:
    def __init__(self, calibration, device):

        self.calibration = calibration
        self.device = device

    def process(self, image):

        result = calibrated_resolve_polarization(image, self.calibration, self.device)

        return result
    

class BatchProcessor:
    def __init__(self, calibration, device):
        
        self.calibration = calibration
        self.device = device

    def process_batch(self, tasks: List[Tuple[NDArray, str]]) -> List[Tuple[NDArray, str]]:

        images = []
        ids = []
        results = []

        for task in tasks:
            images.append(task[0])
            ids.append(task[1])
            
        images = np.stack(images, axis=0)
    
        batch = batch_resolve_polarization(images, self.calibration, self.device)
        batch_list = list(batch)

        for result, cache_id in zip(batch_list, ids):
            results.append((result, cache_id))
            
        return results


@dataclass
class VideoProcessConfig:
    calibration: Calibration

    video_path: Path
    output_directory: Path

    visualisations: list[str]
    cmaps: dict

    batch_size: int


class VideoProcessor:
    def __init__(self, config, device):
        
        self.config = config
        self.device = device

        self.processor = BatchProcessor(self.config.calibration, self.device)
        self.vis_renderer = VisualisationRenderer()
        self._writers = {}

        self._cancelled = False

    def process_video(self):

        with VideoReader(self.config.video_path) as reader:

            self._writers = writers = self._create_writers(
                reader.width,
                reader.height,
                reader.fps
            )

            batch = []

            for frame in reader:

                if self._cancelled:
                    break

                batch.append(frame)

                if len(batch) == self.config.batch_size:
                    self._process_batch(batch, writers)
                    batch.clear()
            
            if batch:
                self._process_batch(batch, writers)

            for writer in writers.values():
                writer.close()

    def _process_batch(self, frames, writers):

        # Create dummy IDs
        ids = ['_'] * len(frames)
        
        results = self.processor.process_batch(list(zip(frames, ids)))

        for result in results:
            vis = self.vis_renderer.render(
                result[0],
                self.config.visualisations,
                self.config.cmaps,
            )
            
            for name, frame in vis.items():
                writers[name].write(frame)

    def _create_writers(self, width, height, fps):

        writers = {}

        for visualisation in self.config.visualisations:

            output_path = (self.config.output_directory /f'{visualisation}.mp4')

            writer = VideoWriter(
                output_path=output_path,
                resolution=(width, height),
                fps=fps
            )

            writer.open()
            writers[visualisation] = writer

        return writers
    
    def cancel(self):

        self._cancelled = True

        for writer in self._writers.values():
            writer.abort()



