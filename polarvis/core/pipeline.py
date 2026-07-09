
# Builtin

# External
import torch
import numpy as np

# Internal
from ..core.image_validation import validate_calibration, ValidationError, ValidationWarning
from .worker import ProcessingWorker
from .processors import SingleProcessor, BatchProcessor, VideoProcessor
from ..app.config.settings import settings


class Pipeline:

    def __init__(self):

        self._workers = set()
        
        if settings.get('processing.use_gpu'):
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print("[Pipeline] Failed to detect cuda, falling back to cpu.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")


    def single_process(self, image, calibration):

        validate_calibration(image, calibration)

        processor = SingleProcessor(calibration, self.device)

        return processor.process(image)

    def start_single_processing(self, image, calibration):

        def task():  # Abstract wrapper, such that now the worker doesn't interract with input arguments
            return self.single_process(image, calibration)
        
        def cleanup(_=None):
            self._workers.discard(worker)
            print(f"[Pipeline] Worker {worker.worker_id} removed.")

        worker = ProcessingWorker(task)
        worker.finished.connect(cleanup)
        worker.error.connect(lambda msg: print(f"[Pipeline] Worker {worker.worker_id} errored: {msg}"))
        worker.error.connect(cleanup)

        self._workers.add(worker)

        return worker

    def batch_process(self, image_list, calibration):
        
        validated_image_list = []

        for image in image_list:
            try:
                validate_calibration(image, calibration)
            except ValidationError as e:
                print(f"[Pipeline] Skipping over an un-validated image: {e}")
                continue
            validated_image_list.append(image)

        images = np.stack(validated_image_list, axis=0)

        processor = BatchProcessor(calibration, self.device)

        return processor.process(images)

    def start_batch_processing(self, image_list, calibration):
        
        def task():
            return self.batch_process(image_list, calibration)
        
        worker = ProcessingWorker(task)

        return worker

    def video_process(self, file_in, calibration):
        ...

    def start_video_processing(self, file_in, calibration):
        ...

