
# Builtin

# External
import torch

# Internal
from ..core.image_validation import validate_calibration, ValidationError, ValidationWarning
from ..core.worker import ProcessingWorker

from ..utils.misc import split_batches
from ..processing.processors import SingleProcessor, BatchProcessor, VideoProcessor, VideoProcessConfig

from ..app.config.settings import settings


class Pipeline:

    def __init__(self, cache_manager):

        self.cache_manager = cache_manager
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
            print(f"[Pipeline] Worker {worker.worker_id} finished.")

        worker = ProcessingWorker(task)
        worker.finished.connect(cleanup)
        worker.error.connect(lambda msg: print(f"[Pipeline] Worker {worker.worker_id} errored: {msg}"))
        worker.error.connect(cleanup)

        self._workers.add(worker)

        return worker

    def batch_process(self, task_list, calibration, cal_id):
        
        validated_task_list = []

        for task in task_list:
            try:
                validate_calibration(task[0], calibration)
            except ValidationError as e:
                print(f"[Pipeline] Skipping over an un-validated image: {e}")
                continue
            validated_task_list.append(task)

        processor = BatchProcessor(calibration, self.device)

        batch_size = settings.get('processing.batch_size')
        for batch in split_batches(validated_task_list, batch_size):

            batch_results = processor.process_batch(batch)

            for result, cache_id in batch_results:
                ID = f"{cache_id}__{cal_id}" 
                self.cache_manager.save_array(ID, result)

        return True

    def start_batch_processing(self, task_list, calibration, cal_id):
        
        def task():
            return self.batch_process(task_list, calibration, cal_id)

        def cleanup(_=None):
            self._workers.discard(worker)
            print(f"[Pipeline] Worker {worker.worker_id} finished.")

        worker = ProcessingWorker(task)
        worker.finished.connect(cleanup)
        worker.error.connect(lambda msg: print(f"[Pipeline] Worker {worker.worker_id} errored: {msg}"))
        worker.error.connect(cleanup)

        self._workers.add(worker)

        return worker

    def video_process(self, config):

        processor = VideoProcessor(config, self.device)
        processor.process_video()

    def start_video_processing(self, file_in, dir_out, visualisations, calibration):
        
        # Set up the config object
        config = VideoProcessConfig(
            calibration=calibration,
            video_path=file_in,
            output_directory=dir_out,
            visualisations=visualisations,
            cmaps=settings.get('visualization.colormaps'),
            batch_size=settings.get('processing.batch_size'),
        )

        def task():
            return self.video_process(config)

        def cleanup(_=None):
            self._workers.discard(worker)
            print(f"[Pipeline] Worker {worker.worker_id} finished.")

        worker = ProcessingWorker(task)
        worker.finished.connect(cleanup)
        worker.error.connect(lambda msg: print(f"[Pipeline] Worker {worker.worker_id} errored: {msg}"))
        worker.error.connect(cleanup)

        self._workers.add(worker)

        return worker


