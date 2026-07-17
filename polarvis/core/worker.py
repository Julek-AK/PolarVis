
# Builtin

# External
from PyQt6.QtCore import QThread, pyqtSignal

# Internal


class ProcessingWorker(QThread):

    _counter = 0

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, function):
        super().__init__()

        self.function = function

        ProcessingWorker._counter += 1
        self.worker_id = ProcessingWorker._counter
        print(f"[Pipeline] Created worker {self.worker_id}.")

    def run(self):

        print(f"[Pipeline] Worker {self.worker_id} started.")

        try:
            result = self.function()
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

        finally:
            print(f"[Pipeline] Worker {self.worker_id} exiting.")

    def report_progress(self, current, total):
        self.progress.emit(current, total)

    def __del__(self):
        print(f"[Pipeline] Worker {self.worker_id} destroyed.")
