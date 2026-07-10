
# Builtin
import subprocess
from pathlib import Path

# External
from numpy.typing import NDArray
import numpy as np
import cv2

# Internal


def check_ffmpeg_available() -> bool:
    """Checks whether FFmpeg is installed and available."""

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )

        return result.returncode == 0

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class VideoReader:
    """
    OpenCV based video reader.
    Yields frames as numpy arrays.
    """
    def __init__(self, filepath: Path, grayscale: bool = True):

        self.filepath = Path(filepath)
        self.grayscale = grayscale

        self.capture = cv2.VideoCapture(str(self.filepath))

        if not self.capture.isOpened():
            raise IOError(f"[VideoReader] Could not open video: {self.filepath}")

    @property
    def fps(self) -> float:
        return self.capture.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int( self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    def read(self) -> NDArray | None:
        """Reads a single frame. Returns frame array or None when video ends."""

        success, frame = self.capture.read()

        if not success:
            return None
        
        if self.grayscale:
            frame = frame[:, :, 0]  # Assumes all channels are same

        return frame

    def __iter__(self):

        while True:

            frame = self.read()

            if frame is None:
                break

            yield frame

    def close(self):
        """
        Releases video resources.
        """

        if self.capture:
            self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class VideoWriter:
    """
    Streams numpy frames into an FFmpeg encoder.
    Expected frame format:
        H x W x 3 uint8 RGB array
    """
    def __init__(self, output_path: Path, resolution: tuple[int, int], fps: float, codec: str = "libx264"):

        self.output_path = Path(output_path)
        self.width, self.height = resolution
        self.fps = fps
        self.codec = codec

        self.process: subprocess.Popen | None = None


    def open(self):
        """Starts FFmpeg process."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-y",

            # Input stream
            "-f",
            "rawvideo",

            "-vcodec",
            "rawvideo",

            "-pix_fmt",
            "rgb24",

            "-s",
            f"{self.width}x{self.height}",

            "-r",
            str(self.fps),

            "-i",
            "-",

            # Encoding
            "-c:v",
            self.codec,

            "-pix_fmt",
            "yuv420p",

            str(self.output_path)
        ]

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def write(self, frame: NDArray):
        """Writes one frame to FFmpeg. frame: RGB uint8 image (H,W,3)"""

        if self.process is None:
            raise RuntimeError("[VideoWriter] VideoWriter has not been opened.")

        if self.process.stdin is None:
            raise RuntimeError("[VideoWriter] FFmpeg stdin unavailable.")

        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"[VideoWriter] Invalid frame shape {frame.shape}, expected {(self.height, self.width, 3)}.")

        self.process.stdin.write(frame.tobytes())

    def close(self):
        """Finishes encoding."""

        if self.process is None:
            return

        if self.process.stdin:
            self.process.stdin.close()

        self.process.wait()

        self.process = None

    def abort(self):
        """Stops encoding immediately."""

        if self.process is None:
            return

        self.process.kill()
        self.process = None