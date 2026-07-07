
# Bultins
import os
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import json
import re
import unicodedata

# Internal support
from ..app.config.settings import settings
from ..app.paths import CALIBRATION_DIR
from ..processing.calibration import Calibration, CalibrationInput, CalibrationConstructor

# External libraries
import numpy as np
from numpy.typing import NDArray, DTypeLike
from PyQt6.QtWidgets import QWidget, QFileDialog
from PyQt6 import QtCore

# TODO improve robustness of metadata/config separation for calibration
# TODO create seperate serializable config parameters that also get embedded in the .calib files
# TODO serialize the used polarization model (sicne in the future many might be available)
# TODO serialize the actual calibration method used (since in the future many might be available)


FILE_FORMAT_VERSION = 1
DEFAULT_CALIBRATION_CHANNELS = settings.get('camera.channel_order')



class CalibrationManager(QtCore.QObject):
    """Orchestrates and manages loading, saving and computing calibration."""
    current_calibration_changed = QtCore.pyqtSignal(str)


    def __init__(self) -> None:
        """Initialize relevant fields, connect to the gui"""
        super().__init__()

        self.cal_dir = CALIBRATION_DIR
        self.cal_dir.mkdir(parents=True, exist_ok=True)

        if not self._has_default():
            default = self._create_default_calibration((2048, 2448))
            self.save_calibration(
                default,
                {   
                    'name': "Default (factory)",
                    'file_type': "polarimeter_calibration",
                    'format_version': FILE_FORMAT_VERSION,
                    'timestamp': "NA",
                    'shape': (2048, 2448),
                    'bit_depth': 8,
                }
            )

        self._metadata_cache: Dict[str, Dict] = {}

        self.current_calibration_id: Optional[str] = None
        self.current_calibration: Optional[Calibration] = None
        self.current_metadata: Optional[Dict] = None

        available = self.list_calibrations()
        if available: self.set_current_calibration(available[0])

    def set_current_calibration(self, cal_id: str):
        """Load and activate a calibration."""

        calibration, metadata = self.load_calibration(cal_id)

        self.current_calibration_id = cal_id
        self.current_calibration = calibration
        self.current_metadata = metadata

        self.current_calibration_changed.emit(cal_id)

    def require_current_calibration(self) -> Tuple[Calibration, str]:

        if self.current_calibration is None:
            raise RuntimeError(
                "[CalibrationManager] No calibration selected."
            )

        return self.current_calibration, self.current_calibration_id

    def refresh_metadata_cache(self):

        self._metadata_cache.clear()

        for cal_id in self.list_calibrations():

            try:
                metadata = self.load_calibration_metadata(cal_id)
                self._metadata_cache[cal_id] = metadata

            except Exception:
                continue    

    def calibrate(self, cal_input: CalibrationInput, metadata: Dict) -> Tuple[Calibration, Dict]:
        """Execute the calibration pipeline."""

        # Compute the calibration
        cal = CalibrationConstructor(cal_input).compute_calibration()

        # Update metadata
        metadata['file_type'] = "polarimeter_calibration"
        metadata['format_version'] = FILE_FORMAT_VERSION
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['shape'] = cal.dark_frame.shape

        return cal, metadata
    
    def get_valid_array(self, img: Image.Image, filepath: Path) -> NDArray:
        """Assert that the image is monochrome and well behaved"""
        arr = np.array(img)

        if arr.ndim == 2:  # Default monochrome case
            pass

        elif arr.ndim == 3 and arr.shape[2] == 3:  # 3 Channels
            if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
                arr = arr[..., 0]
            else:
                raise ValueError(f"[CalibrationManager] {filepath} has RGB channels and is not monochrome.")

        elif arr.ndim == 3 and arr.shape[2] == 1:  # 3rd dimension exists but only 1 channel
            arr = arr[..., 0]

        else:
            raise ValueError(f"[CalibrationManager] Unsupported shape: {arr.shape}.")
        
        return arr
    
    def get_calibration_path(self, cal_id: str) -> Path:
        return self.cal_dir / f"{cal_id}.calib"

    def save_calibration(self, calibration: Calibration, metadata: Dict):
        """Prepare the calibration file and metadata, then save it."""

        cal_id = self.get_calibration_id(metadata)

        file_path = self.get_calibration_path(cal_id)

        with open(file_path, "wb") as f:
            np.savez_compressed(
                f,
                dark_frame=calibration.dark_frame,
                flat_field=calibration.flat_field,
                stokes_reconstruction=calibration.stokes_reconstruction,
                metadata=json.dumps(metadata)
            )
    
    def validate_metadata(self, metadata: Dict):
        if metadata.get('file_type') != "polarimeter_calibration":
            raise RuntimeError(
                "[CalibrationManager] Unrecognised calibration file."
            )

        version = metadata.get('format_version')

        if version is None:
            raise RuntimeError(
                "[CalibrationManager] Missing format version."
            )

        if version != FILE_FORMAT_VERSION:
            raise RuntimeError(
                f"[CalibrationManager] Calibration file has format version {version}, only version {FILE_FORMAT_VERSION} is supported."
            )
        
    def load_calibration_metadata(self, cal_id: str) -> Dict:
        """Load only metadata from a calibration file."""

        file_path = self.get_calibration_path(cal_id)

        with np.load(file_path, allow_pickle=False) as data:

            if 'metadata' not in data:
                raise RuntimeError(
                    f"[CalibrationManager] {cal_id} contains no metadata."
                )

            metadata = json.loads(data['metadata'].item())

            self.validate_metadata(metadata)

        return metadata

    def load_calibration(self, cal_id: str) -> Tuple[Calibration, Dict]:
        """Load a specified calibration file, check that it matches, and return it."""
        
        file_path = self.get_calibration_path(cal_id)

        data = np.load(file_path, allow_pickle=False)
        metadata = json.loads(data['metadata'].item())

        # Consistency validation
        self.validate_metadata(metadata)

        calibration = Calibration(
            dark_frame=data['dark_frame'],
            flat_field=data['flat_field'],
            stokes_reconstruction=data['stokes_reconstruction'],
            resolution=metadata['shape'],
            bit_depth=metadata['bit_depth']
        )

        return calibration, metadata
    
    def list_calibrations(self) -> List[str]:
        """Return a list of all .calib files stored in data/calibration."""
        
        calibrations = []

        for file_path in self.cal_dir.glob("*.calib"):
            if not file_path.is_file():
                continue

            calibrations.append(file_path.stem)

        calibrations.sort(key=str.lower)

        return calibrations

    def get_calibration_id(self, metadata: Dict) -> str:
        """Return a filesystem-safe ID for the calibration file."""

        stem = metadata['name']
        stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")  # Clean up special language characters
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)  # Replace all other weird characters
        stem = re.sub(r"_+", "_", stem)  # Collapse repeated underscores
        stem = stem.strip("_")  # Strip leading and trailing underscores

        timestamp = metadata['timestamp']

        return stem

    def get_compatible(self, img_shape: Tuple[int, int]) -> List[str]:
        """Identify calibration files compatible with given image size."""
        compatible = []

        for cal_id, metadata in self._metadata_cache.items():

            if tuple(metadata.get('shape', ())) == img_shape:
                compatible.append(cal_id)

        return compatible
    
    def get_default(self, img_shape: Tuple[int, int]):
        compatible = self.get_compatible(img_shape)

        if not compatible:
            return None

        # Prefer calibrations starting with "default"
        for cal_id in compatible:
            if "default" in cal_id:
                return cal_id

        return compatible[0]

    def _has_default(self) -> bool:
        ...

    def _create_default_calibration(self, shape: Tuple[int, int]) -> Calibration:
        """Construct the default calibration, with 0 dark frame, 1 flat field, 
        and polarizer angles as per manufacturer specifications."""
        H, W = shape

        # Dark frame
        dark = np.zeros(shape)

        # Flat field
        flat = np.ones(shape)

        # Polarization reconstruction
        PHIS = np.array(DEFAULT_CALIBRATION_CHANNELS)
        PHIS = np.radians(PHIS)

        A = 0.5 * np.stack([
            [1, np.cos(2 * phi), np.sin(2 * phi)]
            for phi in PHIS
        ])  # (4, 3)

        # Reconstruction matrix
        M = np.linalg.pinv(A)  # (3, 4)

        # Tile across image
        n_y = H // 2
        n_x = W // 2

        stokes = np.tile(M, (n_y, n_x, 1, 1))  # (n_y, n_x, 3, 4)

        return Calibration(
            dark_frame=dark,
            flat_field=flat,
            stokes_reconstruction=stokes,
            resolution=shape,
            bit_depth=8,
        )

    def delete_calibration(self, cal_id: str):
        path = self.get_calibration_path(cal_id)
        if path.exists(): path.unlink()
        self._metadata_cache.pop(cal_id, None)

    def get_stats(self):
        """Generates statistics of the calibration folder for user information"""
        files = list(self.cal_dir.iterdir())

        total_size = sum(
            f.stat().st_size for f in files if f.is_file()
        )

        return {
            "file_count": sum(f.is_file() for f in files),
            "total_size_bytes": total_size,
            "path": self.cal_dir,
        }