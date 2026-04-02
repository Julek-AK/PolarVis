
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
from paths import CALIBRATION_DIR, ROOT
from processing.calibration import Calibration, CalibrationInput, CalibrationConstructor

# External libraries
import numpy as np
from numpy.typing import NDArray, DTypeLike
from PyQt6.QtWidgets import QWidget, QFileDialog
from PyQt6 import QtCore


class CalibrationManager:
    """Orchestrates and manages loading, saving and computing calibration."""
    def __init__(self) -> None:
        """Initialize relevant fields, connect to the gui"""

        self.cal_dir = CALIBRATION_DIR
        self.cal_dir.mkdir(parents=True, exist_ok=True)

        if not self._has_default():
            default = self._create_default_calibration((2048, 2448))
            self.save_calibration(
                default,
                {   
                    'name': "Default (factory)",
                    'file_type': "polarimeter_calibration",
                    'format_version': 1,
                    'timestamp': "NA",
                    'shape': (2048, 2448),
                    'bit_depth': 8,
                }
            )

    def calibrate(self, cal_input: CalibrationInput, metadata: Dict) -> Tuple[Calibration, Dict]:
        """Execute the calibration pipeline."""

        # Compute the calibration
        cal = CalibrationConstructor(cal_input).compute_calibration()

        # Update metadata
        metadata['file_type'] = "polarimeter_calibration"
        metadata['format_version'] = 1
        metadata['tiemstamp'] = datetime.now().isoformat()
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

    def save_calibration(self, calibration: Calibration, metadata: Dict):
        """Prepare the calibration file and metadata, then save it."""

        cal_id = self.get_calibration_id(metadata)

        file_path = self.cal_dir / f"{cal_id}.calib"

        with open(file_path, "wb") as f:
            np.savez_compressed(
                f,
                dark_frame=calibration.dark_frame,
                flat_field=calibration.flat_field,
                stokes_reconstruction=calibration.stokes_reconstruction,
                metadata=json.dumps(metadata)
            )

    def load_calibration(self, cal_id: str) -> Tuple[Calibration, Dict]:
        """Load a specified calibration file, check that it matches, and return it."""
        
        file_path = self.cal_dir / f"{cal_id}.calib"

        data = np.load(file_path, allow_pickle=False)
        metadata = json.loads(str(data['metadata']))

        # Consistency validation
        if metadata.get('file_type') != "polarimeter_calibration":
            raise RuntimeError("[CalibrationManager] Attempted to load an unrecognised calibration file.")

        if metadata.get('format_version') != 1:
            raise ValueError(f"[CalibrationManager] Calibration supports format version 1, given: {metadata['format_version']}.")

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



    def get_calibration_id(self, metadata: Dict) -> str:
        """Return a filesystem-safe ID for the calibration file."""

        stem = metadata['name']
        stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")  # Clean up special language characters
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)  # Replace all other weird characters
        stem = re.sub(r"_+", "_", stem)  # Collapse repeated underscores
        stem = stem.strip("_")  # Strip leading and trailing underscores

        timestamp = metadata['timestamp']

        return stem + "_" + timestamp

    def get_compatible(self, img_shape: Tuple[int, int]) -> List[str]:
        """Identify calibration files compatible with given image size."""
        compatible = []

        for cal_id in self.list_calibrations():
            cal, _ = self.load_calibration(cal_id)

            if cal.resolution == img_shape:
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
        PHIS = np.array([
            np.pi / 2,   # channel 0 → top-left
            np.pi / 4,   # channel 1 → top-right
            -np.pi / 4,  # channel 2 → bottom-left
            0.0             # channel 3 → bottom-right
        ])

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
