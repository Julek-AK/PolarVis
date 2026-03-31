
# Bultins
import os
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, Tuple
from datetime import datetime
import json

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
        ...

        self.cal_dir = CALIBRATION_DIR
        self.cal_dir.mkdir(parents=True, exist_ok=True)

        self.angles = []  # TODO implement this lmao
        self.bit_depth = 8  # TODO this as well


    def calibrate(self, cal_input: CalibrationInput) -> Calibration:
        """Execute the calibration pipeline."""
        return CalibrationConstructor(cal_input).compute_calibration()

    def read_input_data(self, parent: Optional[QWidget] = None) -> CalibrationInput:
        """Construct calibration input data by loading appropriate files and aggregating."""
        # Dark current
        dc_folder = self._get_folder_name("Select folder with dark-current images", parent)
        if dc_folder is None:
            raise RuntimeError("[CalibrationManager] Failed to construct calibration input.")
        dc_arr = self._load_folder(dc_folder)

        # Flat field
        ff_folder = self._get_folder_name("Select folder with flat-field images", parent)
        if ff_folder is None: 
            raise RuntimeError("[CalibrationManager] Failed to construct calibration input.")
        ff_arr = self._load_folder(ff_folder)

        # Polarized cases
        angle_dict: Dict[str, NDArray] = {}
        for angle in self.angles:
            angle_folder = self._get_folder_name(f"Select folder for angle {angle} deg", parent)
            if angle_folder is None:
                raise RuntimeError("[CalibrationManager] Failed to construct calibration input.")
            angle_arr = self._load_folder(angle_folder)
            angle_dict[angle] = angle_arr

        return CalibrationInput(
            dark_frame=dc_arr,
            flat_field=ff_arr,
            angle_cases=angle_dict,
            bit_depth=self.bit_depth
        )

    def _get_folder_name(self, caption, parent: Optional[QWidget] = None) -> Optional[Path]:
        folder_path, _ = QFileDialog.getExistingDirectory(
            parent, 
            caption,
            str(ROOT.parent)
            )
        
        if folder_path:
            return Path(folder_path)
        return None

    def _load_folder(self, folder_path: Path) -> NDArray:
        """Grab all the N images from a folder, convert into an array of shape (H, W, N)"""
        files = sorted(folder_path.glob("*"))

        imgs = []
        shapes = []
        dtypes = []

        for filepath in files:

            with Image.open(filepath) as img:
                arr = np.array(img)

            if arr.ndim != 2:
                raise ValueError(f"[CalibrationManager] {filepath} is not monochrome (shape {arr.shape})")

            imgs.append(arr.astype(np.float64))
            shapes.append(arr.shape)
            dtypes.append(arr.dtype)

        # Shape consistency
        if len(set(shapes)) != 1:
            raise ValueError("[CalibrationManager] Inconsistent image dimensions")

        # Data type consistency
        if len(set(dtypes)) != 1:
            raise ValueError("[CalibrationManager] Inconsistent image datatypes")

        return np.stack(imgs, axis=-1)
    
    def save_calibration(self, calibration: Calibration, parent: Optional[QWidget] = None):
        """Prepare the calibration file and metadata, then save it."""

        file_path, _ = QFileDialog.getSaveFileName(
            parent,
            "Save calibration",
            str(self.cal_dir / "calibration.calib"),
            "Calibration Files (*.calib)"
        )

        if not file_path:
            return
        
        if not file_path.endswith(".calib"):
            file_path += ".calib"
        
        metadata = {
            'file_type': "polarimeter_calibration",
            'format_version': 1,
            'sensor_model': "Blackfly S BFS-U3-51S5M",
            'lens_model': "TECHSPEC 16mm C Series Fixed Focal Length Lens",
            'wavelength_filter': "None",
            'temperature': 288.15,
            'timestamp': datetime.now().isoformat(),
            'angles': self.angles,
            'bit_depth': self.bit_depth,
            'shape': calibration.dark_frame.shape
        }

        with open(file_path, "wb") as f:
            np.savez_compressed(
                f,
                dark_frame=calibration.dark_frame,
                flat_field=calibration.flat_field,
                stokes_reconstruction=calibration.stokes_reconstruction,
                metadata=json.dumps(metadata)
            )

    def load_calibration(self, parent: Optional[QWidget] = None) -> Tuple[Calibration, Dict]:
        """Load a specified calibration file, check that it matches, and return it."""
        
        file_path, _ = QFileDialog.getOpenFileName(
            parent,
            "Load calibration",
            str(self.cal_dir),
            "Calibration Files (*.calib)"
        )

        if not file_path:
            raise RuntimeError("[CalibrationManager] No calibration file selected")

        data = np.load(file_path, allow_pickle=False)
        metadata = json.loads(str(data['metadata']))

        # Consistency validation
        if metadata['file_type'] != "polarimeter_calibration":
            raise RuntimeError("[CalibrationManager] Attempted to load an unrecognised calibration file.")

        if metadata['format_version'] != 1:
            raise ValueError(f"[CalibrationManager] Calibration supports format version 1, given: {metadata['format_version']}.")

        calibration = Calibration(
            dark_frame=data['dark_frame'],
            flat_field=data['flat_field'],
            stokes_reconstruction=data['stokes_reconstruction']
        )

        return calibration, metadata
    
    def get_valid_array(self, img: Image.Image, filepath: Path) -> NDArray:
        """Assert that the image is monochrome and well behaved"""
        arr = np.array(img)

        if arr.ndim == 2:  # Default monochrome case
            pass

        elif arr.ndim == 3 and arr.shape[2] == 3:  # 3 Channels
            if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
                arr = arr[..., 0]
            else:
                raise ValueError(f"[CalibrationManager] {filepath} is RGB but not monochrome.")

        elif arr.ndim == 3 and arr.shape[2] == 1:  # 3rd dimension exists but only 1 channel
            arr = arr[..., 0]

        else:
            raise ValueError(f"[CalibrationManager] Unsupported shape: {arr.shape}.")
        
        return arr