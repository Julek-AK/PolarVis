"""
Computational procedure for calibrating the camera sensor using provided calibration cases
"""
# Builtins
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Internal support
from core.utils import raw_to_metapixel_channels

# External libraries
import numpy as np
from numpy.typing import NDArray


@dataclass
class Calibration:
    """Stores processed calibration output"""
    dark_frame: NDArray  # (H, W)
    flat_field: NDArray  # (H, W)
    stokes_reconstruction: NDArray  # (H//2, W//2, 3, 4)

    resolution: Tuple[int, int]  # (H, W)
    bit_depth: int
    sensor_id: Optional[str] = None  # For future use


@dataclass
class CalibrationInput:
    """Stores raw data input used for calibration"""
    dark_frame: NDArray  # (H, W, N)
    flat_field: NDArray  # (H, W, N)
    angle_cases: Dict[
        str,  # describes the polarizer angle setting
        NDArray  # (H, W, N)
    ]
    bit_depth: int  # bit depth used during calibration, needed for normalization


class CalibrationConstructor:
    """Manages the computation of calibration parameters from calibration image data."""
    def __init__(self, inp_data: CalibrationInput) -> None:
        # Input processing
        n_angle_cases = len(inp_data.angle_cases.keys())
        assert n_angle_cases >= 4, f"[Calibration] At least four angle calibration cases must be provided, {n_angle_cases} were given"
        
        self.bit_depth = inp_data.bit_depth

        # Calculate the dark frame
        self.dark_frame = np.mean(inp_data.dark_frame, axis=-1)

        # Calculate the flat field
        flat_frame = np.mean(inp_data.flat_field, axis=-1)
        flat_frame -= self.dark_frame
        flat_field = flat_frame / np.mean(flat_frame)
        self.flat_field = np.clip(flat_field, 1e-6, None)

        # Prepare for polarization computation
        self.max_readout = 2**inp_data.bit_depth - 1
        self.angle_data = inp_data.angle_cases
        self.n_angle_cases = n_angle_cases
        self.cases: List = []

        self.H, self.W = self.dark_frame.shape

        self.I: NDArray  # Intensity matrix (4*n_metapixels, 4)
        self.S: NDArray  # Stokes matrix (4, n_cases)
        self.M: NDArray  # Mueller-weight matrix (4*n_metapixels, 3)

    def _preprocess_frame(self, frame: NDArray) -> NDArray:
        """Apply dark_frame and flat_field corrections"""
        frame = frame.astype(np.float64, copy=True)  # Make a copy just to be safe
        frame -= self.dark_frame
        frame /= self.flat_field
        return frame

    def _compute_stokes(self, theta) -> NDArray:
        stokes = np.array([
            1,
            np.cos(2 * theta),
            np.sin(2 * theta),
        ])
        return stokes

    def _construct_matrices(self) -> None:
        """Use angle polarization data to construct the matrices"""
        intensity_list = []
        stokes_list = []

        for angle, data in self.angle_data.items():
            # Construct the image intensity data
            frame_mean = np.mean(data, axis=-1)
            frame_clean = self._preprocess_frame(frame_mean)
            frame_norm = frame_clean / self.max_readout
            frame_metapx = raw_to_metapixel_channels(frame_norm)

            intensity = np.reshape(frame_metapx, (self.H//2*self.W//2, 4))
            intensity_list.append(intensity)

            # Compute the incident stokes vector
            theta = float(angle) * np.pi/180
            stokes = self._compute_stokes(theta)
            stokes_list.append(stokes)

        self.I = np.stack(intensity_list)  # (n_cases, n_metapixels, 4)
        self.S = np.stack(stokes_list)  # (n_cases, 3)

    def compute_calibration(self) -> Calibration:
        """Execute batched least squares to find the Mueller matrices for each pixel using the metapixel method.
        Strictly speaking, the M matrix in the output is the inverse of a truncated Mueller matrix, as it maps from pixel intensities
        to Stokes parameters neglecting circular polarization, and can thus be used directly during runtime.
        Returns a Calibration object with dark_frame, flat_field and polarized corrections."""
        # Prepare matrices
        self._construct_matrices()

        I = np.transpose(self.I, (1,0,2))  # (n_metapixels, n_cases, 4)
        S = self.S  # (n_cases, 3)

        # Primary computation
        A = np.einsum('pni, pnj -> pij', I, I) # (n_metapixels, 4, 4)
        B = np.einsum('pni, nj -> pij', I, S)  # (n_metapixels, 4, 3)

        A += 1e-8 * np.eye(4)[None, :, :]
        A_inv = np.linalg.inv(A)
        C_t = np.einsum('pij, pjk -> pik', A_inv, B)  # (n_metapixels, 4, 3)

        M = np.transpose(C_t, (0,2,1))  # (n_metapixels, 3, 4)
        self.M = M.reshape(self.H//2, self.W//2, 3, 4)  # (H//2, W//2, 3, 4)

        self.calibration = Calibration(
            self.dark_frame,
            self.flat_field,
            self.M,
            self.dark_frame.shape,
            self.bit_depth
        )
        
        return self.calibration
    