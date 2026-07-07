
import cv2

import numpy as np
from numpy.typing import NDArray

from ..processing.torch_backend import calibrated_resolve_polarization
from ..utils.color_ops import hsv_to_rgb_vec




def arr_to_polarimetric(img_data: NDArray, angle_cmap: str = 'hsv') -> NDArray[np.uint8]:
    """
    Creates an image with total intensity as brightness, DoLP as saturation and polarization angle as hue
    """
    if angle_cmap != 'hsv':
        raise NotImplementedError("[Visualisation] Polarimetric visualisation techniques other than standard HSV are not implemented yet.")

    brightness = img_data[..., 0] / 2 # Intensity

    saturation = img_data[..., 1]  # DOLP

    angle = np.mod(img_data[..., 2], np.pi)  # Theta
    hue = angle / np.pi

    rgb = hsv_to_rgb_vec(np.stack((hue, saturation, brightness), axis=-1))
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    return rgb_uint8


def cleanup_frame(frame: NDArray) -> NDArray:
    if frame.ndim == 2:
        return frame

    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    raise ValueError(
        f'Unsupported frame shape {frame.shape}'
    )   


if __name__ == "__main__":

    VIDEO_PATH = r"C:\Users\juliu\OneDrive - Delft University of Technology\Bureaublad\Honours Programme\Media\Lens Testing Again\polarized_filter.avi"
    cap = cv2.VideoCapture(VIDEO_PATH)

    print('FPS:', cap.get(cv2.CAP_PROP_FPS))
    print('Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Frame count:', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()