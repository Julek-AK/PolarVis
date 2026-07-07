
import cv2

import numpy as np
from numpy.typing import NDArray

from ..processing.image_visualisation import polarimetric_colormap


def arr_to_polarimetric(img_data: NDArray, cmap: str = 'hsv') -> NDArray[np.uint8]:
    """
    Return a numpy array with total intensity as brightness, DoLP as saturation and polarization angle as hue
    """

    result = polarimetric_colormap(img_data, cmap)
    img = result.image
    arr = np.array(img).astype(np.uint8)

    return arr


def cleanup_frame(frame: NDArray) -> NDArray:
    if frame.ndim == 2:
        return frame

    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported frame shape {frame.shape}")  


if __name__ == "__main__":

    VIDEO_PATH = r"C:\Users\juliu\OneDrive - Delft University of Technology\Bureaublad\Honours Programme\Media\Lens Testing Again\polarized_filter.avi"
    cap = cv2.VideoCapture(VIDEO_PATH)

    print('FPS:', cap.get(cv2.CAP_PROP_FPS))
    print('Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Frame count:', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    