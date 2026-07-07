"""
Utility functions for color conversions
"""

# External
import numpy as np
from numpy.typing import NDArray


def hsv_to_rgb_vec(hsv: NDArray) -> NDArray:
    """
    Vectorised HSV -> RGB conversion.
    Input:  (..., 3) HSV in [0,1]
    Output: (..., 3) RGB in [0,1]
    """
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6

    r = np.select(
        [
            i_mod == 0,
            i_mod == 1,
            i_mod == 2,
            i_mod == 3,
            i_mod == 4,
            i_mod == 5,
        ],
        [v, q, p, p, t, v],
    )

    g = np.select(
        [
            i_mod == 0,
            i_mod == 1,
            i_mod == 2,
            i_mod == 3,
            i_mod == 4,
            i_mod == 5,
        ],
        [t, v, v, q, p, p],
    )

    b = np.select(
        [
            i_mod == 0,
            i_mod == 1,
            i_mod == 2,
            i_mod == 3,
            i_mod == 4,
            i_mod == 5,
        ],
        [p, p, t, v, v, q],
    )

    return np.stack([r, g, b], axis=-1)