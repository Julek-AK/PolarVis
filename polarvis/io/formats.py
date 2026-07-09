
# Builtin

# External
import numpy as np
from numpy.typing import NDArray

# Internal




def polar_to_stokes(polar: NDArray) -> NDArray:
    """
    Convert (I, DoLP, AoP) into (S0, S1, S2)
    """

    I = polar[..., 0]
    DoLP = polar[..., 1]
    AoP = polar[..., 2]

    S0 = I
    S1 = I * DoLP * np.cos(2 * AoP)
    S2 = I * DoLP * np.sin(2 * AoP)

    stokes = np.stack((S0, S1, S2), axis=-1)
    return stokes