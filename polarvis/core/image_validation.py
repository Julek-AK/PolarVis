
# Builtin

# External
import numpy as np

# Internal


class ValidationError(Exception):
    pass

class ValidationWarning(Exception):
    pass


def validate_calibration(image: np.ndarray, cal) -> None:
    H, W = image.shape

    # Hard errors
    if cal.dark_frame.shape != (H, W):
        raise ValidationError("Dark frame size mismatch")

    if cal.flat_field.shape != (H, W):
        raise ValidationError("Flat field size mismatch")

    if cal.stokes_reconstruction.shape[:2] != (H//2, W//2):
        raise ValidationError("Stokes matrix size mismatch")

    # Soft warnings
    if np.any(cal.flat_field < 1e-6):
        raise ValidationWarning("Flat field contains near-zero values")