"""
Centralised storage for all system paths
"""

from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent

# Common directories
CORE_DIR = ROOT / "core"
DATA_DIR = ROOT / "data"
GUI_DIR = ROOT / "gui"
PROCESSING_DIR = ROOT / "processing"
RESOURCES_DIR = ROOT / "resources"

CACHE_DIR = ROOT / ".cache"

# Data subdirectories
CALIBRATION_DIR = DATA_DIR / "calibration"
SAMPLES_DIR = DATA_DIR / "samples"

# Resource subdirectories
ICONS_DIR = RESOURCES_DIR / "icons"
STYLES_DIR = RESOURCES_DIR / "styles"
UI_DIR = RESOURCES_DIR / "ui"

# Testing directories
TEST_DIR = ROOT / "tests"
TEST_OUT_DIR = TEST_DIR / "out"


if __name__ == "__main__":
    print("paths.py executed correctly!")

    validation_path = Path(r"C:\Users\juliu\OneDrive - Delft University of Technology\Bureaublad\Honours Programme\Media\CALIBRATION_2\validation_set")

    for i in [0, 15, 20, 45, 60, 75, 90, 105, 120, 135, 150, 165]:
        folder = validation_path / f"{i}_deg"
        folder.mkdir()

