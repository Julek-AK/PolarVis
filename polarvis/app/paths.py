"""
Centralised storage for all system paths
"""

from pathlib import Path

# Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "polarvis"

# Code directories
APP_DIR = PACKAGE_ROOT / "app"
CORE_DIR = PACKAGE_ROOT / "core"
GUI_DIR = PACKAGE_ROOT / "gui"
PROCESSING_DIR = PACKAGE_ROOT / "processing"

# Project directories
DATA_DIR = PROJECT_ROOT / "data"
RESOURCES_DIR = PROJECT_ROOT / "resources"
CACHE_DIR = PROJECT_ROOT / ".cache"
TEST_DIR = PROJECT_ROOT / "tests"

# Data subdirectories
CALIBRATION_DIR = DATA_DIR / "calibration"
SAMPLES_DIR = DATA_DIR / "samples"

# Resource subdirectories
ICONS_DIR = RESOURCES_DIR / "icons"
STYLES_DIR = RESOURCES_DIR / "styles"
UI_DIR = RESOURCES_DIR / "ui"

# Testing directories
TEST_OUT_DIR = TEST_DIR / "out"


if __name__ == "__main__":
    CALIBRATION_DIR.mkdir(exist_ok=True, parents=True)
    SAMPLES_DIR.mkdir(exist_ok=True, parents=True)

    print("paths.py executed correctly!")


