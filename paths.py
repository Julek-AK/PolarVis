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

# Resource subdirectories
ICONS_DIR = RESOURCES_DIR / "icons"
STYLES_DIR = RESOURCES_DIR / "styles"
UI_DIR = RESOURCES_DIR / "ui"

# Testing directories
TEST_DIR = ROOT / "tests"
TEST_OUT_DIR = TEST_DIR / "out"