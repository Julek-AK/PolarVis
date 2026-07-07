"""
Centralised manager for PolarVis system settings
"""

# Builtins
from copy import deepcopy
import json

# Internal
from ..paths import CONFIG_DIR
from .defaults import DEFAULT_SETTINGS


SETTINGS_FILE = CONFIG_DIR / "user_settings.json"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base without mutating inputs"""
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)  # Recursive merge call
        else:
            result[key] = value

    return result


def _resolve_get(data: dict, key: str):
    """Resolve dot-notation key: 'a.b.c'"""
    parts = key.split('.')
    current = data

    for p in parts:
        if p not in current or not isinstance(current, dict):
            return None
        current = current[p]

    return current


def _resolve_set(data: dict, key: str, value):
    """
    Set dot-notation key: 'a.b.c'
    Creates intermediate dictionaries if needed.
    """
    parts = key.split('.')
    current = data

    for p in parts[:-1]:
        if p not in current or not isinstance(current[p], dict):
            current[p] = {}
        current = current[p]

    current[parts[-1]] = value


class SettingsManager:
    def __init__(self):
        self._settings = {}
        self._loaded = False

    def load(self):
        """Load settings from user_settings.json and merge with system defaults"""

        if self._loaded:
            return
        
        base = deepcopy(DEFAULT_SETTINGS)

        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)

                if isinstance(user_data, dict):
                    self._settings = _deep_merge(base, user_data)
                else:
                    self._settings = base

            except (json.JSONDecodeError, OSError):
                # Corrupt file fallback
                self._settings = base
        else:
            self._settings = base

        self._loaded = True
    
    def update(self, new_settings: dict):
        self._settings = new_settings
        self.save()

    def save(self):
        """Save the settings into user_settings.json"""

        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=4, sort_keys=True)
        except OSError as e:
            raise RuntimeError(f"[Settings] Failed to save settings: {e}")
        
    def _check_loaded(self):
        if self._loaded:
            return
        else:
            raise RuntimeError(f"[Settings] Attempted to access settings before loading.")
    
    def get(self, key, default=None):
        self._check_loaded()

        value = _resolve_get(self._settings, key)
        if value is None:
            if default is None:
                raise ValueError(f"[Settings] None type retrieved for setting: {key}")
            return default
        return value

    def set(self, key, value):
        self._check_loaded()
        _resolve_set(self._settings, key, value)

    def reset(self):
        self._check_loaded()
        self._settings = deepcopy(DEFAULT_SETTINGS)


settings = SettingsManager()
