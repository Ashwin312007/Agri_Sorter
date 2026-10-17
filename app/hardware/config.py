"""Hardware configuration loader for ultrasonic, camera, and conveyor subsystems."""
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        raise ImportError("No TOML parser available. Python >=3.11 with tomllib or 'tomli' package required.")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "hardware.toml"

REQUIRED_SECTIONS = ("transmitter", "receiver", "adc", "camera")


def load_config(
    path: Optional[Union[str, Path]] = None,
    mock: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load and validate hardware configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file. If None, uses default
              config/hardware.toml relative to repository root.
        mock: Optional override for mock mode. If provided, sets config['mock'].

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        KeyError: If any required section is missing.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Hardware configuration file not found: {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise KeyError(f"Missing required configuration section: '{section}' in {config_path}")

    if mock is not None:
        config["mock"] = mock

    return config
