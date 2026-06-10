from __future__ import annotations

import PyOpenColorIO as ocio


def load_ocio_config(file_path: str) -> ocio.Config:
    """Load an OCIO config from a file path, or from the environment if no path is given."""
    if file_path:
        return ocio.Config.CreateFromFile(file_path)
    return ocio.Config.CreateFromEnv()


def extract_lists(config: ocio.Config) -> tuple[list[str], list[str], list[str]]:
    """Return (colorspace_names, display_names, role_names) extracted from an OCIO config."""
    colorspace_names = list(config.getColorSpaceNames())
    display_names = list(config.getDisplays())
    role_names = [role for role, _ in config.getRoles()]
    return colorspace_names, display_names, role_names
