from __future__ import annotations

from typing import NamedTuple

import PyOpenColorIO as ocio


class OCIOLists(NamedTuple):
    colorspace_names: list[str]
    display_names: list[str]
    role_names: list[str]


def load_ocio_config(file_path: str | None) -> ocio.Config:
    """Load an OCIO config from a file path, or from the environment if no path is given."""
    if file_path:
        return ocio.Config.CreateFromFile(file_path)
    return ocio.Config.CreateFromEnv()


def extract_lists(config: ocio.Config) -> OCIOLists:
    """Return colorspace, display, and role names extracted from an OCIO config."""
    return OCIOLists(
        colorspace_names=list(config.getColorSpaceNames()),
        display_names=list(config.getDisplays()),
        role_names=[role for role, _ in config.getRoles()],
    )
