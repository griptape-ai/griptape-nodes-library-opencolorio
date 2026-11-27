from __future__ import annotations

import os
from typing import NamedTuple

import PyOpenColorIO as ocio

# Prefix identifying a config that ships inside PyOpenColorIO rather than living on disk.
BUILTIN_URI_PREFIX = "ocio://"


class OCIOLists(NamedTuple):
    colorspace_names: list[str]
    display_names: list[str]
    role_names: list[str]


def load_ocio_config(file_path: str | None) -> ocio.Config:
    """Load an OCIO config from a file path, or from the environment if no path is given."""
    if file_path:
        return ocio.Config.CreateFromFile(file_path)
    return ocio.Config.CreateFromEnv()


def is_builtin_uri(value: str) -> bool:
    """Return whether a value names a config built into PyOpenColorIO."""
    return value.startswith(BUILTIN_URI_PREFIX)


def load_builtin_config(uri: str) -> ocio.Config:
    """Load a config that ships inside PyOpenColorIO, e.g. ``ocio://cg-config-latest``.

    Raises if the registry does not recognise the URI.
    """
    return ocio.Config.CreateFromBuiltinConfig(uri)


def ocio_env_is_set() -> bool:
    """Return whether $OCIO names a config.

    ``CreateFromEnv()`` cannot tell an unset $OCIO from one pointing at a broken
    config: unset returns a raw config with colour management disabled and no
    exception, while a bad path raises. Callers that must treat those two cases
    differently check this first.
    """
    return bool(os.environ.get("OCIO"))


def extract_lists(config: ocio.Config) -> OCIOLists:
    """Return colorspace, display, and role names extracted from an OCIO config."""
    return OCIOLists(
        colorspace_names=list(config.getColorSpaceNames()),
        display_names=list(config.getDisplays()),
        role_names=[role for role, _ in config.getRoles()],
    )
