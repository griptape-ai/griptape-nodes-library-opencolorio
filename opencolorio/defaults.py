"""Resolve source colourspace, display and view when the user has not said.

A node with nothing wired should still render something sensible. OCIO 2 already
carries the answers inside the config, through roles and File Rules, so this
module reads the config's own answers rather than keeping a parallel table.

For each slot the first non-empty answer wins:

1. An explicit value passed by the caller.
2. A library settings override, layering user, project, workspace and env.
3. The config's own answer: roles, File Rules, default display and view.
4. The empty string.

Layer 2 sits above layer 3 so a project can pick Rec.709 output on a config whose
default display is sRGB without forking the config.

Nothing here is cached. ``Config.CreateFromFile`` costs about 4ms and lookups on a
loaded config are free, so the cost is invisible for the interactive case, and a
wrongly keyed cache would mean an artist edits their .ocio and keeps seeing the
old one. The per-frame cost belongs to the transform handler.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import PyOpenColorIO as ocio
from griptape_nodes.retained_mode import griptape_nodes
from griptape_nodes.retained_mode.events.config_events import (
    GetConfigValueRequest,
    GetConfigValueResultSuccess,
)

from opencolorio.ocio_helpers import (
    is_builtin_uri,
    load_builtin_config,
    load_ocio_config,
    ocio_env_is_set,
)

if TYPE_CHECKING:
    from opencolorio.artifacts.ocio_color_params_artifact import OCIOColorParamsArtifact

logger = logging.getLogger("griptape_nodes")

# The literal a dropdown carries to mean "resolve this for me". No colourspace,
# role or display name in the shipped ACES configs contains angle brackets, and
# config.getColorSpace("<default>") returns None, so it cannot collide.
DEFAULT_SENTINEL: Final[str] = "<default>"

# Per-slot provenance. Without these a node cannot tell the user why it chose
# Rec.709, which is the first question anyone asks.
SOURCE_EXPLICIT: Final[str] = "explicit"
SOURCE_SETTINGS: Final[str] = "settings"
SOURCE_SETTINGS_INVALID: Final[str] = "settings-invalid"
SOURCE_CONFIG_ROLE: Final[str] = "config-role"
SOURCE_CONFIG_FILE_RULE: Final[str] = "config-file-rule"
SOURCE_CONFIG_DEFAULT_RULE: Final[str] = "config-default-rule"
SOURCE_CONFIG_DEFAULT_DISPLAY: Final[str] = "config-default-display"
SOURCE_CONFIG_DEFAULT_VIEW: Final[str] = "config-default-view"
SOURCE_FALLBACK: Final[str] = "fallback"
SOURCE_NO_CONFIG: Final[str] = "no-config"

# Library settings keys, read under the "opencolorio" config category.
SETTING_WORKING_SPACE: Final[str] = "default_working_space"
SETTING_DISPLAY: Final[str] = "default_display"
SETTING_VIEW: Final[str] = "default_view"
SETTING_FALLBACK_CONFIG: Final[str] = "fallback_config"

SETTING_KEYS: Final[tuple[str, ...]] = (
    SETTING_WORKING_SPACE,
    SETTING_DISPLAY,
    SETTING_VIEW,
    SETTING_FALLBACK_CONFIG,
)

_SCENE_LINEAR_ROLE: Final[str] = "scene_linear"
_DEFAULT_ROLE: Final[str] = "default"


@dataclass(frozen=True)
class ResolvedColorParams:
    """The four resolved slots, where each came from, and anything worth saying.

    ``working_space`` and ``source_colorspace`` are different things.
    ``DisplayViewTransform`` converts *from* the working space, while a Load node
    tagging incoming pixels wants the per-file answer.

    ``warnings`` carries anything the caller should surface. The resolver has no
    result payload of its own, so a node joins these into its ``result_details``.
    """

    working_space: str
    source_colorspace: str
    display: str
    view: str
    config_path: str | None
    sources: dict[str, str]
    warnings: tuple[str, ...] = ()


def resolve_color_params(
    *,
    explicit: OCIOColorParamsArtifact | None = None,
    config_path: str | None = None,
    context_vars: dict[str, str] | None = None,
    file_path: str | None = None,
) -> ResolvedColorParams:
    """Resolve the colour parameters a node should use.

    Args:
        explicit: Values the user chose. Empty strings mean "not chosen".
        config_path: A config file path or ``ocio://`` URI. None runs the
            discovery chain: $OCIO, then the ``fallback_config`` setting, then
            the raw config.
        context_vars: Carried through for the transform's benefit. Context cannot
            change which names resolve, since none of the four lookups accepts a
            Context, but it does change what those names mean once a processor is
            built.
        file_path: Drives the File Rules lookup for ``source_colorspace``.

    Raises:
        Exception: If a config was named and could not be loaded. A config the
            user pointed at and that does not work is an error, never a fallback.
    """
    del context_vars  # Carried by the caller, not consulted here.

    warnings: list[str] = []
    settings = _read_settings()

    config, resolved_config_path, no_config = _acquire_config(config_path, settings, warnings)

    # The artifact carries no working_space field today, so this is "" in
    # practice. Reading it the same way as the others keeps the four-layer rule
    # above true for every slot, and lights the layer up if the field is added.
    explicit_working_space = _explicit_value(explicit, "working_space")
    explicit_source = _explicit_value(explicit, "source_colorspace")
    explicit_display = _explicit_value(explicit, "display")
    explicit_view = _explicit_value(explicit, "view")

    if no_config:
        return _no_config_result(
            explicit_working_space=explicit_working_space,
            explicit_source=explicit_source,
            explicit_display=explicit_display,
            explicit_view=explicit_view,
            warnings=warnings,
        )

    sources: dict[str, str] = {}

    # Order matters: the view setting is validated against the resolved display,
    # and source_colorspace falls back to the resolved working space.
    working_space, sources["working_space"] = _resolve_working_space(
        config, explicit_working_space, settings[SETTING_WORKING_SPACE], warnings
    )
    display, sources["display"] = _resolve_display(config, explicit_display, settings[SETTING_DISPLAY], warnings)
    view, sources["view"] = _resolve_view(config, display, explicit_view, settings[SETTING_VIEW], warnings)
    source_colorspace, sources["source_colorspace"] = _resolve_source_colorspace(
        config,
        file_path=file_path,
        working_space=working_space,
        working_space_source=sources["working_space"],
        explicit=explicit_source,
        warnings=warnings,
    )

    return ResolvedColorParams(
        working_space=working_space,
        source_colorspace=source_colorspace,
        display=display,
        view=view,
        config_path=resolved_config_path,
        sources=sources,
        warnings=tuple(warnings),
    )


# --- Config acquisition ---


def _acquire_config(
    config_path: str | None,
    settings: dict[str, str],
    warnings: list[str],
) -> tuple[ocio.Config, str | None, bool]:
    """Return the config to resolve against, what it was loaded from, and whether it is raw.

    $OCIO unset and $OCIO wrong behave oppositely. Unset returns a raw config with
    colour management disabled and no exception. A path that does not exist or
    will not parse raises, and that is allowed to propagate.
    """
    if config_path:
        return _load_named_config(config_path), config_path, False

    if ocio_env_is_set():
        return ocio.Config.CreateFromEnv(), os.environ.get("OCIO"), False

    fallback = settings[SETTING_FALLBACK_CONFIG]
    if fallback:
        config = _load_fallback_config(fallback, warnings)
        if config is not None:
            return config, fallback, False

    # No config anywhere. Colour management is off.
    return ocio.Config.CreateFromEnv(), None, True


def _load_named_config(config_path: str) -> ocio.Config:
    if is_builtin_uri(config_path):
        return load_builtin_config(config_path)
    return load_ocio_config(config_path)


def _load_fallback_config(value: str, warnings: list[str]) -> ocio.Config | None:
    """Load the fallback_config setting, or warn and return None.

    This setting only fires when the user has configured nothing at all, so a bad
    value degrades to the raw config rather than raising. Turning it into a hard
    failure would punish the case it exists to soften.

    It accepts ocio:// URIs only. $OCIO stays the one way to point at your own
    config; this chooses among the configs that ship inside PyOpenColorIO, and so
    cannot go stale the way a path can.
    """
    if not is_builtin_uri(value):
        warnings.append(
            f"The opencolorio.fallback_config setting is {value!r}, which is not an ocio:// URI. "
            "Use $OCIO to point at a config file. Ignoring this setting."
        )
        logger.warning("fallback_config %r is not an ocio:// URI; ignoring it.", value)
        return None

    try:
        return load_builtin_config(value)
    except Exception as e:  # noqa: BLE001 - any registry rejection degrades the same way
        warnings.append(f"The opencolorio.fallback_config setting {value!r} could not be loaded: {e}. Ignoring it.")
        logger.warning("fallback_config %r could not be loaded: %s", value, e)
        return None


def _no_config_result(
    *,
    explicit_working_space: str,
    explicit_source: str,
    explicit_display: str,
    explicit_view: str,
    warnings: list[str],
) -> ResolvedColorParams:
    """Build the result for a raw config, where colour management is disabled.

    The raw config does answer sRGB and Raw, but reporting those would make the
    editor look like colour management is on. Every slot the user did not choose
    is empty, which the transform handler already treats as passthrough.
    """
    values = {
        "working_space": explicit_working_space,
        "source_colorspace": explicit_source,
        "display": explicit_display,
        "view": explicit_view,
    }
    sources = {slot: SOURCE_EXPLICIT if value else SOURCE_NO_CONFIG for slot, value in values.items()}

    return ResolvedColorParams(
        working_space=values["working_space"],
        source_colorspace=values["source_colorspace"],
        display=values["display"],
        view=values["view"],
        config_path=None,
        sources=sources,
        warnings=tuple(warnings),
    )


# --- Slot resolvers ---


def _resolve_working_space(
    config: ocio.Config,
    explicit: str,
    setting: str,
    warnings: list[str],
) -> tuple[str, str]:
    if explicit:
        return explicit, SOURCE_EXPLICIT

    config_value, config_source = _config_working_space(config)

    if setting:
        if _is_valid_colorspace(config, setting):
            return setting, SOURCE_SETTINGS
        _warn_invalid_setting(SETTING_WORKING_SPACE, setting, "colourspace or role", warnings)
        return config_value, SOURCE_SETTINGS_INVALID

    return config_value, config_source


def _config_working_space(config: ocio.Config) -> tuple[str, str]:
    """Return the config's own working space.

    Roles are an open dictionary and no standard role is guaranteed present. The
    shipped ACES configs define nine roles and `default` is not among them, so
    every lookup is guarded with hasRole rather than assuming getRoleColorSpace
    succeeds.
    """
    for role in (_SCENE_LINEAR_ROLE, _DEFAULT_ROLE):
        if config.hasRole(role):
            return config.getRoleColorSpace(role), SOURCE_CONFIG_ROLE
    return "", SOURCE_FALLBACK


def _resolve_display(
    config: ocio.Config,
    explicit: str,
    setting: str,
    warnings: list[str],
) -> tuple[str, str]:
    if explicit:
        return explicit, SOURCE_EXPLICIT

    config_value = config.getDefaultDisplay()
    config_source = SOURCE_CONFIG_DEFAULT_DISPLAY if config_value else SOURCE_FALLBACK

    if setting:
        if setting in list(config.getDisplays()):
            return setting, SOURCE_SETTINGS
        _warn_invalid_setting(SETTING_DISPLAY, setting, "display", warnings)
        return config_value, SOURCE_SETTINGS_INVALID

    return config_value, config_source


def _resolve_view(
    config: ocio.Config,
    display: str,
    explicit: str,
    setting: str,
    warnings: list[str],
) -> tuple[str, str]:
    if explicit:
        return explicit, SOURCE_EXPLICIT

    if not display:
        return "", SOURCE_FALLBACK

    # getDefaultView returns "" for a display the config does not know.
    config_value = config.getDefaultView(display)
    config_source = SOURCE_CONFIG_DEFAULT_VIEW if config_value else SOURCE_FALLBACK

    if setting:
        if setting in list(config.getViews(display)):
            return setting, SOURCE_SETTINGS
        _warn_invalid_setting(SETTING_VIEW, setting, f"view for display {display!r}", warnings)
        return config_value, SOURCE_SETTINGS_INVALID

    return config_value, config_source


def _resolve_source_colorspace(
    config: ocio.Config,
    *,
    file_path: str | None,
    working_space: str,
    working_space_source: str,
    explicit: str,
    warnings: list[str],
) -> tuple[str, str]:
    """Resolve the colourspace incoming pixels are in.

    There is no settings layer here: a single default source colourspace would be
    wrong for most files, which is what File Rules exist to answer.
    """
    if explicit:
        return explicit, SOURCE_EXPLICIT

    if file_path is None:
        # Nothing to match a rule against, so the working space is the best answer.
        # Carrying its own source keeps this distinguishable from a Default-rule match.
        return working_space, working_space_source

    # getColorSpaceFromFilepath returns (name, rule_index), not a bare name.
    name, _rule_index = config.getColorSpaceFromFilepath(file_path)

    # OCIO requires every config to terminate in an author-written Default rule.
    # When that rule names sRGB - Display, that is a deliberate instruction for
    # unclassified files, so this only changes the reported provenance, never the
    # value. On cg-config v1 and v2 every path matches only the Default rule.
    source = SOURCE_CONFIG_DEFAULT_RULE if config.filepathOnlyMatchesDefaultRule(file_path) else SOURCE_CONFIG_FILE_RULE

    _warn_if_display_referred(config, name, file_path, warnings)

    return name, source


def _warn_if_display_referred(
    config: ocio.Config,
    name: str,
    file_path: str,
    warnings: list[str],
) -> None:
    """Warn when the resolved source is display-referred.

    Feeding a display-referred space into a display transform is usually wrong,
    but it is what the config author specified, so this warns and does not block.
    getReferenceSpaceType correctly separates "sRGB - Display" from
    "sRGB Encoded Rec.709 (sRGB)", which a name heuristic would confuse.
    """
    colorspace = config.getColorSpace(name)
    if colorspace is None:
        return

    if colorspace.getReferenceSpaceType() == ocio.REFERENCE_SPACE_DISPLAY:
        message = (
            f"The config resolves {file_path!r} to {name!r}, which is display-referred. "
            "Feeding it into a display transform is usually wrong, but this is what the config specifies."
        )
        warnings.append(message)
        logger.warning("%s", message)


# --- Settings ---


def _read_settings() -> dict[str, str]:
    """Read the opencolorio library settings, defaulting every one to empty."""
    return {key: _get_setting(key) for key in SETTING_KEYS}


def _get_setting(key: str) -> str:
    """Read one library setting, or return empty if it is unset or unreadable.

    The sentinel reads as unset. These settings ship as `<default>` rather than
    "" because the engine drops blank strings as each config layer loads, on the
    basis that a layer holding "" states nothing about that setting. Shipping ""
    left the whole category empty, and the editor draws a row per key it finds in
    the config rather than per key in the schema, so the section never appeared.

    GriptapeNodes is reached through its module rather than imported by name, so
    that the autouse fixture patching it at its definition site takes effect.
    A from-import would bind the real class before the patch ran.

    failure_log_level keeps a missing optional key from logging an error on every
    read; absent is the normal case for these.
    """
    try:
        result = griptape_nodes.GriptapeNodes.handle_request(
            GetConfigValueRequest(category_and_key=f"opencolorio.{key}", failure_log_level=logging.DEBUG)
        )
    except Exception as e:  # noqa: BLE001 - no engine running is a normal case for a library
        logger.debug("Could not read the opencolorio.%s setting: %s", key, e)
        return ""

    if isinstance(result, GetConfigValueResultSuccess) and isinstance(result.value, str):
        value = result.value.strip()
        return "" if value == DEFAULT_SENTINEL else value
    return ""


def _warn_invalid_setting(key: str, value: str, expected: str, warnings: list[str]) -> None:
    """Record that a settings value does not exist in the resolved config.

    Settings are free strings typed into a panel with no access to the config, so
    a typo would otherwise surface as an opaque OCIO exception at render time,
    inside a node the user never edited.
    """
    message = (
        f"The opencolorio.{key} setting is {value!r}, which is not a {expected} in this config. "
        "Using the config's own answer instead."
    )
    warnings.append(message)
    logger.warning("%s", message)


# --- Helpers ---


def _explicit_value(explicit: OCIOColorParamsArtifact | None, field: str) -> str:
    """Read one explicit slot, treating the sentinel and blank alike as unset."""
    if explicit is None:
        return ""
    value = getattr(explicit, field, "") or ""
    if value == DEFAULT_SENTINEL:
        return ""
    return str(value)


def _is_valid_colorspace(config: ocio.Config, name: str) -> bool:
    return config.getColorSpace(name) is not None or config.hasRole(name)
