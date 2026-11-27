"""Answer "what colour parameters should I use?" for any library, without an import.

A consumer needs the resolved values before running, to populate its own
dropdowns and show the user what will happen. Resolving inside the transform can
only report after the fact, and a separate request also makes provenance
queryable without running a transform at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from griptape_nodes.retained_mode.events.base_events import (
    RequestPayload,
    ResultPayloadFailure,
    ResultPayloadSuccess,
    WorkflowNotAlteredMixin,
)

from opencolorio.artifacts.ocio_color_params_artifact import OCIOColorParamsArtifact
from opencolorio.defaults import resolve_color_params


@dataclass(kw_only=True)
class ResolveColorParamsRequest(RequestPayload):
    """Ask for resolved source colourspace, display and view.

    Every field is a primitive, deliberately. A library that discovered this type
    by name has never imported our artifact classes and could not construct them,
    which is how griptape-nodes-library-openexr builds ColorspaceTransformRequest
    today.

    Args:
        config_path: An OCIO config path or ocio:// URI. None runs the full
            discovery chain, so a caller with no OCIO nodes wired still gets a
            usable answer.
        context_vars: OCIO context variables, carried through for the caller's
            own transform. They cannot change which names resolve.
        file_path: The file being read, which drives the File Rules lookup.
        explicit_source_colorspace: A value the user already chose. Empty, or the
            <default> sentinel, means "not chosen".
        explicit_display: As above, for the display.
        explicit_view: As above, for the view.
    """

    config_path: str | None = None
    context_vars: dict[str, str] = field(default_factory=dict)
    file_path: str | None = None
    explicit_source_colorspace: str = ""
    explicit_display: str = ""
    explicit_view: str = ""


@dataclass(kw_only=True)
class ResolveColorParamsResultSuccess(WorkflowNotAlteredMixin, ResultPayloadSuccess):
    """Resolved colour parameters, with where each one came from.

    ``sources`` maps each of the four slot names to the layer that answered, so a
    consumer can tell the user why it chose Rec.709.
    """

    working_space: str
    source_colorspace: str
    display: str
    view: str
    config_path: str | None
    sources: dict[str, str]


@dataclass(kw_only=True)
class ResolveColorParamsResultFailure(WorkflowNotAlteredMixin, ResultPayloadFailure):
    """The config could not be loaded, or resolution failed."""


def handle_resolve_color_params(
    req: ResolveColorParamsRequest,
) -> ResolveColorParamsResultSuccess | ResolveColorParamsResultFailure:
    # Raw strings, deliberately. _explicit_value in defaults.py is the one place
    # that decides the sentinel and blank both mean "not chosen"; stripping here
    # too would give that rule two homes that could drift apart. config_path is
    # left off the artifact because the resolver takes it as its own argument and
    # never reads explicit.config_path.
    explicit = OCIOColorParamsArtifact(
        source_colorspace=req.explicit_source_colorspace,
        display=req.explicit_display,
        view=req.explicit_view,
    )

    try:
        resolved = resolve_color_params(
            explicit=explicit,
            config_path=req.config_path,
            context_vars=req.context_vars,
            file_path=req.file_path,
        )
    except Exception as e:
        return ResolveColorParamsResultFailure(
            result_details=(
                f"Attempted to resolve colour parameters with config {req.config_path!r}. Failed due to: {e}"
            ),
            exception=e,
        )

    details = (
        f"Resolved source={resolved.source_colorspace!r}, display={resolved.display!r}, "
        f"view={resolved.view!r}, working space={resolved.working_space!r}"
    )
    if resolved.warnings:
        details += ". Warnings: " + "; ".join(resolved.warnings)

    return ResolveColorParamsResultSuccess(
        working_space=resolved.working_space,
        source_colorspace=resolved.source_colorspace,
        display=resolved.display,
        view=resolved.view,
        config_path=resolved.config_path,
        sources=resolved.sources,
        result_details=details,
    )
