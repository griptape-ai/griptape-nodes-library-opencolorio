---
title: OpenColorIO Library
---

The OpenColorIO library reads an [OpenColorIO](https://opencolorio.org/) (OCIO) config, the file that defines a pipeline's colour spaces, displays and views, and lets you choose a colour setting from it. Its nodes are in the **Colorspace** category.

The library does not change images itself. It provides the config and colour choice that the OpenEXR library's nodes use for pipeline-specific colour. Load OCIO Config is the node to start with.

## Requirements

- An OCIO config file, such as your studio's ACES config. Set the `$OCIO` environment variable to its path before launching Griptape Nodes, or pick the file on the Load OCIO Config node. In a Griptape project, `OCIO` can be set under `environment:` in `project.yml`.
- Griptape Nodes engine 0.106.0 or later.
- The library installs its own Python dependencies: `opencolorio` 2.5.2 or later, `Pillow` and `numpy`.
- The OpenEXR library, to use the result. It is a separate library.

## Nodes

- [Load OCIO Config](nodes/LoadOCIOConfig.md): load the config from `$OCIO` or from a `.ocio` file you pick.
- [OCIO Color Parameters](nodes/OCIOColorParameters.md): choose a source colour space, display and view from that config.

## How they connect

```text
Load OCIO Config -> OCIO Color Parameters -> Read EXR, Display EXR Part,
                                             Display EXR Channel, EXR Sequence to Preview
```

Connect `config` from Load OCIO Config to OCIO Color Parameters, and `color_params` to an OpenEXR node with `color_mode` set to `ocio`. Without these nodes the OpenEXR nodes use their built-in `basic` transforms. The OpenEXR nodes stay on `basic` until you change `color_mode`, whether or not this library is installed.
