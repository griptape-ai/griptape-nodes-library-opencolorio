---
title: OpenColorIO Library
---

Studios keep their colour rules in one file, called an OCIO config (OCIO is short for [OpenColorIO](https://opencolorio.org/), the open-source colour management system). The config lists the colour spaces your pipeline uses. A colour space describes how the numbers in an image map to real colours, such as ACEScg for renders or sRGB for monitors. The config also lists displays (the kinds of monitor or screen you view on) and views (the ways an image is shown on a display, such as an SDR video look).

This library loads that config and lets you pick a colour setting from it: which colour space your image is in, and which display and view to show it through. Its nodes are in the **Colorspace** category.

The library does not change images itself. It provides the config and colour choice that the OpenEXR library's nodes use for pipeline-specific colour. Load OCIO Config is the node to start with.

## Requirements

- An OCIO config file, such as your studio's ACES config. Set the `$OCIO` environment variable to its path before launching Griptape Nodes, set `OCIO` under **Settings → API Keys & Secrets** (no restart needed), or pick the file on the Load OCIO Config node. In a Griptape project, `OCIO` can be set under `environment:` in `project.yml`.
- Griptape Nodes engine 0.106.0 or later.
- The library installs its own Python dependencies: `opencolorio` 2.5.2 or later, `Pillow` and `numpy`.
- The OpenEXR library, to use the result. It is a separate library.

## Nodes

- [Load OCIO Config](nodes/LoadOCIOConfig.md): load the config from `$OCIO` or from a `.ocio` file you pick.
- [OCIO Color Parameters](nodes/OCIOColorParameters.md): choose the colour space your image is in, plus the display and view to show it through, from that config.

## How they connect

```text
Load OCIO Config -> OCIO Color Parameters -> Read EXR, Display EXR Part,
                                             Display EXR Channel, EXR Sequence to Preview
```

Connect `config` from Load OCIO Config to OCIO Color Parameters, and `color_params` to an OpenEXR node with `color_mode` set to `ocio`. Without these nodes the OpenEXR nodes use their built-in `basic` transforms, which are general-purpose colour conversions that do not depend on your config. The OpenEXR nodes stay on `basic` until you change `color_mode`, whether or not this library is installed.
