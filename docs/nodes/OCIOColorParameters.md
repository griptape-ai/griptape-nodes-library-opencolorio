---
description: Bundles a source colour space, a display and a view from an OCIO config into one colour setting that EXR display nodes can use.
when_to_use: Use it when an EXR node should show an image through your pipeline's display transform, such as an ACES view, instead of its built-in transforms.
---

## When to use it

You want an EXR preview to match what your pipeline shows, for example an ACEScg render through the show's SDR view. This node lets you choose the colour space the pixels are in, the display and the view, and outputs the choice as `color_params`. One node can feed several EXR nodes, so the choice is made in one place.

The OpenEXR library's **Read EXR**, **Display EXR Part**, **Display EXR Channel** and **EXR Sequence to Preview** nodes accept `color_params`. On each of them, set `color_mode` to `ocio`, which shows `color_params`, and connect it. The node then transforms the pixels from the source colour space through the display and view you chose. The run stops with a message if `color_mode` is `ocio` and nothing is connected, if a name is not in the config, or if this library is not loaded.

## What it needs

- A config from [Load OCIO Config](LoadOCIOConfig.md) connected to `config`. Without one the dropdowns are empty and the node outputs empty values.
- Run Load OCIO Config first. The dropdowns fill from the config it outputs when it is connected. If Load OCIO Config has not run, the dropdowns stay empty. Disconnecting it empties them again.
- If `display` or `view` is empty, an EXR node leaves the pixels unchanged and applies no display transform.

## Example

An ACES show config is loaded. In **OCIO Color Parameters**, set `source_colorspace` to `scene_linear` for scene-linear renders, or `ACEScg` if you want to name the space directly. Set `display` to the display your monitors use and `view` to the SDR view. The names come from your config. Connect `color_params` to **Read EXR** and **EXR Sequence to Preview**, so the still and the review video look the same.

## Parameters

### source_colorspace

Names the space the pixels are stored in. The list shows the config's roles first, in alphabetical order, such as `aces_interchange`, `compositing_log` and `scene_linear`. After them come every colour space in the config's own order. That includes display-referred spaces such as `sRGB - Display`, which are not the source space of a scene-linear render, and the data space `Raw`. A role name keeps working when a later version of the config renames the colour space behind it. A colour space name used directly may not exist in the next version.

The default is the first role in the list. In the ACES configs that is `aces_interchange`, which is ACES2065-1, not the ACEScg space most renders use. Choose `scene_linear` (ACEScg in these configs) or `ACEScg` yourself for ACEScg renders.

### display

The list holds the displays defined in the config, in the config's order. The default is the first one, `sRGB - Display` in the ACES configs. Choosing a display changes the `view` list.

### view

The list holds the views for the chosen display, in the config's order. The default is the first view of the default display, for example `ACES 2.0 - SDR 100 nits (Rec.709)` in the ACES 2.0 configs and `ACES 1.0 - SDR Video` in the ACES 1.3 CG config. When you change `display`, `view` keeps its value if the new display has it. Otherwise it changes to the first view in the list.

### color_params

Holds the three names and the config file, or `$OCIO` if the config came from the environment.

Only names from the lists can be chosen. When a config is connected or re-run, a saved name that is not in the new lists is replaced by the first entry. If `$OCIO` changes and you run this node without re-running Load OCIO Config, the lists are out of date. A name that is no longer in the config does not stop this node. `result_details` lists it under "Emitted with warnings", and a downstream EXR node that receives it stops with a message.
