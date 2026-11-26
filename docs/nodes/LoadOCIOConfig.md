---
description: Loads the OpenColorIO (OCIO) config that defines your pipeline's colour spaces, displays and views, from $OCIO or from a .ocio file you pick.
when_to_use: "Use it at the start of any colour-managed workflow, to give OCIO Color Parameters and other OCIO nodes the studio or show config."
---

## What it does

This node finds your studio's colour management file and makes it available to the rest of the workflow. That file is an OCIO config (OCIO is short for OpenColorIO). It is a `.ocio` file that lists the colour spaces your pipeline uses. A colour space describes how the numbers in an image map to real colours, for example ACEScg for renders. The file also lists roles, which are standard jobs such as `scene_linear` that point at a colour space, and the displays and views you can show an image through. A display is a kind of monitor or screen. A view is a way of showing an image on that display, for example an SDR video look.

Use it when you want a workflow to follow your studio's or show's colour setup, such as an ACES config, instead of a generic display transform. The node passes the file on as the `config` output. Connect `config` to [OCIO Color Parameters](OCIOColorParameters.md).

## What it needs

- An OCIO config file. By default the node uses the one named by the `$OCIO` environment variable, which most studio workstations already set. You can also set `OCIO` under **Settings → API Keys & Secrets**, which takes effect without a restart. To use a different file, open **Advanced**, set **Override OCIO Config** to on and pick a `.ocio` file.
- OpenColorIO 2.5.2 or later. The library installs it with its other Python dependencies.
- `$OCIO` can also hold the name of one of OpenColorIO's own ACES configs, such as `ocio://default`.
- If `$OCIO` is not set and the override is off, the node still runs, with a minimal config that has one colour space (`raw`), one display (`sRGB`) and one view (`Raw`). A warning on the node tells you `$OCIO` is not set. Set `$OCIO` or use the override to get your real config.
- If the file does not exist or is not a valid config, the run stops and `result_details` shows OpenColorIO's error.

The node does not open the config until you run it. When it succeeds, `result_details` reads "Loaded from" followed by the file path (or `$OCIO`), then the number of colour spaces, displays and roles it found.

## Example

A compositor starts a shot in the show's ACES config. `$OCIO` already points at it, so add **Load OCIO Config**, leave **Advanced** alone and run it. To compare against a candidate config from the colour team, set **Override OCIO Config** to on, pick the candidate `.ocio` file and run again. The rest of the workflow now uses the candidate.

## Parameters

### context_vars

Pairs such as `SHOT` and `SEQ` that some configs use inside file paths. They are stored on the `config` output. OCIO Color Parameters and the EXR display nodes do not use them.

### override_ocio_config

With this set to on, the node loads `file_path` and ignores `$OCIO`. With it off, `$OCIO` is read each time the node runs, so a project that sets `OCIO` in its own environment settings is picked up without reloading the workflow. If this is on but `file_path` is empty, the node uses `$OCIO`. If `file_path` is filled in but the file does not exist, the run stops. It does not fall back to `$OCIO`.

Switching it hides or shows `file_path`. Run the node again afterwards.

### file_path

Only shown while **Override OCIO Config** is on. Its label on the node is **Config File Path**. The picker lists `.ocio` files and opens at the `$OCIO` location when `$OCIO` is set. Run the node again after you change the file.

### config

When the node uses `$OCIO`, the output holds no file path. Each node that receives it reads `$OCIO` again when it runs. When the override is on, the output holds the path you picked.

While the override is off, the node shows the `$OCIO` value it saw when the node was added or the workflow was opened, or a warning if `$OCIO` was not set. If you reopen a workflow and `$OCIO` has changed since the last run, the node warns with the old and new paths. Run it again so downstream nodes pick up the new config.
