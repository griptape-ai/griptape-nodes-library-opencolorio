# OpenColorIO Library for Griptape Nodes

Professional color management nodes for Griptape Nodes, built on [OpenColorIO](https://opencolorio.org/) (OCIO) — the industry-standard color management system used across film, VFX, and animation pipelines.

---

## For Artists

### Prerequisites

This library uses the `$OCIO` environment variable as the primary way to identify your color configuration. Set it in your environment before launching Griptape Nodes:

```bash
export OCIO=/path/to/your/config.ocio
```

Most studio workstations and render environments will have `$OCIO` set already. If you are working with a Griptape **project**, you can set it per-project in `project.yml`:

```yaml
environment:
  OCIO: "{project_dir}/config/aces.ocio"
```

The `$OCIO` value is re-read live at each node execution, so switching projects automatically picks up the new config without reloading the workflow.

### Quick Start

1. Add a **Load OCIO Config** node to your canvas.
   - If `$OCIO` is set, the node detects and displays the path automatically.
   - The colorspace, display, and role dropdowns on downstream nodes populate from the loaded config.
2. Wire the `config` output to any transform node that accepts an `OCIOConfigArtifact`.
3. Run the workflow.

### Overriding the Config (Advanced)

If you need to use a specific config file without changing your environment — for example, to test a new config during look development — open the **Advanced** group on the **Load OCIO Config** node:

- Enable **Override OCIO Config** to reveal a file picker.
- Select your `.ocio` file. The node will use this path instead of `$OCIO`.
- Disable the toggle to return to environment-variable mode.

### Choosing the Config

Three ways to point at a config, in the order a node consults them:

1. **Override OCIO Config** on the **Load OCIO Config** node, saved with the workflow.
2. `$OCIO` in the environment the engine starts in.
3. The **Fallback config** setting, for when `$OCIO` is unset.

There is no per-project pin today. `secrets_to_register` in a project-adjacent
`griptape_nodes_config.json` does not provide one: the engine writes those values to the *global*
`~/.config/griptape_nodes/.env`, and only when the secret is not already set, so the first value
written wins for every project afterwards. Set `$OCIO` in the environment that launches the
engine, or use the node override, which travels with the workflow.

### Default Colour Parameters

Set a parameter to `<default>` and the node resolves it for you. For each value the first answer
available wins:

1. What you picked on the node, when it is not `<default>`.
2. A **Settings → OpenColorIO** override.
3. The config's own answer: the `scene_linear` role, the File Rules for the file being read, and
   the default display and view.
4. Nothing, leaving the value empty, which happens only when no config is in play.

The settings layer sits above the config so that a show can output Rec.709 on a config whose
default display is sRGB, without forking the config. Four settings are available, each shipping
as `<default>`, which means the same as unset:

| Setting | `<default>` means |
|---|---|
| **Working space** | Use the config's `scene_linear` role. |
| **Display** | Use the config's default display. |
| **View** | Use the config's default view for the chosen display. |
| **Fallback config** | Leave colour management disabled when `$OCIO` is unset. |

Clearing one to blank also reads as unset, but the setting then disappears from the panel until
you set it again: the engine drops blank-string entries as it loads each config layer, and the
editor lists the keys it finds in the config rather than the ones the library declares.

A value that does not exist in your config logs a warning and falls through to the config's own
answer rather than failing a render. Nodes report which layer answered, so you can always see why
a value was chosen.

These settings apply to parameters left on `<default>`. A parameter left *blank* is an error, not
a request for a default; the node says so and names the fix.

**Fallback config** accepts `ocio://` URIs only, such as `ocio://cg-config-latest` or a pinned
`ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5`. These are the configs that ship inside OpenColorIO,
so they cannot go stale the way a path can. It applies only when `$OCIO` is unset: a `$OCIO` that
points at a config which will not load stays an error.

### OCIO Color Parameters

The **OCIO Color Parameters** node bundles the three values most transform nodes need — source colorspace, display, and view — into a single reusable `OCIOColorParamsArtifact`.

**Basic usage:**

1. Wire the `config` output of a **Load OCIO Config** node to the `OCIO Config` input.
2. The three dropdowns populate from the connected config.
3. Run the node — it emits an `OCIOColorParamsArtifact` carrying your selections.

**Dropdown behaviour:**

- `<default>` is the first choice in each dropdown and is what a new node ships with. It resolves
  when the node runs, so a node left on `<default>` follows the config rather than freezing
  today's answer. The resolved value is shown greyed as a placeholder, and the badge beside the
  label says which layer it came from.
- **Source Colorspace** lists role aliases next (e.g. `scene_linear`, `compositing_log`) so the
  stable, pipeline-safe names appear near the top, followed by all colorspace names from the config.
- **View** updates automatically whenever you change the **Display** selection.
- Without a connected config the dropdowns offer `<default>` alone, which resolves from `$OCIO`.

**An empty parameter is an error:**

Leaving **Source Colorspace**, **Display** or **View** blank fails the node, with a message naming
the fix: set it to `<default>` or pick a value. Blank is not quietly treated as `<default>`,
because that would change what an existing workflow renders without saying so.

**Validation:**

On execution the node re-validates all three values against the live config. If a value is no longer present (e.g. the config changed, or a value was typed in manually via a wired INPUT), an inline warning appears on the node. Execution still succeeds — the artifact is emitted with the values as-is — but the warning flags the mismatch for the artist.

**Reuse:**

A single **OCIO Color Parameters** node can drive multiple downstream transform nodes. Wire its `color_params` output to each consumer to keep the selection in one place.

### Implemented Nodes

| Node | Category | Description |
|------|----------|-------------|
| Load OCIO Config | Colorspace | Loads an OCIO config from `$OCIO` or an explicit path; emits an `OCIOConfigArtifact` for downstream nodes |
| OCIO Color Parameters | Colorspace | Bundles source colorspace, display, and view into a reusable `OCIOColorParamsArtifact`; dropdowns populate from a connected OCIO config |
| Colorspace Transform | Colorspace | Converts an image from one colorspace to another using a loaded OCIO config |

### Typical Workflow

```
Load OCIO Config → OCIO Color Parameters → (downstream transform nodes)
Load OCIO Config → Colorspace Transform → (output image)
```

---

## For Library Developers

### Installation

Dependencies are declared in `griptape-nodes-library.json` and installed automatically when the library is registered. To install manually for development:

```bash
make install/dev
```

Use the `make` target rather than `uv sync --group dev` directly. Dev tooling belongs in
`.venv-dev`, and plain `uv` would install it into the `.venv` the engine builds beside the
manifest, which makes the library load as FLAWED. See [CONTRIBUTING.md](CONTRIBUTING.md) for why.

### Registering the Library

1. In Griptape Nodes, go to **Settings → App Events → Libraries to Register**.
2. Add the absolute path to `griptape-nodes-library.json` in this repository.
3. Restart Griptape Nodes.
4. Nodes appear under the **Colorspace** category.

### Settings Registration

`griptape-nodes-library.json` carries two `settings` entries.

The first registers `OCIO` as a known environment variable, which surfaces it in the Griptape
Nodes settings UI so users can inspect or set it without leaving the application.

The second declares the `opencolorio` config category holding `default_working_space`,
`default_display`, `default_view` and `fallback_config`. These are top-level scalar strings
because the editor's Library settings page renders those and skips nested dicts. Each has a
`schema` entry giving it a typed control with help text.

Two things to know before adding a key here. Library settings are merged into the user's global
config at registration time, with existing user values winning, and they are never pruned when a
library is unregistered, so a key shipped and later renamed is left behind in every user's config
file. That is why only three colour slots ship today: `thumbnail_colorspace` and `data_colorspace`
wait for the node that first reads them. `tests/unit/test_library_manifest.py` pins the shape.

### Repository Layout

```
opencolorio/
  nodes/
    config/
      load_ocio_config.py         # LoadOCIOConfig node
      ocio_color_parameters.py    # OCIOColorParameters node
  artifacts/
    ocio_config_artifact.py       # OCIOConfigArtifact dataclass
    ocio_color_params_artifact.py # OCIOColorParamsArtifact dataclass
  services/
    colorspace_transform.py       # ColorspaceTransformRequest and its handler
  defaults.py                     # resolve_color_params(), the defaults resolver
  ocio_helpers.py                 # load_ocio_config(), extract_lists()
  advanced_library.py             # AdvancedNodeLibrary registration
griptape-nodes-library.json
```

### Adding a New Node

1. Create the node class under `opencolorio/nodes/<category>/`.
2. Add an entry to the `nodes` array in `griptape-nodes-library.json` (or register via `AdvancedNodeLibrary` if the node depends on services).
3. Write tests under `tests/unit/`, named `test_<module>.py`.

### Testing

```bash
make test/unit
```

A direct `uv run` needs `UV_PROJECT_ENVIRONMENT=.venv-dev`, for the reason above.
