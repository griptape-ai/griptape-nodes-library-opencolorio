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

### Implemented Nodes

| Node | Category | Description |
|------|----------|-------------|
| Load OCIO Config | Colorspace | Loads an OCIO config from `$OCIO` or an explicit path; emits an `OCIOConfigArtifact` for downstream nodes |
| Colorspace Transform | Colorspace | Converts an image from one colorspace to another using a loaded OCIO config |

### Typical Workflow

```
Load OCIO Config → Colorspace Transform → (output image)
```

---

## For Library Developers

### Installation

Dependencies are declared in `griptape-nodes-library.json` and installed automatically when the library is registered. To install manually for development:

```bash
uv sync --group dev
```

### Registering the Library

1. In Griptape Nodes, go to **Settings → App Events → Libraries to Register**.
2. Add the absolute path to `griptape-nodes-library.json` in this repository.
3. Restart Griptape Nodes.
4. Nodes appear under the **Colorspace** category.

### Environment Variable Registration

The library registers `OCIO` as a known environment variable via the `settings` block in
`griptape-nodes-library.json`. This surfaces the variable in the Griptape Nodes settings UI
so users can inspect or set it without leaving the application.

### Repository Layout

```
opencolorio/
  nodes/
    config/
      load_ocio_config.py       # LoadOCIOConfig node
    transform/
      colorspace_transform.py   # ColorspaceTransform node (via AdvancedNodeLibrary)
  artifacts/
    ocio_config_artifact.py     # OCIOConfigArtifact dataclass
  services/
    colorspace_transform_service.py
  ocio_helpers.py               # load_ocio_config(), extract_lists()
  advanced_library.py           # AdvancedNodeLibrary registration
griptape-nodes-library.json
```

### Adding a New Node

1. Create the node class under `opencolorio/nodes/<category>/`.
2. Add an entry to the `nodes` array in `griptape-nodes-library.json` (or register via `AdvancedNodeLibrary` if the node depends on services).
3. Write tests under `tests/nodes/<category>/`.

### Testing

```bash
uv run pytest tests/
```
