- **Breaking:** `OCIO Color Parameters` now fails when `Source Colorspace`, `Display` or `View` is
  left empty, instead of emitting an empty bundle that downstream transforms silently ignored. Each
  dropdown offers a new `<default>` entry as its first choice, which resolves from the OCIO config's
  own roles, File Rules and default display and view, and new nodes ship with all three set to it.
  To fix a workflow that now fails, set the empty parameter to `<default>` or pick a value. Nodes
  show the resolved value as a placeholder and name the layer it came from in a badge beside the
  label. This release also requires Griptape Nodes engine 0.106.0 or newer.
  [#40](https://github.com/griptape-ai/griptape-nodes-library-opencolorio/issues/40)
