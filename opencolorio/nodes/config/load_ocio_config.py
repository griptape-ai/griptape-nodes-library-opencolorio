import os

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import DataNode


class LoadOCIOConfig(DataNode):
    """
    Load and validate OpenColorIO configuration files.

    This node loads an OCIO config file and provides the configuration object
    along with extracted metadata including available color spaces, looks, and displays.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Set category and description
        self.category = "opencolor"
        self.description = "Load and validate an OpenColorIO configuration"

        # Config file input parameter
        config_file_parameter = Parameter(
            name="config_file",
            input_types=["str"],
            type="str",
            default_value="",
            ui_options={"clickable_file_browser": True},
            tooltip="Path to OCIO config file (.ocio)",
        )
        self.add_parameter(config_file_parameter)

        # Config context variable parameter
        context_parameter = Parameter(
            name="context_variables",
            input_types=["dict"],
            type="dict",
            default_value={},
            tooltip="Context variables for the OCIO config (key-value pairs)",
        )
        self.add_parameter(context_parameter)

        # Config output parameter
        config_output_parameter = Parameter(
            name="config", output_type="dict", type="dict", tooltip="Loaded OCIO configuration data"
        )
        self.add_parameter(config_output_parameter)

        # Validation status output
        validation_output_parameter = Parameter(
            name="validation_status",
            output_type="str",
            type="str",
            tooltip="Configuration validation status and messages",
        )
        self.add_parameter(validation_output_parameter)

        # Color spaces output
        colorspaces_output_parameter = Parameter(
            name="colorspaces", output_type="list", type="list", tooltip="List of available color spaces in the config"
        )
        self.add_parameter(colorspaces_output_parameter)

    def process(self) -> None:
        config_file = self.get_parameter_value("config_file")
        context_variables = self.get_parameter_value("context_variables") or {}

        try:
            # Import PyOpenColorIO only when needed
            try:
                import PyOpenColorIO as ocio
            except ImportError as e:
                raise ValueError("PyOpenColorIO is not installed. Please install it to use OpenColorIO nodes.") from e

            # Load the OCIO config
            if not config_file or not os.path.exists(config_file):
                raise ValueError(f"Config file not found: {config_file}")

            config = ocio.Config.CreateFromFile(config_file)

            # Set context variables if provided
            if context_variables:
                context = config.getCurrentContext()
                for key, value in context_variables.items():
                    context.setStringVar(key, str(value))

            # Validate the config
            try:
                config.validate()
                validation_status = "Valid configuration"
            except Exception as e:
                validation_status = f"Config validation warning: {str(e)}"

            # Get color spaces
            colorspaces = []
            for i in range(config.getNumColorSpaces()):
                cs = config.getColorSpace(i)
                colorspaces.append({"name": cs.getName(), "family": cs.getFamily(), "description": cs.getDescription()})

            # Prepare config data for output
            config_data = {
                "file_path": config_file,
                "description": config.getDescription(),
                "working_dir": config.getWorkingDir(),
                "search_path": config.getSearchPath(),
                "num_colorspaces": config.getNumColorSpaces(),
                "context_variables": context_variables,
            }

            # Set outputs
            self.parameter_output_values["config"] = config_data
            self.parameter_output_values["validation_status"] = validation_status
            self.parameter_output_values["colorspaces"] = [cs["name"] for cs in colorspaces]

        except Exception as e:
            error_msg = f"Failed to load OCIO config: {str(e)}"
            self.parameter_output_values["config"] = {}
            self.parameter_output_values["validation_status"] = error_msg
            self.parameter_output_values["colorspaces"] = []
