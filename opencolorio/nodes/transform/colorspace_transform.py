from typing import Any

import numpy as np
import PyOpenColorIO as ocio
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes_library.utils.image_utils import dict_to_image_url_artifact
from PIL import Image


class ColorSpaceTransform(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Set category and description
        self.category = "opencolor"
        self.description = "Transform images between color spaces using OCIO"

        # Input image parameter
        image_parameter = Parameter(
            name="image",
            input_types=["ImageArtifact", "ImageUrlArtifact"],
            type="ImageArtifact",
            default_value=None,
            ui_options={"clickable_file_browser": True, "expander": True},
            tooltip="Input image to transform",
        )
        self.add_parameter(image_parameter)

        # Config parameter
        config_parameter = Parameter(
            name="config",
            input_types=["dict"],
            type="dict",
            default_value={},
            tooltip="OCIO configuration from LoadOCIOConfig node",
        )
        self.add_parameter(config_parameter)

        # Source color space parameter
        source_cs_parameter = Parameter(
            name="source_colorspace",
            input_types=["str"],
            type="str",
            default_value="sRGB",
            tooltip="Source color space name",
        )
        self.add_parameter(source_cs_parameter)

        # Target color space parameter
        target_cs_parameter = Parameter(
            name="target_colorspace",
            input_types=["str"],
            type="str",
            default_value="ACES - ACEScg",
            tooltip="Target color space name",
        )
        self.add_parameter(target_cs_parameter)

        # Output image parameter
        output_parameter = Parameter(
            name="output_image", output_type="ImageUrlArtifact", type="ImageUrlArtifact", tooltip="Transformed image"
        )
        self.add_parameter(output_parameter)

        # Transform info output
        transform_info_parameter = Parameter(
            name="transform_info", output_type="str", type="str", tooltip="Information about the applied transform"
        )
        self.add_parameter(transform_info_parameter)

    def _to_image_artifact(self, image: Any) -> Any:
        """Convert image dict to artifact if needed"""
        if isinstance(image, dict):
            return dict_to_image_url_artifact(image)
        return image

    def _load_image_data(self, image_artifact: Any) -> np.ndarray:
        """Load image data from artifact as numpy array"""
        if hasattr(image_artifact, "to_bytes"):
            from io import BytesIO

            img_bytes = image_artifact.to_bytes()
            pil_image = Image.open(BytesIO(img_bytes))
        else:
            # Handle dict format
            if isinstance(image_artifact, dict) and "value" in image_artifact:
                import base64
                from io import BytesIO

                img_data = base64.b64decode(image_artifact["value"])
                pil_image = Image.open(BytesIO(img_data))
            else:
                raise ValueError("Unsupported image format")

        # Convert to RGB if needed
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert to numpy array (0-1 range)
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        return img_array

    def _array_to_artifact(self, img_array: np.ndarray) -> Any:
        """Convert numpy array back to image artifact"""
        # Clip and convert back to 0-255 range
        img_array = np.clip(img_array, 0, 1)
        img_uint8 = (img_array * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(img_uint8, "RGB")

        # Convert to bytes
        import base64
        from io import BytesIO

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Create artifact dict
        artifact_dict = {"type": "ImageUrlArtifact", "value": img_base64}

        return dict_to_image_url_artifact(artifact_dict)

    def process(self) -> None:
        image = self.get_parameter_value("image")
        config_data = self.get_parameter_value("config")
        source_cs = self.get_parameter_value("source_colorspace")
        target_cs = self.get_parameter_value("target_colorspace")

        try:
            if not image:
                raise ValueError("No input image provided")

            if not config_data or not config_data.get("file_path"):
                raise ValueError("No OCIO config provided")

            # Convert image to artifact if needed
            image_artifact = self._to_image_artifact(image)

            # Load image data
            img_array = self._load_image_data(image_artifact)
            height, width, channels = img_array.shape

            # Load OCIO config
            config = ocio.Config.CreateFromFile(config_data["file_path"])

            # Set context variables if provided
            if config_data.get("context_variables"):
                context = config.getCurrentContext()
                for key, value in config_data["context_variables"].items():
                    context.setStringVar(key, str(value))

            # Create color space transform
            transform = ocio.ColorSpaceTransform()
            transform.setSrc(source_cs)
            transform.setDst(target_cs)

            # Get the processor
            processor = config.getProcessor(transform)
            cpu_processor = processor.getDefaultCPUProcessor()

            # Flatten image for processing
            flat_img = img_array.reshape(-1, channels)

            # Apply transform
            cpu_processor.applyRGB(flat_img)

            # Reshape back
            transformed_img = flat_img.reshape(height, width, channels)

            # Convert back to artifact
            output_artifact = self._array_to_artifact(transformed_img)

            # Create transform info
            transform_info = f"Transformed from '{source_cs}' to '{target_cs}'\n"
            transform_info += f"Image size: {width}x{height}\n"
            transform_info += f"Config: {config_data.get('file_path', 'Unknown')}"

            # Set outputs
            self.parameter_output_values["output_image"] = output_artifact
            self.parameter_output_values["transform_info"] = transform_info

        except Exception as e:
            error_msg = f"Color space transform failed: {str(e)}"
            self.parameter_output_values["output_image"] = None
            self.parameter_output_values["transform_info"] = error_msg

    def after_value_set(self, parameter: Parameter, value: Any, modified_parameters_set: set[str]) -> None:
        if parameter.name == "image":
            image_artifact = self._to_image_artifact(value)
            self.parameter_output_values["output_image"] = image_artifact
        return super().after_value_set(parameter, value, modified_parameters_set)
