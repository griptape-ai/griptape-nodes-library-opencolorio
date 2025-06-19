from typing import Any
import os
from pathlib import Path
import numpy as np
from PIL import Image
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes_library.utils.image_utils import dict_to_image_url_artifact


class EXRLoader(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # Set category and description
        self.category = "opencolor"
        self.description = "Load OpenEXR files for HDR color workflows"
        
        # EXR file input parameter
        exr_file_parameter = Parameter(
            name="exr_file",
            input_types=["ImageArtifact", "ImageUrlArtifact"],
            type="ImageArtifact",
            default_value=None,
            ui_options={"clickable_file_browser": True, "expander": True, "file_extensions": [".exr"]},
            tooltip="EXR file to load"
        )
        self.add_parameter(exr_file_parameter)
        
        # Channel selection parameter
        channel_selection_parameter = Parameter(
            name="channel_selection",
            input_types=["str"],
            type="str",
            default_value="RGB",
            tooltip="Which channels to load from multi-channel EXR"
        )
        self.add_parameter(channel_selection_parameter)
        
        # Preserve HDR parameter
        preserve_hdr_parameter = Parameter(
            name="preserve_hdr",
            input_types=["bool"],
            type="bool",
            default_value=True,
            tooltip="Preserve high dynamic range values (don't clamp to 0-1)"
        )
        self.add_parameter(preserve_hdr_parameter)
        
        # Output image parameter
        output_image_parameter = Parameter(
            name="output_image",
            output_type="ImageUrlArtifact",
            type="ImageUrlArtifact",
            tooltip="Loaded EXR image"
        )
        self.add_parameter(output_image_parameter)
        
        # EXR metadata output
        exr_metadata_parameter = Parameter(
            name="exr_metadata",
            output_type="dict",
            type="dict",
            tooltip="EXR file metadata and channel information"
        )
        self.add_parameter(exr_metadata_parameter)
        
        # File info output
        file_info_parameter = Parameter(
            name="file_info",
            output_type="str",
            type="str",
            tooltip="File information summary"
        )
        self.add_parameter(file_info_parameter)

    def _get_file_path_from_artifact(self, artifact: Any) -> str:
        """Extract file path from artifact"""
        if hasattr(artifact, 'value'):
            # Handle URL artifact
            if isinstance(artifact.value, str):
                return artifact.value
        elif hasattr(artifact, 'to_bytes'):
            # Handle binary artifact - save to temp file
            import tempfile
            img_bytes = artifact.to_bytes()
            with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as temp_file:
                temp_file.write(img_bytes)
                return temp_file.name
        elif isinstance(artifact, dict) and 'value' in artifact:
            # Handle dict format
            return artifact['value']
        else:
            # Fallback - assume it's a file path string
            return str(artifact)

    def _load_exr_with_imageio(self, file_path: str, channels: str, preserve_hdr: bool) -> tuple:
        """Load EXR file using imageio library"""
        try:
            import imageio.v3 as iio
            
            # Read EXR file
            image_data = iio.imread(file_path)
            
            # Get metadata
            metadata = {}
            try:
                meta = iio.immeta(file_path)
                metadata.update(meta)
            except Exception:
                pass
            
            # Handle channel selection
            if len(image_data.shape) == 3:  # Multi-channel
                if channels == "RGB" and image_data.shape[2] >= 3:
                    image_data = image_data[:, :, :3]
                elif channels == "RGBA" and image_data.shape[2] >= 4:
                    image_data = image_data[:, :, :4]
            
            # Handle HDR preservation
            if not preserve_hdr:
                # Clamp to 0-1 range for standard display
                image_data = np.clip(image_data, 0, 1)
            
            # Ensure we have at least RGB channels
            if len(image_data.shape) == 2:
                # Grayscale to RGB
                image_data = np.stack([image_data] * 3, axis=-1)
            elif len(image_data.shape) == 3 and image_data.shape[2] == 1:
                # Single channel to RGB
                image_data = np.repeat(image_data, 3, axis=-1)
            
            return image_data, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load EXR with imageio: {e}")

    def _numpy_to_artifact(self, image_array: np.ndarray) -> Any:
        """Convert numpy array to image artifact"""
        # Ensure we have valid data
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32)
        
        # For display, we need to convert to 0-255 range
        if image_array.max() <= 1.0 and image_array.min() >= 0.0:
            # Standard 0-1 range
            image_uint8 = (image_array * 255).astype(np.uint8)
        else:
            # HDR data - normalize for display while preserving relative values
            img_min, img_max = image_array.min(), image_array.max()
            if img_max > img_min:
                normalized = (image_array - img_min) / (img_max - img_min)
            else:
                normalized = image_array
            image_uint8 = (normalized * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if len(image_uint8.shape) == 3 and image_uint8.shape[2] >= 3:
            pil_image = Image.fromarray(image_uint8[:, :, :3], 'RGB')
        else:
            pil_image = Image.fromarray(image_uint8, 'RGB')
        
        # Convert to bytes
        from io import BytesIO
        import base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Create artifact dict
        artifact_dict = {
            "type": "ImageUrlArtifact",
            "value": img_base64
        }
        
        return dict_to_image_url_artifact(artifact_dict)

    def process(self) -> None:
        exr_file = self.get_parameter_value("exr_file")
        channel_selection = self.get_parameter_value("channel_selection")
        preserve_hdr = self.get_parameter_value("preserve_hdr")
        
        try:
            if not exr_file:
                raise ValueError("EXR file is required")
            
            # Get file path from artifact
            file_path = self._get_file_path_from_artifact(exr_file)
            
            # Load EXR file
            image_array, exr_metadata = self._load_exr_with_imageio(
                file_path, channel_selection, preserve_hdr
            )
            
            # Convert to artifact
            output_artifact = self._numpy_to_artifact(image_array)
            
            # Prepare metadata output
            filename = Path(file_path).name
            metadata_output = {
                "file_path": file_path,
                "filename": filename,
                "channels_loaded": channel_selection,
                "preserve_hdr": preserve_hdr,
                "dimensions": {
                    "width": image_array.shape[1],
                    "height": image_array.shape[0],
                    "channels": image_array.shape[2] if len(image_array.shape) > 2 else 1
                },
                "data_range": {
                    "min": float(image_array.min()),
                    "max": float(image_array.max()),
                    "mean": float(image_array.mean())
                },
                "exr_metadata": exr_metadata
            }
            
            # Create file info summary
            info_lines = []
            info_lines.append(f"# EXR File Loaded")
            info_lines.append("")
            info_lines.append(f"**File:** {filename}")
            info_lines.append(f"**Dimensions:** {metadata_output['dimensions']['width']} x {metadata_output['dimensions']['height']}")
            info_lines.append(f"**Channels:** {channel_selection}")
            info_lines.append(f"**HDR Preserved:** {preserve_hdr}")
            info_lines.append("")
            info_lines.append("## Data Range")
            info_lines.append(f"- **Min Value:** {metadata_output['data_range']['min']:.6f}")
            info_lines.append(f"- **Max Value:** {metadata_output['data_range']['max']:.6f}")
            info_lines.append(f"- **Mean Value:** {metadata_output['data_range']['mean']:.6f}")
            
            if not preserve_hdr and (metadata_output['data_range']['max'] > 1.0 or metadata_output['data_range']['min'] < 0.0):
                info_lines.append("")
                info_lines.append("⚠️ **Note:** HDR data was clamped to 0-1 range for display compatibility.")
            
            file_info = "\n".join(info_lines)
            
            # Set outputs
            self.parameter_output_values["output_image"] = output_artifact
            self.parameter_output_values["exr_metadata"] = metadata_output
            self.parameter_output_values["file_info"] = file_info
            
        except ImportError as e:
            error_msg = f"Missing dependency: {str(e)}. Install imageio[pyexr]: pip install 'imageio[pyexr]'"
            self.parameter_output_values["output_image"] = None
            self.parameter_output_values["exr_metadata"] = {}
            self.parameter_output_values["file_info"] = f"# Import Error\n\n{error_msg}"
            
        except Exception as e:
            error_msg = f"Failed to load EXR: {str(e)}"
            self.parameter_output_values["output_image"] = None
            self.parameter_output_values["exr_metadata"] = {}
            self.parameter_output_values["file_info"] = f"# EXR Loading Error\n\n{error_msg}" 