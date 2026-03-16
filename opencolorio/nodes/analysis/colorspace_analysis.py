from typing import Any

import numpy as np
import PyOpenColorIO as ocio
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes_library.utils.image_utils import dict_to_image_url_artifact
from PIL import Image


class ColorSpaceAnalysis(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Set category and description
        self.category = "opencolor"
        self.description = "Analyze color properties of images using OCIO"

        # Input image parameter
        image_parameter = Parameter(
            name="image",
            input_types=["ImageArtifact", "ImageUrlArtifact"],
            type="ImageArtifact",
            default_value=None,
            ui_options={"clickable_file_browser": True, "expander": True},
            tooltip="Input image to analyze",
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

        # Color space parameter
        colorspace_parameter = Parameter(
            name="colorspace", input_types=["str"], type="str", default_value="sRGB", tooltip="Color space for analysis"
        )
        self.add_parameter(colorspace_parameter)

        # Analysis options (backward compatible parameter names)
        analyze_histogram_parameter = Parameter(
            name="include_histogram",
            input_types=["bool"],
            type="bool",
            default_value=True,
            tooltip="Include histogram analysis",
        )
        self.add_parameter(analyze_histogram_parameter)

        # Histogram bins parameter
        histogram_bins_parameter = Parameter(
            name="histogram_bins",
            input_types=["int"],
            type="int",
            default_value=256,
            tooltip="Number of bins for histogram analysis",
        )
        self.add_parameter(histogram_bins_parameter)

        # Context variables parameter (backward compatible name)
        context_vars_parameter = Parameter(
            name="context_vars",
            input_types=["dict"],
            type="dict",
            default_value={},
            tooltip="Context variables for the OCIO config",
        )
        self.add_parameter(context_vars_parameter)

        # Status message parameter
        status_message_parameter = Parameter(
            name="status_message",
            input_types=["str"],
            type="str",
            default_value="Ready to analyze image colors",
            tooltip="Current status message",
        )
        self.add_parameter(status_message_parameter)

        # Statistics output
        statistics_parameter = Parameter(
            name="statistics", output_type="dict", type="dict", tooltip="Color statistics and analysis results"
        )
        self.add_parameter(statistics_parameter)

        # Analysis report output
        analysis_report_parameter = Parameter(
            name="analysis_report", output_type="str", type="str", tooltip="Detailed analysis report"
        )
        self.add_parameter(analysis_report_parameter)

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

    def _analyze_color_statistics(self, img_array: np.ndarray) -> dict:
        """Analyze basic color statistics"""
        height, width, channels = img_array.shape

        # Basic statistics per channel
        stats = {}
        channel_names = ["Red", "Green", "Blue"]

        for i, channel_name in enumerate(channel_names):
            channel_data = img_array[:, :, i].flatten()
            stats[channel_name] = {
                "min": float(np.min(channel_data)),
                "max": float(np.max(channel_data)),
                "mean": float(np.mean(channel_data)),
                "std": float(np.std(channel_data)),
                "median": float(np.median(channel_data)),
            }

        # Overall statistics
        stats["Overall"] = {
            "width": width,
            "height": height,
            "total_pixels": width * height,
            "channels": channels,
            "min_luminance": float(np.min(img_array)),
            "max_luminance": float(np.max(img_array)),
            "mean_luminance": float(np.mean(img_array)),
        }

        return stats

    def _analyze_histogram(self, img_array: np.ndarray, bins: int = 256) -> dict:
        """Analyze color histogram"""
        height, width, channels = img_array.shape
        histograms = {}

        channel_names = ["Red", "Green", "Blue"]

        for i, channel_name in enumerate(channel_names):
            channel_data = img_array[:, :, i].flatten()
            hist, bin_edges = np.histogram(channel_data, bins=bins, range=(0, 1))
            histograms[channel_name] = {"histogram": hist.tolist(), "bin_edges": bin_edges.tolist()}

        return histograms

    def _check_color_ranges(self, img_array: np.ndarray) -> dict:
        """Check if colors are within expected ranges"""
        range_analysis = {}

        # Check for out-of-gamut values
        below_zero = np.any(img_array < 0)
        above_one = np.any(img_array > 1)

        range_analysis["out_of_range"] = {
            "below_zero": bool(below_zero),
            "above_one": bool(above_one),
            "min_value": float(np.min(img_array)),
            "max_value": float(np.max(img_array)),
        }

        # Check for clipping
        zero_pixels = np.sum(img_array == 0)
        one_pixels = np.sum(img_array == 1)
        total_pixels = img_array.size

        range_analysis["clipping"] = {
            "zero_pixels": int(zero_pixels),
            "one_pixels": int(one_pixels),
            "zero_percentage": float(zero_pixels / total_pixels * 100),
            "one_percentage": float(one_pixels / total_pixels * 100),
        }

        return range_analysis

    def process(self) -> None:
        image = self.get_parameter_value("image")
        config_data = self.get_parameter_value("config")
        colorspace = self.get_parameter_value("colorspace")
        include_histogram = self.get_parameter_value("include_histogram")
        histogram_bins = self.get_parameter_value("histogram_bins")
        context_vars = self.get_parameter_value("context_vars")

        try:
            if not image:
                raise ValueError("No input image provided")

            # Convert image to artifact if needed
            image_artifact = self._to_image_artifact(image)

            # Load image data
            img_array = self._load_image_data(image_artifact)

            # If config is provided, transform to specified colorspace first
            if config_data and config_data.get("file_path") and colorspace:
                try:
                    config = ocio.Config.CreateFromFile(config_data["file_path"])

                    # Set context variables if provided
                    if context_vars:
                        context = config.getCurrentContext()
                        for key, value in context_vars.items():
                            context.setStringVar(key, str(value))

                    # Transform to specified colorspace (assuming sRGB input)
                    transform = ocio.ColorSpaceTransform()
                    transform.setSrc("sRGB")  # Assume input is sRGB
                    transform.setDst(colorspace)

                    processor = config.getProcessor(transform)
                    cpu_processor = processor.getDefaultCPUProcessor()

                    # Apply transform
                    height, width, channels = img_array.shape
                    flat_img = img_array.reshape(-1, channels)
                    cpu_processor.applyRGB(flat_img)
                    img_array = flat_img.reshape(height, width, channels)

                except Exception:
                    # If transform fails, continue with original image
                    pass

            # Perform analysis
            statistics = {}

            # Basic color statistics
            color_stats = self._analyze_color_statistics(img_array)
            statistics.update(color_stats)

            # Histogram analysis
            if include_histogram:
                histogram_data = self._analyze_histogram(img_array, bins=histogram_bins)
                statistics["Histograms"] = histogram_data

            # Range analysis
            range_data = self._check_color_ranges(img_array)
            statistics["Range Analysis"] = range_data

            # Create detailed report
            report_lines = []
            report_lines.append("# Color Space Analysis Report")
            report_lines.append("")
            report_lines.append(f"**Analysis Color Space:** {colorspace}")
            report_lines.append(
                f"**Image Dimensions:** {statistics['Overall']['width']} x {statistics['Overall']['height']}"
            )
            report_lines.append(f"**Total Pixels:** {statistics['Overall']['total_pixels']:,}")
            report_lines.append("")

            # Color channel statistics
            report_lines.append("## Color Channel Statistics")
            for channel in ["Red", "Green", "Blue"]:
                stats = statistics[channel]
                report_lines.append(f"### {channel} Channel")
                report_lines.append(f"- **Min:** {stats['min']:.4f}")
                report_lines.append(f"- **Max:** {stats['max']:.4f}")
                report_lines.append(f"- **Mean:** {stats['mean']:.4f}")
                report_lines.append(f"- **Std Dev:** {stats['std']:.4f}")
                report_lines.append(f"- **Median:** {stats['median']:.4f}")
                report_lines.append("")

            # Range analysis
            range_info = statistics["Range Analysis"]
            report_lines.append("## Range Analysis")
            if range_info["out_of_range"]["below_zero"] or range_info["out_of_range"]["above_one"]:
                report_lines.append("⚠️ **Out-of-range values detected!**")
                report_lines.append(f"- **Below 0:** {'Yes' if range_info['out_of_range']['below_zero'] else 'No'}")
                report_lines.append(f"- **Above 1:** {'Yes' if range_info['out_of_range']['above_one'] else 'No'}")
            else:
                report_lines.append("✅ **All values within normal range (0-1)**")

            report_lines.append(f"- **Actual Min:** {range_info['out_of_range']['min_value']:.6f}")
            report_lines.append(f"- **Actual Max:** {range_info['out_of_range']['max_value']:.6f}")
            report_lines.append("")

            # Clipping analysis
            clipping = range_info["clipping"]
            report_lines.append("## Clipping Analysis")
            report_lines.append(
                f"- **Black Pixels (0.0):** {clipping['zero_pixels']:,} ({clipping['zero_percentage']:.2f}%)"
            )
            report_lines.append(
                f"- **White Pixels (1.0):** {clipping['one_pixels']:,} ({clipping['one_percentage']:.2f}%)"
            )

            analysis_report = "\n".join(report_lines)

            # Set outputs
            self.parameter_output_values["statistics"] = statistics
            self.parameter_output_values["analysis_report"] = analysis_report

        except Exception as e:
            error_msg = f"Color analysis failed: {str(e)}"
            self.parameter_output_values["statistics"] = {}
            self.parameter_output_values["analysis_report"] = error_msg

    def after_value_set(self, parameter: Parameter, value: Any, modified_parameters_set: set[str]) -> None:
        if parameter.name == "image":
            # Trigger analysis when image changes
            self.process()
        return super().after_value_set(parameter, value, modified_parameters_set)
