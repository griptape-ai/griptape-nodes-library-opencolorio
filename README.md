# OpenColorIO Library for Griptape Nodes

Professional color management tools for film, VFX, and animation using OpenColorIO (OCIO).

## Overview

This library provides a comprehensive set of Griptape Nodes for working with OpenColorIO, the industry-standard color management system. It enables visual node-based workflows for color space conversions, look development, display transforms, and advanced color analysis.

## Features

### Core Capabilities
- **OCIO Configuration Management**: Load and analyze OCIO config files
- **Color Space Transforms**: Convert images between different color spaces
- **Look Development**: Apply artistic looks and creative color transforms
- **Display Transforms**: Prepare images for specific display devices
- **Color Analysis**: Analyze gamut coverage and color properties
- **LUT Operations**: Generate, load, and save Look-Up Tables
- **Batch Processing**: Apply transforms to multiple images efficiently

### Node Categories
All nodes are organized under the **OpenColor** category and include:

#### Configuration Nodes
- **Load OCIO Config**: Load and validate OCIO configuration files
- **OCIO Config Info**: Extract detailed information from OCIO configs

#### Transform Nodes  
- **Color Space Transform**: Convert images between color spaces
- **Apply Look**: Apply artistic looks to images
- **Display Transform**: Apply display transforms for specific devices
- **Custom Transform**: Apply custom transform matrices or operations

#### Analysis Nodes
- **Color Space Analysis**: Analyze image color properties and statistics
- **Gamut Check**: Check for out-of-gamut pixels and generate warnings

#### Utility Nodes
- **LUT Generator**: Generate LUTs from OCIO transforms
- **Batch Color Transform**: Apply color transforms to multiple images

#### I/O Nodes
- **Load LUT**: Load LUT files for use in transforms
- **Save LUT**: Save LUT data to various formats

## Installation

This library requires the following dependencies:
- `PyOpenColorIO>=2.3.0` - Official OpenColorIO Python bindings
- `Pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical operations

Dependencies are automatically installed when the library is registered in Griptape Nodes.

## Usage

### Basic Color Space Conversion Workflow

1. **Load OCIO Config** → Load your OCIO configuration file
2. **Color Space Transform** → Connect input image and config, select source/destination color spaces
3. Output the transformed image

### Look Development Workflow  

1. **Load OCIO Config** → Load configuration with defined looks
2. **Apply Look** → Connect image, config, and select desired look
3. **Display Transform** → Apply display transform for preview
4. Compare results visually

### Batch Processing Workflow

1. **Load OCIO Config** → Load your configuration  
2. **Batch Color Transform** → Connect multiple input images and transform settings
3. Process all images with consistent color transforms

### Analysis Workflow

1. **Load OCIO Config** → Load configuration
2. **Color Space Analysis** → Analyze color properties of input images
3. **Gamut Check** → Identify out-of-gamut pixels for quality control

## OCIO Configuration Files

This library works with standard OCIO configuration files (.ocio). Popular configs include:

- **ACES** - Academy Color Encoding System configs
- **OpenColorIO-Configs** - Community-maintained configurations  
- **Studio-specific** - Custom configurations for specific pipelines

Config files define:
- Available color spaces (sRGB, Rec709, ACES, etc.)
- Artistic looks and creative transforms
- Display devices and viewing transforms
- LUT-based transforms

## Advanced Features

### Dynamic Parameter Population
- Color space dropdowns automatically populate from loaded configs
- Look and display options update based on config contents
- Context variable support for flexible configurations

### Performance Optimization
- Processor caching for repeated transforms
- Batch processing with parallel execution
- Memory-efficient handling of large images

### Professional Workflows
- Soft-proofing for different display devices
- Gamut mapping and out-of-gamut warnings
- LUT generation for offline processing
- Statistical analysis of color data

## Node Library Registration

To use this library in Griptape Nodes:

1. Ensure the OpenColorIO library is installed in your workspace
2. Navigate to Settings → App Events → Libraries to Register
3. Add the absolute path to: `opencolorio/griptape-nodes-library.json`
4. Restart Griptape Nodes
5. Find the nodes under the "OpenColor" category in the Libraries panel

## Example Workflows

### Film Dailies Pipeline
```
Load OCIO Config → Color Space Transform (Log to sRGB) → Display Transform (Rec709) → Output
```

### VFX Shot Preparation  
```
Load OCIO Config → Color Space Transform (sRGB to ACES) → Apply Look (Film Emulation) → Save LUT
```

### Quality Control
```
Load OCIO Config → Gamut Check → Color Space Analysis → Generate Report
```

## Technical Requirements

- **OpenColorIO**: Version 2.3.0 or higher
- **Python**: 3.8+ with numpy and Pillow
- **Griptape Nodes**: Compatible with current engine version
- **Memory**: Sufficient RAM for processing large images (4GB+ recommended)

## Supported Formats

### Images
- Common formats: JPEG, PNG, TIFF, EXR, DPX
- HDR formats: EXR, TIFF (16-bit+)
- Sequence processing: Individual frames or batches

### LUTs
- **.cube** - Industry standard 3D LUT format
- **.3dl** - Autodesk/Discreet 3D LUT format  
- **.lut** - Various 1D/3D LUT formats
- **Custom formats** via OCIO transform definitions

## Color Spaces

Supports all color spaces defined in your OCIO configuration, typically including:

- **Scene-referred**: ACES2065-1, ACEScg, Linear sRGB
- **Display-referred**: sRGB, Rec709, P3-D65, Rec2020
- **Log formats**: LogC, Log3G10, SLog3
- **Film emulation**: Various film stock simulations
- **Custom**: Studio-specific working spaces

## Contributing

This library follows Griptape Node development best practices:

- Each node is self-contained with clear parameter definitions
- Comprehensive error handling for OCIO operations
- Performance optimization for production use
- Extensive parameter validation and user feedback

## License

This library is released under the same license as Griptape Nodes. OpenColorIO is used under its BSD license.

## Support

For issues specific to this OpenColorIO library:
- Check OCIO configuration file validity
- Verify color space names match config definitions
- Ensure image formats are supported by Pillow
- Check memory usage for large batch operations

For general Griptape Nodes support, refer to the main Griptape documentation.
