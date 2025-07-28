# FPN-YOLO-RS

A Rust implementation combining Feature Pyramid Networks (FPN) with YOLO object detection, providing both research-quality FPN architecture and practical object detection capabilities.

[![License](https://img.shields.io/github/license/ionthedev/YOLO-FPN-RS)](LICENSE)
[![ONNX](https://img.shields.io/badge/ONNX-grey)](https://onnx.ai/)

## Features

- **Complete FPN Implementation**: Research-grade Feature Pyramid Network following the original paper architecture
- **YOLO Integration**: YOLOv8 model support via ONNX for practical object detection
- **GPU Acceleration**: CUDA support for both FPN computation and YOLO inference
- **Multi-modal Processing**: Single images, webcam feed, and batch dataset processing
- **Python Bindings**: Optional Python interface for easy integration
- **WebGPU Support**: Experimental WebGPU compute shaders (feature flag)

## Architecture

The project implements the FPN architecture as described in ["Feature Pyramid Networks for Object Detection"](https://arxiv.org/abs/1612.03144):

- **ResNet-50 Backbone**: Proper bottleneck blocks with residual connections
- **Multi-scale Features**: C2, C3, C4, C5 feature maps at strides 4, 8, 16, 32
- **Top-down Pathway**: Lateral connections with 3x3 anti-aliasing convolutions
- **RPN Integration**: Region Proposal Network working across pyramid levels
- **Fast R-CNN Head**: Multi-scale object detection and classification

## Quick Start

### Prerequisites

- Rust 1.70+
- OpenCV 4.5+ (with optional CUDA support)
- YOLO model file: `yolov8m.onnx` (download from Ultralytics)

### Installation

```bash
git clone https://github.com/ionthedev/FPN-YOLO-RS.git
cd FPN-YOLO-RS
cargo build --release
```

### Usage

**Single Image Detection:**
```bash
cargo run --release -- image -i input.jpg -o output.jpg
```

**Webcam Processing:**
```bash
cargo run --release -- webcam -c 0 -p balanced
```

**Batch Dataset Processing:**
```bash
cargo run --release -- dataset -i ./images/ -o ./outputs/ -c 4
```

### Performance Modes

- `turbo`: Process every 8th frame, 640x480, confidence 0.7
- `fast`: Process every 4th frame, 640x480, confidence 0.5  
- `balanced`: Process every 2nd frame, 960x540, confidence 0.35
- `quality`: Process every frame, 1280x720, confidence 0.25

## Technical Details

### FPN Implementation

The FPN implementation follows the research paper specifications:

```rust
// Multi-scale feature extraction
let backbone_features = backbone.forward(&input_tensor)?;
let pyramid_features = fpn.forward(&backbone_features)?;

// Multi-scale object detection
let proposals = rpn.forward(&pyramid_features, image_size)?;
let detections = fast_rcnn.forward(&pyramid_features, &proposals, image_size)?;
```

### GPU Acceleration

- **CUDA Backend**: Automatic detection and utilization for both Candle (FPN) and OpenCV (YOLO)
- **Memory Pooling**: Efficient GPU memory management with tensor caching
- **Parallel Processing**: Multi-threaded CPU fallback with optimized thread pool

### System Configuration

The system automatically detects optimal configuration:

```rust
let system_config = SystemConfig::detect_optimal()?;
// Automatically configures:
// - CUDA vs CPU device selection
// - Optimal thread pool size
// - GPU memory pool allocation
// - Parallel processing streams
```

## Build Features

- `python-bindings`: Enable Python interface via PyO3
- `webgpu-shaders`: Experimental WebGPU compute shader support

```bash
# Build with Python bindings
cargo build --release --features python-bindings

# Build with WebGPU support
cargo build --release --features webgpu-shaders
```

## Dataset Support

The included Oxford RobotCar dataset samples are provided for testing. The dataset is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License and is intended for academic use only.

For commercial use, contact [Oxford Robotics Institute](https://robotcar-dataset.robots.ox.ac.uk).

## Performance

- **FPN Forward Pass**: ~15-30ms (GPU) / ~50-100ms (CPU)
- **YOLO Inference**: ~10-25ms (GPU) / ~40-80ms (CPU)  
- **Total Pipeline**: ~25-55ms (GPU) / ~90-180ms (CPU)

*Benchmarks on RTX 3080 / Intel i7-10700K*

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request
6. Hire Me

## References

- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

**Key License Points:**
- Free to use, modify, and distribute under AGPL-3.0 terms
- If you deploy this as a network service, you must release your modifications
- Chosen for compatibility with YOLOv8's AGPL-3.0 license requirements
- For commercial closed-source use, contact Ultralytics for enterprise licensing
