#!/bin/bash

echo "🚀 Building FPN-YOLO with GPU acceleration..."

# Build with WebGPU acceleration enabled for maximum performance
cargo build --release --features webgpu-shaders

if [ $? -eq 0 ]; then
    echo "✅ Build successful with WebGPU acceleration!"
    echo "🔥 FPN pipeline now uses:"
    echo "   - CUDA for tensor operations"
    echo "   - WebGPU shaders for image processing"
    echo "   - GPU-accelerated NMS"
    echo "   - Multi-scale FPN features"
    echo ""
    echo "🎯 Ready for high-performance real-time detection!"
else
    echo "❌ Build failed. Falling back to CPU-only build..."
    cargo build --release
fi 