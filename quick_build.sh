#!/bin/bash
# Quick development build script for FPN-YOLO

echo "Building FPN-YOLO Python package..."

# Build and install in development mode
echo "Building package..."
maturin develop --features python-bindings

if [[ $? -eq 0 ]]; then
    echo "Package built and installed successfully!"
    echo ""
    echo "Test: python -c 'import fpn_yolo_rs; print(fpn_yolo_rs.PyFPNYOLO().get_system_info())'"
else
    echo "Build failed!"
    exit 1
fi 