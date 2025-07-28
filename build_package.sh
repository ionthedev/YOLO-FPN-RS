#!/bin/bash
set -e  # Exit on any error

# FPN-YOLO Python Package Build Script
# This script compiles, builds, and exports the FPN-YOLO Rust library as a Python package

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PACKAGE_NAME="fpn-yolo-rs"
VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
BUILD_TYPE="release"  # release or debug
UPLOAD_TO_PYPI=false
CLEAN_BEFORE_BUILD=false
CREATE_TARBALL=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="debug"
            shift
            ;;
        -c|--clean)
            CLEAN_BEFORE_BUILD=true
            shift
            ;;
        --upload)
            UPLOAD_TO_PYPI=true
            shift
            ;;
        --no-tarball)
            CREATE_TARBALL=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --debug      Build in debug mode (default: release)"
            echo "  -c, --clean      Clean build artifacts before building"
            echo "  --upload         Upload to PyPI after building"
            echo "  --no-tarball     Don't create distribution tarball"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}$1${NC}"
}

log_success() {
    echo -e "${GREEN}$1${NC}"
}

log_warning() {
    echo -e "${YELLOW}$1${NC}"
}

log_error() {
    echo -e "${RED}$1${NC}"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if maturin is installed
    if ! command -v maturin &> /dev/null; then
        log_error "maturin not found. Installing..."
        pip install maturin
    fi
    
    # Check if build tools are available
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo (Rust) not found. Please install Rust toolchain."
        exit 1
    fi
    
    # Check if required files exist
    if [[ ! -f "Cargo.toml" ]]; then
        log_error "Cargo.toml not found. Are you in the project root?"
        exit 1
    fi
    
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "pyproject.toml not found. Are you in the project root?"
        exit 1
    fi
    
    log_success "All dependencies verified"
}

clean_build() {
    if [[ "$CLEAN_BEFORE_BUILD" == true ]]; then
        log_info "Cleaning previous build artifacts..."
        
        # Clean Rust artifacts
        if [[ -d "target" ]]; then
            cargo clean
            log_success "Cleaned Rust build artifacts"
        fi
        
        # Clean Python artifacts
        if [[ -d "dist" ]]; then
            rm -rf dist/
            log_success "Cleaned Python dist directory"
        fi
        
        if [[ -d "build" ]]; then
            rm -rf build/
            log_success "Cleaned Python build directory"
        fi
        
        # Clean Python cache
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        
        log_success "Build environment cleaned"
    fi
}

build_package() {
    log_info "Building $PACKAGE_NAME package (version: $VERSION, mode: $BUILD_TYPE)..."
    
    # Set build flags based on build type
    if [[ "$BUILD_TYPE" == "release" ]]; then
        BUILD_FLAGS="--release"
        log_info "Building in release mode for optimal performance"
    else
        BUILD_FLAGS=""
        log_info "Building in debug mode"
    fi
    
    # Build the wheel
    log_info "Compiling Rust code and building Python wheel..."
    maturin build --features python-bindings $BUILD_FLAGS --out dist
    
    if [[ $? -eq 0 ]]; then
        log_success "Package built successfully"
    else
        log_error "Package build failed"
        exit 1
    fi
}

create_source_distribution() {
    log_info "Creating source distribution..."
    
    # Create source tarball using maturin
    maturin sdist --out dist
    
    if [[ $? -eq 0 ]]; then
        log_success "Source distribution created"
    else
        log_warning "Source distribution creation failed (continuing anyway)"
    fi
}

verify_package() {
    log_info "Verifying built package..."
    
    # Check if wheel was created
    WHEEL_FILE=$(find dist -name "*.whl" | head -1)
    if [[ -z "$WHEEL_FILE" ]]; then
        log_error "No wheel file found in dist/"
        exit 1
    fi
    
    log_success "Found wheel: $(basename "$WHEEL_FILE")"
    
    # Check wheel contents
    log_info "Checking wheel contents..."
    python -m zipfile -l "$WHEEL_FILE" | head -20
    
    # Test installation in a temporary virtual environment
    log_info "Testing package installation..."
    TEMP_VENV=$(mktemp -d)
    python -m venv "$TEMP_VENV"
    source "$TEMP_VENV/bin/activate"
    
    # Install dependencies first
    pip install numpy opencv-python
    
    # Install our package
    pip install "$WHEEL_FILE"
    
    # Test import
    python -c "
import fpn_yolo_rs
print('✅ Package import successful')
detector = fpn_yolo_rs.PyFPNYOLO()
print('✅ PyFPNYOLO class instantiation successful')
print('📊 System info:')
print(detector.get_system_info())
"
    
    if [[ $? -eq 0 ]]; then
        log_success "Package verification completed successfully"
    else
        log_error "Package verification failed"
        deactivate
        rm -rf "$TEMP_VENV"
        exit 1
    fi
    
    deactivate
    rm -rf "$TEMP_VENV"
}

create_distribution_package() {
    if [[ "$CREATE_TARBALL" == true ]]; then
        log_info "Creating distribution package..."
        
        DIST_NAME="${PACKAGE_NAME}-${VERSION}"
        DIST_DIR="dist/${DIST_NAME}"
        
        # Create distribution directory
        mkdir -p "$DIST_DIR"
        
        # Copy built wheels and source distribution
        cp dist/*.whl "$DIST_DIR/" 2>/dev/null || true
        cp dist/*.tar.gz "$DIST_DIR/" 2>/dev/null || true
        
        # Copy essential files
        cp README.md "$DIST_DIR/" 2>/dev/null || true
        cp pyproject.toml "$DIST_DIR/"
        cp Cargo.toml "$DIST_DIR/"
        
        # Create installation script
        cat > "$DIST_DIR/install.sh" << 'EOF'
#!/bin/bash
echo "Installing FPN-YOLO Python package..."
pip install *.whl
echo "Installation complete!"
EOF
        chmod +x "$DIST_DIR/install.sh"
        
        # Create usage example
        cat > "$DIST_DIR/example.py" << 'EOF'
#!/usr/bin/env python3
"""
FPN-YOLO Usage Example
"""
import fpn_yolo_rs

# Create detector
detector = fpn_yolo_rs.PyFPNYOLO()

# Print system info
print(detector.get_system_info())

# Example detection (requires image file)
# result = detector.detect("your_image.jpg", "output.jpg")
print("Ready for object detection!")
EOF
        
        # Create tarball
        cd dist
        tar -czf "${DIST_NAME}.tar.gz" "${DIST_NAME}/"
        cd ..
        
        log_success "Distribution package created: dist/${DIST_NAME}.tar.gz"
    fi
}

upload_to_pypi() {
    if [[ "$UPLOAD_TO_PYPI" == true ]]; then
        log_info "Uploading to PyPI..."
        
        # Check if twine is installed
        if ! command -v twine &> /dev/null; then
            log_info "Installing twine..."
            pip install twine
        fi
        
        # Upload to PyPI
        log_warning "This will upload to PyPI. Make sure you have proper credentials configured."
        read -p "Continue? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            twine upload dist/*.whl dist/*.tar.gz
            log_success "Package uploaded to PyPI"
        else
            log_info "Upload cancelled"
        fi
    fi
}

print_summary() {
    echo
    log_success "Build Summary"
    echo "=============================================="
    echo "Package: $PACKAGE_NAME"
    echo "Version: $VERSION"
    echo "Build Type: $BUILD_TYPE"
    echo "Output Directory: $(pwd)/dist"
    echo
    echo "Files created:"
    ls -la dist/ | grep -E '\.(whl|tar\.gz)$' || echo "No distribution files found"
    echo
    
    if [[ "$CREATE_TARBALL" == true ]]; then
        echo "Distribution package:"
        ls -la dist/*.tar.gz 2>/dev/null || echo "No distribution tarball found"
        echo
    fi
    
    echo "To install locally:"
    echo "  pip install dist/*.whl"
    echo
    echo "To install from tarball:"
    echo "  tar -xzf dist/*.tar.gz && cd fpn-yolo-rs-* && ./install.sh"
    echo
    log_success "Build completed successfully!"
}

# Main execution
main() {
    echo "FPN-YOLO Python Package Builder"
    echo "================================"
    echo
    
    check_dependencies
    clean_build
    build_package
    create_source_distribution
    verify_package
    create_distribution_package
    upload_to_pypi
    print_summary
}

# Error handling
trap 'log_error "Build failed! Check the output above for details."; exit 1' ERR

# Run main function
main "$@" 