#!/bin/bash
# Build script for the Rust realtime backend
# This script compiles the Rust backend and installs it as a Python module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/src/realtime_backend"

echo "Building Rust realtime backend..."
echo "================================"

# Check for Rust toolchain
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust toolchain not found. Please install Rust from https://rustup.rs"
    exit 1
fi

# Check for maturin
if ! command -v maturin &> /dev/null; then
    echo "maturin not found, installing..."
    pip install maturin
fi

# Build and install the backend
cd "$BACKEND_DIR"

echo ""
echo "Compiling Rust backend with maturin..."
echo ""

# Use --release for optimized builds
maturin develop --release

echo ""
echo "================================"
echo "Build complete!"
echo ""
echo "The realtime_backend module is now available in Python."
echo "You can verify by running:"
echo "  python -c 'import realtime_backend; print(\"Rust backend loaded successfully\")'"
echo ""
