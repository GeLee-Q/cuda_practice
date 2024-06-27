#!/bin/bash

set -e

# Check if CMake is installed
if ! command -v cmake >/dev/null 2>&1; then
  echo "Error: CMake is not installed."
  exit 1
fi

# Create the build directory if it does not exist
rm -rf build
if [ ! -d "build" ]; then
  mkdir build
fi

# Change to the build directory
cd build

# # Run CMake
cmake ..

# # Build the project
make

# ctest -R test_gemm
ctest -R test_relu