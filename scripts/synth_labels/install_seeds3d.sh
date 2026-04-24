#!/usr/bin/env bash
# Install python_3d_seeds (https://github.com/zch0414/3d-seeds) into the
# active conda environment.
#
# Usage:
#   conda activate <your-env>
#   bash scripts/synth_labels/install_seeds3d.sh

set -e

# ---- Active conda env required -----------------------------------------------
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: no conda environment active. Run: conda activate <env>" >&2
    exit 1
fi
echo "Conda env: $CONDA_PREFIX"

# ---- g++ required ------------------------------------------------------------
if ! command -v g++ &>/dev/null; then
    echo "ERROR: g++ not found. Install with: conda install -c conda-forge gxx_linux-64" >&2
    exit 1
fi

# ---- OpenCV headers + libs ---------------------------------------------------
if [ ! -f "$CONDA_PREFIX/include/opencv4/opencv2/core.hpp" ]; then
    echo "OpenCV   : not found, installing libopencv ..."
    # libopencv = C++ library only (no Python bindings), much faster to solve
    # Prefer mamba (fast solver) over conda
    if command -v mamba &>/dev/null; then
        mamba install -y -c conda-forge libopencv 2>&1 | tail -5
    else
        conda install -y -c conda-forge libopencv 2>&1 | tail -5
    fi
fi

if [ ! -f "$CONDA_PREFIX/include/opencv4/opencv2/core.hpp" ]; then
    echo "ERROR: opencv install failed." >&2; exit 1
fi
echo "OpenCV   : $CONDA_PREFIX/include/opencv4"

# ---- pybind11 ----------------------------------------------------------------
pip install pybind11 --quiet

# ---- Build and install -------------------------------------------------------
echo "Building python_3d_seeds ..."
CFLAGS="-I$CONDA_PREFIX/include/opencv4" \
CXXFLAGS="-I$CONDA_PREFIX/include/opencv4" \
LDFLAGS="-L$CONDA_PREFIX/lib -lopencv_core -lopencv_imgproc -Wl,-rpath,$CONDA_PREFIX/lib" \
CXX=g++ CC=gcc \
pip install --no-build-isolation git+https://github.com/zch0414/3d-seeds

echo ""
echo "Done. Verify with:"
echo "  python -c \"import python_3d_seeds; print('ok')\""
