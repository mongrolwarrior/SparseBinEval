#!/usr/bin/env bash
# Build a CPU-only somoclu (sparse kernel) for the comparison harness — no autotools needed.
#
# somoclu's build normally uses autoconf/automake (often absent in minimal containers) and its
# sources rely on an implicit `using namespace std;`. We side-step both: clone, then compile the
# CPU sources directly with g++/OpenMP, injecting a small std prelude. The GPU (dense-only) and
# MPI paths are #ifdef-guarded out by simply not defining CUDA / HAVE_MPI.
#
# Usage: tools/build_somoclu.sh [SRC_DIR]   (default /workspaces/somoclu)
#   -> produces $SRC_DIR/somoclu_cpu
set -euo pipefail

SRC="${1:-/workspaces/somoclu}"
REPO="https://github.com/peterwittek/somoclu"

if [ ! -d "$SRC" ]; then
    echo "cloning somoclu -> $SRC"
    git clone --depth 1 "$REPO" "$SRC"
fi

PRELUDE="$(mktemp)"
cat > "$PRELUDE" <<'EOF'
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
using namespace std;
EOF

echo "compiling somoclu CPU (sparse + dense CPU kernels) ..."
g++ -O3 -fopenmp -DNDEBUG -include "$PRELUDE" \
    "$SRC/src/somoclu.cpp" "$SRC/src/io.cpp" "$SRC/src/training.cpp" \
    "$SRC/src/sparseCpuKernels.cpp" "$SRC/src/denseCpuKernels.cpp" \
    "$SRC/src/mapDistanceFunctions.cpp" "$SRC/src/uMatrix.cpp" \
    -o "$SRC/somoclu_cpu"
rm -f "$PRELUDE"
echo "built $SRC/somoclu_cpu"
"$SRC/somoclu_cpu" 2>&1 | head -1 || true
