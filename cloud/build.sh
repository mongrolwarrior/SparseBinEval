#!/usr/bin/env bash
# Stage the three first-party repos into a clean build context and build the repro image.
# Run from a machine that has all repos checked out (defaults assume the /workspaces layout).
# Source is rsync'd WITHOUT build/, .git/, data/ or runs, so the context stays small.
#
#   cloud/build.sh                       # -> sparsesom-repro:latest
#   REPOS=/path/to/parent cloud/build.sh # if the repos live elsewhere
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_ROOT="$(dirname "$HERE")"
REPOS="${REPOS:-$(dirname "$EVAL_ROOT")}"          # parent dir holding the repos
IMAGE="${IMAGE:-sparsesom-repro:latest}"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

stage() {                                          # stage $1 (a repo) into the context, source-only
    echo "staging $1 ..."
    rsync -a --exclude='.git' --exclude='build' --exclude='data' --exclude='runs_compare' \
          --exclude='__pycache__' "$REPOS/$1/" "$CTX/$1/"
}
stage SparseSOM
stage StandardSparseSOM
stage SparseBinEval

cp "$HERE/Dockerfile" "$CTX/Dockerfile"
echo "building $IMAGE (multi-arch; this takes a while) ..."
docker build ${DOCKER_BUILD_ARGS:-} -t "$IMAGE" "$CTX"
echo "done: $IMAGE"
echo "next: docker run --gpus all -v \$PWD/data:/data $IMAGE   # /data holds the corpus split"
