#!/usr/bin/env bash
# Build a SAB benchmarking image for a hardware target.
#
# Usage:
#   docker/build.sh t4|ai1 [extra docker build args...]
#
# Bakes the current git commit into the image (SAB_GIT_SHA) so results carry
# provenance. Refuses to build from a dirty worktree: the baked SHA must
# describe the code actually in the image.
set -euo pipefail

if [[ $# -lt 1 || ! "$1" =~ ^(t4|ai1)$ ]]; then
    echo "usage: docker/build.sh t4|ai1 [extra docker build args...]" >&2
    exit 1
fi

target="$1"
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "error: worktree is dirty; commit or stash before building so SAB_GIT_SHA is truthful" >&2
    exit 1
fi

sha="$(git rev-parse HEAD)"

exec docker build \
    -f "docker/Dockerfile.${target}" \
    --build-arg SAB_GIT_SHA="${sha}" \
    -t "sab-${target}:${sha:0:12}" \
    -t "sab-${target}:latest" \
    "$@" \
    .
