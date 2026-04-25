#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash rebuild.sh [extra cmake args...]

Environment variables:
  BUILD_DIR   Build directory name, default: build
  BUILD_TYPE  CMAKE_BUILD_TYPE value, default: Release
EOF
  exit 0
fi

CACHE_FILE="${SCRIPT_DIR}/${BUILD_DIR}/CMakeCache.txt"
if [[ -f "${CACHE_FILE}" ]]; then
  CACHED_SOURCE_DIR="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${CACHE_FILE}")"
  if [[ -n "${CACHED_SOURCE_DIR}" && "${CACHED_SOURCE_DIR}" != "${SCRIPT_DIR}" ]]; then
    echo "Detected stale CMake cache for: ${CACHED_SOURCE_DIR}"
    echo "Removing ${BUILD_DIR} and reconfiguring for: ${SCRIPT_DIR}"
    rm -rf "${SCRIPT_DIR}/${BUILD_DIR}"
  fi
fi

cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" "$@"
cmake --build "${SCRIPT_DIR}/${BUILD_DIR}" -j"$(nproc)"
