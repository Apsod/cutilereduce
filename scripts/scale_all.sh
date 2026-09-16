#!/usr/bin/env bash
# Usage: scripts/scale_all.sh N [N ...] --hardware HARDWARE [example options...]
# Example: scripts/scale_all.sh 1024 2048 4096 --hardware rtx5080 --torch-compile
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
status=0
for script in scale_xentropy.sh scale_attention.sh scale_affine_attention.sh; do
    if ! "$script_dir/$script" "$@"; then
        printf '%s failed; continuing with the remaining workloads.\n' "$script" >&2
        status=1
    fi
done
exit "$status"
