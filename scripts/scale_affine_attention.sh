#!/usr/bin/env bash
# Usage: scripts/scale_affine_attention.sh N [N ...] --hardware HARDWARE [example options...]
set -euo pipefail

sizes=()
while (($#)) && [[ $1 != --hardware ]]; do
    [[ $1 =~ ^[1-9][0-9]*$ ]] || {
        printf 'expected a positive size before --hardware, got %q\n' "$1" >&2
        exit 2
    }
    sizes+=("$1")
    shift
done
((${#sizes[@]})) || { printf 'provide at least one size\n' >&2; exit 2; }
[[ ${1-} == --hardware && -n ${2-} ]] || {
    printf 'usage: %s N [N ...] --hardware HARDWARE [example options...]\n' "$0" >&2
    exit 2
}
hardware=$2
shift 2

for n in "${sizes[@]}"; do
    result="affine_attention_${n}.results"
    printf 'affine_attention N=%s -> %s\n' "$n" "$result"
    if uv run -m examples.affine_attention \
        --hardware "$hardware" \
        --length "$n" \
        --right "$n" \
        "$@" >"$result" 2>&1; then
        :
    else
        status=$?
        printf 'affine_attention N=%s failed; see %s\n' "$n" "$result" >&2
        tail -40 "$result" >&2
        exit "$status"
    fi
done
