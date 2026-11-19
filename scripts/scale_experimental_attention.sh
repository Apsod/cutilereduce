#!/usr/bin/env bash
# Usage: scripts/scale_experimental_attention.sh N [N ...] [example options...]
set -euo pipefail

sizes=()
while (($#)) && [[ $1 != --* ]]; do
    [[ $1 =~ ^[1-9][0-9]*$ ]] || {
        printf 'expected a positive size, got %q\n' "$1" >&2
        exit 2
    }
    sizes+=("$1")
    shift
done
((${#sizes[@]})) || { printf 'provide at least one size\n' >&2; exit 2; }

for n in "${sizes[@]}"; do
    result="results/experimental_attention_${n}.results"
    printf 'experimental attention N=%s -> %s\n' "$n" "$result"
    if uv run -m examples.experimental_attention \
        --length "$n" --right "$n" "$@" >"$result" 2>&1; then
        :
    else
        status=$?
        printf 'experimental attention N=%s failed; see %s\n' "$n" "$result" >&2
        tail -40 "$result" >&2
        exit "$status"
    fi
done
