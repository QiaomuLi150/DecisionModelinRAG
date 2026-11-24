#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--gpu" ]]; then
  gpu="$2"
  shift 2
else
  echo "usage: e5_a_gpu_guard.sh --gpu 5|6 -- python arguments..." >&2
  exit 2
fi
if [[ "${1:-}" != "--" ]]; then
  echo "usage: e5_a_gpu_guard.sh --gpu 5|6 -- python arguments..." >&2
  exit 2
fi
shift

case "$gpu" in
  5) expected_uuid="<REDACTED_GPU_UUID>";;
  6) expected_uuid="<REDACTED_GPU_UUID>";;
  *) echo "only frozen E5-A GPUs 5 and 6 are permitted" >&2; exit 2;;
esac

device_row=""
for attempt in 1 2 3 4 5; do
  if raw_devices="$(nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    device_row="$(printf '%s\n' "$raw_devices" | awk -F',' -v u="$expected_uuid" '$2 ~ u {print; found=1} END {if (!found) exit 1}')" || true
    [[ -n "$device_row" ]] && break
  fi
  sleep 1
done
if [[ -z "$device_row" ]]; then
  echo "E5-A GPU guard refused: could not verify $expected_uuid after five nvidia-smi attempts" >&2
  exit 4
fi
process_rows=""
for attempt in 1 2 3 4 5; do
  if raw_processes="$(<GPU_PROCESS_LISTING_OMITTED>
    process_rows="$(printf '%s\n' "$raw_processes" | awk -F',' -v u="$expected_uuid" '$1 ~ u {print}')"
    break
  fi
  sleep 1
done
if [[ -n "$process_rows" ]]; then
  echo "E5-A GPU guard refused: $expected_uuid has compute processes:" >&2
  echo "$process_rows" >&2
  exit 3
fi

export E5_A_EXTERNAL_PREFLIGHT=1
export E5_A_PREFLIGHT_GPU_UUID="$expected_uuid"
export E5_A_PREFLIGHT_PROCESS_COUNT=0
export E5_A_PREFLIGHT_DEVICE_ROW="$device_row"
export E5_A_PHYSICAL_GPU="$gpu"
exec "$@"
