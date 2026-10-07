#!/usr/bin/env bash
# Convert exactly the 20 RoboTwin Clean tasks reported in JEPA-WAM Table 14.
set -euo pipefail

mode="${1:-plan}"
if [[ "$mode" != plan && "$mode" != run ]]; then
  echo "Usage: $0 [plan|run]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(dirname "$repo_root")"
robotwin_root="${ROBOTWIN_ROOT:-$project_root/RoboTwin}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$project_root/robotwin_lerobot}"
repo_id="robotwin2_clean20_v21"
target="$HF_LEROBOT_HOME/$repo_id"

tasks=(
  adjust_bottle beat_block_hammer click_alarmclock click_bell dump_bin_bigbin
  grab_roller handover_mic lift_pot place_bread_basket place_bread_skillet
  place_burger_fries place_cans_plasticbox place_empty_cup place_object_basket
  place_shoe press_stapler shake_bottle_horizontally shake_bottle
  stack_bowls_three stack_bowls_two
)

patterns=()
shopt -s nullglob
for task in "${tasks[@]}"; do
  source_dir="$robotwin_root/data/demo_clean/$task/aloha_agilex/data"
  if [[ ! -d "$source_dir" ]]; then
    echo "Missing Clean task data: $source_dir" >&2
    exit 1
  fi
  files=("$source_dir"/*.hdf5 "$source_dir"/*.h5)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No HDF5 episodes in: $source_dir" >&2
    exit 1
  fi
  printf '%-30s %s episodes\n' "$task" "${#files[@]}"
  patterns+=("demo_clean.$task.aloha_agilex")
done

echo "Tasks: ${#patterns[@]}; output: $target"
if [[ "$mode" == plan ]]; then
  exit 0
fi
if [[ -e "$target" ]]; then
  echo "Refusing to overwrite existing LeRobot dataset: $target" >&2
  exit 1
fi

# The converter itself deletes an existing repo_id, so the check above is required.
python_bin="${PYTHON_BIN:-python}"
cd "$robotwin_root"
"$python_bin" XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "${patterns[@]}" \
  --repo_id "$repo_id" \
  --max_episode 1000000 \
  --resolution 240x320
