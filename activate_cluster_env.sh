#!/usr/bin/env bash
#
# Usage (must be sourced):
#   source activate_cluster_env.sh
#
# What it does:
# - Ensures the "module" command is available (Compute Canada / Alliance clusters)
# - Loads gcc + Arrow (PyArrow) module BEFORE activating venv (required on Alliance)
# - Creates/activates a local venv at ./.venv
# - Installs Python deps from minimal_gpu_tuner/requirements.txt
# - Leaves you in the activated venv in the current shell
#
# Optional overrides:
#   VENV_DIR=".venv" ARROW_MODULE="arrow/22.0.0" source activate_cluster_env.sh
#

# Ensure this script is sourced (so `activate` persists).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: This script must be sourced, not executed."
  echo "Run: source ${BASH_SOURCE[0]}"
  exit 1
fi

_repo_root() {
  # Resolve to the directory containing this script.
  local src="${BASH_SOURCE[0]}"
  while [[ -L "$src" ]]; do
    src="$(readlink "$src")"
  done
  cd "$(dirname "$src")" >/dev/null 2>&1 && pwd -P
}

_ensure_module_cmd() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi

  # Common init scripts on Alliance/CC clusters.
  local init_candidates=(
    "/cvmfs/soft.computecanada.ca/config/profile/bash.sh"
    "/etc/profile.d/modules.sh"
    "/usr/share/lmod/lmod/init/bash"
  )
  local f
  for f in "${init_candidates[@]}"; do
    if [[ -f "$f" ]]; then
      # shellcheck source=/dev/null
      source "$f"
      break
    fi
  done

  command -v module >/dev/null 2>&1
}

_pick_latest_arrow_module() {
  # Picks the newest arrow/<version> visible to `module avail`.
  # Prints the module name (e.g., arrow/22.0.0) to stdout.
  module -t avail 2>&1 \
    | sed 's/[[:space:]]//g' \
    | sed 's/(default)//g' \
    | awk -F/ '$1=="arrow" && $2!="" {print $0}' \
    | sort -V \
    | tail -n 1
}

_main() {
  local root venv_dir req_file arrow_mod
  root="$(_repo_root)"
  venv_dir="${VENV_DIR:-$root/.venv}"
  req_file="$root/minimal_gpu_tuner/requirements.txt"

  if [[ ! -f "$req_file" ]]; then
    echo "ERROR: Can't find requirements file: $req_file"
    return 1
  fi

  if ! _ensure_module_cmd; then
    echo "ERROR: 'module' command not found. Run this in a login shell (e.g., 'bash -l') or on the cluster."
    return 1
  fi

  # Deactivate any existing venv so Arrow is loaded first (Alliance requirement).
  if type deactivate >/dev/null 2>&1; then
    deactivate || true
  fi

  arrow_mod="${ARROW_MODULE:-}"
  if [[ -z "$arrow_mod" ]]; then
    arrow_mod="$(_pick_latest_arrow_module)"
  fi
  if [[ -z "$arrow_mod" ]]; then
    echo "ERROR: No arrow/<version> module found. Try: module spider arrow"
    return 1
  fi

  echo "Loading modules: gcc + $arrow_mod"
  module load gcc "$arrow_mod"

  # Create venv if needed.
  if [[ ! -d "$venv_dir" ]]; then
    echo "Creating venv: $venv_dir"
    python -m venv "$venv_dir"
  fi

  # Activate venv in current shell.
  # shellcheck source=/dev/null
  source "$venv_dir/bin/activate"

  # Keep pip reasonably up to date (but don't fail hard if offline).
  python -m pip install -U pip >/dev/null 2>&1 || true

  # Sanity check: PyArrow should import from the module.
  python - <<'PY'
import pyarrow
print("pyarrow import OK, version =", pyarrow.__version__)
PY

  echo "Installing Python deps: $req_file"
  python -m pip install -r "$req_file"

  echo
  echo "Done. Venv is active: $VIRTUAL_ENV"
}

_main


