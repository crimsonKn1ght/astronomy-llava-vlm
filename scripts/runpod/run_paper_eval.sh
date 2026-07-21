#!/usr/bin/env bash
set -euo pipefail

original_args=("$@")
print_recovery() {
  status=$?
  if [[ ${status} -ne 0 ]]; then
    printf '\nPaper evaluation stopped (exit %s). Recovery command:\n  bash %q' "${status}" "$0" >&2
    has_resume=false
    recovery_command="all"
    for argument in "${original_args[@]}"; do
      printf ' %q' "${argument}" >&2
      if [[ "${argument}" == "--resume" ]]; then
        has_resume=true
      fi
      case "${argument}" in
        all|preflight|prepare|download|smoke|run|analyze|package)
          recovery_command="${argument}"
          ;;
      esac
    done
    if [[ "${has_resume}" == "false" && "${recovery_command}" =~ ^(all|smoke|run)$ ]]; then
      printf ' --resume' >&2
    fi
    printf '\n' >&2
  fi
}
trap print_recovery EXIT

# One-command RunPod bootstrap for the paper evaluation. It is intentionally
# idempotent: environments and model/data downloads are reused on --resume.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

print_wrapper_help() {
  cat <<'EOF'
Usage: bash scripts/runpod/run_paper_eval.sh [command] [options]

Commands:
  all (default), preflight, prepare, download, smoke, run, analyze, package

Common options:
  --suites internal,deepsdo|astrovlbench
  --models all|astraq_stage1,astraq_stage2,astrollava,qwen3_vl_4b
  --resume
  --lock-astrovlbench
  --dry-run
  --allow-dirty                 Diagnostic only; not valid for paper evidence
  --skip-hardware-check         Diagnostic only
  --diagnostic-allow-partial    Diagnostic only
  -h, --help

Examples:
  bash scripts/runpod/run_paper_eval.sh --suites internal,deepsdo --models all --resume
  HF_TOKEN=... bash scripts/runpod/run_paper_eval.sh --lock-astrovlbench
  HF_TOKEN=... bash scripts/runpod/run_paper_eval.sh --suites astrovlbench --models all --resume

The wrapper never trains a checkpoint. A definitive paper run requires a clean
Git worktree and the pinned environments/protocol in configs/paper_eval_v2.yaml.
EOF
}

for argument in "${original_args[@]}"; do
  if [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
    print_wrapper_help
    exit 0
  fi
done

VENV_ROOT="${PAPER_EVAL_VENV_ROOT:-${REPO_ROOT}/.paper_eval_venvs}"
RUNTIME_ROOT="${PAPER_EVAL_RUNTIME_ROOT:-${REPO_ROOT}/.paper_eval_runtime}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"
MODERN_VENV="${VENV_ROOT}/modern"
ASTRO_VENV="${VENV_ROOT}/astrollava"
ASTRO_SOURCE="${RUNTIME_ROOT}/AstroLLaVA"
ASTRO_REVISION="697cfbf11fbe16ce326dbbdab06bd9d93ccba3e9"
PYTHON_VERSION="3.11.15"
UV_VERSION="0.11.30"
PIP_VERSION="25.3"
MODERN_TORCH_VERSION="2.8.0"
MODERN_TORCHVISION_VERSION="0.23.0"
MODERN_TORCH_INDEX="https://download.pytorch.org/whl/cu128"
ASTRO_TORCH_VERSION="2.1.2"
ASTRO_TORCHVISION_VERSION="0.16.2"
ASTRO_TORCH_INDEX="https://download.pytorch.org/whl/cu121"

yaml_section_value() {
  section="$1"
  key="$2"
  awk -v section="${section}" -v key="${key}" '
    $1 == section ":" {inside=1; next}
    inside && $1 == key ":" {value=$2; gsub(/^"|"$/, "", value); print value; exit}
    inside && $0 !~ /^[[:space:]]/ {inside=0}
  ' configs/paper_eval_v2.yaml
}

configured_uv="$(yaml_section_value runtime bootstrap_uv_version)"
configured_modern_python="$(yaml_section_value modern_generation python)"
configured_modern_pip="$(yaml_section_value modern_generation pip)"
configured_modern_torch="$(yaml_section_value modern_generation torch)"
configured_modern_torchvision="$(yaml_section_value modern_generation torchvision)"
configured_modern_index="$(yaml_section_value modern_generation torch_index_url)"
configured_astro_python="$(yaml_section_value astrollava_generation python)"
configured_astro_pip="$(yaml_section_value astrollava_generation pip)"
configured_astro_torch="$(yaml_section_value astrollava_generation torch)"
configured_astro_torchvision="$(yaml_section_value astrollava_generation torchvision)"
configured_astro_index="$(yaml_section_value astrollava_generation torch_index_url)"
if [[ "${configured_uv}" != "${UV_VERSION}" || \
      "${configured_modern_python}" != "${PYTHON_VERSION}" || \
      "${configured_astro_python}" != "${PYTHON_VERSION}" || \
      "${configured_modern_pip}" != "${PIP_VERSION}" || \
      "${configured_astro_pip}" != "${PIP_VERSION}" || \
      "${configured_modern_torch}" != "${MODERN_TORCH_VERSION}" || \
      "${configured_modern_torchvision}" != "${MODERN_TORCHVISION_VERSION}" || \
      "${configured_astro_torch}" != "${ASTRO_TORCH_VERSION}" || \
      "${configured_astro_torchvision}" != "${ASTRO_TORCHVISION_VERSION}" || \
      "${configured_modern_index}" != "${MODERN_TORCH_INDEX}" || \
      "${configured_astro_index}" != "${ASTRO_TORCH_INDEX}" ]]; then
  echo "RunPod wrapper bootstrap constants do not match configs/paper_eval_v2.yaml." >&2
  exit 2
fi

requested_command="all"
allow_dirty=false
skip_hardware=false
dry_run=false
lock_requested=false
suites_explicit=false
for argument in "${original_args[@]}"; do
  case "${argument}" in
    all|preflight|prepare|download|smoke|run|analyze|package) requested_command="${argument}" ;;
    --allow-dirty) allow_dirty=true ;;
    --skip-hardware-check) skip_hardware=true ;;
    --dry-run) dry_run=true ;;
    --lock-astrovlbench) lock_requested=true ;;
    --suites|--suites=*) suites_explicit=true ;;
  esac
done
lock_only=false
if [[ "${lock_requested}" == "true" && "${requested_command}" == "all" && "${suites_explicit}" == "false" ]]; then
  lock_only=true
fi
requires_inference=false
if [[ "${requested_command}" =~ ^(all|preflight|smoke|run)$ && "${lock_only}" == "false" ]]; then
  requires_inference=true
fi
requires_generation_environment=false
if [[ "${requested_command}" =~ ^(all|smoke|run)$ && "${lock_only}" == "false" ]]; then
  requires_generation_environment=true
fi

# A dry run is an inspection operation: do not bootstrap CUDA environments or
# download assets merely to print the execution plan. RunPod base images
# normally provide these lightweight imports; otherwise report the exact
# prerequisite and leave the filesystem untouched.
if [[ "${dry_run}" == "true" ]]; then
  planner_python=""
  for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1 && \
      "${candidate}" -c 'import numpy, PIL, psutil, yaml' >/dev/null 2>&1; then
      planner_python="${candidate}"
      break
    fi
  done
  if [[ -z "${planner_python}" ]]; then
    echo "--dry-run requires python3 with numpy, Pillow, psutil, and PyYAML; no CUDA environment was installed." >&2
    exit 2
  fi
  exec "${planner_python}" scripts/run_paper_eval.py "$@"
fi

# Fail fast before installing two large CUDA environments.
if [[ "${allow_dirty}" == "false" && -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Paper mode requires a clean Git worktree before environment setup." >&2
  git status --short >&2
  exit 2
fi
if [[ "${skip_hardware}" == "false" && "${lock_only}" == "false" ]]; then
  minimum_disk_gib="$(awk '$1 == "minimum_disk_gib:" {print $2}' configs/paper_eval_v2.yaml)"
  minimum_ram_gib="$(awk '$1 == "minimum_ram_gib:" {print $2}' configs/paper_eval_v2.yaml)"
  available_kib="$(df -Pk "${REPO_ROOT}" | awk 'NR == 2 {print $4}')"
  total_ram_kib="$(awk '$1 == "MemTotal:" {print $2}' /proc/meminfo)"
  if (( available_kib < minimum_disk_gib * 1024 * 1024 )); then
    echo "Insufficient disk before setup: need ${minimum_disk_gib} GiB free." >&2
    exit 2
  fi
  if (( total_ram_kib < minimum_ram_gib * 1024 * 1024 )); then
    echo "Insufficient RAM before setup: need ${minimum_ram_gib} GiB." >&2
    exit 2
  fi
fi
if [[ "${skip_hardware}" == "false" && "${requires_inference}" == "true" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for an inference run." >&2
    exit 2
  fi
  minimum_memory_mib="$(awk '$1 == "minimum_gpu_memory_mib:" {print $2}' configs/paper_eval_v2.yaml)"
  minimum_capability="$(awk '$1 == "minimum_compute_capability:" {print $2}' configs/paper_eval_v2.yaml)"
  maximum_capability="$(awk '$1 == "maximum_compute_capability_exclusive:" {print $2}' configs/paper_eval_v2.yaml)"
  eligible_gpu=false
  while IFS=',' read -r memory_mib compute_capability; do
    memory_mib="${memory_mib//[[:space:]]/}"
    compute_capability="${compute_capability//[[:space:]]/}"
    if awk -v memory="${memory_mib}" -v minimum_memory="${minimum_memory_mib}" \
      -v capability="${compute_capability}" -v minimum_capability="${minimum_capability}" \
      -v maximum_capability="${maximum_capability}" \
      'BEGIN {exit !(memory >= minimum_memory && capability >= minimum_capability && capability < maximum_capability)}'; then
      eligible_gpu=true
    fi
  done < <(nvidia-smi --query-gpu=memory.total,compute_cap --format=csv,noheader,nounits)
  if [[ "${eligible_gpu}" == "false" ]]; then
    echo "No GPU meets the ${minimum_memory_mib} MiB, compute-capability >=${minimum_capability} and <${maximum_capability} gate." >&2
    exit 2
  fi
fi

mkdir -p "${VENV_ROOT}" "${RUNTIME_ROOT}" "${HF_HOME}"

UV_INSTALL_DIR="${RUNTIME_ROOT}/uv-bin"
UV_BIN="${UV_INSTALL_DIR}/uv"
if [[ ! -x "${UV_BIN}" || "$(${UV_BIN} --version 2>/dev/null || true)" != "uv ${UV_VERSION}"* ]]; then
  mkdir -p "${UV_INSTALL_DIR}"
  installer="$(mktemp "${RUNTIME_ROOT}/uv-install.XXXXXX.sh")"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "${installer}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${installer}" "https://astral.sh/uv/${UV_VERSION}/install.sh"
  else
    echo "curl or wget is required to bootstrap pinned uv ${UV_VERSION}." >&2
    exit 2
  fi
  env UV_INSTALL_DIR="${UV_INSTALL_DIR}" UV_NO_MODIFY_PATH=1 sh "${installer}"
  rm -f "${installer}"
fi
if [[ "$(${UV_BIN} --version)" != "uv ${UV_VERSION}"* ]]; then
  echo "Pinned uv bootstrap failed: expected ${UV_VERSION}." >&2
  exit 2
fi
export UV_PYTHON_INSTALL_DIR="${RUNTIME_ROOT}/python"

ensure_venv() {
  target="$1"
  if [[ -x "${target}/bin/python" && "$(${target}/bin/python -c 'import platform; print(platform.python_version())' 2>/dev/null || true)" == "${PYTHON_VERSION}" ]]; then
    return
  fi
  "${UV_BIN}" venv --clear --managed-python --python "${PYTHON_VERSION}" "${target}"
}

ensure_venv "${MODERN_VENV}"
if [[ "${requires_generation_environment}" == "true" ]]; then
  ensure_venv "${ASTRO_VENV}"
fi

environment_hash() {
  python_bin="$1"
  requirements_file="$2"
  torch_version="$3"
  torchvision_version="$4"
  wheel_index="$5"
  "${python_bin}" - "${PYTHON_VERSION}" "${PIP_VERSION}" "${torch_version}" \
    "${torchvision_version}" "${wheel_index}" "${requirements_file}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

python_version, pip_version, torch_version, vision_version, index, requirements = sys.argv[1:]
payload = {
    "python": python_version,
    "pip": pip_version,
    "torch": torch_version,
    "torchvision": vision_version,
    "index": index,
    "requirements_sha256": hashlib.sha256(Path(requirements).read_bytes()).hexdigest(),
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
PY
}

verify_environment() {
  python_bin="$1"
  requirements_file="$2"
  torch_version="$3"
  torchvision_version="$4"
  "${python_bin}" - "${PYTHON_VERSION}" "${PIP_VERSION}" "${torch_version}" \
    "${torchvision_version}" "${requirements_file}" <<'PY'
from importlib import metadata
import platform
import sys
from pathlib import Path

python_version, pip_version, torch_version, vision_version, requirements = sys.argv[1:]
if platform.python_version() != python_version:
    raise SystemExit(1)
expected = {"pip": pip_version, "torch": torch_version, "torchvision": vision_version}
for raw in Path(requirements).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if "==" not in line:
        raise SystemExit(f"Unpinned requirement: {line}")
    name, version = line.split("==", 1)
    expected[name.strip()] = version.strip()
for name, version in expected.items():
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        raise SystemExit(1)
    if actual != version and not (name in {"torch", "torchvision"} and actual.startswith(version + "+")):
        raise SystemExit(1)
PY
}

modern_hash="$(environment_hash "${MODERN_VENV}/bin/python" requirements-paper-modern.txt \
  "${MODERN_TORCH_VERSION}" "${MODERN_TORCHVISION_VERSION}" "${MODERN_TORCH_INDEX}")"
if [[ ! -f "${MODERN_VENV}/.paper-requirements-${modern_hash}" ]] || \
  ! verify_environment "${MODERN_VENV}/bin/python" requirements-paper-modern.txt \
    "${MODERN_TORCH_VERSION}" "${MODERN_TORCHVISION_VERSION}"; then
  "${MODERN_VENV}/bin/python" -m pip install --upgrade "pip==${PIP_VERSION}"
  "${MODERN_VENV}/bin/python" -m pip install \
    --index-url "${MODERN_TORCH_INDEX}" \
    "torch==${MODERN_TORCH_VERSION}" "torchvision==${MODERN_TORCHVISION_VERSION}"
  "${MODERN_VENV}/bin/python" -m pip install -r requirements-paper-modern.txt
  verify_environment "${MODERN_VENV}/bin/python" requirements-paper-modern.txt \
    "${MODERN_TORCH_VERSION}" "${MODERN_TORCHVISION_VERSION}"
  rm -f "${MODERN_VENV}"/.paper-requirements-* || true
  touch "${MODERN_VENV}/.paper-requirements-${modern_hash}"
fi

if [[ "${requires_generation_environment}" == "true" ]]; then
  astro_hash="$(environment_hash "${ASTRO_VENV}/bin/python" requirements-paper-astrollava.txt \
    "${ASTRO_TORCH_VERSION}" "${ASTRO_TORCHVISION_VERSION}" "${ASTRO_TORCH_INDEX}")"
  if [[ ! -f "${ASTRO_VENV}/.paper-requirements-${astro_hash}" ]] || \
    ! verify_environment "${ASTRO_VENV}/bin/python" requirements-paper-astrollava.txt \
      "${ASTRO_TORCH_VERSION}" "${ASTRO_TORCHVISION_VERSION}"; then
    "${ASTRO_VENV}/bin/python" -m pip install --upgrade "pip==${PIP_VERSION}"
    "${ASTRO_VENV}/bin/python" -m pip install \
      --index-url "${ASTRO_TORCH_INDEX}" \
      "torch==${ASTRO_TORCH_VERSION}" "torchvision==${ASTRO_TORCHVISION_VERSION}"
    "${ASTRO_VENV}/bin/python" -m pip install -r requirements-paper-astrollava.txt
    verify_environment "${ASTRO_VENV}/bin/python" requirements-paper-astrollava.txt \
      "${ASTRO_TORCH_VERSION}" "${ASTRO_TORCHVISION_VERSION}"
    rm -f "${ASTRO_VENV}"/.paper-requirements-* || true
    touch "${ASTRO_VENV}/.paper-requirements-${astro_hash}"
  fi

  if [[ ! -d "${ASTRO_SOURCE}/.git" ]]; then
    git clone https://github.com/UniverseTBD/AstroLLaVA.git "${ASTRO_SOURCE}"
  fi
  git -C "${ASTRO_SOURCE}" fetch --quiet origin "${ASTRO_REVISION}"
  git -C "${ASTRO_SOURCE}" checkout --quiet --detach "${ASTRO_REVISION}"
  resolved_astro_revision="$(git -C "${ASTRO_SOURCE}" rev-parse HEAD)"
  if [[ "${resolved_astro_revision}" != "${ASTRO_REVISION}" ]]; then
    echo "AstroLLaVA source revision mismatch: ${resolved_astro_revision}" >&2
    exit 3
  fi
  if [[ -n "$(git -C "${ASTRO_SOURCE}" status --porcelain --untracked-files=all)" ]]; then
    echo "Pinned AstroLLaVA source tree is dirty; remove local changes before evaluation." >&2
    git -C "${ASTRO_SOURCE}" status --short >&2
    exit 4
  fi
  astro_code_marker="${ASTRO_VENV}/.paper-astrollava-code-${ASTRO_REVISION}"
  verify_astrollava_install() {
    "${ASTRO_VENV}/bin/python" - "${ASTRO_REVISION}" <<'PY'
from importlib import metadata
import json
import sys

revision = sys.argv[1]
distribution = metadata.distribution("llava")
if distribution.version != "1.2.2.post1":
    raise SystemExit(1)
payload = json.loads(distribution.read_text("direct_url.json") or "{}")
resolved = ((payload.get("vcs_info") or {}).get("commit_id") or "").lower()
if resolved != revision.lower():
    raise SystemExit(1)
PY
  }
  if [[ ! -f "${astro_code_marker}" ]] || ! verify_astrollava_install; then
    "${ASTRO_VENV}/bin/python" -m pip install --no-deps --force-reinstall \
      "git+https://github.com/UniverseTBD/AstroLLaVA.git@${ASTRO_REVISION}"
    verify_astrollava_install
    rm -f "${ASTRO_VENV}"/.paper-astrollava-code-* || true
    touch "${astro_code_marker}"
  fi
fi

# COCO METEOR uses Java. Install it when the pod permits apt; otherwise the
# scorer will fail explicitly before paper tables are produced.
if [[ "${requested_command}" =~ ^(all|analyze)$ && "${lock_only}" == "false" ]] && \
  ! command -v java >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" == "0" ]]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openjdk-17-jre-headless
fi

export HF_HOME
export PAPER_MODERN_PYTHON="${MODERN_VENV}/bin/python"
export PAPER_ASTROLLAVA_PYTHON="${ASTRO_VENV}/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

"${MODERN_VENV}/bin/python" -m pytest -q \
  tests/test_paper_*.py \
  tests/test_prepare_deepsdo.py \
  tests/test_astrovlbench_adapter.py \
  tests/test_decode.py \
  tests/test_internal_paper_manifest.py \
  tests/test_model_revision_propagation.py \
  tests/test_strict_lora_loading.py

"${MODERN_VENV}/bin/python" scripts/run_paper_eval.py "$@"
