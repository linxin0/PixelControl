#!/usr/bin/env bash
set -u

# Code/data/asset preflight. This intentionally does not import torch or load
# multi-GB models; it is safe to run on a login node before reserving GPUs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0
warn=0
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN=python; else PYTHON_BIN=python3; fi
fi
required=(
  README.md requirements.txt
  pixdit_core/pixeldit_t2i_control.py
  t2i/train_control.py t2i/infer_threecontrol_val.py
  t2i/configs_t2i/pixeldit_seg_control_v1_first200.yaml
  t2i/configs_t2i/pixeldit_edge_control_v1_first200.yaml
  t2i/configs_t2i/pixeldit_threecontrol_v1_mixed_cycle005_from_mixed2k.yaml
)

for path in "${required[@]}"; do
  if [[ -f "$path" ]]; then
    printf '[ok]   %s\n' "$path"
  else
    printf '[fail] missing %s\n' "$path"
    fail=1
  fi
done

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  "$PYTHON_BIN" --version
  if "$PYTHON_BIN" -m compileall -q pixdit_core t2i eval reference_innovation_code; then
    printf '[ok]   Python syntax compilation\n'
  else
    printf '[fail] Python syntax compilation\n'
    fail=1
  fi
  "$PYTHON_BIN" - <<'PY'
import importlib.util
packages = ("torch", "torchvision", "yaml", "pyrallis", "transformers", "PIL", "cv2")
missing = [name for name in packages if importlib.util.find_spec(name) is None]
if missing:
    print("[warn] missing Python packages:", ", ".join(missing))
    print("       install them with: pip install -r requirements.txt")
else:
    print("[ok]   core Python packages are discoverable")
PY
else
  printf '[fail] Python interpreter not found (tried %s)\n' "$PYTHON_BIN"
  fail=1
fi

# These are intentionally warnings: the release does not bundle large assets.
for path in \
  "t2i/pixeldit_t2i_v1.pth" \
  "${CLIP_MODEL:-t2i/pretrained/clip-vit-large-patch14}" \
  "${SAM2_MODEL:-t2i/pretrained/sam2.1-hiera-large}" \
  "${DA3_MODEL:-t2i/pretrained/DA3NESTED-GIANT-LARGE-1.1}"; do
  if [[ -e "$path" ]]; then
    printf '[ok]   asset %s\n' "$path"
  else
    printf '[warn] external asset not found: %s\n' "$path"
    warn=1
  fi
done

printf '\nPreflight finished: %d fatal issue(s), %d asset/package warning(s).\n' "$fail" "$warn"
printf 'Warnings are expected on a code-only checkout; see docs/02_USAGE.md.\n'
exit "$fail"
