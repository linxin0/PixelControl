# 08 — Troubleshooting

## `ModuleNotFoundError` or import conflicts

Use a fresh Python 3.10 environment and run from the repository root. The
launchers set `PYTHONNOUSERSITE=1` and
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` to avoid common user-site
tokenizers and protobuf conflicts. For DA3/SAM2 evaluation, follow the `deco`
environment instructions instead of forcing the main environment variables.

```bash
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/check_setup.sh
```

## `FileNotFoundError` for a checkpoint or condition map

Check the four roots and checkpoint variables printed by the launcher:

```bash
printf '%s\n' "$CKPT_THREE" "$EVAL_IMAGE_ROOT" "$EVAL_DEPTH_ROOT" \
  "$EVAL_SEG_ROOT" "$EVAL_EDGE_ROOT"
```

The YAML paths are relative to `t2i/`, while `scripts/_env.sh` resolves paths
relative to the repository root. A moved dataset requires deleting its cached
index JSON under `outputs/`.

## CUDA out-of-memory

Lower `BATCH_SIZE` for inference, reduce `cycle_subbatch_size` or increase
`cycle_apply_every` for training, and enable gradient checkpointing (already
enabled in the release configs). Do not silently change resolution or sampler
steps when collecting a comparable metric.

## DDP/NCCL hangs

Use a free `MASTER_PORT`, ensure `NP` equals the number of comma-separated GPU
IDs, and launch only one job per port:

```bash
GPUS=2,3 NP=2 MASTER_PORT=29541 bash scripts/train_threecontrol.sh
```

If a process was interrupted, inspect the previous log and checkpoint before
restarting; do not start a second copy against the same work directory.

## Empty or unexpectedly small metric tables

The evaluators match filenames by `sa_XXXXXX` stem and optional control suffix.
Check that the generation folder contains at least the requested number of
files and that `SUFFIX` matches the mode (`depth`, `seg`, or `edge`). Use the
`--dry_run` option of the SAM2/edge evaluators to inspect the manifest before
loading a large model.

## DA3 or SAM2 cannot load

These are optional external evaluators, not part of the base Python dependency
set. Confirm the local source/weight paths, use the `deco` environment, and
check that the weight format matches the evaluator script. The visual-quality
and edge evaluators can still be run independently.

## Unexpected result differences

Record all of: git commit, YAML, checkpoint path/EMA choice, dataset roots,
sample count, seed, CFG, solver steps, dtype, device, and evaluator versions.
Differences in any of these can change FID or condition-fidelity metrics even
when the model weights are identical.
