# 06 — Parameters Reference

All parameters live in the YAML configs under `t2i/configs_t2i/`. This doc
explains the control-specific ones. Values shown are the **final** settings for
each model.

---

## 1. Control-injection parameters (`model.extra`)

These configure how condition maps are encoded and injected into the backbone.

| Param | Final value | Meaning |
|-------|-------------|---------|
| `control_mode` | `single` (seg/edge) / `multi` (three-control) | single = one branch reusing `depth_*` module names; multi = independent depth/seg/edge branches + gate. |
| `n_local_controls` | 1 (single) / 3 (multi) | number of independent control branches. |
| `control_names` | `[depth, seg, edge]` | fixed branch order; `control_keep[b]` and gate columns follow this order. |
| `depth_channels` | 1 | channels per control map (depth/seg/edge are each 1-channel). |
| `depth_base_channels` | 64 | first conv width of each control encoder. |
| `depth_max_channels` | 512 | max conv width of each control encoder. |
| `inject_every` | 1 | inject after every patch-level block → **14 injection sites** (blocks 0–13). |
| `inject_layer_indices` | (derived: 0..13) | explicit override of injection sites; null → use `inject_every`. |
| `init_gate` | 1.0 | scalar init for the per-adapter `gate` parameter (zero-init residual path scale; the adapter `gate` itself starts at 0 for zero-init safety, `init_gate` scales the legacy single path). |
| `init_gate_logits` | `[0.5, 0.0, -0.5]` | **layer-wise fusion gate** init, one row `[depth, seg, edge]` repeated over 14 layers. Depth starts favored (0.5), edge down-weighted (−0.5) to protect the pretrained depth branch when multiple controls are active. Shape becomes `[14, 3]`. |
| `enable_structure_inject` | true | master switch for Sobel structure modulation. |
| `control_structure_inject` | `[true, true, false]` | per-branch structure injection: depth ✓, seg ✓, **edge ✗**. Edge maps are already pure high-frequency; extra Sobel boosting caused messy artifacts, so it is disabled for edge (and the edge-only config sets `enable_structure_inject:false`, `alpha_inject:0.0`). |
| `alpha_inject` | 2.0 | structure-modulation strength: `residual *= (1 + alpha_inject · sobel_map)`. With `alpha_inject=2.0` a full-structure token can triple the residual; `0.0` disables modulation. |
| `freeze_backbone` | true | freeze the base T2I model; train only control branches + gate. |
| `freeze_control_branches` | `[]` (or `["depth"]` early experiments) | optionally freeze named branches entirely. |
| `pretrained_ckpt` | base / prior ckpt | weights loaded at init. |
| `load_strict` | false | non-strict load (control branches are new). |
| `load_prefix` | `"core."` | strip this prefix when matching backbone keys. |
| `skip_pretrained_modules` | (per stage) | module-name substrings to NOT load (re-initialise seg/edge when initializing from a depth-only ckpt). |

### Per-branch learning-rate scales (`control` / `model.extra`)
Multiply the base LR (`train.optimizer.lr = 2e-5`). Final mixed model:

| Param | Value | Effect |
|-------|-------|--------|
| `depth_branch_lr_scale` | 0.05 | barely perturb the strong pretrained depth branch |
| `seg_branch_lr_scale`   | 0.10 | let seg adapt |
| `edge_branch_lr_scale`  | 0.10 | let edge adapt |
| `gate_lr_scale`         | 0.50 | gate learns moderately fast |

---

## 2. Control-mode sampling (`control`)

| Param | Final value | Meaning |
|-------|-------------|---------|
| `enable_control_dropout` | true (multi) / false (single) | sample a random mode per step vs always full. |
| `num_controls` | 3 (multi) / 1 (single) | |
| `control_modes` | `[depth, seg, edge, depth_seg, depth_edge, seg_edge, depth_seg_edge]` | the 7 modes. |
| `control_probs` | `[0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.19]` | per-step sampling probability; **broadcast from rank 0** so all DDP ranks agree. |

After backward, inactive branches' grads are zeroed; the gate updates only when
≥2 controls are active (`_mask_inactive_control_grads`).

---

## 3. Cycle / consistency loss parameters (`control.cycle_loss`)

Top-level scheduling (`control`):

| Param | Final value | Meaning |
|-------|-------------|---------|
| `cycle_weight` | 0.005 (mixed) / 0.02 (seg) / 0.01 (edge) | overall weight of the cycle term added to flow-matching loss. |
| `cycle_t_min`, `cycle_t_max` | 0.3, 1.0 | only apply the cycle loss when the sampled noise level σ is in this window (cleaner `pred_x0`). |
| `cycle_subbatch_size` | 2 | number of samples per step used for the (expensive) cycle pass. |
| `cycle_apply_every` | 1 | apply every N steps (raise to save VRAM). |

The wrapper is `MultiConditionCycleLoss(depth_cycle_loss, seg_cycle_loss,
edge_cycle_loss, depth_weight, seg_weight, edge_weight)` which dispatches by the
active `control_mode`. Depth and seg compare the **generated image's** extracted
structure to the **condition label** (the label is enough — no extra GT pass).
Edge compares generated RGB vs **GT RGB** because the offline Canny labels use a
random threshold (see docs/04 §2).

### 3a. Pyramid layer parameters (shared by all three cycle losses)

The image-cycle losses operate on an **image pyramid**: the generated and target
maps are resized to several scales and an L1/SmoothL1 is taken at each, weighted:

| Param | Final value | Meaning |
|-------|-------------|---------|
| `enable_pyramid_cycle_loss` | true | turn on multi-scale comparison. |
| `cycle_scales` | `[512, 256, 128, 64]` | pyramid resolutions (px). |
| `cycle_scale_weights` | `[0.1, 0.25, 1.0, 0.25]` (seg uses `[0.1,0.25,0.75,0.25]`) | per-scale weight. **128 px dominates** (weight 1.0) — mid-frequency structure is the main signal; 512 (fine) is down-weighted to avoid chasing pixel noise, 64 (coarse) anchors global layout. |
| `loss_res` | 128 | base resolution for the non-pyramid term. |
| `smooth_l1_beta` | 0.05 | SmoothL1 transition point (small → near-L1, robust to outliers). |

### 3b. Coarse-to-fine / boundary weighting (depth DA3 + seg SAM2)

| Param | Final value | Meaning |
|-------|-------------|---------|
| `enable_coarse_to_fine_cycle` | true | use a special weighted term at the finest (512) scale. |
| `enable_fine_sobel_weight` | true | at 512, weight the loss by the **target's Sobel edges** so boundaries (where structure matters) are emphasized. |
| `alpha_fine_sobel` | 0.3 | strength of that boundary up-weighting. |
| `enable_fine_grad_loss` | false (mixed) | optional gradient-matching term at fine scale. |
| `lambda_fine_grad` | 0.0 / 0.01 | weight of the fine gradient term when enabled. |

### 3c. SoftCanny edge-cycle parameters (`SoftCannyImagePyramidCycleLoss`)

Differentiable soft Canny on gen vs GT RGB at a shared random threshold:

| Param | Final value | Meaning |
|-------|-------------|---------|
| `gaussian_kernel` | 11 | blur kernel before soft-Sobel (matches the k=11 edge recipe). Must be odd. |
| `threshold_min` / `threshold_max` | 0.2745 / 0.5882 | the random soft-threshold is sampled uniformly in this range each step (= the 70/255 … 150/255 Canny range used to build edge labels). The **same** threshold is applied to gen and GT, so the loss is invariant to the random choice. |
| `temperature` | 0.03 | sigmoid sharpness of the soft threshold (smaller = harder, more Canny-like). |
| `cycle_scales` / `cycle_scale_weights` | `[512,256,128,64]` / `[0.1,0.25,1.0,0.25]` | same pyramid as above. |

### 3d. DA3 depth-cycle specifics (`DA3*DepthCycleLoss`)

| Param | Final value | Meaning |
|-------|-------------|---------|
| `process_res` | 504 | DA3 inference resolution (multiple of 14), matching the training-GT pipeline. |
| `silog_lambda` | 0.85 | scale-invariant log term balance. |
| `gt_norm` / `pred_norm` | `minmax` | normalize both depths before comparison. |
| `affine_align` | true | per-image (scale, shift) alignment before the loss. |

---

## 4. Sampling / scheduler parameters (`scheduler`, `validation`)

| Param | Value | Meaning |
|-------|-------|---------|
| `flow_shift` | 4.0 (single) / 3.0 (mixed) | flow-matching timestep shift. |
| `weighting_scheme` | `logit_normal` | σ sampling distribution. |
| `logit_mean` / `logit_std` | 0.0/1.0 (single), −0.8/0.8 (mixed) | bias σ sampling toward higher noise in mixed training. |
| `num_sampling_steps` | 50 | DPM-Solver steps at inference. |
| `cfg_scale` | 2.75 | classifier-free guidance scale. |
| `seed` | 2025 | per-image deterministic noise seed offset. |

---

## 5. Quick "what to change" guide
- **Protect a pretrained branch** → lower its `*_branch_lr_scale` (e.g. depth 0.05) or list it in `freeze_control_branches`.
- **Edge too noisy** → keep `control_structure_inject` edge = false, `alpha_inject` low/0.
- **OOM during cycle** → raise `cycle_apply_every`, lower `cycle_subbatch_size`.
- **Re-init seg/edge from a depth-only ckpt** → add their module names to `skip_pretrained_modules`.
- **Change modality balance in multi** → edit `control_probs` and/or `init_gate_logits`.
