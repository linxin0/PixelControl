# 01 — Overview & Innovations

## 1. What this model is

A **text-to-image PixelDiT** backbone (pixel-space MMDiT + PiT refinement,
~1.0 B backbone params, hidden size 1536, 14 patch-level blocks) extended with
**spatial control** for three structural modalities:

- **depth** (DepthAnything-3 relative depth, 1 channel)
- **seg** (SAM2 segmentation label map, 1 channel)
- **edge** (Gaussian-blur + Canny edges, 1 channel)

The model can be driven by any single modality, any pair, or all three at once.

## 2. The core innovation: independent branches + layer-wise gated fusion

Most multi-control methods concatenate all condition maps and feed them through
one shared encoder. That entangles modalities: improving edge control degrades
depth control, and single-condition behaviour drifts as you add modalities.

**Our design keeps each modality fully independent and fuses only at the end:**

```
                depth map ─► depth_encoder ─► depth_adapters[L] ─┐
                seg   map ─► seg_encoder   ─► seg_adapters[L]   ─┤
                edge  map ─► edge_encoder  ─► edge_adapters[L]  ─┤
                                                                 ▼
   for each injection layer L:  hidden += Σ_b  w[L,b] · residual_b
```

- One encoder + one stack of 14 zero-initialised adapters **per modality**.
- A learnable gate matrix `control_gate_logits` of shape `[14, 3]`
  (`[num_layers, num_controls]`).
- Fusion weight selection (`_per_layer_per_sample_weights`):
  - **single active control** → **hard select** that branch, weight = 1, the
    gate is *ignored entirely*. This guarantees depth-only / seg-only /
    edge-only behaviour is never disturbed by the other branches.
  - **multiple active controls** → **masked softmax over only the active
    branches** at each layer. Inactive branches get weight 0.

Formally, per sample `b`, layer `L`, control `c`:

```
keep[b,c] ∈ {0,1}                      # which controls are active for this sample
if Σ_c keep[b,c] == 1:  w[L,b,c] = keep[b,c]                  # hard select
else:                   w[L,b,c] = softmax_c( logit[L,c] · keep[b,c] )  # active-only
residual = Σ_c  keep[b,c] · w[L,b,c] · adapter_c[L](encoder_c(map_c))
hidden  += residual
```

### Why this matters (motivation)
- **No capability bleed.** A pretrained depth branch keeps its quality even
  while seg/edge branches are still learning.
- **Per-layer modality preference.** The gate learns, *per layer*, how much
  each modality should contribute when several are present — and we log the
  weights for interpretability.
- **Clean training signal.** Inactive branches receive **no gradient** for a
  given step (see gradient masking below), so each branch only learns from the
  samples that actually use it.

## 3. Structure-aware injection (Sobel modulation)

Each adapter is a `StructureAwareGatedZeroAdapter`:

```
residual = gate · Linear(LayerNorm(cond_tokens))
if structure_inject:  residual *= (1 + alpha_inject · sobel_structure_map(cond))
```

`sobel_structure_map` produces a per-token [0,1] map of local gradient
magnitude from the condition image, up-weighting the residual on
high-structure regions (object boundaries). Per-modality flags
`control_structure_inject = (depth=True, seg=True, edge=False)` — **edge
injection is disabled** because edge maps are already pure high-frequency, and
extra Sobel boosting produced "messy" high-frequency artifacts. See
`docs/06_PARAMETERS.md`.

## 4. Training strategy

- **Backbone frozen.** Only control branches + gate train. The base T2I model
  (`t2i/pixeldit_t2i_v1.pth`) provides generation quality for free.
- **Per-branch LR scales.** Final mixed model uses `depth=0.05`,
  `seg=0.10`, `edge=0.10`, `gate=0.50` × base LR (2e-5), so the pretrained
  depth branch is barely perturbed while seg/edge/gate adapt.
- **7-mode dropout sampling per step** (DDP-synchronised so all ranks agree):
  `depth, seg, edge, depth_seg, depth_edge, seg_edge, depth_seg_edge` with
  probabilities `[0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.19]`.
- **Gradient masking.** After backward, branches not in the sampled mode have
  their grads zeroed; the gate updates only for multi-control steps.
- **Cycle (consistency) losses**, small weight (0.005), close the loop with the
  same extractors used for evaluation — DA3 for depth, SAM2-target for seg,
  SoftCanny image-cycle for edge. See `docs/04` & `docs/06`.

The single-control seg / edge models are the same machinery in
`control_mode="single"` with `n_local_controls=1`.

## 5. Robustness details worth knowing
- Corrupt / truncated condition maps are **skipped and replaced** by a random
  valid sample, not silently loaded (`ImageFile.LOAD_TRUNCATED_IMAGES=False`).
- The control-mode is broadcast from rank 0 every step so DDP never desyncs.
- Edge maps that are missing on disk are recomputed from RGB on the fly.

## 6. Where to read the code
| Concern | File |
|---------|------|
| Control model, branches, gate, forward | `pixdit_core/pixeldit_t2i_control.py` |
| Encoders + structure-aware adapter | `pixdit_core/depth_condition_encoder.py` |
| Trainer wrapper / config knobs | `t2i/diffusion/model/control_trainer.py` |
| Training loop, mode sampling, grad mask | `t2i/train_control.py` |
| Datasets (train + eval, single + three) | `t2i/diffusion/data/datasets/control_datasets.py` |
| Cycle losses | `t2i/diffusion/losses/*.py` |
| Framework-free extract of the idea | `reference_innovation_code/independent_gated_control.py` |
