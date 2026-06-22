# Paper Notes: Independent-Branch Multi-Control PixelDiT

This document is a detailed paper-writing guide for the user's own network. It summarizes the motivation, architecture, training strategy, data, evaluation, and ablation directions. It is intended as input for Codex or another writing agent to draft a paper.

Related code-level summary:

```text
docs/my_network_minimal_implementation.md
```

## 1. Working Title

Possible titles:

```text
Independent-Branch Gated Fusion for Multi-Condition Pixel-Level Diffusion Control
```

```text
Layer-Wise Gated Multi-Control PixelDiT with Modality-Isolated Control Branches
```

```text
Towards Robust Multi-Condition Text-to-Image Control via Independent Residual Branches
```

Internal method name:

```text
my-network-pixeldit-three-control
```

## 2. One-Paragraph Abstract Draft

We propose a multi-condition controllable text-to-image diffusion model built on PixelDiT. Instead of concatenating depth, segmentation, and edge maps into a shared control encoder, we introduce independent modality-specific control branches and fuse their per-layer residuals through a learnable layer-wise gate. To preserve single-condition controllability, single-control samples bypass the learned gate and directly select the corresponding branch, while multi-condition samples use a masked softmax over only the active controls. This design isolates modality-specific learning, prevents single-condition degradation from multi-condition fusion, and provides interpretable per-layer control weights. We further introduce branch-specific learning-rate scaling, selective structure injection, robust multi-control data loading, and a soft-Canny image-cycle loss for edge consistency. Experiments on depth, segmentation, and edge controls demonstrate improved robustness for both single-control and multi-control generation.

## 3. Problem Definition

The task is text-to-image generation with optional structural controls. Each sample may contain one or more of:

```text
depth map
segmentation map
edge map
```

The model must support seven modes:

```text
depth
seg
edge
depth_seg
depth_edge
seg_edge
depth_seg_edge
```

The central problem is that different control modalities have different semantics and frequency characteristics:

- Depth is global and geometric.
- Segmentation is region-level semantic structure.
- Edge is high-frequency local structure.

A naive shared control encoder can cause interference:

- Segmentation or edge training can damage pretrained depth behavior.
- Edge high-frequency signals can dominate residual injection.
- A learned fusion gate can perturb single-condition outputs.
- DDP training can become unstable if ranks activate different branches.

## 4. Core Contributions

The paper can claim these contributions:

1. Independent modality-specific control branches for PixelDiT.
2. Strict single-vs-multi behavior separation:
   single-control samples hard-select the active branch; multi-control samples use learnable gated fusion.
3. Layer-wise scalar gated fusion over active controls only.
4. Branch-specific training controls:
   freezing, LR scaling, selective checkpoint loading, and per-modality structure injection.
5. Robust multi-control dataset construction requiring complete RGB/caption/depth/seg/edge tuples and skipping corrupt edge maps.
6. SoftCanny image-cycle consistency for edge control under stochastic Canny preprocessing.
7. Practical DDP-safe control-mode sampling and gradient masking to align branch updates with active controls.

## 5. Architecture Overview

Base model:

```text
PixelDiT text-to-image diffusion backbone
```

Control extension:

```text
Depth branch: encoder + residual adapters
Seg branch: encoder + residual adapters
Edge branch: encoder + residual adapters
Layer-wise gate: logits [num_layers, 3]
```

In the current implementation, there are 14 control injection sites matching the PixelDiT patch blocks:

```text
num_inject_layers = 14
```

Each branch outputs residuals:

```text
R_depth_l, R_seg_l, R_edge_l
```

where `l` is a transformer layer/injection site.

## 6. Why Independent Branches

The original multi-control alternative was to concatenate controls:

```text
control = concat(depth, seg, edge)
shared_control_encoder(control)
```

This was rejected because it creates a single representation space for signals with very different meanings and statistics. In practice, the user observed:

```text
depth sky artifacts after mixed training
edge outputs becoming messy and overly high-frequency
```

Independent branches instead make each branch specialize:

```text
depth branch learns global geometry
seg branch learns semantic region constraints
edge branch learns boundary/local structure
```

Fusion is moved to residual space, where branches are already aligned with the backbone hidden representation.

## 7. Single-Condition Behavior

For single-condition inputs, the model should behave like a normal single-control model.

Example:

```text
depth only: control_keep = [1, 0, 0]
```

The model directly uses:

```text
R_l = R_depth_l
```

No softmax gate is used. This is essential because the gate is trained on multi-condition combinations and should not alter single-condition behavior.

The same rule holds for seg-only and edge-only:

```text
seg only:  R_l = R_seg_l
edge only: R_l = R_edge_l
```

Paper phrasing:

```text
For single-control samples, we bypass the fusion module entirely and enforce a deterministic one-hot branch selection. This decouples unimodal controllability from the learned multimodal fusion parameters.
```

## 8. Multi-Condition Behavior

For multi-condition samples, the gate is active. For each layer:

```text
gate_l = softmax(mask(control_gate_logits_l, active_controls))
```

Inactive controls are masked to negative infinity before softmax. Therefore the weights are normalized only over active controls.

Fusion:

```text
R_l = keep_d * g_d_l * R_depth_l
    + keep_s * g_s_l * R_seg_l
    + keep_e * g_e_l * R_edge_l
```

The gate is layer-wise, not patch-wise. This keeps the method simple, stable, and interpretable.

Paper phrasing:

```text
The layer-wise gate allows the model to adaptively choose which control modality to emphasize at different depths of the denoising network, while avoiding the instability of token-wise or patch-wise fusion.
```

## 9. Gate Initialization

Initial gate logits:

```yaml
depth: 0.5
seg: 0.0
edge: -0.5
```

Earlier version:

```yaml
depth: 0.5
seg: 0.0
edge: 0.0
```

Reason:

- Depth branch starts from a pretrained depth-control checkpoint.
- Depth should dominate at initialization.
- Edge should not dominate early because it carries high-frequency signals.

Important:

```text
Gate initialization affects only multi-condition inputs. Single-condition inputs ignore the gate.
```

## 10. Structure Injection

Structure-aware adapter residual:

```text
residual = gate * projection(norm(condition_tokens))
residual = residual * (1 + alpha * structure_map)
```

Final mixed model uses:

```yaml
control_structure_inject: [true, true, false]
alpha_inject: 2.0
```

Interpretation:

- Depth structure injection: enabled.
- Seg structure injection: enabled.
- Edge structure injection: disabled.

Why edge injection is disabled:

```text
Edge maps already contain strong high-frequency structure. Additional Sobel-style injection overemphasized boundaries and degraded visual quality.
```

This is an important paper point: modality-specific injection policy.

## 11. Checkpoint Strategy

The depth branch is initialized from a strong depth-control model:

```text
t2i/universal_pix_t2i_workdirs/exp_pixeldit_depth_control_v1_512_bs16x2_acc4_cycle001_nograd_l128075_from8k/checkpoints/epoch_5_step_10000.pth
```

The branch names are kept compatible:

```text
depth_encoder
depth_adapters
```

This enables loading old depth-control checkpoints without manual remapping.

Seg and edge branches can be:

- randomly initialized in the first three-control model,
- loaded from later checkpoints,
- selectively skipped during checkpoint loading for reinitialization.

Selective skipping example:

```yaml
skip_pretrained_modules: [edge_encoder, edge_adapters]
```

Used when reinitializing edge branch while keeping backbone/depth/seg/gate.

## 12. Training Mode Sampling

Original requested sampling:

```yaml
depth: 0.25
seg: 0.20
edge: 0.20
depth_seg: 0.125
depth_edge: 0.10
seg_edge: 0.075
depth_seg_edge: 0.05
```

Final mixed model sampling:

```yaml
depth: 0.15
seg: 0.15
edge: 0.15
depth_seg: 0.12
depth_edge: 0.12
seg_edge: 0.12
depth_seg_edge: 0.19
```

Rationale:

- Preserve single-control modes through frequent unimodal training.
- Ensure all multi-condition combinations are trained.
- Increase all-three mode probability in later training for full multi-control fusion.

## 13. DDP Safety

Problem:

```text
If different DDP ranks sample different control modes, different ranks activate different branches.
This can desynchronize gradient reduction and cause hangs/SIGABRT.
```

Solution:

```text
Rank 0 samples the control mode and broadcasts it to all ranks.
```

This is not just implementation detail. It is required by conditional branch execution.

Paper or appendix note:

```text
For distributed training, we synchronize the sampled control subset across workers to ensure identical branch participation in every update.
```

## 14. Gradient Masking

Even though inactive branches are skipped in forward, the training code explicitly masks gradients:

```text
depth-only: update depth branch only
seg-only: update seg branch only
edge-only: update edge branch only
multi: update active branches plus gate
```

The gate is trained only when more than one control is active.

This enforces the intended training invariant:

```text
unimodal batches do not update unrelated branches or fusion parameters.
```

## 15. Data

Training data roots:

```text
RGB/caption: data/blip/extracted
Depth:       data/blip_depth_da3_nested_giant_large_1_1
Seg:         data/blip_sam2_large_extracted
Edge:        data/blip_edge
```

Final mixed model training shards:

```text
sa_000000 through sa_000199
```

Validation roots:

```text
RGB/caption: data/blip/extracted_new/sa_000201
Depth:       data/blip_depth_da3_nested_giant_large_1_1/sa_000201
Seg:         data/blip_sam2_large_extracted/sa_000201
Edge:        data/blip_edge/sa_000201
```

Validation uses:

```yaml
max_samples: 50
batch_size: 8
num_sampling_steps: 50
cfg_scale: 2.75
seed: 2025
```

Dataset completeness rule:

```text
Only use samples that have RGB + caption + depth + seg + edge.
```

This is important for fair multi-condition training.

## 16. Edge Data

Offline edge maps are generated using Gaussian blur plus Canny.

User reference script:

```text
scripts/generate_blip_edge_labels.py
```

Regeneration script:

```text
scripts/run_regenerate_blip_edge_201_first500_gpu0.sh
```

The user noted:

```text
offline edge generation uses k=5 Gaussian blur before Canny
```

The training loss uses a differentiable soft-Canny transform with:

```yaml
gaussian_kernel: 11
threshold_min: 0.2745
threshold_max: 0.5882
temperature: 0.03
```

The thresholds correspond to:

```text
70 / 255 = 0.2745
150 / 255 = 0.5882
```

## 17. Robust Data Loading

Corrupt edge maps caused:

```text
OSError: image file is truncated
```

Final behavior:

- Keep `ImageFile.LOAD_TRUNCATED_IMAGES = False`.
- Let corrupt files raise an error.
- In training, catch the error and randomly sample another image.
- In validation, fallback behavior can recompute edge from RGB if needed.

This prevents a single corrupt file from killing distributed training.

## 18. Cycle Losses

Final mixed cycle setting:

```yaml
cycle_weight: 0.005
cycle_t_min: 0.3
cycle_t_max: 1.0
cycle_subbatch_size: 2
cycle_apply_every: 1
```

Depth cycle:

```text
generated RGB -> DA3 -> predicted depth
compare predicted depth to depth condition label
```

Seg cycle:

```text
generated RGB -> segmentation consistency path
compare predicted segmentation structure to segmentation condition label
```

Edge cycle:

```text
generated RGB -> soft Canny
GT RGB -> soft Canny
compare generated edge to detached GT edge
```

Why edge uses GT RGB:

```text
Offline Canny labels are sensitive to random/variable Canny parameters.
Comparing generated RGB and GT RGB under the same soft-Canny transform gives a more consistent target.
```

## 19. Training Stages

### Stage 0: Depth-Only Initialization

Checkpoint:

```text
exp_pixeldit_depth_control_v1_512_bs16x2_acc4_cycle001_nograd_l128075_from8k/checkpoints/epoch_5_step_10000.pth
```

Purpose:

```text
Initialize depth branch from a strong depth-control model.
```

### Stage 1: Initial Three-Control From Depth

Config:

```text
t2i/configs_t2i/pixeldit_threecontrol_v1.yaml
```

Observation:

```text
Depth started degrading by later checkpoints; sky false-shadow artifacts appeared around 8000.
Edge remained messy.
```

### Stage 2: Edgefix From 6k

Config:

```text
t2i/configs_t2i/pixeldit_threecontrol_v1_edgefix_from6k.yaml
```

Strategy:

```text
Freeze/protect depth, reduce seg LR, disable edge injection, disable cycle loss.
```

### Stage 3: Edge-Only Reinitialization

Config:

```text
t2i/configs_t2i/pixeldit_threecontrol_v1_edgeonly_reinit_from10k.yaml
```

Strategy:

```text
Freeze depth and seg. Skip loading edge branch. Train fresh edge branch and gate.
```

### Stage 4: Edge-Only SoftCanny

Config:

```text
t2i/configs_t2i/pixeldit_threecontrol_v1_edgeonly_softcanny_from10k.yaml
```

Strategy:

```text
Add SoftCanny cycle weight 0.02 for edge consistency.
```

### Stage 5: Mixed Small-LR

Config:

```text
t2i/configs_t2i/pixeldit_threecontrol_v1_mixed_smalllr_from_softcanny2k.yaml
```

Strategy:

```text
Unfreeze all branches with small LR; restore all seven training modes.
```

### Stage 6: Final Mixed Cycle005 First200

Config:

```text
t2i/configs_t2i/pixeldit_threecontrol_v1_mixed_cycle005_from_mixed2k.yaml
```

Final checkpoint:

```text
t2i/universal_pix_t2i_workdirs/exp_pixeldit_threecontrol_v1_mixed_cycle005_first200_from_mixed2k/checkpoints/epoch_1_step_10000.pth
```

Strategy:

```text
200 shards, all branches small LR, gate LR 0.5 scale, weak cycle loss 0.005 for depth/seg/edge.
```

## 20. Final Model Settings

Final workdir:

```text
t2i/universal_pix_t2i_workdirs/exp_pixeldit_threecontrol_v1_mixed_cycle005_first200_from_mixed2k
```

Final checkpoint:

```text
checkpoints/epoch_1_step_10000.pth
```

Final LR scales:

```yaml
base_lr: 2.0e-5
depth_branch_lr_scale: 0.05
seg_branch_lr_scale: 0.1
edge_branch_lr_scale: 0.1
gate_lr_scale: 0.5
```

Effective LR:

```text
depth: 1.0e-6
seg:   2.0e-6
edge:  2.0e-6
gate:  1.0e-5
```

Backbone:

```text
frozen
```

## 21. Single-Control Baselines

To fairly compare with the original depth-only model, additional single-control models were trained:

Depth:

```text
exp_pixeldit_depth_control_v1_512_bs16x2_acc4_cycle001_nograd_l128075_from8k
```

Seg with cycle/injection:

```text
exp_pixeldit_seg_control_v1_512_bs16x2_acc4_cycle002_first200
```

Edge without injection but with SoftCanny 0.01:

```text
exp_pixeldit_edge_control_v1_512_bs16x2_acc4_noinj_softcanny001_first200
```

Pure seg no injection/no cycle:

```text
exp_pixeldit_seg_control_v1_512_bs16x2_acc4_noinj_nocycle_first200
```

These are useful for:

- single-condition upper bounds,
- ablation against multi-control branch sharing,
- checking whether multi-control training damages single-control ability.

## 22. Evaluation

Qualitative validation:

```text
Generate depth, seg, edge, depth_seg, depth_edge, seg_edge, depth_seg_edge on sa_000201.
```

Depth consistency:

Use DA3 to predict depth from generated image and compare with depth condition.

Original script:

```text
t2i/third_party/PixelGen/scripts/eval_depth_consistency_da3.py
```

Baseline single-control upload/eval outputs:

```text
outputs/baseline_eval
```

HF dataset:

```text
https://huggingface.co/datasets/<your-hf-user>/<your-eval-dataset>
```

Uploaded structure:

```text
baseline_eval_single_controls/<method>/sa_000201_first2000/{depth,seg,edge}/
```

## 23. Ablation Ideas

Important ablations for paper:

1. Shared concatenated encoder vs independent branches.
2. Gate active for all samples vs hard selection for single-condition samples.
3. Layer-wise scalar gate vs fixed average fusion.
4. Depth-favored gate initialization vs uniform initialization.
5. Edge structure injection on vs off.
6. No cycle loss vs depth/seg/edge cycle.
7. Edge direct label matching vs SoftCanny generated/GT consistency.
8. 50 shards vs 200 shards.
9. Freeze depth branch vs small depth LR.
10. With/without gradient masking for inactive branches.

Expected qualitative claims:

- Independent branches reduce cross-modality interference.
- Hard selection preserves unimodal controllability.
- Edge injection off improves image naturalness.
- SoftCanny improves edge consistency without forcing noisy label matching.
- Weak final cycle loss balances controllability and visual quality.

## 24. Figures To Make

Suggested figures:

1. Architecture diagram:
   PixelDiT backbone with three parallel control branches and layer-wise gate.
2. Single vs multi behavior:
   one-hot hard selection vs masked softmax fusion.
3. Qualitative grid:
   same prompt with depth, seg, edge, and combined controls.
4. Gate visualization:
   per-layer weights for depth_seg, depth_edge, seg_edge, depth_seg_edge.
5. Ablation grid:
   edge injection on/off, cycle on/off, shared vs independent.
6. Failure case:
   depth sky artifacts from earlier mixed model.

## 25. Table Suggestions

Table 1: Method comparison.

Columns:

```text
Method
Branch design
Single-control behavior
Fusion
Depth score
Seg score
Edge score
Image quality
```

Table 2: Ablation.

Rows:

```text
shared encoder
independent branches + average
independent branches + gate
gate + hard single select
gate + hard single select + cycle
final
```

Table 3: Training strategy.

Rows:

```text
initial three-control
edgefix
edge reinit
edge softcanny
mixed small LR
final cycle005 first200
```

## 26. Limitations

Potential limitations to mention:

- Layer-wise scalar gate is global over spatial positions, so it cannot resolve local conflicts where depth and edge are reliable in different regions.
- Edge control remains sensitive to preprocessing and threshold choices.
- Training uses several staged refinements, which increases engineering complexity.
- Cycle losses improve consistency but can reduce naturalness if weighted too strongly.
- Validation currently emphasizes a fixed shard `sa_000201`; broader benchmark coverage is needed.

## 27. Future Work

Possible future directions:

- Patch-wise or token-wise fusion after stable layer-wise gate.
- Gate regularization to encourage interpretable modality usage.
- Learned confidence maps for each control condition.
- Better edge representation, e.g. soft boundaries or HED-like maps.
- Unified control dropout curriculum.
- Automatic conflict resolution when controls disagree.

## 28. Codex Prompt For Paper Draft

Use this prompt to ask Codex to write the paper:

```text
Write a research paper draft for a text-to-image diffusion control method named "Independent-Branch Gated Multi-Control PixelDiT".

The method extends PixelDiT with three independent control branches for depth, segmentation, and edge maps. Each branch has its own encoder and residual adapters. For single-condition samples, the model bypasses the learned fusion gate and hard-selects the active branch, preserving unimodal controllability. For multi-condition samples, the model uses a layer-wise learnable gate with masked softmax over active controls only. The final residual at each layer is a weighted sum of active branch residuals.

Emphasize the motivation: shared concatenated control encoders cause cross-modality interference, especially depth degradation and noisy edge effects. Explain why depth, segmentation, and edge have different statistics. Describe the DDP-safe mode sampling, gradient masking for inactive branches, branch-specific LR scaling, edge injection disabled, and SoftCanny edge cycle loss.

Include sections: Abstract, Introduction, Related Work, Method, Training Strategy, Experiments, Ablations, Limitations, Conclusion. Add equations for hard single selection and masked softmax fusion. Add paper-style ablation descriptions using the notes in docs/my_network_paper_writeup.md.
```

## 29. Most Important Sentences

If the paper must be short, preserve these points:

```text
Unlike prior multi-control designs that concatenate all conditions into one encoder, our method assigns each condition an independent residual branch and only fuses their outputs at injection layers.
```

```text
We explicitly decouple single-control and multi-control behavior: single-control samples hard-select the corresponding branch, while multi-control samples use masked layer-wise softmax fusion.
```

```text
This design preserves unimodal controllability while allowing the model to learn how to combine active controls in multi-condition settings.
```

```text
For edge control, we disable additional structure injection and use a weak soft-Canny image-cycle loss to improve consistency without overemphasizing high-frequency artifacts.
```

