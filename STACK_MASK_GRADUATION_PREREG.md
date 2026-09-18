# Stack-input mask graduation (locked protocol)

Status: preregistered before the first training run.

## Question

The diagnostic stack probe suggested that the shallow/middle Zipformer
representations carry most of the speaker-count signal.  That probe changed
the head input width, parameter count, and initialization, so it cannot support
an architecture claim.  This graduation asks the narrower causal question:

> With the complete six-stack TCN graph, parameterization, initialization,
> optimizer, data order, and endpoint held fixed, is suppressing selected
> stack inputs non-inferior to using all six?

This experiment is a representation-view ablation.  It does **not** reduce
backbone compute or deployment latency.  A slim/early-exit backbone is a
separate follow-up and is authorized only by the strict non-inferiority rule
below.

## Arms

The only model/training-semantic configuration field allowed to differ is
`model.head.stack_input_mask`, a six-element binary list applied after
per-stack normalization and before gated fusion.  Artifact identity fields
such as `training.log_dir` necessarily carry the arm name:

| arm | mask for stacks 0..5 | interpretation |
|---|---|---|
| `all6` | `[1, 1, 1, 1, 1, 1]` | incumbent full hypercolumn view |
| `only1` | `[0, 1, 0, 0, 0, 0]` | stack 1 alone |
| `pre012` | `[1, 1, 1, 0, 0, 0]` | pre-bottleneck stacks 0–2 |

The mask must not remove modules, parameters, or persistent state.  A mandatory
preflight rebuilds every arm from the same RNG state, loads the same v3
backbone, and requires identical:

- state-dict keys, shapes, dtypes, and tensor-value SHA;
- named-parameter keys, shapes, trainability, and tensor-value SHA;
- total and trainable parameter counts;
- missing/unexpected keys when loading the v3 backbone.

It also perturbs included and excluded stack slices to verify that each mask is
active and has the intended semantics.  Failure of any invariant aborts before
training.

## Locked training and evaluation protocol

- Seeds: `1234`, `2345`, `3456`.
- Initialization: `logs/r8_v3_diverse/best_macro_f1.pt`.
- Backbone: fully frozen and kept in evaluation mode.
- Head: the complete six-stack causal `TCNCountHead`, `d_model=192`.
- Alignment: historical `average`; no final-output residual.
- Objective: the full legacy v4 loss block, KD and boundary T-MSE off.
- Training: 3,000 steps, batch 24, AdamW at `1e-3`, 500-step warmup,
  cosine schedule, AMP initial scale 4096.
- Any AMP overflow invalidates the run (`max_amp_overflows=0`).
- Complete v3 backbone loading is mandatory
  (`require_backbone_complete=true`).
- RNG is reset after initialization.
- Endpoint: **locked step 3000**; no best-checkpoint selection.
- Train manifest:
  `/media/edabk/500GB Hard Disk/data_diar/manifests/train_manifest_v3.json`.
- Training validation manifest:
  `/home/edabk/hoangbpm/diar/diar_new/data/meetings/val_manifest.json`.
- Primary evaluation manifest:
  `/media/edabk/500GB Hard Disk/data_diar/voxconverse/vox_sel.json`.

Run order is a fixed Latin square to balance wall-clock position:

1. seed 1234: `all6`, `only1`, `pre012`;
2. seed 2345: `only1`, `pre012`, `all6`;
3. seed 3456: `pre012`, `all6`, `only1`.

All configs, initialization, manifests, evaluator, training/model/data/loss
sources, dynamic Zipformer recipe sources, this preregistration, and the runner
are SHA-bound.  Each completion sidecar additionally binds the result,
checkpoint, and on-disk config.  Inputs are re-hashed after evaluation; source
or data drift during a run leaves it uncertified.  Existing artifacts are
reused only when both sides of this lineage match exactly; otherwise they are
moved to timestamped `.stale.*` paths.

## Statistics and decisions

The primary endpoint is pooled macro-F1.  Paired deltas retain seed identity.
A paired bootstrap averages the three seeds within each recording, then
resamples the 116 recordings 20,000 times with RNG seed 0.  Recording sets must
match exactly; intersection is forbidden.

Each sparse-view arm is compared with `all6`.  The runner reports:

- **Operational mean-only NI** (compatibility diagnostic):
  mean pooled delta `>= -0.002`.  This alone is not a scientific claim.
- **Strict non-inferiority**: every paired seed delta is `>= -0.002` and the
  recording-bootstrap lower 95% bound is `> -0.002`.
- **REAL positive evidence**: all three seed deltas are positive and the
  recording-bootstrap lower bound is above zero.
- **NEGATIVE evidence**: all three seed deltas are negative and the upper
  bound is below zero.
- **Practical adoption R2**: mean pooled delta `>= +0.005`.
- **Practical adoption R3**: class-2 or OSD mean delta `>= +0.010`, while
  pooled macro remains operationally non-inferior (`>= -0.002`).

Strict NI authorizes, but does not itself adopt, a later capacity-matched
slim/early-exit experiment.  The current Phase-8 base is never rewritten by
this runner.

Candidate selection is fixed:

1. If neither sparse view is strict-NI, retain `all6`; no slim follow-up.
2. If exactly one is strict-NI, that arm is the slim follow-up candidate.
3. If both are strict-NI, prefer simpler `only1` unless `pre012` beats
   `only1` directly through R2 or R3.
4. A mask is recommended for current-model adoption only if its direct
   contrast against `all6` passes R2 or R3.  Non-inferiority alone is not
   called an improvement.

Class-wise and OSD metrics are secondary.  No AMI, teacher, probe, or
intermediate-checkpoint result may alter the locked endpoint or rules above.
