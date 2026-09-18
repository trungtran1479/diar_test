# Gated pyramid ordinal segmentation roadmap

Status: **implementation and staged-ablation specification**.

This document supersedes the model direction in `EVENT_STATE_ROADMAP.md`.
Phase 7 remains a valid negative result for exact-frame
`DOWN/STAY/UP` cross-entropy plus a Markov filter, but it did not test the
architecture proposed in `chatgpt_recommend.txt`.

## Scientific question

The current model is already good at finding overlap, but its frame decisions
flicker and its transition timing is weak.  Speaker count is piecewise
constant:

```text
count:       1 1 1 1 | 2 2 2 2 | 1 1 1
interior:    --------   -------   ------
boundary:            ^         ^
```

One representation should therefore solve two different sub-problems:

1. classify the stable state inside each segment;
2. localize the sparse boundaries at which the state may change.

The working hypothesis is that the current system loses useful boundary
information in its multiscale alignment and asks a single framewise objective
to solve both tasks.  The proposed system preserves phase-sensitive encoder
features, predicts count cumulatively, and supervises boundaries and segment
interiors explicitly.

## What “change the backbone” means here

Zipformer remains the pretrained acoustic encoder.  Replacing it with a new
Transformer/Mamba encoder would discard a strong initialization and introduce
data scale as a confound.  The backbone-side change in this phase is the
trainable path from Zipformer stacks to the temporal decoder:

- preserve all six multiscale stack outputs;
- align them to 25 Hz using Zipformer's learned output downsampler rather than
  fixed 50/50 averaging;
- include the encoder's real final 512-dimensional output as a residual path;
- normalize and nonlinearly project every stack;
- choose stack evidence with a content-dependent gate at every frame;
- fine-tune this path and Zipformer end to end only after the head-only screen.

The pretrained output downsampler currently places approximately 1.4%/98.6%
mass on its two phases.  Fixed average pooling changes that to 50%/50%, which
can blur a 40 ms transition without crashing or appearing in the logs.

## Target architecture

```text
Zipformer stack outputs (50 Hz)
  | learned phase alignment to 25 Hz
  v
per-stack LN -> Linear -> SiLU
  | frame-wise softmax gate
  + final-output residual projection
  v
causal dual-dilation TCN, stage 1
  |--> CORN count P(N=0,1,2,3+)
  |--> boundary P(change)
  `--> direction P(up | change)
          |
          + logits/posteriors + hidden features
          v
causal dual-dilation TCN, stage 2
  |--> final CORN count
  |--> final boundary
  `--> final direction
```

Both stages use six causal dual-dilation blocks with dilations
`1,2,4,8,16,32`, kernel 3, and width 192 by default.  Stage 2 is a refinement
stage, not a Markov state integrator.  It receives stage-1 predictions as
features and can correct them using normal absolute-count evidence.

The model remains causal.  With two stages its temporal receptive field is
approximately 253 frames, or 10.1 seconds of past context at 25 Hz.  This does
not add lookahead or change the 16-output-frame/640 ms streaming cadence.
Both stages must cache their own causal context.

### Frame-wise pyramid neck

For aligned stack feature `x_i(t)`:

```text
z_i(t) = SiLU(W_i LN_i(x_i(t)))
g_i(t) = softmax_i(score_i(z_i(t), r(t)))
h(t)   = LN(W_final r(t) + sum_i g_i(t) z_i(t))
```

`r(t)` is the actual final Zipformer output.  Static global gates remain
available as the exact legacy control.  The dynamic gate must expose
per-recording and per-class gate statistics for mechanism analysis.

### True cumulative ordinal output

The main count output uses three conditional CORN logits:

```text
c1 = P(N >= 1)
c2 = P(N >= 2 | N >= 1)
c3 = P(N >= 3 | N >= 2)

q1 = c1
q2 = c1*c2
q3 = c1*c2*c3

p0 = 1-q1
p1 = q1-q2
p2 = q2-q3
p3 = q3
```

The public four-class posterior is therefore valid and the cumulative
probabilities are monotone by construction.  No independent VAD/OSD heads are
needed for the new system: `q1` is VAD and `q2` is OSD.

### Boundary and direction heads

The edge representation is causal:

```text
e_t = [h_t, h_(t-1), h_t - h_(t-1)]
```

It predicts:

- `P(change_t)` for boundary localization;
- `P(up_t | change_t)` only where the ground-truth count changes.

This factorization avoids the 99%-STAY class collapse observed in Phase 7.
There is no three-way STAY-dominated cross-entropy and no Markov filter.

## Structured loss

Every loss uses one strict `valid_frame` mask and one
`valid_pair = valid_frame[t-1] & valid_frame[t]` mask.  Shape mismatch, labels
outside `0..3`, and lengths outside `0..T` raise an error.  Padding, internal
invalid gaps, empty samples, and an all-invalid batch must remain finite and
must contribute exactly zero.

### Main CORN loss

The three conditional binary risks are:

- risk 1: all valid frames, target `N >= 1`;
- risk 2: valid frames with `N >= 1`, target `N >= 2`;
- risk 3: valid frames with `N >= 2`, target `N >= 3`.

Each risk is normalized by its own active population.  Computation and
posterior reconstruction are float32 under AMP.

### Soft boundary loss

A true count change produces the local target:

```text
offset:  -2    -1     0    +1    +2
target: .25   .75   1.0   .75   .25
```

At 25 Hz this gives ±80 ms tolerance.  The envelope must never cross padding
or an internal invalid gap.  Positive and negative soft-BCE contributions are
normalized separately so sparse boundaries do not require an unstable giant
`pos_weight`.

### Boundary-aware temporal smoothness

For adjacent log posteriors:

```text
d_t = min(mean_c((log p_t(c) - log p_(t-1)(c))^2), tau)
L_tmse = mean((1 - soft_boundary_t) * d_t), tau=4
```

It suppresses flicker inside a segment without penalizing a legitimate sharp
transition.  This is a new, clean key; legacy `lambda_smooth` must not activate
it.

### Segment consistency

Within each contiguous ground-truth segment, predictions are encouraged to
agree with their segment mean.  Segments are averaged equally, not in
proportion to duration, so one long class-1 segment cannot dominate many short
overlap segments.  No segment may cross padding or an invalid gap.

### Optional delta consistency

This is disabled until the boundary head passes its own localization gate:

```text
mu_t   = expected count from the ordinal posterior
dmu_t  = clip(mu_t - mu_(t-1), -1, 1)
dpi_t  = P(up_t) - P(down_t)

L_delta = 0.5 Huber(dmu_t, stopgrad(dpi_t))
        + 0.5 Huber(stopgrad(dmu_t), dpi_t)
```

The stop-gradient symmetry lets the two views teach each other without a
degenerate feedback loop.

Starting auxiliary weights, subject to a real-batch gradient-ratio audit:

```yaml
lambda_stage1: 0.20
lambda_boundary: 0.10
lambda_direction: 0.05
lambda_t_mse: 0.05
lambda_segment: 0.05
lambda_delta: 0.00
```

Each auxiliary loss must contribute no more than 20% of the shared decoder
gradient norm produced by the main final-stage loss.  Weights are adjusted
from gradient ratios before the locked runs, not from evaluation scores.

## Staged ablation

The implementation may contain all switches, but experiments change one
scientific factor at a time.

| Phase | Control | Treatment | Question |
|---|---|---|---|
| P8A0 | fixed 50/50 alignment | learned Zipformer alignment | was phase blur hurting boundaries? |
| P8A1 | static stack gate | frame-wise gate + final residual | does content-dependent scale selection help? |
| P8B | SORD on four logits | cumulative loss on the same four logits | is the gain from loss geometry alone? |
| P8B2 | best four-logit output | native CORN conditional output | does architectural monotonicity add value? |
| P8C | identical edge module, loss off | soft boundary loss on | does boundary supervision localize changes? |
| P8D1 | no temporal smoothing | boundary-aware T-MSE | does it reduce interior flicker? |
| P8D2 | best D1 | segment consistency | does equal-segment supervision stabilize plateaus? |
| P8E | one-stage | two-stage refinement | does iterative refinement beat added capacity? |
| P8E-cap | one-stage + parameter-matched TCN | two-stage | refinement versus parameter count |
| P8F | best system, delta off | detached delta consistency | only after boundary gate passes |

Do not enable F0/F1 or another acoustic branch in P8.  After the structural
system is frozen, test acoustic information separately in this order:
spectral flux/flatness/energy, multi-harmonic salience and voicing confidence,
scalar F0 with validity, then F1/LPC as an exploratory control.  Scalar F0 and
F1 are unreliable in overlapped reverberant speech and are already partly
encoded by log-mel features.

## Training protocol

For structural screening:

- initialize the encoder from the same seed-matched checkpoint;
- freeze the entire encoder and keep it in eval mode;
- train the candidate task stack for 3,000 steps at `1e-3`;
- use paired seeds `1234,2345,3456`, paired data order, and reset RNG after
  module construction;
- evaluate a locked endpoint, never select a different step per arm.

Graduation:

- initialize each arm from its own paired head-only checkpoint;
- full-model fine-tune for 800 steps at uniform `3e-5`;
- evaluate steps 200/400/800, with step 400 locked as primary;
- report all three seed deltas and paired recording bootstrap confidence
  intervals;
- benchmark AMI/DipCo/VoxConverse/MSDWild only after the configuration freezes.

`vox_sel` has already influenced model decisions and is development data.  A
paper-quality final claim requires a newly locked holdout.

Go/no-go:

- continue if mean macro-F1 improves by at least `+0.005`, or class-2/OSD by
  at least `+0.010`, while macro-F1 falls by no more than `0.002`;
- require all paired seeds to have the expected sign for a strong claim;
- require at least `+0.010` macro-F1 for the final added-complexity system;
- stop pursuing effects in the millipoint range unless the change materially
  reduces latency, memory, or fragmentation.

Boundary supervision graduates only if boundary AP reaches at least `0.05`
(Phase 7 was approximately `0.0236` at prevalence `0.0093`) and transition
recall improves by at least `0.05` at a fixed false-transition rate.

## Invariants and required tests

### Data/label invariants

- Teacher groups and labels remain sample-aligned under every batch ordering.
- Source-deny masks, teacher-presence masks, temporal masks, and confidence
  masks are independent and cannot leak through multiplication by NaN.
- No loss silently truncates labels or clamps an unexpected target.
- Adjacent-frame losses require both frames valid.
- Boundary envelopes and segments cannot cross internal invalid gaps.

### Numerical invariants

- float16 input gives finite float32 loss and finite gradients;
- every all-invalid or empty-risk case returns a differentiable zero;
- masked values use `torch.where`, not `NaN * False`;
- global gradient finiteness is checked even when gradient clipping is off;
- CORN posteriors are non-negative, sum to one, and yield monotone cumulative
  probabilities.

### Causality/streaming invariants

- perturbing future input cannot change past output;
- full forward equals chunk-16 streaming forward;
- full forward equals ragged-chunk streaming forward;
- warm-cache output differs from cold-cache output;
- reset restores cold behavior;
- both stage caches have the expected receptive field;
- the 640 ms output cadence is unchanged.

### Compatibility invariants

- all legacy configs instantiate the exact legacy model/loss path;
- new behavior requires an explicit config switch;
- checkpoint loading reports exact expected missing and unexpected keys;
- each ablation records config and checkpoint hashes and refuses to overwrite
  existing artifacts.

## Runtime acceptance

The 640 ms cadence stays fixed.  Measure rather than infer:

- cold-start end-to-end latency;
- warm streaming compute p50/p95;
- peak GPU memory;
- real-time factor;
- cache size.

Acceptance: warm p95 compute below 640 ms and at most 1.10x the paired
incumbent; peak memory at most 1.15x.  The frontend requires approximately
130 ms of fbank tail/lookahead, so cadence and end-to-end latency must be
reported separately.

## Execution record

- [x] Recommendation and existing implementation audited.
- [x] Phase-7 negative result re-scoped correctly.
- [x] Learned phase-alignment defect identified.
- [x] Config-gated architecture implemented.
- [x] Structured loss implemented.
- [x] Unit, fp16, causality, and streaming tests pass: 64 repo-owned tests.
- [x] Real mixed-batch forward/backward smoke passes: 595 output frames,
      969 finite gradient tensors, 0 bad gradients.
- [x] Parameter/memory audit: 66,937,139 total and 2,381,116 head parameters
      for the target width-192/edge-width-96 config; 0.753 GiB allocated for
      a batch-one 23.9-second forward/backward smoke.
- [x] Gradient-ratio audit: stage-1 coefficient reduced from 0.30 to 0.20;
      boundary/direction/T-MSE/segment weighted shared-gradient ratios were
      0.5%/0.7%/2.0%/<0.1% of the main count gradient on the smoke batch.
- [x] P8A runner preregistered and preflighted.
- [x] P8A launched in tmux session `p8a_alignment` on 2026-07-23;
      first arm `p8a_average_s1234` passed initialization and finite training
      startup.
