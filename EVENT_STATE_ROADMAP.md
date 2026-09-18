# Event-driven speaker-count roadmap

Status: **completed negative experiment; retained as an audit record**.
The oracle numbers below are diagnostic results, not deployable model results.
The next model direction is specified in `PYRAMID_ORDINAL_ROADMAP.md`.

## Why this direction

Speaker count is a piecewise-constant state.  Most frames should keep the same
count, while a sparse event changes it:

```text
state:   1 1 1 1 1 2 2 2 2 1 1 1
event:   . . . . . UP  . . . DOWN . .
```

On `vox_sel` after the existing 100 Hz -> 25 Hz majority alignment, a read-only
label audit found approximately:

- 1,894,232 valid frames;
- 0.9269% boundary frames;
- 99.812% of observed count changes have magnitude one.

The existing P6c TCN still fragments predictions.  More importantly, a
ground-truth-boundary oracle on `p6c_tcnft_s1234/step400.pt` found:

| Decoder on existing logits | pooled macro-F1 | OSD F1 | recording-mean macro-F1 |
|---|---:|---:|---:|
| frame argmax | 0.577388 | 0.540389 | 0.509013 |
| GT boundaries + causal evidence accumulation | 0.618850 | 0.591297 | 0.555045 |
| GT boundaries + whole-segment pooling | 0.667576 | 0.708371 | 0.596269 |

The causal oracle uses only evidence observed since the last true boundary.
The whole-segment oracle also uses future frames inside the segment.  Both use
ground-truth boundaries, so they are upper bounds.  They show that structural
headroom is measured in percentage points rather than the millipoint effects
seen in recent decoder/loss ablations.

## Proposed model

The TCN produces two views of the same hidden sequence:

1. an absolute count emission over `0/1/2/3+`;
2. a signed event distribution over `DOWN/STAY/UP`.

A causal differentiable Markov filter combines the current emission, the event
conditioned transition prior, and the previous filtered state:

```text
TCN h_t --> absolute emission p_t ---------------------+
       \-> event probability DOWN/STAY/UP --> A_t -----+--> filtered q_t
previous filtered state q_(t-1) -----------------------+
```

The absolute emission is retained at every frame so the system can recover
after a missed or false event.  Pure integration of count deltas is forbidden:
one missed onset would otherwise leave the rest of the stream permanently
offset.  The recurrence is computed in float32 and caches only the last
four-state posterior in addition to the existing TCN caches.  It must not add
future context or algorithmic latency.

Boundary means an **observed count change**, not a speaker-identity change.  A
speaker hand-off that keeps count one is not a target event.  `3+` is a censored
state: hidden changes within `3+` cannot be supervised and must not be invented.

## Phase 7 arms

The deployment-relevant test uses the clean Phase-6c graduation regime.  Each
seed starts from its own trained `p5c_tcn192_s{seed}/step3000.pt`, then all
65.8M parameters are fine-tuned for 800 steps with uniform LR `3e-5`, warmup
500, cosine schedule, paired seeds `1234/2345/3456`, and identical batch order
and stochastic-model RNG.  KD and boundary T-MSE stay off; legacy KL
smoothness stays at its historical `0.02` in every arm.

| Arm | Public count output | Extra supervision | Purpose |
|---|---|---|---|
| C0 `tcn` | raw emission | none | newly rerun Phase-6c control |
| C1 `event_aux` | raw emission | signed event CE | supervision/extra-parameter control |
| C2 `event_filter` | filtered state | signed event CE + raw-emission anchor | primary system |

Primary endpoint is **step 400**, fixed before training.  Steps 200 and 800 are
secondary learning-curve diagnostics and may not replace it after results are
seen.  A C2 claim requires both `C2 > C1` (the complete structured system adds
value beyond event supervision) and `C2 > C0` (it beats the incumbent).  C1
and C2 instantiate the same module and parameter count; C2 additionally makes
the filter public and anchors its raw absolute emission.  Therefore C2-vs-C1
is a *structured-system* contrast, not a filter-only attribution.

C0 is deliberately rerun instead of reusing historical P6c numbers.  Module
construction of the event arms consumes extra RNG, so every arm resets Python,
NumPy, Torch, and CUDA RNG immediately after checkpoint loading.  The data
loader retains its own seed.  Initialization guards require C0 to have no
missing/unexpected keys and C1/C2 to have exactly
`head.event_proj.weight`/`head.event_proj.bias` missing.

Default new-loss proposal, subject only to finite-gradient scale checks before
the long launch:

```yaml
lambda_event: 0.2
event_class_weights: [12.0, 1.0, 12.0]  # DOWN, STAY, UP; capped sqrt imbalance
lambda_raw_anchor: 0.25                  # C2 only
```

Weighted CE is cost-sensitive, not probability-calibrated: at its optimum the
event softmax is proportional to `class_weight * true_probability`.  Before
constructing a transition matrix, C2 therefore subtracts the log class weights
from the event logits.  A config guard requires filter weights to exactly match
loss weights.  With 0.9269% observed boundaries, the STAY classifier bias is
initialized to `2.9`; after undoing the 12x rare-event weighting this implies
approximately the empirical 99% STAY prior rather than hallucinating a change
on roughly one frame in five.  The new event projection weight is initialized
to zero, so its first output is this feature-independent prior rather than
random boundaries injected into an already-strong checkpoint.

The filtered recurrent loss is evaluated in float32, but under AMP its
gradient must still cross fp16 activations.  A real 400-frame smoke batch found
the default GradScaler scale 65536 overflowed two gradient tensors while 4096
and 16384 were finite.  Phase 7 locks `amp_init_scale: 4096` for every arm and
checks the unscaled global gradient norm before advancing either optimizer or
scheduler.  The 800-step run is shorter than the 2000-step scale-growth
interval, so this scale cannot silently grow back during Phase 7.

The transition prior is tridiagonal: DOWN/STAY/UP means exactly
`i->i-1/i/i+1`, with impossible edge moves folded into STAY.  Full
inverse-frequency weighting is forbidden because transitions are below
1% of frames and would encourage false boundaries.  Event targets are derived
after 25 Hz label alignment.  Both adjacent frames must be valid; crop frame
zero and padding never receive an event target.

## Metrics and decision rule

Primary metric: `vox_sel` recording-level macro-F1 with three paired seeds and
paired recording bootstrap, matching Phase 5.  Mechanism metrics are computed
on reconstructed recording timelines, not independently averaged 30-second
windows:

- strict one-to-one signed transitions for `0<->1`, `1<->2`, and `2<->3+`;
- boundary precision/recall/F1 at 40/80/160/250 ms;
- signed early/late error median and p90;
- false transitions per minute inside true segment interiors;
- split/merge count and overlap-segment coverage by duration;
- frame macro-F1, class-2/class-3 F1, OSD F1, and class-0 false speech.

Go to full-backbone graduation only if C2 has all paired seed deltas positive,
the recording bootstrap interval excludes zero, and the mean gain is at least
`+0.010` macro-F1 or there is a comparably clear class-2/OSD mechanism gain.
Effects below `0.005` are not worth another architecture cycle.  Domain claims
require later confirmation on AMI/DipCo and VoxConverse/MSDWild separately.

## Acoustic side-information phase (only after C2)

Do not begin with scalar F0 or raw first formant F1.  The ordered probe is:

1. energy, causal spectral flux, and spectral flatness from the existing fbank;
2. voicing/periodicity confidence and a multi-harmonic salience representation
   retaining at least the top two pitch hypotheses;
3. scalar F0 plus validity/confidence as an explanatory control;
4. F1/LPC statistics as exploratory negative controls only.

Single-F0 trackers are ill-defined precisely during overlapping speech.  The
useful evidence is harmonic multiplicity, top-2 salience, voicing confidence,
and instability, not one Hz value.  Raw F1 mostly represents vocal-tract and
phonetic content, is unstable in mixtures/reverberation, and is already partly
represented by log-mel features.

Acoustic cues enter a small causal, zero-initialized side branch feeding the
event potential first.  Required controls are parameter-matched `zero`, cue
shifted by at least ten seconds inside the recording, and correctly aligned
cue.  Only `aligned > shifted` demonstrates time-local information.  Feature
extraction must happen after waveform augmentation and must pass an explicit
future-perturbation test; centered pitch windows or whole-sequence Viterbi
smoothing are forbidden.

If synchronized multichannel audio is in scope, spatial IPD/GCC-PHAT is a
separate, potentially higher-upside path for AMI/DipCo.  It must not be mixed
into the single-channel experiment because the current loader downmixes audio.

## Required tests before long training

- transition matrices are non-negative, row-normalized, and directionally
  correct at boundary states;
- invalid/padded pairs and crop frame zero contribute no event loss;
- float16 inputs produce finite filter output, loss, and gradients;
- future perturbations do not change past outputs;
- full forward equals cached fixed-chunk and ragged-chunk forward;
- warm cache differs from cold cache and reset restores cold behavior;
- checkpoint initialization reports only the expected new event parameters as
  missing;
- a short real-batch training smoke test updates the event parameters;
- throughput and memory remain acceptable for the full run;
- the tmux runner checks disk, GPU, manifests, checkpoints, and refuses to
  overwrite an existing non-empty run directory.

## Execution record

Fill this section as work proceeds.

- [x] Label audit and GT-boundary oracle completed (diagnostic, seed 1234).
- [x] Event-state module implemented.
- [x] Loss/config plumbing implemented.
- [x] Streaming and fp16 unit tests pass.
- [x] Real-batch AMP smoke passes: finite loss/69 gradient tensors, corrected
      prior `[0.004542, 0.990916, 0.004542]`, event weight updated.
- [x] Head throughput check passes: batch 24 x 750 C2 forward+backward ~0.46 s,
      0.83 GiB peak for the head-only benchmark on RTX 3090.
- [x] Phase 7 runner preflight passes: CUDA 22.8/23.6 GiB free, disk 34 GiB.
- [x] Phase 7 completed: 9/9 runs clean at the locked step 400.
- [x] Result: C0 `0.5611`, C1 `0.5600`, C2 `0.5337` recording macro-F1.
- [x] Interpretation: exact-frame event auxiliary was null and the Markov
      filter was harmful.  This does not test or reject soft boundary
      supervision, CORN output, segment consistency, or two-stage refinement.
