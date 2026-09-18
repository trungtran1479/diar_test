"""Post-hoc frame-level decoders for the speaker-count logit sequence.

All three operate on a single recording's ALREADY-COMPUTED per-frame class
probabilities (the frozen model's argmax/softmax output) -- none of them
retrain anything. They exist to test a specific hypothesis: how much of the
gap to DiariZen is temporal-consistency/flicker that a smarter decode-time
constraint can recover, versus how much is a genuine per-frame
classification error no amount of smoothing can fix (see
zipcount-wavlm-offline-progress memory, "user's methodological critique").

Increasing structure, in the order they should be tried:
  1. hysteresis_min_duration -- cheap frame-local smoothing, no learned prior.
  2. viterbi_decode -- HMM-style: an emission (the model's own probabilities)
     plus a learned 4x4 FRAME-transition prior (self-loop-heavy; see
     fit_decoder_priors.py's `log_trans_frame`). Implicitly geometric
     segment durations.
  3. semimarkov_decode -- explicit per-class duration prior instead of the
     implicit geometric one, PLUS a separate SEGMENT-transition prior with
     same-class transitions forbidden (`log_trans_segment`, diagonal =
     -inf) -- using the frame-level self-loop-heavy matrix here would let
     the DP split one long true segment into several same-class hidden
     segments purely to re-collect the duration prior's peak repeatedly
     (caught in review before this ever ran on real data).

Gap handling: NONE of these functions may see a `valid` array with False
frames in the middle and be expected to keep the two True-runs on either
side independent -- every one of them is a stateful sequence model (sticky
argmax, Viterbi DP, segmental DP) and state trivially crosses a masked-out
region even if that region's own EMISSION is zeroed. The caller MUST use
`decode_respecting_gaps` (below), which slices out each contiguous valid
span and decodes it in isolation, rather than calling these three functions
directly on a `valid` array containing gaps. This is enforced structurally:
each function still accepts `valid` (for the all-valid common case and for
defensive clarity) but callers going through real cached recordings with
partial coverage MUST go through the wrapper.

All three low-level functions:
  - take `log_probs`/`probs`: [T, C] and `valid: [T] bool` (all-True in the
    normal, gap-free call pattern -- see above).
  - return `decoded: [T] int64`.
  - never mutate their inputs.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

NEG_INF = -1e18


def _safe_log(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.log(np.clip(p, eps, 1.0))


def contiguous_valid_spans(valid: np.ndarray) -> List[Tuple[int, int]]:
    """[(start, end)] of maximal contiguous True runs in `valid`, end excl."""
    spans = []
    in_run = False
    start = 0
    for i, v in enumerate(valid):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            spans.append((start, i))
            in_run = False
    if in_run:
        spans.append((start, len(valid)))
    return spans


def decode_respecting_gaps(
    decode_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    probs_or_log_probs: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Decode each contiguous valid span independently and reassemble.

    `decode_fn` must have signature (span_probs, span_valid_all_true) ->
    span_decoded, i.e. it is one of hysteresis_min_duration / viterbi_decode
    / semimarkov_decode (or a closure fixing their other arguments) called
    on a span that is entirely valid. Gap frames in the output are left as
    0 -- never read, since every caller (src/utils/decode_scoring.py) masks
    by the SAME `valid` array before scoring.
    """
    T = len(valid)
    decoded = np.zeros(T, dtype=np.int64)
    for s, e in contiguous_valid_spans(valid):
        span_valid = np.ones(e - s, dtype=bool)
        decoded[s:e] = decode_fn(probs_or_log_probs[s:e], span_valid)
    return decoded


def hysteresis_min_duration(
    probs: np.ndarray,
    valid: np.ndarray,
    switch_margin: float = 0.15,
    min_duration_frames: int = 5,
) -> np.ndarray:
    """Sticky argmax + minimum-duration merge.

    Step 1 (hysteresis): start on frame 0's argmax; at each subsequent frame,
    only switch away from the CURRENT class if some other class's
    probability exceeds the current class's probability by `switch_margin`.
    This is the natural multi-class generalisation of a two-state
    on/off hysteresis gate (no single global on/off threshold makes sense
    for a 4-way class).

    Step 2 (minimum duration): any run shorter than `min_duration_frames` is
    reassigned to whichever of its two neighbouring CLASSES better explains
    the SHORT RUN'S OWN frames -- i.e. compare
    `probs[run_span, left_class].mean()` vs `probs[run_span, right_class]
    .mean()`, NOT each neighbour's confidence over its OWN frames (an
    earlier version compared the neighbours' self-confidence, which answers
    "which neighbour is more sure of itself" rather than "which neighbour's
    class better fits this run" -- caught in review with a counter-example
    where the correct call flips). Ties -> earlier neighbour. Runs are
    re-collapsed and the check repeats until no short run remains or the
    whole sequence collapses to one run.

    See decode_respecting_gaps -- this function assumes `valid` is all-True
    (a single contiguous span); it does not itself special-case gaps.
    """
    T = probs.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    argmax = probs.argmax(-1)
    decoded = np.empty(T, dtype=np.int64)
    cur = int(argmax[0])
    decoded[0] = cur
    for t in range(1, T):
        p_cur = probs[t, cur]
        best_c = int(probs[t].argmax())
        if probs[t, best_c] - p_cur > switch_margin:
            cur = best_c
        decoded[t] = cur

    # merge short runs
    while True:
        runs = _runs(decoded)
        short = [i for i, (_, s, e) in enumerate(runs) if (e - s) < min_duration_frames]
        if not short or len(runs) <= 1:
            break
        i = short[0]
        _, s, e = runs[i]
        seg_probs = probs[s:e]  # this run's OWN frames -- evaluated under
                                  # each CANDIDATE class below, not read here
        left_class = runs[i - 1][0] if i > 0 else None
        right_class = runs[i + 1][0] if i + 1 < len(runs) else None
        if left_class is None:
            target = right_class
        elif right_class is None:
            target = left_class
        else:
            left_support = seg_probs[:, left_class].mean()
            right_support = seg_probs[:, right_class].mean()
            target = left_class if left_support >= right_support else right_class
        decoded[s:e] = target

    decoded[~valid] = 0
    return decoded


def _runs(seq: np.ndarray):
    """[(label, start, end)] for consecutive equal-label runs, end exclusive."""
    out = []
    if len(seq) == 0:
        return out
    s = 0
    cur = seq[0]
    for i in range(1, len(seq)):
        if seq[i] != cur:
            out.append((int(cur), s, i))
            s = i
            cur = seq[i]
    out.append((int(cur), s, len(seq)))
    return out


def viterbi_decode(
    log_probs: np.ndarray,
    valid: np.ndarray,
    log_trans_frame: np.ndarray,
    trans_weight: float = 1.0,
) -> np.ndarray:
    """Standard first-order Viterbi (frame-level HMM).

    log_probs: [T, C] log emission probabilities (the model's own).
    log_trans_frame: [C, C] log P(class_t | class_{t-1}) FIT PER FRAME (row
        = from, col = to), i.e. self-loop-heavy (see fit_decoder_priors.py's
        `log_trans_frame`) -- this is the correct prior for this function,
        which models an implicitly-geometric segment duration through the
        self-transition probability. Do NOT pass the segment-transition
        prior here (that one forbids self-transitions and belongs to
        semimarkov_decode only).
    trans_weight: scales the transition term's influence relative to the
        emission term (locked hyperparameter, tuned on DEV) -- 0 reduces to
        per-frame argmax, 1 uses the fitted prior at full strength.

    See decode_respecting_gaps -- this function assumes `valid` is all-True
    (a single contiguous span); the DP state has no gap-awareness of its
    own.
    """
    T, C = log_probs.shape
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    lt = log_trans_frame * trans_weight
    emit = np.where(valid[:, None], log_probs, 0.0)

    V = np.full((T, C), NEG_INF)
    ptr = np.zeros((T, C), dtype=np.int64)
    V[0] = emit[0]
    for t in range(1, T):
        # V[t-1, :, None] + lt[:, :]  -> [from, to]
        scores = V[t - 1][:, None] + lt  # [C_from, C_to]
        ptr[t] = scores.argmax(axis=0)
        V[t] = scores.max(axis=0) + emit[t]

    decoded = np.zeros(T, dtype=np.int64)
    decoded[T - 1] = int(V[T - 1].argmax())
    for t in range(T - 2, -1, -1):
        decoded[t] = ptr[t + 1, decoded[t + 1]]
    decoded[~valid] = 0
    return decoded


def semimarkov_decode(
    log_probs: np.ndarray,
    valid: np.ndarray,
    log_trans_segment: np.ndarray,
    duration_log_pmf: np.ndarray,
    max_dur: int = 250,
    trans_weight: float = 1.0,
    duration_weight: float = 1.0,
) -> np.ndarray:
    """Explicit-duration (semi-Markov) segmental Viterbi, exact DP.

    log_trans_segment: [C, C] log P(next SEGMENT's class | this segment's
        class), fit ONLY over adjacent DIFFERENT-class segment pairs (see
        fit_decoder_priors.py's `log_trans_segment`) -- the diagonal is
        -inf, enforced both when fitting and again defensively here
        (`np.fill_diagonal(lt, NEG_INF)`), so the DP structurally CANNOT
        chain two adjacent hidden segments of the same visible class. An
        earlier version reused the frame-level self-loop-heavy transition
        matrix here, which barely penalised same-class adjacency and let
        one long true run get carved into several short hidden segments
        purely to re-collect the duration prior's peak repeatedly --
        verified with `max_dur=2` producing a 6-frame class-0 run in the
        DECODED (visible) output, i.e. duration_log_pmf was not actually
        bounding what a human/scorer would see as one run's length. Fixed
        by construction here, not by a smarter prior alone.

    duration_log_pmf: [C, max_dur] log P(segment has length d | class c) for
        d = 1..max_dur (index d-1). max_dur is a HARD cap on any single
        DECODED segment: a true segment longer than max_dur frames is
        structurally impossible to represent as one segment here (the
        forbidden-self-transition fix above means the DP cannot paper over
        this by silently splitting into same-class pieces the way the
        buggy version did) and will be decoded as >=2 segments bridged by a
        brief different-class detour, or as whatever the DP finds cheapest
        under the fitted priors. This is a real, bounded approximation, not
        swept under an inaccurate "clamped to the last bin" claim -- pick
        max_dur comfortably larger than the corpus's real segment lengths
        (see fit_decoder_priors.py's --max-dur-frames and its printed mean
        durations) to keep this rare in practice, and treat any adoption
        decision as conditional on checking how often max_dur actually
        binds on the corpus being decoded.

    State: V[t, c] = best log-score of a segmentation of frames [0, t) whose
    LAST segment has class c. The naive recurrence maximises over both the
    segment start s=t-d AND the predecessor class c' at s, which is
    O(T * max_dur * C^2) if done directly. Standard trick used here: since
    the transition term only depends on (c', c) and not on d, precompute
    M[s, c] = max_c' (V[s, c'] + log_trans_segment[c', c]) -- an O(T * C^2)
    pass, trivial for C=4 -- then the per-(t, c) maximisation over d needs
    only M[t-d, c], dropping the C^2 factor from the inner loop. The d-loop
    itself is vectorised with numpy (one array op per t, not per (t, d)).

    See decode_respecting_gaps -- this function assumes `valid` is all-True
    (a single contiguous span).
    """
    T, C = log_probs.shape
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    emit = np.where(valid[:, None], log_probs, 0.0)
    cum = np.zeros((T + 1, C))
    cum[1:] = np.cumsum(emit, axis=0)

    lt = log_trans_segment.copy() * trans_weight
    np.fill_diagonal(lt, NEG_INF)  # defensive: enforced again regardless of what was passed in
    dur_lp = duration_log_pmf * duration_weight  # [C, max_dur]

    V = np.full((T + 1, C), NEG_INF)
    V[0] = 0.0  # empty prefix: any starting class is free
    seg_start = np.zeros((T + 1, C), dtype=np.int64)   # best d's s = t-d, per (t,c)
    prev_class = np.zeros((T + 1, C), dtype=np.int64)  # best c' feeding M[s,c]

    # M[s, c] = max_c' (V[s, c'] + log_trans_segment[c', c]), c' != c enforced
    # via the -inf diagonal above. M[0, c] = 0 (no predecessor at s=0).
    M = np.full((T + 1, C), NEG_INF)
    M_argc = np.zeros((T + 1, C), dtype=np.int64)
    M[0] = 0.0

    for t in range(1, T + 1):
        D = min(max_dur, t)
        s_vals = np.arange(t - D, t)                    # ascending, length D
        d_vals = t - s_vals                              # D..1
        dur_idx = np.clip(d_vals, 1, max_dur) - 1         # [D]
        seg_emit = cum[t][None, :] - cum[s_vals]          # [D, C]
        dur_term = dur_lp[:, dur_idx].T                   # [D, C]
        m_vals = M[s_vals]                                 # [D, C]
        total = m_vals + seg_emit + dur_term               # [D, C]
        best_i = total.argmax(axis=0)                      # [C]
        V[t] = total[best_i, np.arange(C)]
        seg_start[t] = s_vals[best_i]
        prev_class[t] = M_argc[s_vals[best_i], np.arange(C)]

        # extend M for position t: M[t, c] = max_c' (V[t, c'] + lt[c', c]),
        # c' != c because lt's diagonal is -inf.
        scores = V[t][:, None] + lt  # [C_from, C_to]
        M_argc[t] = scores.argmax(axis=0)
        M[t] = scores.max(axis=0)

    decoded = np.zeros(T, dtype=np.int64)
    c = int(V[T].argmax())
    t = T
    while t > 0:
        s = int(seg_start[t, c])
        decoded[s:t] = c
        c = int(prev_class[t, c])
        t = s
    decoded[~valid] = 0
    return decoded
