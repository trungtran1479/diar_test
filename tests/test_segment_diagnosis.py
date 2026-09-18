"""CPU-only regression tests for paper-grade segment diagnostics.

Run:
    PYTHONPATH=. python tests/test_segment_diagnosis.py
"""
import os
import sys
import unittest
import importlib.util

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
# Some environments install an unrelated top-level package named ``scripts``.
# Load this repository's CLI by path so the regression test cannot resolve the
# wrong module merely because site-packages precedes a namespace package.
spec = importlib.util.spec_from_file_location(
    "zipcount_segment_diagnosis", os.path.join(ROOT, "scripts", "segment_diagnosis.py"))
sd = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = sd
spec.loader.exec_module(sd)

_empty_system = sd._empty_system
TimelineWindow = sd.TimelineWindow
boundary_f1 = sd.boundary_f1
change_events = sd.change_events
delay_summary = sd.delay_summary
expand_output_to_reference = sd.expand_output_to_reference
match_events = sd.match_events
match_signed_events = sd.match_signed_events
overlap_transition_events = sd.overlap_transition_events
parse_tolerances = sd.parse_tolerances
positive_runs = sd.positive_runs
precision_recall_f1 = sd.precision_recall_f1
stitch_recording = sd.stitch_recording
transition_recall = sd.transition_recall
update_system = sd.update_system


class SegmentDiagnosisTest(unittest.TestCase):
    def test_empty_empty_is_perfect_not_zero(self):
        m = match_events([], [], tol=25)
        self.assertEqual((m.tp, m.fp, m.fn), (0, 0, 0))
        self.assertEqual(precision_recall_f1(m.tp, m.fp, m.fn), (1.0, 1.0, 1.0))
        f1, ng, npred = boundary_f1(np.zeros(20), np.zeros(20), 25)
        self.assertEqual((f1, ng, npred), (1.0, 0, 0))

    def test_one_prediction_cannot_match_two_references(self):
        m = match_events([10, 12], [11], tol=2)
        self.assertEqual((m.tp, m.fp, m.fn), (1, 0, 1))
        self.assertEqual(len(m.delays), 1)

    def test_strict_transition_does_not_credit_already_positive(self):
        # GT enters overlap at frame 2. Prediction was already in overlap from
        # frame 0, so the old `seg.any()` implementation falsely called a hit.
        gt = np.array([0, 0, 2, 2, 0, 0])
        pred = np.array([2, 2, 2, 2, 0, 0])
        self.assertEqual(transition_recall(gt, pred, tol=2, into_overlap=True), (0, 1))
        self.assertEqual(transition_recall(gt, pred, tol=0, into_overlap=False), (1, 1))

    def test_signed_matching_rejects_wrong_direction(self):
        gt_i, gt_s = np.array([10]), np.array([1])
        pr_i, pr_s = np.array([10]), np.array([-1])
        m = match_signed_events(gt_i, gt_s, pr_i, pr_s, tol=0)
        self.assertEqual((m.tp, m.fp, m.fn), (0, 1, 1))

    def test_positive_runs_count_only_true_islands(self):
        self.assertEqual(positive_runs(np.zeros(5, bool)), [])
        self.assertEqual(positive_runs(np.ones(5, bool)), [(0, 5)])
        self.assertEqual(
            positive_runs(np.array([0, 1, 1, 0, 1, 0], bool)), [(1, 3), (4, 5)])

    def test_fragmentation_and_binary_overlap_segments(self):
        # One GT overlap run (2->3 does not split it), two predicted positive
        # runs. Generic label-segment counting would give the wrong ratio.
        gt = np.array([0, 2, 2, 3, 3, 0, 0])
        pred = np.array([0, 2, 0, 2, 2, 0, 0])
        a = _empty_system([0])
        update_system(a, gt, pred, [0])
        self.assertEqual(a["gt_overlap_runs"], 1)
        self.assertEqual(a["pred_overlap_runs"], 2)

    def test_aggregate_counts_not_mean_of_window_f1(self):
        a = _empty_system([0])
        # First item: perfect boundary. Second: one missed GT boundary and two
        # false predicted boundaries. Aggregate must be TP=1 FP=2 FN=1.
        update_system(a, np.array([0, 0, 2, 2]), np.array([0, 0, 2, 2]), [0])
        update_system(a, np.array([0, 0, 2, 2]), np.array([0, 1, 1, 0]), [0])
        m = a["boundary"][0]
        self.assertEqual((m.tp, m.fp, m.fn), (1, 2, 1))
        self.assertAlmostEqual(precision_recall_f1(m.tp, m.fp, m.fn)[2], 0.4)

    def test_tolerance_sweep_is_monotonic_and_keeps_delay_sign(self):
        gt = np.array([10, 30])
        pred = np.array([12, 25])
        small = match_events(gt, pred, tol=2)
        large = match_events(gt, pred, tol=5)
        self.assertEqual(small.tp, 1)
        self.assertEqual(large.tp, 2)
        self.assertEqual(large.delays, [2, -5])
        d = delay_summary(large.delays)
        self.assertEqual(d["n"], 2)
        self.assertAlmostEqual(d["median_ms"], -15.0)
        self.assertAlmostEqual(d["late_frac"], 0.5)

    def test_event_extractors(self):
        y = np.array([0, 0, 2, 3, 1, 1])
        idx, sign = change_events(y)
        np.testing.assert_array_equal(idx, [2, 3, 4])
        np.testing.assert_array_equal(sign, [1, 1, -1])
        np.testing.assert_array_equal(overlap_transition_events(y, 1), [2])
        np.testing.assert_array_equal(overlap_transition_events(y, -1), [4])

    def test_primary_tolerance_is_always_in_sweep(self):
        self.assertEqual(parse_tolerances(250, "500,100,500"), [100, 250, 500])

    def test_inverse_array_split_expansion_reaches_exact_tail(self):
        # np.array_split(arange(5), 2) has bucket sizes [3, 2]. A blind x4
        # repeat followed by truncation cannot reproduce this timeline.
        np.testing.assert_array_equal(
            expand_output_to_reference(np.array([1, 2]), 5), [1, 1, 1, 2, 2])

    def test_30_second_join_neither_creates_nor_drops_transition(self):
        def window(wid, start, value):
            y = np.full(3000, value, dtype=np.int64)
            return TimelineWindow("rec", wid, start, y, y.copy(), y.copy())

        # Sorting is intentional: input order must not define the timeline.
        constant = stitch_recording([
            window("rec_w0001", 3000, 1),
            window("rec_w0000", 0, 1),
        ])
        self.assertEqual(len(constant.reference), 6000)
        self.assertTrue(constant.valid.all())
        self.assertEqual(len(change_events(constant.reference)[0]), 0)
        a = _empty_system([0])
        update_system(a, constant.reference, constant.zipcount, [0], constant.valid)
        self.assertEqual((a["boundary"][0].tp, a["boundary"][0].fp,
                          a["boundary"][0].fn), (0, 0, 0))

        # A genuine 1->2 transition exactly at 30.00 s was invisible when
        # each window was scored independently. Recording reconstruction must
        # retain it exactly once, for both boundary and overlap-onset metrics.
        transition = stitch_recording([
            window("rec_w0001", 3000, 2),
            window("rec_w0000", 0, 1),
        ])
        np.testing.assert_array_equal(change_events(transition.reference)[0], [3000])
        a = _empty_system([0])
        update_system(a, transition.reference, transition.zipcount, [0], transition.valid)
        self.assertEqual((a["boundary"][0].tp, a["boundary"][0].fp,
                          a["boundary"][0].fn), (1, 0, 0))
        self.assertEqual((a["onset"][0].tp, a["onset"][0].fp,
                          a["onset"][0].fn), (1, 0, 0))

    def test_masked_timeline_gap_does_not_invent_join_transition(self):
        left = np.ones(2999, dtype=np.int64)
        right = np.full(3000, 2, dtype=np.int64)
        timeline = stitch_recording([
            TimelineWindow("rec", "rec_w0000", 0, left, left, left),
            TimelineWindow("rec", "rec_w0001", 3000, right, right, right),
        ])
        self.assertEqual(timeline.gap_frames, 1)
        a = _empty_system([0])
        update_system(a, timeline.reference, timeline.zipcount, [0], timeline.valid)
        self.assertEqual((a["boundary"][0].tp, a["boundary"][0].fp,
                          a["boundary"][0].fn), (0, 0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
