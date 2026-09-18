"""Unit tests for the phase-8 chain driver's DECISION and LINEAGE logic —
the parts two external reviews found bugs in.  No training happens here; arms
are synthetic result JSONs.
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import phase8_chain_driver as drv  # noqa: E402


# ---------------------------------------------------------------- adoption

def stats(per_seed, f12=0.0, osd=0.0, ci=(0.0, 0.0)):
    return {"per_seed": list(per_seed),
            "mean_pooled_delta": float(np.mean(per_seed)),
            "delta_f1_2": f12, "delta_osd": osd,
            "boot_mean": float(np.mean(per_seed)), "ci": ci,
            "n_improved": 0, "n_recordings": 116}


def test_millipoint_real_is_labelled_but_not_adopted():
    s = stats([0.0001, 0.0002, 0.0003], ci=(0.0001, 0.0004))
    assert drv.evidence_label(s) == "REAL"
    adopted, _ = drv.adoption(s)
    assert not adopted, "millipoint REAL must not pass the adoption bar"


def test_r2_adopts_and_r3_respects_margin():
    assert drv.adoption(stats([0.006, 0.005, 0.004]))[0]
    assert drv.adoption(stats([-0.001, -0.001, -0.001], f12=0.012))[0]
    assert not drv.adoption(stats([-0.003, -0.003, -0.003], f12=0.012))[0], \
        "R3 must respect the -0.002 macro margin"


# ---------------------------------------------------------------- compare

def _write_result(path, macro, recs):
    payload = {
        "pooled": {"macro_f1": macro, "f1_2": 0.5, "osd_f1": 0.5},
        "per_recording": {r: {"macro_f1": macro} for r in recs},
    }
    json.dump(payload, open(path, "w"))


def test_compare_refuses_mismatched_recording_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    recs_a = [f"rec{i}" for i in range(10)]
    recs_b = recs_a[:-1] + ["other"]
    a_paths, b_paths = [], []
    for seed in range(3):
        pa, pb = tmp_path / f"a{seed}.json", tmp_path / f"b{seed}.json"
        _write_result(pa, 0.5, recs_a)
        _write_result(pb, 0.5, recs_b if seed == 2 else recs_a)
        a_paths.append(pa.name)
        b_paths.append(pb.name)
    with pytest.raises(ValueError, match="recording sets differ"):
        drv.compare(a_paths, b_paths)


# ---------------------------------------------------------------- lineage

def _fake_arm(tmp_path, tag, content="{}"):
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / drv.ART).mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / tag).mkdir(parents=True, exist_ok=True)
    result = f"results/{tag}_voxsel.json"
    (tmp_path / result).write_text(content)
    (tmp_path / "logs" / tag / f"step{drv.STEPS}.pt").write_bytes(b"ckpt")
    (tmp_path / drv.ART / f"{tag}.yaml").write_text("a: 1\n")
    return result



def _armset(tmp_path, prefix):
    return [_fake_arm(tmp_path, f"{prefix}_s{seed}") for seed in drv.SEEDS]


def test_replay_refuses_mismatched_parent(tmp_path, monkeypatch):
    cfg = drv.base_config()
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    paths = _armset(tmp_path, "p8p_trt")
    prior = {"phase": "d1", "adopt": False,
             "parent_state": {"config_sha": "not-the-real-sha",
                              "results": [{"result": p} for p in paths]},
             "output_state": drv.state_fingerprint(cfg, paths),
             "carried_results": paths}
    with pytest.raises(SystemExit, match="config sha changed"):
        drv.replay("d1", prior, copy.deepcopy(cfg), paths)


def test_replay_adopted_full_roundtrip_and_output_check(tmp_path, monkeypatch):
    cfg = drv.base_config()
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    parent_paths = _armset(tmp_path, "p8q_ctl")
    child_paths = _armset(tmp_path, "p8q_trt")
    mutated = copy.deepcopy(cfg)
    drv.mut_d1aux(mutated)
    prior = {"phase": "d1aux", "adopt": True,
             "parent_state": drv.state_fingerprint(cfg, parent_paths),
             "output_state": drv.state_fingerprint(mutated, child_paths),
             "carried_results": child_paths}
    out_cfg, out_res = drv.replay(
        "d1aux", prior, copy.deepcopy(cfg), parent_paths)
    assert drv.sha_of_config(out_cfg) == drv.sha_of_config(mutated)
    assert out_res == child_paths
    # corrupting the recorded output must be caught
    prior["output_state"]["config_sha"] = "corrupted"
    with pytest.raises(SystemExit, match="config sha changed"):
        drv.replay("d1aux", prior, copy.deepcopy(cfg), parent_paths)


def test_replay_not_adopted_keeps_carried_state(tmp_path, monkeypatch):
    cfg = drv.base_config()
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    paths = _armset(tmp_path, "p8r_ctl")
    state = drv.state_fingerprint(cfg, paths)
    prior = {"phase": "d1", "adopt": False,
             "parent_state": state, "output_state": state,
             "carried_results": paths}
    out_cfg, out_res = drv.replay("d1", prior, copy.deepcopy(cfg), paths)
    assert drv.sha_of_config(out_cfg) == drv.sha_of_config(cfg)
    assert out_res == paths


# ---------------------------------------------------------------- phase C/E

def test_phase_c_supervises_boundary_only():
    cfg = drv.base_config()
    drv.mut_b_sord(cfg)
    drv.mut_c(cfg)
    assert cfg["loss"]["lambda_boundary"] == 0.10
    assert cfg["loss"]["lambda_direction"] == 0.0, \
        "direction supervision belongs to phase c2, not c"
    drv.mut_c2(cfg)
    assert cfg["loss"]["lambda_direction"] == 0.05


# --------------------------------------------------------- rev-8 aux bridge

def test_order_and_generic_muts_are_the_rev8_legacy_bridge():
    assert drv.ORDER == [
        "a0", "a1", "a2", "b", "caux", "c2aux", "d1aux", "d2aux",
    ]
    assert set(drv.GENERIC_MUTS) == {"a1", "a2", "caux", "c2aux",
                                     "d1aux", "d2aux"}
    for retired in ("b2", "c", "c2", "d1", "d2", "e"):
        assert retired not in drv.ORDER
        assert retired not in drv.GENERIC_MUTS


def test_aux_mutations_never_touch_the_legacy_loss_type_or_keys():
    cfg = drv.base_config()
    assert cfg["loss"]["type"] == "legacy"
    before = copy.deepcopy(cfg["loss"])

    mutated = copy.deepcopy(cfg)
    drv.mut_caux(mutated)
    drv.mut_c2aux(mutated)
    drv.mut_d1aux(mutated)
    drv.mut_d2aux(mutated)

    assert mutated["loss"]["type"] == "legacy"
    assert mutated["loss"]["aux_lambda_boundary"] == 0.10
    assert mutated["loss"]["aux_lambda_direction"] == 0.05
    assert mutated["loss"]["aux_lambda_t_mse"] == 0.05
    assert mutated["loss"]["aux_lambda_segment"] == 0.05
    # every pre-existing legacy key/value is untouched (additive only)
    for key, value in before.items():
        assert mutated["loss"][key] == value, key


def test_aux_mutations_compound_incrementally_like_the_retired_ladder():
    cfg = drv.base_config()
    drv.mut_caux(cfg)
    assert "aux_lambda_direction" not in cfg["loss"]
    drv.mut_c2aux(cfg)
    assert cfg["loss"]["aux_lambda_boundary"] == 0.10
    assert cfg["loss"]["aux_lambda_direction"] == 0.05


def test_matched_cap_width_tracks_lineage():
    cfg = drv.base_config()
    width_plain, err_plain = drv.matched_cap_width(cfg)
    assert err_plain <= 0.01
    with_final = copy.deepcopy(cfg)
    drv.mut_a2(with_final)          # adds the final residual (the 2.4% case)
    width_final, err_final = drv.matched_cap_width(with_final)
    assert err_final <= 0.01
    # the third review measured d263/d260 for the two lineages: a hardcoded
    # width CANNOT serve both, so the search must return different widths
    assert width_plain != width_final, (width_plain, width_final)


def test_phase_a_splits_framewise_gate_from_final_residual():
    """Each adoption contrast changes one factor, not gate+feature source."""
    gate_only = drv.base_config()
    drv.mut_a1(gate_only)
    assert gate_only["model"]["head"]["gate_mode"] == "framewise"
    assert not gate_only["model"]["encoder"]["multiscale_include_final"]

    final_only = drv.base_config()
    drv.mut_a2(final_only)
    assert final_only["model"]["head"]["gate_mode"] == "global"
    assert final_only["model"]["encoder"]["multiscale_include_final"]


# ------------------------------------------------- bridge / ladder / promote

def test_resolve_bridge_is_symmetric():
    good = stats([0.001, 0.001, 0.001])
    bad = stats([-0.010, -0.010, -0.010])
    # candidate passes -> chosen, no fallback
    assert drv.resolve_bridge("cum", good, "sord", bad) == ("cum", False)
    # candidate fails, alternate passes -> ALWAYS falls back (rev-3 bug: it
    # skipped this branch whenever cumulative had earned the head-to-head)
    assert drv.resolve_bridge("cum", bad, "sord", good) == ("sord", True)
    # both fail -> stop
    assert drv.resolve_bridge("cum", bad, "sord", bad) == (None, False)


def test_ladder_cannot_jump_a_failed_rung():
    """4th review reproduction: cap fails vs carried at +0.0049 while
    stage2-vs-carried would pass at +0.0050 — but its true refinement over cap
    is +0.0001.  The strict ladder must STOP at cap and never consult any
    carried-level shortcut for stage2."""
    calls = []
    def cmp(control, rung):
        calls.append((control, rung))
        table = {("carried", "cap"): stats([0.0049, 0.0049, 0.0049])}
        return table[(control, rung)]     # KeyError == illegal contrast
    holder, trail = drv.climb_ladder(
        "carried", ["cap", "stage2_nods", "stage2_ds"], cmp)
    assert holder == "carried"
    assert calls == [("carried", "cap")], \
        "ladder consulted a contrast beyond the failed rung"
    assert len(trail) == 1 and not trail[0]["adopted"]


def test_ladder_rejects_millipoint_refinement():
    table = {
        ("carried", "cap"): stats([0.0051, 0.0051, 0.0051]),
        ("cap", "stage2_nods"): stats([0.0001, 0.0001, 0.0001],
                                      ci=(0.00005, 0.0002)),
    }
    holder, trail = drv.climb_ladder(
        "carried", ["cap", "stage2_nods", "stage2_ds"],
        lambda holder, rung: table[(holder, rung)])
    assert holder == "cap",         "a +0.0001 refinement over cap must NOT hand the chain to stage2"


def test_ladder_ties_keep_the_simpler_system():
    zero = stats([0.0, 0.0, 0.0])
    holder, _ = drv.climb_ladder(
        "carried", ["cap", "stage2_nods", "stage2_ds"],
        lambda holder, rung: zero)
    assert holder == "carried"


def test_ladder_climbs_when_bar_is_met():
    table = {
        ("carried", "cap"): stats([0.006, 0.006, 0.006]),
        ("cap", "stage2_nods"): stats([0.006, 0.006, 0.006]),
        ("stage2_nods", "stage2_ds"): stats([0.001, 0.001, 0.001]),
    }
    holder, _ = drv.climb_ladder(
        "carried", ["cap", "stage2_nods", "stage2_ds"],
        lambda holder, rung: table[(holder, rung)])
    assert holder == "stage2_nods",         "DS must not be added unless it beats noDS under the practical bar"


def test_promote_keeps_root_below_gate():
    assert drv.promote(stats([0.005, 0.005, 0.005])) == ("root", False)
    assert drv.promote(stats([0.011, 0.011, 0.011])) == ("candidate", True)
    assert drv.promote(None) == ("root", True)


# ------------------------------------------------- state fingerprints (B1)

def test_verify_state_catches_regenerated_results(tmp_path, monkeypatch):
    cfg = drv.base_config()
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    paths = [_fake_arm(tmp_path, f"p8x_trt_s{seed}") for seed in drv.SEEDS]
    expected = drv.state_fingerprint(cfg, paths)
    drv.verify_state(expected, cfg, paths, where="test")   # clean -> passes
    # regenerate ONE result file with different content, same config
    (tmp_path / paths[1]).write_text('{"regenerated": true}')
    with pytest.raises(SystemExit, match="result_sha drifted"):
        drv.verify_state(expected, cfg, paths, where="test")


def test_verify_state_catches_regenerated_checkpoint(tmp_path, monkeypatch):
    cfg = drv.base_config()
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    paths = [_fake_arm(tmp_path, f"p8y_trt_s{seed}") for seed in drv.SEEDS]
    expected = drv.state_fingerprint(cfg, paths)
    tag = drv.tag_of_result(paths[0])
    (tmp_path / "logs" / tag / f"step{drv.STEPS}.pt").write_bytes(b"NEW")
    with pytest.raises(SystemExit, match="checkpoint_sha drifted"):
        drv.verify_state(expected, cfg, paths, where="test")


def test_verify_state_rejects_short_sha_vector(tmp_path, monkeypatch):
    cfg = drv.base_config()
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    paths = [_fake_arm(tmp_path, f"p8z_trt_s{seed}") for seed in drv.SEEDS]
    expected = drv.state_fingerprint(cfg, paths)
    expected["results"] = expected["results"][:2]      # the zip-truncation bug
    with pytest.raises(SystemExit, match="expected 3 result entries"):
        drv.verify_state(expected, cfg, paths, where="test")


# ---------------------------------------------------------------- boundary AP

def test_tie_aware_ap_is_permutation_invariant():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eval_boundary_ap import tie_aware_ap, recall_at_fp_budget
    scores = np.array([1.0, 1.0, 1.0, 0.0])
    targets = np.array([1, 0, 0, 1])
    base = tie_aware_ap(scores, targets)
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = rng.permutation(4)
        assert tie_aware_ap(scores[perm], targets[perm]) == pytest.approx(base)
    # perfect ranking -> AP 1; budget recall behaves atomically on tie groups
    assert tie_aware_ap(np.array([3.0, 2.0, 1.0]), np.array([1, 1, 0])) == 1.0
    assert recall_at_fp_budget(np.array([3.0, 2.0, 1.0]),
                               np.array([1, 0, 1]), budget=0) == 0.5
    assert recall_at_fp_budget(np.array([3.0, 2.0, 1.0]),
                               np.array([1, 0, 1]), budget=1) == 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ------------------------------------------------- root seal (BL2, rev-5)

def _fake_root(tmp_path):
    for rel in drv.ROOT_RESULTS + drv.ROOT_CHECKPOINTS + drv.ROOT_CONFIGS:
        full = tmp_path / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(f"content-of-{rel}")


def test_seal_root_then_verify_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    (tmp_path / drv.ART).mkdir(parents=True, exist_ok=True)
    manifest = str(tmp_path / drv.ART / "root_manifest.json")
    _fake_root(tmp_path)
    drv.seal_root_command(manifest)
    drv.verify_root(manifest)                       # clean round-trip
    with pytest.raises(SystemExit, match="refusing to reseal"):
        drv.seal_root_command(manifest)             # exclusive
    # tampering any sealed artifact must be caught
    (tmp_path / drv.ROOT_CHECKPOINTS[1]).write_text("TAMPERED")
    with pytest.raises(SystemExit, match="ROOT VIOLATION"):
        drv.verify_root(manifest)


def test_verify_root_rejects_truncated_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    (tmp_path / drv.ART).mkdir(parents=True, exist_ok=True)
    manifest = str(tmp_path / drv.ART / "root_manifest.json")
    _fake_root(tmp_path)
    json.dump({}, open(manifest, "w"))              # the fail-open {} case
    with pytest.raises(SystemExit, match="ROOT VIOLATION"):
        drv.verify_root(manifest)


def test_verify_root_requires_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    _fake_root(tmp_path)
    with pytest.raises(SystemExit, match="root manifest missing"):
        drv.verify_root(str(tmp_path / drv.ART / "root_manifest.json"))


def test_seal_refuses_missing_or_empty_root(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    (tmp_path / drv.ART).mkdir(parents=True, exist_ok=True)
    manifest = str(tmp_path / drv.ART / "root_manifest.json")
    _fake_root(tmp_path)
    (tmp_path / drv.ROOT_CHECKPOINTS[0]).write_text("")   # empty file
    with pytest.raises(SystemExit, match="missing or empty"):
        drv.seal_root_command(manifest)


# ------------------------------------------------- seed identity / fail-closed

def test_seed_identity_rejects_duplicates_and_order():
    good = [f"results/p8x_trt_s{s}_voxsel.json" for s in drv.SEEDS]
    drv.assert_seed_identity(good, "test")
    dup = [good[0], good[0], good[2]]
    with pytest.raises(SystemExit, match="seed identity"):
        drv.assert_seed_identity(dup, "test")
    swapped = [good[1], good[0], good[2]]
    with pytest.raises(SystemExit, match="seed identity"):
        drv.assert_seed_identity(swapped, "test")


def test_fingerprint_fails_closed_on_missing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    paths = _armset(tmp_path, "p8w_trt")
    tag = drv.tag_of_result(paths[0])
    os.remove(tmp_path / "logs" / tag / f"step{drv.STEPS}.pt")
    with pytest.raises(SystemExit, match="checkpoint missing"):
        drv.result_fingerprint(paths[0])


# ------------------------------------------------- rev-6: driver<->evaluator
# contract, sidecar outputs, backbone-complete knob

def test_boundary_evaluator_accepts_every_driver_flag(tmp_path):
    """6th review: 95 unit tests missed that the driver passed --protocol-sha
    to an evaluator that did not declare it (guaranteed phase-C crash).  This
    invokes the REAL script with the driver's exact flag list; reaching the
    config-loading stage proves argparse accepted the contract."""
    import subprocess
    repo = os.path.join(os.path.dirname(__file__), "..")
    flags = ["--manifest", str(tmp_path / "nope.json"),
             "--checkpoint", str(tmp_path / "nope.pt"),
             "--config", str(tmp_path / "nope.yaml"),
             "--checkpoint-sha", "x", "--config-sha", "y",
             "--protocol-sha", "z",
             "--out", str(tmp_path / "out.json")]
    proc = subprocess.run(
        [sys.executable, "scripts/eval_boundary_ap.py", *flags],
        cwd=repo, env={**os.environ, "PYTHONPATH": "."},
        capture_output=True, text=True)
    assert "unrecognized arguments" not in proc.stderr, proc.stderr
    assert proc.returncode != 0            # dies later, at the missing config
    assert "nope.yaml" in proc.stderr      # ...which is exactly what we want


def test_driver_protocol_flags_match_evaluator(tmp_path):
    """The driver's subprocess argv for the boundary evaluator must be a
    subset of the evaluator's declared options."""
    import re
    repo = os.path.join(os.path.dirname(__file__), "..")
    driver_src = open(os.path.join(repo, "scripts/phase8_chain_driver.py")).read()
    eval_src = open(os.path.join(repo, "scripts/eval_boundary_ap.py")).read()
    declared = set(re.findall(r'add_argument\("(--[a-z-]+)"', eval_src))
    block = driver_src[driver_src.index("eval_boundary_ap.py"):]
    used = set(re.findall(r'"(--[a-z-]+)"', block[:1200]))
    assert used <= declared, f"driver passes undeclared flags: {used - declared}"


def test_sidecar_outputs_gate_reuse(tmp_path, monkeypatch):
    """5th review B2: editing result.json (or the checkpoint) under an
    unchanged YAML/inputs must NOT be reusable."""
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    tag = "p8m_trt_s1234"
    result_rel = _fake_arm(tmp_path, tag, content='{"pooled": 1}')
    outputs = drv.arm_outputs(tag, result_rel)
    assert set(outputs) == {"result_sha", "checkpoint_sha", "arm_config_sha"}
    before = dict(outputs)
    (tmp_path / result_rel).write_text('{"pooled": 2}')     # tampered result
    after = drv.arm_outputs(tag, result_rel)
    assert after["result_sha"] != before["result_sha"]
    assert after["checkpoint_sha"] == before["checkpoint_sha"]


def test_require_backbone_complete_knob_is_enforced():
    """B3: the config key the chain emits must actually be read by train.py."""
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "src/train.py")).read()
    assert 'require_backbone_complete' in src
    assert 'backbone_missing' in src


def _stub_fresh_arm(tmp_path, monkeypatch, *, tamper_yaml=False):
    """Run run_arm's real fresh-artifact path with train/eval subprocesses
    replaced by tiny file-producing stubs."""
    monkeypatch.setattr(drv, "REPO", str(tmp_path))
    monkeypatch.setattr(drv, "check_disk", lambda: None)
    (tmp_path / drv.ART).mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "results").mkdir()

    def fake_inputs(config):
        return {
            "config_sha": drv.sha_of_config(config),
            "init_sha": "init",
            "train_manifest_sha": "train",
            "eval_manifest_sha": "eval",
            "source_sha": "source",
        }
    monkeypatch.setattr(drv, "arm_inputs", fake_inputs)

    tag = "p8x_fresh_s1234"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if "src/train.py" in argv:
            log_dir = tmp_path / "logs" / tag
            log_dir.mkdir()
            (log_dir / f"step{drv.STEPS}.pt").write_bytes(b"checkpoint")
        else:
            out = argv[argv.index("--out") + 1]
            (tmp_path / out).write_text('{"completed": true}')
            if tamper_yaml:
                cfg_path = Path(argv[argv.index("--config") + 1])
                cfg_path.write_text("tampered: true\n")
        return None

    monkeypatch.setattr(drv.subprocess, "run", fake_run)
    config = {"training": {}, "data": {}}
    return tag, calls, lambda: drv.run_arm("x", "fresh", config, 1234)


def test_run_arm_fresh_path_writes_bound_completion_sidecar(
        tmp_path, monkeypatch):
    """A behavioral fresh-run test catches stale variable names and verifies
    that the sidecar certifies matching input/output config identities."""
    tag, calls, invoke = _stub_fresh_arm(tmp_path, monkeypatch)
    result_rel = invoke()

    assert result_rel == f"results/{tag}_voxsel.json"
    assert len(calls) == 2
    sidecar = json.loads(
        (tmp_path / drv.ART / f"{tag}.lineage.json").read_text())
    assert sidecar["inputs"]["config_sha"] \
        == sidecar["outputs"]["arm_config_sha"]
    assert sidecar["outputs"] == drv.arm_outputs(tag, result_rel)


def test_run_arm_refuses_on_disk_yaml_toctou(tmp_path, monkeypatch):
    """Changing the YAML during train/eval must leave no completion sidecar;
    re-hashing the unchanged in-memory dict alone would miss this."""
    tag, _, invoke = _stub_fresh_arm(
        tmp_path, monkeypatch, tamper_yaml=True)

    with pytest.raises(SystemExit, match="config YAML changed WHILE"):
        invoke()
    assert not (tmp_path / drv.ART / f"{tag}.lineage.json").exists()
