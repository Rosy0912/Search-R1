"""Independent read-only join of actual local copies; no remote/runtime acceptance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "final-acceptance-candidate"
SNAPSHOT = HERE / "actual-snapshots/3261-20261005T210622178545Z/original-files"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_rejection(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Changed evidence was accepted")


shared = load("independent_current_observer", CANDIDATE / "author200_observation.py")
native = load("independent_current_native", CANDIDATE / "inspect_author200_checkpoint_cpu.py")
hf = load("independent_current_hf", CANDIDATE / "hf_author200_cpu.py")
gate_path = ROOT / "opsd-e1-300/search_r1/baselines/checkpoint_gate.py"
assert hashlib.sha256(gate_path.read_bytes()).hexdigest() == shared.GATE_SHA
gate = load("independent_pinned_gate", gate_path)
workers = []
for rank in range(8):
    p = SNAPSHOT / f"checkpoints/actor/global_step_5/training_state/rank_{rank}.json"
    r = json.loads(p.read_bytes())
    assert r["rank"] == rank and r["step"] == 5 and r["status"] == "EXACT_FULL_STATE_RELOAD_OK"
    workers.append(dict(rank=rank, pid=r["pid"], object_ids=r["object_ids"]))
context = dict(gate=gate, workers=workers,
    checkpoint_warning_path=Path("/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py"))
prefixes = {}
parsed = {}
for role, name in (("driver", "ray-stdout-pid-1533312.out"), ("aggregate", "train.log")):
    path = SNAPSHOT / name
    raw = path.read_bytes()
    last = raw.rfind(b"\n") + 1
    assert last > 0
    raw = raw[:last]
    prefixes[role] = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    parsed[role] = shared.parse_records(raw, context)
joined = shared.join_log_prefixes(parsed["driver"], parsed["aggregate"], context, 5)
assert joined["authoritative_last_outer"] == 8 and joined["aggregate_last_outer"] == 7
assert joined["aggregate_forwarding_lag"] == 1
evaluation = shared.check_evaluation_sequence(parsed["driver"], 8)
assert [(v["kind"], v["step"], v["count"]) for v in evaluation["events"]] == [("dev", 0, 512), ("stage", 5, 64)]

# Actual transport corner, with strict counterexamples using the same pinned gate.
original = (SNAPSHOT / "train.log").read_bytes().splitlines(keepends=True)[444]
assert b"FULL_STATE_RANK_JSON=" in original and b"[repeated 7x across cluster]" in original
one = shared.parse_records(original, context)
assert len(one["native_console"]) == 1
assert one["native_transport"][0]["observed_repeat_tag"] == 7
assert one["native_transport"][0]["raw"].encode() == original.rstrip(b"\n")
wrong_lines = [
    original.replace(b"[repeated 7x across cluster]", b"[repeated 8x across cluster]"),
    original.replace(b"[repeated 7x across cluster]", b"[repeated 0x across cluster]"),
    original.replace(b"torch/utils/checkpoint.py:1399", b"torch/foreign/checkpoint.py:1399"),
    original.replace(b"(WorkerDict pid=1533910)", b"(WorkerDict pid=9999999)"),
    original.rstrip(b"\n") + b" arbitrary suffix\n",
    original.replace(b"[repeated 7x across cluster]", b"[repeated 7x across cluster]x"),
]
for raw in wrong_lines:
    expect_rejection(lambda raw=raw: shared.parse_records(raw, context))

# Direct cross-interface shape of the observation inventory, without pretending
# fixture future reports are actual saved-state evidence.
inventory = dict(version=1, kind=shared.KIND, source_commit=shared.SOURCE, job_id="3261",
    run_root=str(shared.RUN), checkpoint_step=25, config_sha256=shared.CONFIG_SHA,
    profile="author200", horizon=200, world_size=8,
    native_import_closure_sha256=shared.CLOSURE_SHA, formal_manifest_sha256=shared.FORMAL_SHA,
    accepted5=dict(manifest_sha256=shared.MANIFEST5_SHA, signed_receipt_sha256=shared.RECEIPT5_SHA,
        public_key_sha256=shared.PUBLIC_SHA, driver_pid=shared.DRIVER_PID,
        driver_start_ticks=shared.DRIVER_START, workers=workers),
    rank_reports=[dict(rank=rank) for rank in range(8)],
    driver_report=dict(status="DRIVER_STATE_SERIALIZED_OK"),
    raw_statuses=dict(ranks=["FULL_STATE_SERIALIZED_OK"] * 8, driver="DRIVER_STATE_SERIALIZED_OK"))
clock = native.validate_inventory(inventory, 25)
assert clock["adam_steps"] == 400 and clock["cursor"] == 12800
for change in (lambda x: x.update(version=True), lambda x: x["rank_reports"][0].update(rank=False),
    lambda x: x["accepted5"].update(driver_start_ticks=str(shared.DRIVER_START)),
    lambda x: x["raw_statuses"]["ranks"].__setitem__(0, "EXACT_FULL_STATE_RELOAD_OK")):
    altered = copy.deepcopy(inventory)
    change(altered)
    expect_rejection(lambda altered=altered: native.validate_inventory(altered, 25))

report = dict(version=3, rank=0, world_size=8, step=25, pid=workers[0]["pid"],
    scheduler_last_epoch=25, optimizer_state_entries=37, model_dtensor_entries=435,
    status="FULL_STATE_SERIALIZED_OK", algorithm="pure_opsd", model_state_dict_type="SHARDED_STATE_DICT",
    object_ids=workers[0]["object_ids"])
native.validate_rank_report(report, 0, 25, workers[0]["pid"])
for key in ("rank", "version", "world_size", "step", "pid", "scheduler_last_epoch", "optimizer_state_entries", "model_dtensor_entries"):
    changed = copy.deepcopy(report)
    changed[key] = float(changed[key])
    expect_rejection(lambda changed=changed: native.validate_rank_report(changed, 0, 25, workers[0]["pid"]))

result = dict(version=1, scope="INDEPENDENT_LOCAL_ORIGINAL8_LOG_REPLAY_AND_SOFTWARE_INTERFACE_NOT25_ACCEPTANCE",
    original_snapshot=str(SNAPSHOT), prefix_files=prefixes, joined=joined,
    evaluation_events=[dict(kind=v["kind"], step=v["step"], count=v["count"]) for v in evaluation["events"]],
    actual_transport_line=445, actual_transport_single_record=True, inferred_hidden_rank_count=0,
    exact_repeat_tag_and_full_raw_preserved=True, transport_negative_cases=len(wrong_lines),
    fixture_cross_interface_validated=True, fixture_checkpoint_step=25,
    strict_inventory_negative_cases=4, strict_rank_report_negative_cases=8,
    source_pins={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        CANDIDATE / "author200_observation.py", CANDIDATE / "inspect_author200_checkpoint_cpu.py",
        CANDIDATE / "hf_author200_cpu.py", gate_path]},
    limitations=["No actual25 state/PT or HF model inspected", "No remote Slurm/GPU invocation",
        "Only actual copied8 logs are replayed; future25 inventory is a declared schema fixture",
        "No state reload, scientific effect, terminal or resource acceptance"])
snapshot25 = HERE / "actual-snapshots/3261-20261005T214741940075Z/original-files"
parsed25 = {}
prefixes25 = {}
for role, name in (("driver", "ray-stdout-pid-1533312.out"), ("aggregate", "train.log")):
    path = snapshot25 / name
    raw = path.read_bytes()
    raw = raw[:raw.rfind(b"\n") + 1]
    prefixes25[role] = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    parsed25[role] = shared.parse_records(raw, context)
join25 = shared.join_log_prefixes(parsed25["driver"], parsed25["aggregate"], context, 25)
assert join25["authoritative_last_outer"] == 25 and join25["aggregate_last_outer"] == 24
assert join25["aggregate_forwarding_lag"] == 1
eval25 = shared.check_evaluation_sequence(parsed25["driver"], 25)
reports25 = []
for rank in range(8):
    path = snapshot25 / f"checkpoints/actor/global_step_25/training_state/rank_{rank}.json"
    report = json.loads(path.read_bytes())
    native.validate_rank_report(report, rank, 25, workers[rank]["pid"])
    assert report["object_ids"] == workers[rank]["object_ids"]
    assert report["teacher"]["counters"]["cuda_update_entries"] == 25
    reports25.append(report)
visible25 = [r for r in parsed25["aggregate"]["native_console"] if r["step"] == 25]
assert [r["rank"] for r in visible25] == [1, 0]
assert all(shared.canonical(r) == shared.canonical(reports25[r["rank"]]) for r in visible25)
driver25 = json.loads((snapshot25 / "full_state/global_step_25/driver.json").read_bytes())
assert driver25["pid"] == shared.DRIVER_PID and driver25["step"] == 25 and driver25["job_id"] == "3261"
assert driver25["status"] == "DRIVER_STATE_SERIALIZED_OK" and type(driver25["pid"]) is int
result["actual25_log_and_original_json_only"] = dict(
    scope="ACTUAL25_COPIED_LOG_AND_SERIALIZED_JSON_CONTINUITY_NOT_NATIVE_PT_ACCEPTANCE",
    snapshot=str(snapshot25), log_prefixes=prefixes25, join=join25,
    original_rank_json_count=8, raw_saved_rank_statuses=[r["status"] for r in reports25],
    raw_saved_driver_status=driver25["status"], visible_console_ranks=[r["rank"] for r in visible25],
    hidden_ranks_inferred=False, original8_rank_json_metadata_valid=True,
    native_pt_or_hf_model_loaded=False,
    evaluation_events=[dict(kind=v["kind"], step=v["step"], count=v["count"]) for v in eval25["events"]])
print(json.dumps(result, indent=2, sort_keys=True))
