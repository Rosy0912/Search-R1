"""Independent pipeline checks in isolated temporary directories, no Slurm/GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def context(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("independent_pipeline", Path(__file__).parent / "checkpoint_cpu_pipeline_candidate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    run = tmp_path / "run"
    out = run / "evidence" / "checkpoint-25-cpu-inspection-fixture"
    out.mkdir(parents=True)
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "inspect_author200_checkpoint_cpu.py").write_text("# never executed\n")
    monkeypatch.setattr(m, "RUN", run)
    monkeypatch.setattr(m, "REPO", tmp_path / "repo")
    monkeypatch.setattr(m, "HERE", tools)
    def write(path, value):
        with path.open("xb") as handle:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    launch = SimpleNamespace(write=write, now=lambda: "fixture-utc", process=lambda pid, **kw: {"pid": pid, "fixture": True},
        physical=lambda path: Path(path), unique=lambda raw: json.loads(raw), load_request=lambda *a: None)
    cfg = {"checkpoint_step": 25, "log_prefixes": {
        "training": {"path": str(run / "train.log"), "bytes": 17, "sha256": "a" * 64},
        "driver": {"path": str(tmp_path / "driver-1533312.out"), "bytes": 91, "sha256": "b" * 64}}}
    exec_ctx = {"job_id": "3261", "step_id": "3", "process": {"pid": os.getpid(), "fixture": True}}
    monkeypatch.setattr(m, "load_launcher", lambda *a: (launch, cfg, out))
    monkeypatch.setattr(m, "execution_context", lambda *a: exec_ctx)
    return SimpleNamespace(m=m, out=out, cfg=cfg, launch=launch, exec_ctx=exec_ctx, tmp=tmp_path)


def local_command(code=0, sig=None, done=None):
    raw = "import os,sys,time; print('real local stdout',flush=True); print('real local stderr',file=sys.stderr,flush=True); time.sleep(.03); "
    if done:
        raw += f"open({str(done)!r},'w').write('done'); "
    raw += f"os.kill(os.getpid(),{sig})" if sig else f"sys.exit({code})"
    return [sys.executable, "-c", raw]


@pytest.mark.parametrize("code,sig", [(0, None), (11, None), (0, signal.SIGTERM)])
def test_run_one_real_child_waits_original_exit_and_streams(context, code, sig):
    c = context
    if code or sig:
        with pytest.raises(ValueError, match="failed"):
            c.m.run_one(c.launch, c.out, "native", local_command(code, sig))
    else:
        c.m.run_one(c.launch, c.out, "native", local_command(code, sig))
    record = json.loads((c.out / "native-exit-original.json").read_bytes())
    assert record["returncode"] == (-int(sig) if sig else code)
    assert type(record["returncode"]) is int
    assert record["waited_by_pipeline_pid"] == os.getpid()
    assert (c.out / "native-stdout-original.txt").read_text() == "real local stdout\n"
    assert (c.out / "native-stderr-original.txt").read_text() == "real local stderr\n"


def test_run_one_failed_metadata_does_not_abandon_actual_child(context):
    c = context
    orig = c.launch.write
    def fail_metadata(path, value):
        if path.name == "native-process-original.json":
            raise OSError("fixture metadata write failed")
        orig(path, value)
    c.launch.write = fail_metadata
    done = c.tmp / "child_done"
    with pytest.raises(OSError, match="metadata"):
        c.m.run_one(c.launch, c.out, "native", local_command(done=done))
    assert done.read_text() == "done"
    assert not (c.out / "native-exit-original.json").exists()


def test_all_arguments_are_literal_and_context_separate(context):
    c = context
    observed = c.m.observer_command(c.cfg, c.out)
    for role, stem in (("training", "--training-log"), ("driver", "--driver-log")):
        p = c.cfg["log_prefixes"][role]
        assert observed[observed.index(stem):observed.index(stem) + 6] == [stem, p["path"], stem + "-sha256", p["sha256"], stem + "-prefix-bytes", str(p["bytes"])]
    inventory = c.out / "observation" / "observed-checkpoint-inventory.json"
    native = c.m.native_command(c.cfg, c.out, inventory, "c" * 64, c.exec_ctx)
    assert native[native.index("--step") + 1] == "25"
    assert native[native.index("--execution-job-id") + 1] == "3261"
    assert native[native.index("--execution-step-id") + 1] == "3"
    assert "--gres" not in native and not any(x.startswith("--jobid") for x in native)
    hf = c.m.hf_command(c.out, inventory, "c" * 64, "d" * 64, c.exec_ctx)
    assert hf[hf.index("--native-inspector-sha256") + 1] == c.m.sha(c.m.HERE / "inspect_author200_checkpoint_cpu.py")
    assert hf[hf.index("--execution-step-id") + 1] == "3"


def fake_run_one(c, calls, fail=None, bad=None):
    def execute(launcher, output, name, command):
        calls.append(name)
        if name == fail:
            raise ValueError("injected actual nonzero " + name)
        if name == "observer":
            (output / "observation").mkdir()
            value = dict(kind="OBSERVED_AUTHOR200_CHECKPOINT_INVENTORY_NOT_TRAINING_MANIFEST",
                source_commit=c.m.SOURCE, job_id="3261", checkpoint_step=c.cfg["checkpoint_step"])
            path = output / "observation/observed-checkpoint-inventory.json"
        elif name == "native":
            value = dict(scope="NATIVE_CPU_STORED_CHECKPOINT_CONTENT_INSPECTION", passed=True,
                source_commit=c.m.SOURCE, job_id="3261", checkpoint_step=c.cfg["checkpoint_step"], cuda_is_initialized=False)
            path = output / "native.json"
        else:
            value = dict(scope="HF200_WHOLE_CPU_MODEL_LOAD_NOT_TRAINING_RESTORE", passed=True,
                source_commit=c.m.SOURCE, job_id="3261", checkpoint_step=200, cuda_is_initialized=False)
            path = output / "hf.json"
        if bad and name == bad[0]:
            value[bad[1]] = bad[2]
        launcher.write(path, value)
    return execute


@pytest.mark.parametrize("step", [25, 50, 75, 100, 125, 150, 175, 200])
def test_pipeline_success_scope_and_hf_only200(context, monkeypatch, step):
    c = context
    c.cfg["checkpoint_step"] = step
    calls = []
    monkeypatch.setattr(c.m, "run_one", fake_run_one(c, calls))
    c.m.pipeline(c.tmp / "request.json", "e" * 64)
    assert calls == ["observer", "native"] + (["hf"] if step == 200 else [])
    completed = json.loads((c.out / "inspection-completed-review-required.json").read_bytes())
    assert completed["independent_acceptance_pending"] is True
    assert completed["scope"] == "AUTHOR200_CPU_STORED_CONTENT_INSPECTION_NOT_TRAINING_CONTROL"
    assert completed["checkpoint_step"] == step
    assert completed["training_job_id"] == "3261"
    assert completed["execution_context"] == c.exec_ctx
    assert (completed["whole_hf200_report_sha256"] is not None) == (step == 200)
    assert not any("RELOAD" in str(v) or "RESTORED" in str(v) for v in completed.values())


@pytest.mark.parametrize("fail", ["observer", "native", "hf"])
def test_pipeline_actual_failure_stops_and_never_fabricates_completion(context, monkeypatch, fail):
    c = context
    c.cfg["checkpoint_step"] = 200
    calls = []
    monkeypatch.setattr(c.m, "run_one", fake_run_one(c, calls, fail=fail))
    with pytest.raises(ValueError, match="actual nonzero"):
        c.m.pipeline(c.tmp / "request.json", "e" * 64)
    assert calls == ["observer", "native", "hf"][:["observer", "native", "hf"].index(fail) + 1]
    assert not (c.out / "inspection-completed-review-required.json").exists()


@pytest.mark.parametrize("bad", [("observer", "checkpoint_step", True), ("native", "passed", 1),
    ("native", "checkpoint_step", True), ("native", "cuda_is_initialized", 0),
    ("native", "source_commit", "wrong"), ("hf", "cuda_is_initialized", True),
    ("hf", "scope", "LIVE_RELOAD"), ("hf", "checkpoint_step", 200.0)])
def test_pipeline_identity_type_drift_rejected_without_completion(context, monkeypatch, bad):
    c = context
    c.cfg["checkpoint_step"] = 200
    calls = []
    monkeypatch.setattr(c.m, "run_one", fake_run_one(c, calls, bad=bad))
    with pytest.raises(ValueError, match="differs"):
        c.m.pipeline(c.tmp / "request.json", "e" * 64)
    assert not (c.out / "inspection-completed-review-required.json").exists()


def test_pipeline_final_recheck_rejects_actual_report_change(context, monkeypatch):
    c = context
    calls = []
    monkeypatch.setattr(c.m, "run_one", fake_run_one(c, calls))
    def altered_request(*a):
        with (c.out / "native.json").open("ab") as handle:
            handle.write(b"\n")
    c.launch.load_request = altered_request
    with pytest.raises(ValueError, match="changed during"):
        c.m.pipeline(c.tmp / "request.json", "e" * 64)
    assert not (c.out / "inspection-completed-review-required.json").exists()
