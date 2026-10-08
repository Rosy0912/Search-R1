"""Independent local-only tests. No SSH, Slurm command execution or GPU work."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("independent_overlap", HERE / "checkpoint_overlap_candidate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    run = tmp_path / "run"
    evidence = run / "evidence"
    evidence.mkdir(parents=True)
    ray = tmp_path / "ray"
    ray.mkdir()
    training = run / "train.log"
    driver = ray / "worker-owned-1533312.out"
    training.write_bytes(b"aggregate complete row\n")
    driver.write_bytes(b"driver complete row\n")
    monkeypatch.setattr(m, "RUN", run)
    monkeypatch.setattr(m, "REPO", tmp_path / "repo")
    monkeypatch.setattr(m, "PYTHON", Path(sys.executable))
    monkeypatch.setattr(m, "HERE", tmp_path / "tools")
    m.HERE.mkdir()
    monkeypatch.setattr(m, "host", lambda: None)
    def physical(path, directory=False):
        p = Path(path)
        m.need(p.is_absolute() and p.resolve(strict=True) == p, "physical fixture path")
        m.need(p.is_dir() if directory else p.is_file(), "fixture path kind")
        return p
    monkeypatch.setattr(m, "physical", physical)
    # Only translate the fixed remote ray directory into the fixture directory.
    monkeypatch.setattr(m, "Path", lambda value: ray if str(value) == "/tmp/o4e-3261/ray" else Path(value))
    monkeypatch.setattr(m, "process", lambda pid, **kw: {"pid": pid, "present": True, "start_ticks": "123", "fixture": True})
    monkeypatch.setattr(m, "allocation", lambda: {"fixture_only": True, "driver": {"pid": 1533312, "start_ticks": "1829085316"}})
    original_readlink = os.readlink
    monkeypatch.setattr(m.os, "readlink", lambda p: str(driver) if str(p) == "/proc/1533312/fd/1" else original_readlink(p))
    prefixes = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": m.sha(path)}
        for name, path in (("training", training), ("driver", driver))}
    config = {"version": 1, "scope": m.SCOPE, "source_commit": m.SOURCE,
        "training_job_id": "3261", "checkpoint_step": 25, "tools_manifest_sha256": "a" * 64,
        "output_directory": str(evidence / "checkpoint-25-cpu-inspection-fixture"), "log_prefixes": prefixes}
    marker = evidence / "checkpoint-cpu-25-launch.json"
    def bypass_tools(pin):
        m.pin(pin)
        return {"fixture_only": True}
    real_tools = m.tools
    monkeypatch.setattr(m, "tools", bypass_tools)
    return SimpleNamespace(m=m, run=run, evidence=evidence, driver=driver,
        training=training, config=config, marker=marker, real_tools=real_tools)


def args(c):
    return SimpleNamespace(checkpoint_step=c.config["checkpoint_step"],
        tools_manifest_sha256=c.config["tools_manifest_sha256"],
        output_directory=c.config["output_directory"],
        log_prefixes_json=json.dumps(c.config["log_prefixes"]))


def command(exit_code=0, signal_number=None, done=None):
    script = "import os,sys,time; print('actual fixture stdout',flush=True); print('actual fixture stderr',file=sys.stderr,flush=True); time.sleep(.03); "
    if done:
        script += f"open({str(done)!r},'w').write('done'); "
    script += f"os.kill(os.getpid(),{signal_number})" if signal_number else f"sys.exit({exit_code})"
    return [sys.executable, "-c", script]


@pytest.mark.parametrize("which,code,sig", [("controller", 0, None), ("controller", 7, None),
    ("controller", 0, signal.SIGTERM), ("supervise", 0, None), ("supervise", 9, None),
    ("supervise", 0, signal.SIGTERM)])
def test_real_local_child_wait_and_status(ctx, monkeypatch, which, code, sig):
    m = ctx.m
    output = Path(ctx.config["output_directory"])
    output.mkdir()
    cmd = command(code, sig)
    monkeypatch.setattr(m, "cpu_command", lambda *a: cmd)
    monkeypatch.setattr(m, "child", lambda *a: cmd)
    result = getattr(m, which)(ctx.config, output, ctx.marker, "a" * 64)
    label = "srun" if which == "controller" else "controller"
    exit_record = json.loads((output / f"{label}-exit-original.json").read_bytes())
    raw = -int(sig) if sig else code
    assert type(exit_record["returncode"]) is int and exit_record["returncode"] == raw
    assert result == (128 + int(sig) if sig else code)
    assert exit_record[f"waited_by_{'controller' if which == 'controller' else 'supervisor'}_pid"] == os.getpid()
    assert (output / f"{label}-stdout-original.txt").read_text() == "actual fixture stdout\n"
    assert (output / f"{label}-stderr-original.txt").read_text() == "actual fixture stderr\n"


@pytest.mark.parametrize("which,metadata", [("controller", "srun-start-original.json"),
    ("supervise", "controller-launch-original.json")])
def test_metadata_write_failure_still_waits_actual_child(ctx, monkeypatch, which, metadata):
    m = ctx.m
    output = Path(ctx.config["output_directory"])
    output.mkdir()
    done = ctx.run / "actual-child-finished"
    cmd = command(done=done)
    monkeypatch.setattr(m, "cpu_command", lambda *a: cmd)
    monkeypatch.setattr(m, "child", lambda *a: cmd)
    original = m.write
    def fail_metadata(path, value):
        if path.name == metadata:
            raise OSError("injected metadata publication failure")
        original(path, value)
    monkeypatch.setattr(m, "write", fail_metadata)
    with pytest.raises(OSError, match="metadata"):
        getattr(m, which)(ctx.config, output, ctx.marker, "a" * 64)
    assert done.read_text() == "done"
    label = "srun" if which == "controller" else "controller"
    assert not (output / f"{label}-exit-original.json").exists()


def test_exact_no_gpu_command_and_argument_boundaries(ctx):
    m = ctx.m
    hostile = ctx.evidence / "request with spaces;$(never).json"
    result = m.cpu_command(hostile, "b" * 64)
    assert result[:9] == ["srun", "--jobid=3261", "--overlap", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=1", "--gres=none", "--nodelist=h100-10", "env"]
    assert result[9:15] == ["CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=" + str(m.REPO)]
    assert result[-4:] == ["--request", str(hostile), "--request-sha256", "b" * 64]


@pytest.mark.parametrize("field,value", [("version", True), ("checkpoint_step", True),
    ("checkpoint_step", 5), ("checkpoint_step", 201), ("training_job_id", 3261)])
def test_invalid_typed_request_rejected_without_mutation(ctx, field, value):
    ctx.config[field] = value
    with pytest.raises(ValueError):
        ctx.m.validate_request(ctx.config, ctx.marker, output_exists=False)
    assert not ctx.marker.exists() and not Path(ctx.config["output_directory"]).exists()


@pytest.mark.parametrize("mutation", ["missing", "boolean_bytes", "bad_sha", "partial_line", "foreign_driver"])
def test_start_invalid_prefix_prevalidated_before_marker_or_output(ctx, monkeypatch, mutation):
    if mutation == "missing":
        ctx.config["log_prefixes"].pop("driver")
    elif mutation == "boolean_bytes":
        ctx.config["log_prefixes"]["driver"]["bytes"] = True
    elif mutation == "bad_sha":
        ctx.config["log_prefixes"]["training"]["sha256"] = "a" * 64
    elif mutation == "partial_line":
        ctx.driver.write_bytes(b"partial")
        ctx.config["log_prefixes"]["driver"].update(bytes=7, sha256=ctx.m.sha(ctx.driver))
    else:
        ctx.config["log_prefixes"]["driver"]["path"] = str(ctx.training)
    monkeypatch.setattr(ctx.m.subprocess, "Popen", lambda *a, **kw: pytest.fail("Invalid config launched process"))
    with pytest.raises(ValueError):
        ctx.m.start(args(ctx))
    assert not ctx.marker.exists() and not Path(ctx.config["output_directory"]).exists()


def test_reserved_output_alias_rejected_before_mutation(ctx):
    ctx.config["output_directory"] = str(ctx.marker)
    with pytest.raises(ValueError, match="reserved marker"):
        ctx.m.start(args(ctx))
    assert not ctx.marker.exists()


def test_duplicate_json_and_nonfinite_rejected(ctx):
    for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e309}'):
        with pytest.raises(ValueError):
            ctx.m.unique(raw)


def test_live_append_beyond_complete_pinned_prefix_is_allowed(ctx):
    with ctx.driver.open("ab") as handle:
        handle.write(b"future partial row")
    with ctx.training.open("ab") as handle:
        handle.write(b"future complete row\n")
    assert ctx.m.validate_request(ctx.config, ctx.marker, output_exists=False)[0] == ctx.config


def test_symlink_log_is_rejected(ctx):
    alias = ctx.driver.parent / "other-1533312.out"
    alias.symlink_to(ctx.driver)
    ctx.config["log_prefixes"]["driver"]["path"] = str(alias)
    with pytest.raises(ValueError, match="physical"):
        ctx.m.validate_request(ctx.config, ctx.marker, output_exists=False)


def test_unknown_process_creation_preserves_once_marker_and_original_streams(ctx, monkeypatch):
    def unknown(*a, **kw):
        raise OSError("unknown process creation")
    monkeypatch.setattr(ctx.m.subprocess, "Popen", unknown)
    with pytest.raises(OSError, match="unknown"):
        ctx.m.start(args(ctx))
    marker_bytes = ctx.marker.read_bytes()
    output = Path(ctx.config["output_directory"])
    assert (output / "supervisor-stdout-original.txt").is_file()
    assert (output / "supervisor-stderr-original.txt").is_file()
    assert not (output / "durable-launch-original.json").exists()
    second = args(ctx)
    second.output_directory = str(ctx.evidence / "checkpoint-25-cpu-inspection-second")
    with pytest.raises(ValueError, match="Existing launch"):
        ctx.m.start(second)
    assert ctx.marker.read_bytes() == marker_bytes
    assert not Path(second.output_directory).exists()


def tool_manifest(ctx):
    m = ctx.m
    required = ["checkpoint_overlap.py", "checkpoint_cpu_pipeline.py", "observe_author200_checkpoint.py",
        "inspect_author200_checkpoint_cpu.py", "hf_author200_cpu.py"]
    for name in required:
        (m.HERE / name).write_text("# pinned local dummy; never executed\n")
    return {"version": 1, "source_commit": m.SOURCE, "kind": "AUTHOR200_EXTERNAL_CHECKPOINT_TOOLS",
        "scope": m.SCOPE, "tools": {name: m.sha(m.HERE / name) for name in required}, "dependencies": {},
        "contract": {"training_job_id": "3261", "horizon": 200, "checkpoint_steps": list(m.STEPS),
            "global_mini_batch": 32, "adam_steps_per_rank_per_outer": 16,
            "warmup_steps": 0, "learning_rate": 5e-6, "cuda_work": False}}


@pytest.mark.parametrize("mutation", ["version_bool", "warmup_bool", "cuda_int", "step_float"])
def test_manifest_counter_types_cannot_alias_bool_int_float(ctx, mutation):
    value = tool_manifest(ctx)
    if mutation == "version_bool":
        value["version"] = True
    elif mutation == "warmup_bool":
        value["contract"]["warmup_steps"] = False
    elif mutation == "cuda_int":
        value["contract"]["cuda_work"] = 0
    else:
        value["contract"]["checkpoint_steps"][0] = 25.0
    path = ctx.m.HERE / "manifest.json"
    path.write_bytes(ctx.m.canonical(value))
    with pytest.raises(ValueError):
        ctx.real_tools(ctx.m.sha(path))


def test_good_manifest_is_accepted(ctx):
    value = tool_manifest(ctx)
    path = ctx.m.HERE / "manifest.json"
    path.write_bytes(ctx.m.canonical(value))
    assert ctx.real_tools(ctx.m.sha(path)) == value


def test_loaded_request_must_be_the_exact_hashed_bytes_even_during_replacement(ctx, monkeypatch):
    output = Path(ctx.config["output_directory"])
    output.mkdir()
    ctx.marker.write_bytes(ctx.m.canonical(ctx.config))
    old_sha = ctx.m.sha(ctx.marker)
    replacement = copy.deepcopy(ctx.config)
    new_output = ctx.evidence / "checkpoint-25-cpu-inspection-replaced"
    new_output.mkdir()
    replacement["output_directory"] = str(new_output)
    original_sha = ctx.m.sha
    def replace_after_sha(path):
        actual = original_sha(path)
        if Path(path) == ctx.marker:
            ctx.marker.write_bytes(ctx.m.canonical(replacement))
        return actual
    monkeypatch.setattr(ctx.m, "sha", replace_after_sha)
    with pytest.raises(ValueError, match="Immutable|changed|pinned|SHA"):
        ctx.m.load_request(ctx.marker, old_sha)


def test_tool_manifest_must_be_the_exact_hashed_bytes_even_during_replacement(ctx, monkeypatch):
    value = tool_manifest(ctx)
    path = ctx.m.HERE / "manifest.json"
    path.write_bytes(ctx.m.canonical(value))
    old_sha = ctx.m.sha(path)
    replacement = copy.deepcopy(value)
    # A new dependency is legitimate schema, but was absent from reviewed bytes.
    extra = ctx.run / "unreviewed_dependency.py"
    extra.write_text("# never executed\n")
    replacement["dependencies"][str(extra)] = ctx.m.sha(extra)
    original_sha = ctx.m.sha
    def replace_after_sha(p):
        actual = original_sha(p)
        if Path(p) == path:
            path.write_bytes(ctx.m.canonical(replacement))
        return actual
    monkeypatch.setattr(ctx.m, "sha", replace_after_sha)
    with pytest.raises(ValueError, match="changed|pinned|SHA"):
        ctx.real_tools(old_sha)
