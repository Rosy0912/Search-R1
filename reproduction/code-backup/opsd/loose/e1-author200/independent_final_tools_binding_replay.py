"""Read-only local frozen-byte/dependency-map check, not remote deployment."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "final-acceptance-8441e10fa456"
EXPECTED_MANIFEST = "8441e10fa4565bf53f7baf24a57a60ddbe9844dc979a2c9a0c97250219df7019"
REMOTE_ROOT = "/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006"
REMOTE_REPO = REMOTE_ROOT + "/repo-188f4e5e7b63"
REMOTE_RUN = REMOTE_ROOT + "/runs/formal-3261"
SNAPSHOT = HERE / "actual-snapshots/3261-20261005T214741940075Z/original-files"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def parse(path):
    def pairs(items):
        out = {}
        for k, v in items:
            assert k not in out
            out[k] = v
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)


manifest_path = BUNDLE / "manifest.json"
assert sha(manifest_path) == EXPECTED_MANIFEST
m = parse(manifest_path)
assert set(m) == {"version", "kind", "source_commit", "scope", "contract", "tools", "dependencies", "review_records", "limits"}
assert type(m["version"]) is int and m["version"] == 1
assert m["kind"] == "AUTHOR200_EXTERNAL_CHECKPOINT_TOOLS"
assert m["scope"] == "AUTHOR200_CPU_STORED_CONTENT_INSPECTION_NOT_TRAINING_CONTROL"
assert m["source_commit"] == "188f4e5e7b63cf479290d0888a4a075499886750"
contract = dict(adam_steps_per_rank_per_outer=16, checkpoint_steps=[25, 50, 75, 100, 125, 150, 175, 200],
    cuda_work=False, global_mini_batch=32, horizon=200, learning_rate=5e-6, training_job_id="3261", warmup_steps=0)
assert json.dumps(m["contract"], sort_keys=True) == json.dumps(contract, sort_keys=True)
assert set(p.name for p in BUNDLE.iterdir()) == {"manifest.json"} | set(m["tools"]) | set(m["review_records"])
for name, expected in {**m["tools"], **m["review_records"]}.items():
    p = BUNDLE / name
    assert Path(name).name == name and not p.is_symlink() and p.resolve() == p
    assert sha(p) == expected
wrapper = parse(BUNDLE / "CHECKPOINT_CPU_WRAPPERS_INDEPENDENT_REVIEW.json")
helpers = parse(BUNDLE / "EXTERNAL_CHECKPOINT_HELPERS_INDEPENDENT_REVIEW.json")
assert wrapper["status"] == "PASS_INDEPENDENT_LOCAL_CPU_WRAPPER_REVIEW_NOT_RUNTIME_ACCEPTANCE" and not wrapper["blockers"]
assert helpers["status"] == "PASS_INDEPENDENT_EXTERNAL_CHECKPOINT_HELPER_SOFTWARE_REVIEW_NOT_NATIVE25_ACCEPTANCE" and not helpers["blockers"]
expected_tools = dict(helpers["candidate_files_sha256"])
expected_tools["checkpoint_overlap.py"] = wrapper["files_sha256"]["checkpoint_overlap_candidate.py"]
expected_tools["checkpoint_cpu_pipeline.py"] = wrapper["files_sha256"]["checkpoint_cpu_pipeline_candidate.py"]
assert m["tools"] == expected_tools and len(expected_tools) == 7

accepted = parse(SNAPSHOT / "full_state/global_step_5/manifest.json")
receipt = parse(SNAPSHOT / "acceptance-5-receipt.json")
assert sha(SNAPSHOT / "full_state/global_step_5/manifest.json") == "8c6e17b2af0026b1267a2422c1437aa00461da46e0a1911caea3633bfef473d3"
assert sha(SNAPSHOT / "acceptance-5-receipt.json") == "812f0baf1d70ee87e847865f62ff03151e1de0c4a4c0eab2e9ae72e0dbb3793b"
spec = importlib.util.spec_from_file_location("frozen_gate_binding_only", HERE.parent / "opsd-e1-300/search_r1/baselines/checkpoint_gate.py")
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
assert sha(spec.origin) == "c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb"
gate.verify_rsa_signature(receipt["payload"], receipt["signature_hex"], parse(SNAPSHOT / "bindings/public.json"))
assert receipt["payload"]["checkpoint_files"] == accepted["files"]
dependencies = {}
def add(path, digest):
    assert path not in dependencies or dependencies[path] == digest
    assert Path(path).is_absolute() and len(digest) == 64 and all(x in "0123456789abcdef" for x in digest)
    dependencies[path] = digest
closure_path = SNAPSHOT / "bindings/import_closure.json"
assert sha(closure_path) == "3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d"
closure = parse(closure_path)
assert set(closure) == {"modules"} and len(closure["modules"]) == 210
for rec in closure["modules"].values(): add(rec["path"], rec["sha256"])
for dirname, manifest_name, expected in [
    ("formal-bb5f4872142f", "manifest.json", "bb5f4872142f34a330abab14b5ae3869bc212c2e1eb7e247cf895f5aa3319d7c"),
    ("first5-native-1f209cab7e02", "candidate-manifest.json", "1f209cab7e025439584a938b4aab7d71358db66ee3f7d63ab348285247ebbd65"),
    ("first5-12424d01025e", "manifest.json", "12424d01025e9759d495500eef87de37712a29e1dfe989d3772d05cfa3bfe330")]:
    p = HERE / dirname / manifest_name
    assert sha(p) == expected
    old = parse(p)
    add(REMOTE_ROOT + "/tools/" + dirname + "/" + manifest_name, expected)
    for name, digest in old["tools"].items():
        assert sha(p.parent / name) == digest
        add(REMOTE_ROOT + "/tools/" + dirname + "/" + name, digest)
    for name, digest in old.get("source_files", {}).items():
        assert sha(HERE.parent / "opsd-e1-300" / name) == digest
        add(REMOTE_REPO + "/" + name, digest)
for role, rec in accepted["files"].items():
    # Old large HF/export and eight raw rankPT are signed historical references,
    # not bootstrap dependencies. Actual driverPT is required for sampler join.
    if role.startswith("actor/") and not ("/training_state/" in role and role.endswith(".json")):
        continue
    add(rec["path"], rec["sha256"])
    local = SNAPSHOT / Path(rec["path"]).relative_to(REMOTE_RUN)
    if local.is_file(): assert sha(local) == rec["sha256"]
add(REMOTE_RUN + "/full_state/global_step_5/manifest.json", sha(SNAPSHOT / "full_state/global_step_5/manifest.json"))
add(REMOTE_RUN + "/acceptance-5-receipt.json", sha(SNAPSHOT / "acceptance-5-receipt.json"))
add("/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json", accepted["files"]["model_manifest"]["sha256"])
audit = receipt["payload"]["evidence"]["rank_audit"]
assert audit["sha256"] == "b50c6ed9995f6148ac250e75f20721bafc95d952f4d841c24542dce31f4adcb2"
add(audit["path"], audit["sha256"])
assert sha(SNAPSHOT / Path(audit["path"]).relative_to(REMOTE_RUN)) == audit["sha256"]
assert dependencies == m["dependencies"] and len(dependencies) == 254
assert not any("global_step_25" in p or "global_step_200" in p for p in dependencies)
assert not any(p.endswith(".safetensors") or "/training_state/rank_" in p and p.endswith(".pt") for p in dependencies)
print(json.dumps(dict(version=1, status="PASS_INDEPENDENT_LOCAL_FROZEN_TOOL_BINDING_NOT_REMOTE_EXECUTION",
    frozen_manifest_sha256=EXPECTED_MANIFEST, source_commit=m["source_commit"], tool_count=7,
    review_record_count=6, dependency_count=254, exact_dependency_union_matches=True,
    accepted5_signature_verified=True, signed_native_audit_dependency_added=True,
    actual_native_manifest_basename="candidate-manifest.json",
    remote_dependency_full_byte_verification_performed=False,
    no_future_checkpoint_pass_or_nonce=True,
    limitations=["No upload/remote CPU/GPU or Slurm command executed", "Root must verify all254 remote files before manifest-last publication",
        "No actual25 PT/HF/native2.4 or terminal acceptance"]), indent=2))
