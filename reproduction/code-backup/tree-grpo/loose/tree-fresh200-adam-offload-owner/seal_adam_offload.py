"""Bind the local two-file fix and actual tiny CPU evidence, without execution release."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OWNER = Path(__file__).resolve().parent
SOURCE = ROOT / "work/tree-fresh200-adam-offload-source"
BASE_SOURCE = ROOT / "work/tree-behavior-mu-utf8-v2"
BASE_COMMIT = "02b1ea2688ff3fb032ee6699003a7e09d651f166"
PRIOR_PATH = ROOT / "work/tree-fresh200-utf8-v2-owner/UTF8_V2_SOURCE_INTERFACE_CPU_RECEIPT.json"
TERMINAL_PATH = ROOT / "work/tree-fresh200-freeze-02b1-v11/actual-zero-dispatch/TERMINAL3314_AUDIT.json"
CHANGED = ["verl/utils/fsdp_utils.py", "verl/workers/fsdp_workers.py"]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OWNER/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n")


def node(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def dump(value):
    return ast.dump(value, include_attributes=False)


def strip_docstrings(tree):
    for entry in ast.walk(tree):
        if isinstance(entry, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if entry.body and isinstance(entry.body[0], ast.Expr) \
                    and isinstance(entry.body[0].value, ast.Constant) \
                    and isinstance(entry.body[0].value.value, str):
                entry.body.pop(0)
    return tree


assert git(BASE_SOURCE, "rev-parse", "HEAD") == BASE_COMMIT
assert not git(BASE_SOURCE, "status", "--porcelain")
assert not git(SOURCE, "status", "--porcelain")
assert git(SOURCE, "rev-parse", "HEAD^") == BASE_COMMIT
commit = git(SOURCE, "rev-parse", "HEAD")
assert git(SOURCE, "diff", "--name-only", BASE_COMMIT, "HEAD").splitlines() == CHANGED
subprocess.check_call(["git", "-C", str(SOURCE), "diff", "--check", BASE_COMMIT, "HEAD"])
prior = json.loads(PRIOR_PATH.read_text())
old_interface = {**prior["source_files"], **prior["additional_runtime_files"]}
assert len(old_interface) == 19
assert prior["code_commit"] == BASE_COMMIT
assert all(sha(BASE_SOURCE/name) == digest for name, digest in old_interface.items())
source_files = {name: sha(SOURCE/name) for name in old_interface}
source_files["verl/utils/fsdp_utils.py"] = sha(SOURCE/"verl/utils/fsdp_utils.py")
assert len(source_files) == 20
assert all(source_files[name] == digest for name, digest in old_interface.items()
           if name != "verl/workers/fsdp_workers.py")
before_workers = ast.parse((BASE_SOURCE/CHANGED[1]).read_text())
after_workers = ast.parse((SOURCE/CHANGED[1]).read_text())
inspector_before = dump(node(before_workers, "inspect_fresh_worker_state"))
inspector_after = dump(node(after_workers, "inspect_fresh_worker_state"))
assert inspector_before == inspector_after
assert dump(strip_docstrings(copy.deepcopy(before_workers))) == \
       dump(strip_docstrings(copy.deepcopy(after_workers)))
before_utils = ast.parse((BASE_SOURCE/CHANGED[0]).read_text())
after_utils = ast.parse((SOURCE/CHANGED[0]).read_text())
guards = []
for name in ("offload_fsdp_optimizer", "load_fsdp_optimizer"):
    before = node(before_utils, name)
    after = node(after_utils, name)
    loop = after.body[0].body[0]
    guard = loop.body.pop(0)
    expected = ast.parse("if param not in optimizer.state:\n    continue\n").body[0]
    assert dump(guard) == dump(expected)
    assert dump(before) == dump(after)
    guards.append(name)
assert dump(before_utils) == dump(after_utils)
write("PRESERVED_SOURCE_AST.json", dict(
    code_commit=commit, base_commit=BASE_COMMIT,
    inspector_ast_sha256=hashlib.sha256(inspector_after.encode()).hexdigest(),
    inspect_fresh_worker_state_byte_ast_unchanged=True,
    fsdp_workers_all_executable_ast_unchanged=True,
    fsdp_utils_all_ast_unchanged_except_exact_membership_continue=guards,
    existing_mapping_iteration_tensor_to_nonblocking_and_empty_cache_ast_unchanged=True,
    no_state_clear_or_state_get_added=True,
    strict_entire_state_nonempty_and_scheduler_zero_refusals_cpu_passed=True))
run = json.loads((OWNER/"FROZEN_TINY_CPU_RUN.json").read_text())
assert run["status"] == "TINY_CPU_PASSED" and run["actual_test_count"] == 5
assert run["source_files"] == {name: source_files[name] for name in CHANGED}
assert run["actual_tiny_cpu_adamw_step_calls"] == 1
assert not run["actual_gpu_executed"] and not run["actual_remote_executed"]
diff = subprocess.check_output(["git", "-C", str(SOURCE), "diff", BASE_COMMIT, "HEAD"], text=True)
(OWNER/"SOURCE_DIFF.patch").write_text(diff)
affected = {name: dict(base_sha256=sha(BASE_SOURCE/name), source_sha256=source_files[name])
            for name in CHANGED}
manifest = dict(base_commit=BASE_COMMIT, code_commit=commit,
    full_git_tree_sha1=git(SOURCE, "rev-parse", "HEAD^{tree}"),
    branch=git(SOURCE, "branch", "--show-current"), source_directory=str(SOURCE),
    source_clean=True, changed_files=CHANGED, affected_source=affected,
    original_source_clean=True, original_source_commit=BASE_COMMIT,
    legacy_source_interface_count=19, source_interface_count=20, source_files=source_files,
    source_interface_count_semantics="Total unique source interfaces: prior 15 primary + 4 additional runtime + new fsdp_utils",
    primary_source_interface_count=16,
    primary_source_files={name: source_files[name] for name in [*prior["source_files"], "verl/utils/fsdp_utils.py"]},
    additional_runtime_files={name: source_files[name] for name in prior["additional_runtime_files"]},
    new_source_interface_modules_required=["verl/utils/fsdp_utils.py"])
write("AFFECTED_SOURCE_MANIFEST.json", manifest)
artifacts = {}
for name in ["BASE_02B1_COUNTEREXAMPLE.json", "INITIAL_CPU_FIXTURE_FAILURE.json",
             "test_adam_offload_cpu.py", "seal_adam_offload.py", "FROZEN_TINY_CPU_RUN.json",
             "FROZEN_TINY_CPU_ACTUAL.txt", "PRESERVED_SOURCE_AST.json", "SOURCE_DIFF.patch",
             "AFFECTED_SOURCE_MANIFEST.json"]:
    artifacts[name] = dict(path=str(OWNER/name), sha256=sha(OWNER/name))
legacy_runs = {name: dict(prior[name], rerun_for_this_commit=False,
    scope="Historical base-02b1 CPU results for unchanged semantics only; not a new full-suite pass")
    for name in ("latest_actual_source_cpu", "baseline_budget_checkpoint_cpu")}
receipt = dict(manifest,
    created_utc=datetime.now(timezone.utc).isoformat(),
    status="FRESH200_ADAM_OFFLOAD_TINY_CPU_OWNER_VERIFIED_REVIEW_PENDING",
    root_cause="AdamW.state is a defaultdict; the original optimizer.state[param] lookup created empty mapping entries before any optimizer.step. The existing init_model offload invokes it before the strict fresh-state inspector.",
    historical_actual_failure_input=dict(path=str(TERMINAL_PATH), sha256=sha(TERMINAL_PATH),
        job_id=3314, terminal_state="FAILED", slurm_exit_code="1:0", runtime_seconds=258,
        formal_outer_progress="0/200", original_eight_pid_start_times_preserved=True,
        cleanup_residual={}, scope="Read-only previously captured terminal diagnostic metadata; no owner remote/GPU execution"),
    fix="Both optimizer transfer helpers test param membership before reading state[param]; missing keys are skipped, existing {} remains, invalid None/nonmapping still raises via state.items(), real moments use the unchanged tensor migration logic.",
    source_owner_verification_completed=True, source_cpu_tests_passed=True,
    source_independent_review_completed=False, source_independent_review_accepted=False,
    training_release=False, actual_3b_training_accepted=False,
    actual_3b_training_optimizer_calls=0, actual_3b_training_outer_updates=0,
    actual_tiny_cpu_adamw_step_calls=1, actual_remote_executed=False,
    actual_slurm_executed=False, actual_gpu_executed=False,
    actual_ray_executed=False, actual_native_vllm_executed=False,
    actual_official_model_or_dataset_assets_loaded=False,
    actual_official_tokenizer_executed=False,
    actual_historical_raw_or_tensor_assets_loaded=False,
    latest_actual_source_cpu=dict(run,
        command=[run["python_executable"], "-B", str(OWNER/"test_adam_offload_cpu.py"),
                 "--source", str(SOURCE), "--json-output", str(OWNER/"FROZEN_TINY_CPU_RUN.json")],
        code_commit=commit, source_clean=True,
        log_path=str(OWNER/"FROZEN_TINY_CPU_ACTUAL.txt"), log_sha256=sha(OWNER/"FROZEN_TINY_CPU_ACTUAL.txt")),
    inherited_base_cpu_evidence=legacy_runs,
    inherited_legacy_test_files=prior["test_files"],
    inherited_legacy_test_files_scope="Byte-checked against the clean base and unchanged candidate; historical tests not rerun here",
    prior_source_interface_receipt=dict(path=str(PRIOR_PATH), sha256=sha(PRIOR_PATH),
        code_commit=BASE_COMMIT, old_primary_interface_count=15,
        old_additional_runtime_interface_count=4, old_total_interface_count=19),
    new_source_interface_scope="Original 15 primary + 4 runtime interfaces preserved; fsdp_utils added and all 20 byte hashes bound to this new clean commit",
    unchanged_runtime_files=[name for name in old_interface if name not in CHANGED],
    new_config_keys_required=[],
    probability_tolerance=prior["probability_tolerance"],
    target_old_policy_kl=prior["target_old_policy_kl"],
    rows_per_opportunity=prior["rows_per_opportunity"], rows_per_rank=prior["rows_per_rank"],
    opportunities_per_complete_outer_per_rank=prior["opportunities_per_complete_outer_per_rank"],
    opportunity_accounting=prior["opportunity_accounting"],
    unchanged_protocol_scope="20 actually processed opportunities with >=1 successful Adam write completes one outer and advances scheduler once under guard continue; all-skip or abandoned gives no outer/warmup; partial writes remain and writes reported separately. No actor, scheduler, guard, math, mu, raw-token, config, model or data behavior modified.",
    verification_scope="Real torch2.8 AdamW tiny CPU only, exact AST-extracted transfer helpers and unchanged actual worker inspector; clean source byte closure and existing AST logic preservation",
    limitations=["Five new CPU tests only; 52 affected/14 budget and prior143/Gloo results are inherited base evidence, never a new suite pass.",
        "Tensor device calls exercised CPU->CPU only; no CUDA, real FSDP, cross-device, distributed, Ray, Slurm, model/data/tokenizer assets or 5120 historical calibration.",
        "One actual tiny CPU AdamW.step is not actual3B optimizer or training acceptance.",
        "Initial CPU fixture import-registry failure is preserved separately; fixed by constructing optimizers before sys.modules patch, not by stubbing AdamW or changing source.",
        "Independent source acceptance pending; no READY/GPU/probability/or production threshold pass claimed."],
    artifacts=artifacts)
assert all(sha(SOURCE/name) == digest for name, digest in source_files.items())
assert all(sha(SOURCE/name) == digest for name, digest in prior["test_files"].items())
assert not git(SOURCE, "status", "--porcelain")
write("ADAM_OFFLOAD_SOURCE_INTERFACE_CPU_RECEIPT.json", receipt)
print(json.dumps(dict(code_commit=commit, source_interface_count=20, actual_test_count=5,
    source_clean=True, receipt=str(OWNER/"ADAM_OFFLOAD_SOURCE_INTERFACE_CPU_RECEIPT.json")), indent=2))
