"""Fixed-weight diagnostic worker. No optimizer, backward, training, or launcher."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import subprocess

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import append_json, atomic_json, file_sha, load_science, object_sha
    from admission import RetrieverAdmission
    from token_probe import FullVocabularyBudget, probe
    from trace_agent import TraceSearchAgent
    from weight_identity import VERSIONS, check_weight_manifest
else:
    from .common import append_json, atomic_json, file_sha, load_science, object_sha
    from .admission import RetrieverAdmission
    from .token_probe import FullVocabularyBudget, probe
    from .trace_agent import TraceSearchAgent
    from .weight_identity import VERSIONS, check_weight_manifest


def arguments():
    parser = argparse.ArgumentParser()
    for key in ("runtime", "cases-dir", "model", "weight-manifest", "retriever-admission", "output"):
        parser.add_argument("--" + key, required=True)
    for version in VERSIONS:
        parser.add_argument("--adapter-" + version, required=True)
    parser.add_argument("--retriever-url", default="http://10.120.17.101:18005/retrieve")
    parser.add_argument("--allow-fallback-18004", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--full-vocab-case-id", action="append", default=[])
    parser.add_argument("--full-vocab-versions", default="epoch1")
    parser.add_argument("--full-vocab-max-bytes", type=int, default=2 << 30)
    return parser.parse_args()


def verify_cases(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "cases.jsonl"
    if manifest["selected_cases"] != 192 or manifest["per_cell"] != 32 or manifest["seed"] != 42 or not manifest["strict_reward_checked"]:
        raise ValueError("Required actual 192-case selector identity is missing")
    if file_sha(path) != manifest["cases"]["sha256"]:
        raise ValueError("Diagnostic case bytes changed")
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    if len(cases) != 192 or [case["case_index"] for case in cases] != list(range(192)):
        raise ValueError("Invalid diagnostic case indices")
    for case in cases:
        claimed = case["case_semantic_sha256"]
        original = {key: value for key, value in case.items() if key != "case_semantic_sha256"}
        if object_sha(original) != claimed:
            raise ValueError("Case semantic source hash changed")
        if case["group"] == "all5_wrong" and case["original_pair"] is not None:
            raise ValueError("Hard case was assigned a fabricated reference")
    return cases, dict(path=str(path), sha256=manifest["cases"]["sha256"],
                       manifest_sha256=file_sha(directory / "manifest.json"), manifest=manifest)


def current_gpu_memory(torch):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    rows = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-gpu=uuid,memory.free,memory.used,memory.total",
                                    "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError("Worker GPU inventory is not uniquely assigned")
    uuid, free, used, total = [value.strip() for value in rows[0].split(",")]
    cuda_free, cuda_total = torch.cuda.mem_get_info()
    return dict(sample_unix=time.time(), gpu_uuid=uuid, nvidia_smi_actual_free_mib=int(free),
                nvidia_smi_used_mib=int(used), nvidia_smi_total_mib=int(total),
                cuda_free_bytes=cuda_free, cuda_total_bytes=cuda_total,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    args = arguments()
    if args.workers != 3 or not 0 <= args.worker_index < 3:
        raise ValueError("Independent debug assigns exactly three workers")
    if args.full_vocab_max_bytes < 0:
        raise ValueError("Invalid full vocabulary cap")
    versions = args.full_vocab_versions.split(",") if args.full_vocab_versions else []
    if any(version not in VERSIONS for version in versions):
        raise ValueError("Unknown full vocabulary adapter version")
    admission = RetrieverAdmission(args.retriever_admission, args.retriever_url, args.allow_fallback_18004)
    startup_health = admission.check_live()
    cases, cases_identity = verify_cases(args.cases_dir)
    if not set(args.full_vocab_case_id).issubset({case["case_id"] for case in cases}):
        raise ValueError("A full vocabulary case is outside the fixed diagnostic sample")
    adapters = {version: getattr(args, "adapter_" + version) for version in VERSIONS}
    weights = check_weight_manifest(args.weight_manifest, args.model, adapters)
    core, opsd, rollout, science = load_science(args.runtime, with_torch=True)
    # The selector and worker must import exactly the same frozen scoring/loss.
    for relative, identity in cases_identity["manifest"]["science"].items():
        if identity["sha256"] != science[relative]["sha256"]:
            raise ValueError("Case selector/worker scientific identity mismatch")
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    for version in VERSIONS:
        tensors = load_file(str(Path(adapters[version]) / "adapter_model.safetensors"), device="cpu")
        if not tensors or not all(torch.isfinite(tensor).all().item() for tensor in tensors.values()):
            raise FloatingPointError("Nonfinite fixed adapter: " + version)
        if version == "source100":
            b = [value for key, value in tensors.items() if "lora_B" in key]
            if len(b) != 252 or any(torch.count_nonzero(value).item() for value in b):
                raise ValueError("source100 adapter is not the verified zero-B baseline")
        del tensors
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Root must assign exactly one GPU per worker with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    directory = Path(args.output).resolve() / f"worker-{args.worker_index}"
    directory.mkdir(parents=True, exist_ok=False)
    status = dict(schema="independent-fixed-weight-opsd-debug-v1", status="INITIALIZING", started_unix=time.time(),
                  uid=os.getuid(), pid=os.getpid(), ppid=os.getppid(), pgid=os.getpgid(0),
                  proc_starttick=int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()[19]),
                  worker_index=args.worker_index, workers=args.workers, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  weights=weights, science=science, cases=cases_identity,
                  retriever=dict(admission.identity, startup_health=startup_health),
                  no_training=True, seed=42, actual_generation_temperature=0.0,
                  full_vocab=dict(global_cap_bytes=args.full_vocab_max_bytes, worker_partition_bytes=args.full_vocab_max_bytes // args.workers,
                                  cases=args.full_vocab_case_id, versions=versions))
    atomic_json(directory / "status.json", status)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    model = PeftModel.from_pretrained(base, adapters["source100"], adapter_name="source100", is_trainable=False)
    for version in VERSIONS[1:]:
        model.load_adapter(adapters[version], adapter_name=version, is_trainable=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    budget = FullVocabularyBudget(args.full_vocab_max_bytes, args.workers, args.full_vocab_case_id, versions)
    agent = TraceSearchAgent(model, tokenizer, core, rollout, args.retriever_url, admission.check_live)
    assigned = [case for case in cases if case["case_index"] % args.workers == args.worker_index]
    desired = ("correct_correct", "correct_wrong", "all5_wrong")[args.worker_index]
    priority = set(args.full_vocab_case_id)
    first = min((case for case in assigned if case["group"] == desired),
                key=lambda case: (case["case_id"] not in priority, case["case_index"]))
    assigned = [first] + sorted((case for case in assigned if case is not first),
                               key=lambda case: (case["case_id"] not in priority, case["case_index"]))
    status.update(status="RUNNING", assigned_cases=len(assigned), completed_cases=0, gpu=current_gpu_memory(torch),
                  execution_order=[case["case_id"] for case in assigned],
                  execution_order_reason="First case group rotates correct_correct/correct_wrong/all5_wrong by worker; then requested full-vocabulary cases first; fixed greedy weights, no training")
    atomic_json(directory / "status.json", status)
    with (directory / "case-results.jsonl").open("x") as results:
        for case in assigned:
            case_directory = directory / case["case_id"]
            case_directory.mkdir()
            try:
                health = admission.check_live()
                atomic_json(case_directory / "admitted-retriever-health.json", health)
                source = case["source_train_row"]["row"]
                question, answers = source["question"], source["answers"]
                pair = case["original_pair"]
                reference = pair["row"]["reference"] if pair is not None else None
                model.set_adapter("source100")
                if reference is None:
                    teacher = dict(kind="actual_autoregressive_generation", status="not_applicable_no_successful_reference",
                                   privileged_reference=False, reason="All five original source100 siblings failed; no successful teacher reference exists")
                else:
                    with model.disable_adapter():
                        teacher = agent.run(question, answers, reference)
                    teacher.update(role="frozen_source100_privileged_teacher", active_adapter="disabled", fixed_weight_source="source100")
                atomic_json(case_directory / "teacher-generation.json", teacher)
                teacher_probe = None
                if reference is not None:
                    teacher_probe = probe(model, tokenizer, core, opsd, teacher["response"], teacher["policy_spans"],
                                          question, reference, "source100", case["case_id"],
                                          case_directory / "tokens-source100-actual-teacher-generation", budget,
                                          response_origin="new_teacher_generation")
                    atomic_json(case_directory / "token-summary-source100-actual-teacher-generation.json", teacher_probe)
                versions_report = {}
                for version in VERSIONS:
                    model.set_adapter(version)
                    generation = agent.run(question, answers)
                    generation.update(role="ordinary_student", active_adapter=version, fixed_weight_source="source100")
                    atomic_json(case_directory / f"student-generation-{version}.json", generation)
                    summaries = []
                    if pair is not None:
                        trajectories = [("original-mined-student", pair["row"]["student"], pair["row"]["student_policy_spans"])]
                    else:
                        trajectories = [(f"original-failed-sibling-{item['row']['sibling']}", item["row"]["response"], item["row"]["policy_spans"])
                                        for item in case["original_siblings"]]
                    for name, response, spans in trajectories:
                        summary = probe(model, tokenizer, core, opsd, response, spans, question, reference,
                                        version, case["case_id"], case_directory / f"tokens-{version}-{name}", budget,
                                        response_origin="preserved_student" if pair is not None else "preserved_failed_sibling")
                        summary["preserved_trajectory_name"] = name
                        atomic_json(case_directory / f"token-summary-{version}-{name}.json", summary)
                        summaries.append(summary)
                    actual_probe = probe(model, tokenizer, core, opsd, generation["response"], generation["policy_spans"],
                                         question, reference, version, case["case_id"],
                                         case_directory / f"tokens-{version}-actual-student-generation", budget,
                                         response_origin="new_student_generation")
                    atomic_json(case_directory / f"token-summary-{version}-actual-student-generation.json", actual_probe)
                    versions_report[version] = dict(generation_em=generation["protocol_em"], generation_termination=generation["termination"],
                                                   generation_search_calls=generation["search_calls"], token_probes=summaries,
                                                   actual_student_response_teacher_forced_probe=actual_probe)
                report = dict(case_id=case["case_id"], case_index=case["case_index"], source_case_sha256=case["case_semantic_sha256"],
                              dataset=case["dataset"], group=case["group"], status="complete", teacher_generation=teacher,
                              actual_teacher_response_teacher_forced_probe=teacher_probe,
                              versions=versions_report, completed_unix=time.time())
                append_json(results, report)
                status["completed_cases"] += 1
                status["last_case"] = case["case_id"]
                status["updated_unix"] = time.time()
                status["gpu"] = current_gpu_memory(torch)
                atomic_json(directory / "status.json", status)
                print(json.dumps(dict(case_id=case["case_id"], completed=status["completed_cases"], assigned=len(assigned))), flush=True)
            except BaseException as error:
                failure = dict(status="FAILED", case_id=case["case_id"], time_unix=time.time(),
                               error=repr(error), traceback=traceback.format_exc())
                atomic_json(case_directory / "failure.json", failure)
                status.update(status="FAILED", last_failure=failure)
                atomic_json(directory / "status.json", status)
                raise
    # Recheck attested inputs; do not call a complete debug run if any changed.
    check_weight_manifest(args.weight_manifest, args.model, adapters)
    load_science(args.runtime)
    verify_cases(args.cases_dir)
    admission.check_live()
    status.update(status="COMPLETE", ended_unix=time.time(), full_vocab_worker_bytes_reserved=budget.used,
                  results_sha256=file_sha(directory / "case-results.jsonl"),
                  diagnostic_sample_only=True)
    atomic_json(directory / "status.json", status)


if __name__ == "__main__":
    main()
