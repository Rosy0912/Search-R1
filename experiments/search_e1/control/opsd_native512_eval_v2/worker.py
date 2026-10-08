"""One physical GPU, fixed adapters, frozen native SearchAgent, transparent hooks.

Invoked only with a live controller's immutable launch ticket. No training,
optimizer, backward, adapter merge, new generation defaults, or retries.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import append_json, atomic_json, checked_json, file_sha, load_support, object_sha, proc_identity
    from identity import VERSIONS, PHASES, check_adapter_labels, cohort, native_files, rank_slots, validate_config
    from gates import publication, require
    from hooks import capture_native, snapshot
else:
    from .utils import append_json, atomic_json, checked_json, file_sha, load_support, object_sha, proc_identity
    from .identity import VERSIONS, PHASES, check_adapter_labels, cohort, native_files, rank_slots, validate_config
    from .gates import publication, require
    from .hooks import capture_native, snapshot


def arguments():
    parser = argparse.ArgumentParser()
    for name in ("config", "config-sha256", "launch-ticket", "launch-ticket-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--worker-index", type=int, choices=range(3), required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    return parser.parse_args()


def gpu_memory(torch, physical_gpu, expected_uuid):
    raw = subprocess.check_output(["nvidia-smi", "-i", str(physical_gpu),
                                   "--query-gpu=uuid,memory.free,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
    uuid, free, used, total = [part.strip() for part in raw.split(",")]
    require(uuid == expected_uuid, "Actual assigned GPU UUID differs")
    cuda_free, cuda_total = torch.cuda.mem_get_info()
    return dict(sample_unix=time.time(), gpu_uuid=uuid, actual_free_mib=int(free), used_mib=int(used), total_mib=int(total),
                cuda_free_bytes=cuda_free, cuda_total_bytes=cuda_total,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(), torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def rng_snapshot(torch):
    return (random.getstate(), torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())


def restore_rng(torch, state):
    random.setstate(state[0])
    torch.set_rng_state(state[1])
    torch.cuda.set_rng_state(state[2])


def rng_digest(state):
    return dict(python=hashlib.sha256(repr(state[0]).encode()).hexdigest(),
                torch_cpu=hashlib.sha256(bytes(state[1].tolist())).hexdigest(),
                torch_cuda=hashlib.sha256(bytes(state[2].tolist())).hexdigest())


def decode_protocol(core, result, answers):
    # The frozen reward uses the last authored action span, not whole-transcript
    # tag balancing. Read precisely the same parser; no new scoring rule.
    from search_r1.qa_protocol import extract_final_answer
    spans = result["policy_spans"]
    last = result["response"][spans[-1][0]:spans[-1][1]] if spans else ""
    answer = extract_final_answer([last] if spans else [])
    score = core.reward(result["response"], answers, spans)
    if score not in (0.0, 1.0):
        raise ValueError("Frozen native EM must be binary")
    return dict(protocol_em=int(score), protocol_valid_final_action=answer is not None,
                extracted_final_answer=answer, gold_normalized=[core.normalize(text) for text in answers],
                native_last_authored_action=last,
                answer_tag_termination_is_not_equivalent_to_protocol_validity=True)


def main():
    args = arguments()
    config, config_identity = checked_json(dict(path=args.config, sha256=args.config_sha256))
    validate_config(config)
    require(os.getuid() == config["uid"], "Wrong actual worker UID")
    publication_proof = publication(config, Path(__file__).parent)
    support = load_support(config)
    ticket, ticket_identity = checked_json(dict(path=args.launch_ticket, sha256=args.launch_ticket_sha256))
    controller = ticket["controller"]
    current = proc_identity(controller["pid"])
    require(all(current[key] == controller[key] for key in ("pid", "uid", "starttick")), "Original controller no longer owns the launch")
    require(ticket["accepted"] is True and ticket["config_sha256"] == config_identity["sha256"] and ticket["phase"] == args.phase,
            "Unaccepted/mismatched immutable launch ticket")
    require(Path(args.output).resolve() == Path(ticket["phase_output"]).resolve(), "Worker output differs from controller ticket")
    require(os.environ.get("E1_RUN_ID") == ticket["run_id"], "Worker run identity differs")
    gpu = config["physical_gpus"][args.worker_index]
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu), "Root must uniquely assign this worker its GPU")
    rows, validation_proof = cohort(config)
    native_values, native_proof = native_files(config)
    weights = support["weight_identity"].check_weight_manifest(config["weight_manifest"], config["model"], config["adapters"])
    check_adapter_labels(weights)
    admission = support["admission"].RetrieverAdmission(config["retriever_admission"], config["retriever_url"])
    health = admission.check_live()
    core, opsd, rollout, science = support["common"].load_science(config["runtime"], with_torch=True)
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Worker must see exactly one CUDA device")
    require(torch.__version__.split("+")[0] == "2.4.0", "Different native torch version")
    import importlib.metadata
    for package, version in {"transformers": "4.46.3", "peft": "0.13.2"}.items():
        require(importlib.metadata.version(package) == version, "Different native package: " + package)
    # Read all fixed checkpoint tensors on CPU, before model initialization.
    tensor_proof = {}
    for version in VERSIONS:
        values = load_file(str(Path(config["adapters"][version]) / "adapter_model.safetensors"), device="cpu")
        require(len(values) == 504 and all(value.dtype == torch.float32 and torch.isfinite(value).all().item() for value in values.values()),
                "Fixed adapter is not finite504 FP32 tensors: " + version)
        b = [value for name, value in values.items() if "lora_B" in name]
        zero = len(b) == 252 and all(torch.count_nonzero(value).item() == 0 for value in b)
        require(version != "source100" or zero, "Source100 baseline B is not allzero")
        tensor_proof[version] = dict(tensors=504, dtype="torch.float32", finite=True, lora_B_count=len(b), lora_B_all_zero=zero)
        del values
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    random.seed(42)
    directory = Path(args.output) / f"worker-{args.worker_index}"
    directory.mkdir(parents=True, exist_ok=False)
    own = proc_identity(os.getpid())
    status = dict(schema="native512-m4-worker-v1", status="INITIALIZING", phase=args.phase, run_id=ticket["run_id"],
                  process=own, worker_index=args.worker_index, physical_gpu=gpu, gpu_uuid=config["gpu_uuids"][args.worker_index],
                  started_unix=time.time(), logical_ranks=rank_slots(args.worker_index), rank_seed=42,
                  versions=list(PHASES[args.phase]), completed_model_question_slots=0, no_training=True,
                  validation=validation_proof, native_files=native_proof, weights=weights, tensor_proof=tensor_proof,
                  science=science, publication=publication_proof, config=config_identity, launch_ticket=ticket_identity,
                  retriever=dict(admission.identity, startup_health=health))
    atomic_json(directory / "status.json", status)
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(config["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        require(base.generation_config.to_dict().get("repetition_penalty") == 1.05 and base.generation_config.to_dict().get("eos_token_id") == [151645, 151643],
                "Loaded actual GenerationConfig lost native repetition/EOS defaults")
        model = PeftModel.from_pretrained(base, config["adapters"]["source100"], adapter_name="source100", is_trainable=False)
        for version in VERSIONS[1:]:
            model.load_adapter(config["adapters"][version], adapter_name=version, is_trainable=False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        adapter_dtypes = {version: dict(Counter(str(parameter.dtype) for name, parameter in model.named_parameters()
                                               if "." + version + "." in name and "lora_" in name)) for version in VERSIONS}
        require(all(counts == {"torch.float32": 504} for counts in adapter_dtypes.values()), "Loaded fixed adapter dtype/count differs from native FP32")
        require(model.get_base_model().get_output_embeddings().weight.dtype == torch.bfloat16, "Base output head is not BF16")
        agent = rollout.SearchAgent(model, tokenizer, config["retriever_url"], max_new_tokens=500, max_context=8192)
        status.update(status="RUNNING", actual_generation_config=model.generation_config.to_dict(),
                      actual_loaded_adapter_dtype_counts=adapter_dtypes,
                      actual_base_output_head_dtype=str(model.get_base_model().get_output_embeddings().weight.dtype),
                      actual_cuda_flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                             cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
                      gpu=gpu_memory(torch, gpu, config["gpu_uuids"][args.worker_index]),
                      expected_model_question_slots=len(rank_slots(args.worker_index)) * 64 * 2)
        atomic_json(directory / "status.json", status)
        with (directory / "results.jsonl").open("x") as results:
            # Each logical rank preserves validation[rank::8] and the same42 seed.
            # Model versions use independent rank-local RNG streams. Greedy
            # consumption is recorded; null-effect checks verify that hook/plain
            # have the same post-state, not an assumed no-consumption property.
            for rank in rank_slots(args.worker_index):
                rank_rng = {}
                for version in PHASES[args.phase]:
                    torch.manual_seed(42)
                    random.seed(42)
                    rank_rng[version] = rng_snapshot(torch)
                for slot, index in enumerate(range(rank, 512, 8)):
                    row = rows[index]
                    for version in PHASES[args.phase]:
                        model.set_adapter(version)
                        for parameter in model.parameters():
                            parameter.requires_grad_(False)
                        model.eval()
                        restore_rng(torch, rank_rng[version])
                        admission.check_live()
                        case_directory = directory / f"rank-{rank}" / f"slot-{slot:02d}" / version
                        case_directory.mkdir(parents=True, exist_ok=False)
                        before_rng = rng_snapshot(torch)
                        native_plain = plain_rng = None
                        if slot < config["equivalence_cases_per_rank"]:
                            native_plain = agent.run(row["question"], temperature=0)
                            plain_rng = rng_snapshot(torch)
                            atomic_json(case_directory / "native-unhooked-result.json", dict(result=native_plain, protocol=decode_protocol(core, native_plain, row["answers"]),
                                                                                          before_rng=rng_digest(before_rng), after_rng=rng_digest(plain_rng),
                                                                                          excluded_from_512_denominator=True, no_parameter_updates=True))
                            restore_rng(torch, before_rng)
                            admission.check_live()
                        try:
                            with capture_native(agent, rollout.requests) as trace:
                                result = agent.run(row["question"], temperature=0)
                        except BaseException:
                            atomic_json(case_directory / "native-capture-failed.json", dict(trace=trace, question_id=row["question_id"],
                                                                                         validation_index=index, logical_rank=rank, rank_slot=slot,
                                                                                         version=version, excluded_from_completed_512_slots=True))
                            raise
                        after_rng = rng_snapshot(torch)
                        rank_rng[version] = after_rng
                        protocol = decode_protocol(core, result, row["answers"])
                        equivalence = dict(required=native_plain is not None, accepted=None)
                        if native_plain is not None:
                            accepted = object_sha(native_plain) == object_sha(result) and rng_digest(plain_rng) == rng_digest(after_rng)
                            equivalence = dict(required=True, accepted=accepted, plain_result_sha256=object_sha(native_plain),
                                               hooked_result_sha256=object_sha(result), plain_rng_after=rng_digest(plain_rng), hooked_rng_after=rng_digest(after_rng))
                            atomic_json(case_directory / "null-effect-equivalence.json", equivalence)
                            require(accepted, "Actual same-rank native/hooked result or RNG differs; preserve results and HOLD")
                        trace.update(kind="actual_autoregressive_generation", teacher_forced=False, question=row["question"], answers=row["answers"],
                                     question_id=row["question_id"], validation_index=index, logical_rank=rank, rank_slot=slot,
                                     rank_seed=42, version=version, native_result=result, protocol=protocol, equivalence=equivalence,
                                     model_generation_config=status["actual_generation_config"],
                                     rng_before=rng_digest(before_rng), rng_after=rng_digest(after_rng),
                                     greedy_rng_unchanged_before_after=rng_digest(before_rng) == rng_digest(after_rng),
                                     native_temperature=0, max_new_tokens=500, max_context=8192,
                                     native_result_unmodified=True, no_training=True)
                        trace_path = case_directory / "native-capture.json"
                        atomic_json(trace_path, trace)
                        record = dict(status="complete", version=version, validation_index=index, logical_rank=rank, rank_slot=slot,
                                      rank_seed=42, worker_index=args.worker_index, question_id=row["question_id"], dataset=row["data_source"],
                                      row_sha256=validation_proof["row_sha256"][index], row_semantic_sha256=validation_proof["row_semantic_sha256"][index],
                                      protocol_em=protocol["protocol_em"], protocol_valid_final_action=protocol["protocol_valid_final_action"],
                                      termination=result["termination"], search_calls=result["search_calls"],
                                      native_result_sha256=object_sha(result), trace=dict(path=str(trace_path), sha256=file_sha(trace_path)),
                                      null_effect_equivalence=equivalence, completed_unix=time.time())
                        append_json(results, record)
                        status.update(completed_model_question_slots=status["completed_model_question_slots"] + 1,
                                      last_slot=dict(rank=rank, slot=slot, validation_index=index, version=version), updated_unix=time.time(),
                                      gpu=gpu_memory(torch, gpu, config["gpu_uuids"][args.worker_index]))
                        atomic_json(directory / "status.json", status)
                        print(json.dumps(dict(version=version, index=index, em=record["protocol_em"], completed=status["completed_model_question_slots"])), flush=True)
        require(status["completed_model_question_slots"] == status["expected_model_question_slots"], "Worker incomplete logical slices")
        support["weight_identity"].check_weight_manifest(config["weight_manifest"], config["model"], config["adapters"])
        support["common"].load_science(config["runtime"], with_torch=True)
        cohort(config)
        native_files(config)
        publication(config, Path(__file__).parent)
        admission.check_live()
        status.update(status="COMPLETE", ended_unix=time.time(), results_sha256=file_sha(directory / "results.jsonl"),
                      final_source_weights_validation_science_publication_admission_verified=True,
                      results_fresh_after_all_checks=True)
        atomic_json(directory / "status.json", status)
    except BaseException as error:
        status.update(status="FAILED", failed_unix=time.time(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(directory / "status.json", status)
        raise


if __name__ == "__main__":
    main()
