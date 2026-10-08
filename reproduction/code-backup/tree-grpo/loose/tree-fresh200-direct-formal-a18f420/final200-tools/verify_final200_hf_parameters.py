"""After-training CPU HF reload and parameter-only comparison on h100-9.

Standalone: no training imports, scheduling, export, inference, or dataset access.
Prepared code is not evidence. A success receipt exists only after real execution.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time

# Set before importing any model library; never allocate a CUDA device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

SOURCE = "e7236b7d8729482aad864419fe986066643f3be5"
PACKAGE_MANIFEST_SHA256 = "995819deb2d8ba85f4da92a4ab6ed7abc7eb4887e566cda730297f1e86305760"
OFFICIAL_IDENTITY = "744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0"
OFFICIAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
OFFICIAL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
DATA_IDENTITY = "0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24"
ROOT = Path("/dataset/shaoshitong/search-r1-rl/tree-grpo")
RUN_DIRECTORY = ROOT / "runs/tree-current200-continue-3333"
OUTER = 200
CHUNK_ELEMENTS = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def file_hash(path):
    before = fingerprint(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    require(fingerprint(path) == before, "File changed during hashing")
    return h.hexdigest(), before


def read_json(path, expected_sha):
    require(path.is_file(), "Required JSON does not exist")
    before = fingerprint(path)
    raw = path.read_bytes()
    require(fingerprint(path) == before, "JSON changed during reading")
    sha = hashlib.sha256(raw).hexdigest()
    require(sha == expected_sha, "JSON SHA pin differs")
    return json.loads(raw), sha


def hf_identity(directory):
    require(directory.is_dir(), "HF directory does not exist")
    require((directory / "config.json").is_file()
            and (directory / "tokenizer_config.json").is_file(),
            "HF config/tokenizer files are incomplete")
    hashes, fingerprints = {}, {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        require(path.resolve().is_relative_to(directory.resolve()), "HF file escapes directory")
        name = str(path.relative_to(directory))
        hashes[name], fingerprints[name] = file_hash(path)
    # Match current source select_teacher.model_manifest's exact JSON hash rule.
    sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"files": hashes, "sha256": sha}, fingerprints


def unchanged(directory, fingerprints):
    current = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    require(current == set(fingerprints), "HF file set changed during loading")
    require(all(fingerprint(directory / name) == value
                for name, value in fingerprints.items()), "HF file changed during loading")


def safe_weight_map(directory, safe_open):
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text()).get("weight_map")
        require(isinstance(mapping, dict) and mapping, "Invalid safetensors shard index")
    else:
        path = directory / "model.safetensors"
        require(path.is_file(), "Complete safetensors HF export is required")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            mapping = {name: "model.safetensors" for name in handle.keys()}
    observed = {}
    for filename in sorted(set(mapping.values())):
        path = (directory / filename).resolve()
        require(path.is_relative_to(directory.resolve()) and path.is_file(),
                "Missing/outside safetensors shard")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                require(name not in observed, "Duplicate serialized tensor key")
                observed[name] = filename
    require(observed == mapping, "Shard tensor keys differ from index")
    return mapping


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run(args, report):
    report["stage"] = "host_and_metadata_binding"
    require(socket.gethostname().split(".")[0] == "h100-9", "Only h100-9 may read model assets")
    require(1 <= args.threads <= 192, "Invalid CPU thread count")
    require(args.package_sha256 == PACKAGE_MANIFEST_SHA256, "Current V9 package SHA pin differs")
    require(len(args.config_sha256) == 64
            and all(character in "0123456789abcdef" for character in args.config_sha256),
            "Actual native3333 config SHA256 is required; placeholders are not accepted")
    package, package_sha = read_json(args.package_manifest, args.package_sha256)
    require(package.get("packet_revision") == 9 and package.get("code_commit") == SOURCE
            and package.get("scope") == "CURRENT_FRESH200_CHAIN_CONTINUATION"
            and package.get("training_protocol") == "tree_fresh200_five_leaf_v1"
            and package.get("execution_mode") == "HUMAN_AUTHORIZED_DIRECT_TRAIN200",
            "Package is not the currente723/V9 fresh200 chain")
    require(package["model_identity"]["sha256"] == OFFICIAL_IDENTITY
            and package["common_data_identity"] == DATA_IDENTITY, "Official model/data pin differs")
    run_directory = args.run_directory.resolve()
    require(run_directory == RUN_DIRECTORY.resolve(), "Run is not the own current3333 chain")
    job_id = run_directory.name.rsplit("-", 1)[1]
    require(job_id.isdigit(), "Run job identity is invalid")
    base = Path(package["official_model_path"]).resolve()
    checkpoint = run_directory / "checkpoints/actor" / f"global_step_{OUTER}"
    state_path = run_directory / "checkpoints/training_state" / f"global_step_{OUTER}" / "manifest.json"
    saved, state_sha = read_json(state_path, args.final_state_manifest_sha256)
    identity = saved["identity"]
    expected = {"job_id": job_id, "run_id": run_directory.name,
                "code_commit": SOURCE, "config_sha256": args.config_sha256,
                "data_identity": DATA_IDENTITY, "model_identity": OFFICIAL_IDENTITY,
                "completed_step": OUTER, "actual_update_limit": OUTER}
    require(saved.get("status") == "COMPLETE_SAME_WORLD_STATE_SAVED"
            and all(identity.get(key) == value for key, value in expected.items())
            and saved["driver"]["next_step"] == OUTER + 1
            and Path(saved["hf_checkpoint"]).resolve() == checkpoint.resolve(),
            "Saved final boundary/source/HF reference differs")
    require(args.output.resolve().is_relative_to(run_directory)
            and not args.output.exists(), "Output must be a new JSON in this run")
    report["provenance"] = dict(expected,
        package_manifest={"path": str(args.package_manifest.resolve()), "sha256": package_sha},
        final_state_manifest={"path": str(state_path), "sha256": state_sha},
        official_model=OFFICIAL_MODEL, official_revision=OFFICIAL_REVISION,
        base=str(base), checkpoint=str(checkpoint), node="h100-9",
        config_sha256_scope="Caller supplies actual native3333 value, matched exactly against permanent final state identity; no parent config pin is substituted",
        completed_step_scope="Saved-manifest boundary binding; actual200 accounting remains the V9 native final ledger/terminal receipt")

    report["stage"] = "full_file_identity"
    base_identity, base_fingerprints = hf_identity(base)
    checkpoint_identity, checkpoint_fingerprints = hf_identity(checkpoint)
    require(base_identity == package["model_identity"], "Offline official directory bytes differ from V9 pin")
    require(checkpoint_identity == saved["hf_identity"], "Saved HF bytes differ from final manifest")
    report["official_file_identity"] = base_identity
    report["checkpoint_file_identity"] = checkpoint_identity

    import torch
    import transformers
    import safetensors
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    report["library_versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                                  "safetensors": safetensors.__version__}
    report["stage"] = "offline_tokenizer_and_architecture"
    base_tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    require(base_tokenizer.get_vocab() == tokenizer.get_vocab(), "Tokenizer vocabulary differs")
    require(all(getattr(base_tokenizer, key) == getattr(tokenizer, key)
                for key in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")),
            "Tokenizer special-token identities differ")
    require(base_tokenizer.special_tokens_map == tokenizer.special_tokens_map,
            "Tokenizer special-token semantics differ")
    require(json.loads(base_tokenizer.backend_tokenizer.to_str())
            == json.loads(tokenizer.backend_tokenizer.to_str()), "Fast-tokenizer semantics differ")
    report["tokenizer"] = {"vocabulary_equal": True, "special_tokens_equal": True,
        "fast_tokenizer_semantics_equal": True, "vocabulary_entries": len(tokenizer.get_vocab()),
        "vocabulary_sha256": canonical_hash(tokenizer.get_vocab()),
        "special_tokens_sha256": canonical_hash(tokenizer.special_tokens_map)}
    base_config = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    checkpoint_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    architecture_fields = ("model_type", "architectures", "vocab_size", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_act", "max_position_embeddings",
        "rms_norm_eps", "rope_theta", "rope_scaling", "tie_word_embeddings", "attention_bias", "attention_dropout")
    require(all(getattr(base_config, key, None) == getattr(checkpoint_config, key, None)
                for key in architecture_fields), "Official and saved architecture differ")

    report["stage"] = "full_saved_hf_cpu_reload"
    model, loading = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, attn_implementation="eager", output_loading_info=True)
    loading_counts = {key: len(loading.get(key, []))
                      for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    require(not any(loading_counts.values()), "Incomplete/incompatible saved HF reload")
    require(all(parameter.device.type == "cpu" for parameter in model.parameters())
            and not torch.cuda.is_initialized(), "Reload must remain entirely on CPU")
    model.eval()
    report["full_hf_reloaded"] = True
    report["hf_loading_info_counts"] = loading_counts

    report["stage"] = "all_unique_model_parameter_values"
    base_map = safe_weight_map(base, safe_open)
    checkpoint_map = safe_weight_map(checkpoint, safe_open)
    aliases = {}
    named_aliases = list(model.named_parameters(remove_duplicate=False))
    for name, parameter in named_aliases:
        aliases.setdefault(id(parameter), []).append(name)
    counts = dict(unique_parameter_tensors=0, parameter_elements=0, changed_parameter_tensors=0,
                  changed_parameter_elements=0, logical_parameter_names=len(named_aliases),
                  excluded_buffer_tensors=len(list(model.named_buffers())), tied_parameter_aliases=0)
    max_delta, sum_abs_delta, sum_squared_delta = 0.0, 0.0, 0.0
    loaded_hash, reference_hash = hashlib.sha256(), hashlib.sha256()
    supported = (torch.float16, torch.bfloat16, torch.float32)
    for name, parameter in model.named_parameters():
        names = aliases[id(parameter)]
        base_names = [key for key in names if key in base_map]
        checkpoint_names = [key for key in names if key in checkpoint_map]
        require(len(base_names) == len(checkpoint_names) == 1,
                "Missing/ambiguous serialized parameter alias; cannot claim complete comparison")
        require(parameter.dtype == torch.float32, "CPU reload parameter dtype differs")
        with safe_open(str(base / base_map[base_names[0]]), framework="pt", device="cpu") as before_file, \
                safe_open(str(checkpoint / checkpoint_map[checkpoint_names[0]]), framework="pt", device="cpu") as saved_file:
            before = before_file.get_tensor(base_names[0])
            serialized = saved_file.get_tensor(checkpoint_names[0])
            require(before.shape == serialized.shape == parameter.shape
                    and before.dtype in supported and serialized.dtype in supported,
                    "Parameter shape/dtype is unsupported or differs")
            after = parameter.detach().reshape(-1)
            before = before.reshape(-1)
            serialized = serialized.reshape(-1)
            header = json.dumps([name, list(parameter.shape), "float32"], separators=(",", ":")).encode()
            loaded_hash.update(header + b"\0")
            reference_hash.update(header + b"\0")
            changed_in_tensor = 0
            for start in range(0, after.numel(), CHUNK_ELEMENTS):
                stop = min(start + CHUNK_ELEMENTS, after.numel())
                current = after[start:stop]
                original = before[start:stop].to(torch.float32)
                stored = serialized[start:stop].to(torch.float32)
                require(bool(torch.isfinite(current).all()) and bool(torch.isfinite(original).all())
                        and bool(torch.isfinite(stored).all()), "Nonfinite model parameter encountered")
                require(torch.equal(current, stored), "Cold loaded parameter differs from serialized HF values")
                changed_in_tensor += int(torch.count_nonzero(current != original))
                difference = current.to(torch.float64) - original.to(torch.float64)
                absolute = difference.abs()
                max_delta = max(max_delta, float(absolute.max()))
                sum_abs_delta += float(absolute.sum())
                sum_squared_delta += float(difference.square().sum())
                loaded_hash.update(current.contiguous().view(torch.uint8).numpy().tobytes())
                reference_hash.update(original.contiguous().view(torch.uint8).numpy().tobytes())
            counts["unique_parameter_tensors"] += 1
            counts["parameter_elements"] += after.numel()
            counts["changed_parameter_tensors"] += int(changed_in_tensor > 0)
            counts["changed_parameter_elements"] += changed_in_tensor
            counts["tied_parameter_aliases"] += len(names) - 1
    require(counts["unique_parameter_tensors"] > 0 and counts["changed_parameter_elements"] > 0
            and max_delta > 0 and all(math.isfinite(x) for x in (max_delta, sum_abs_delta, sum_squared_delta)),
            "No finite nonzero model parameter change from official weights")
    report["parameter_comparison"] = dict(counts, all_model_parameters_finite=True,
        all_loaded_parameters_equal_serialized_saved_values=True, nonzero_parameter_delta_observed=True,
        max_abs_parameter_delta=max_delta, mean_abs_parameter_delta=sum_abs_delta / counts["parameter_elements"],
        l2_parameter_delta=math.sqrt(sum_squared_delta), loaded_parameter_values_sha256=loaded_hash.hexdigest(),
        official_parameter_values_sha256=reference_hash.hexdigest(), comparison_dtype="float64_difference_of_exact_fp32_values",
        scope="Unique named_parameters only; buffers, optimizer and duplicate tied aliases excluded")
    unchanged(base, base_fingerprints)
    unchanged(checkpoint, checkpoint_fingerprints)
    report.update(status="FINAL200_HF_CPU_PARAMETER_VALUES_VERIFIED", stage="complete", success=True,
        all_model_parameters_finite=True, nonzero_parameter_delta_observed=True,
        model_file_fingerprints_unchanged_during_run=True, gpu_allocated_by_helper=False,
        inference_or_evaluation_executed=False, optimizer_or_rng_restore_verified=False,
        exact_executor_resume=False, whole_training_goal_accepted_by_this_helper=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--final-state-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    started, cpu_started = time.perf_counter(), time.process_time()
    report = {"schema_version": 1, "status": "FINAL200_HF_CPU_PARAMETER_VALUES_FAILED",
              "success": False, "full_hf_reloaded": False,
              "all_model_parameters_finite": None, "nonzero_parameter_delta_observed": None,
              "execution_scope": "Actual CPU process only; aggregate metadata, no tensor/token/dataset output"}
    exit_code = 0
    try:
        run(args, report)
    except Exception as error:
        report["failure_type"] = type(error).__name__
        report["failure_scope"] = "The named stage did not complete; no success/whole-goal acceptance is implied"
        exit_code = 1
    report["wall_seconds"] = time.perf_counter() - started
    report["driver_process_cpu_seconds"] = time.process_time() - cpu_started
    report["driver_process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report["measurement_scope"] = "Linux helper process and its CPU threads; not GPU use, Slurm capacity, training cost, or host peak memory"
    # Node/output checks are repeated for publication even if an earlier stage failed.
    if socket.gethostname().split(".")[0] != "h100-9" or not args.output.resolve().is_relative_to(args.run_directory.resolve()):
        print(json.dumps(report, allow_nan=False))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(report, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
