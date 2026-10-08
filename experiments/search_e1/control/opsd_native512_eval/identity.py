"""Pinned native cohort, generation defaults, and complete merge invariants."""
from collections import Counter
import json
from pathlib import Path

try:
    from .utils import checked_json, file_sha, object_sha, stat_identity
except ImportError:
    from utils import checked_json, file_sha, object_sha, stat_identity

VERSIONS = ("source100", "update1", "update5", "epoch1")
PHASES = {"endpoints": ("source100", "epoch1"), "early": ("update1", "update5")}
PINNED = {
    "validation": "75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76",
    "data_manifest": "c67f5ecba92a04869a2a26c6e23b39a65219a1320b517fddf1e691470c33c164",
    "native_config": "84a6167b9216c65fe0e810d03a8489f263d7feccddface605fa0ab6e726e8640",
    "native_history": "44b2776b9a278518dc9c459cae0f224e5ed1a5cd47f8da1fc9ace05d836d7aa9",
    "generation_config": "5cbc3d054b73e70801535a65ffc9b427481257fa29f896e48f68e9ea3c285b96",
}
ADAPTER_SHA = {
    "source100": "8cbebc6cb10203bf93a2c8b7271d31020ab1d1e7f1958a6f94f5890af3206c3d",
    "update1": "a36ba7b25c6d57083fcd1ab5a699f4aedeca52d8a8aa2fb4d994647f791aee01",
    "update5": "44efab6b2a7e92fec4237275dada2e27b47d665e7162220cdc9db6bf594d3157",
    "epoch1": "8eef75aafa06eb98dc21a6b0ab85195800fc373692d633faa0ecf461fa45afd0",
}
ADAPTER_CONFIG_SHA = "6f5704adcea5a0c8498ff7a789c456b9ab774255e9571a4118ce966b3ba61783"
GPU_UUIDS = ["GPU-1b18ee4f-7486-4183-bf3e-3290d8133db6", "GPU-0f337443-2d5c-cac7-51e9-1f628f90a2bf", "GPU-2d1c2f87-07d3-0c7a-ef9c-20f7ee656e45"]


def validate_config(config):
    if config["schema"] != "native512-m4-config-v1":
        raise ValueError("Unknown M4 configuration schema")
    fixed = dict(uid=1005, physical_gpus=[4, 5, 6], gpu_uuids=GPU_UUIDS,
                 logical_world_size=8, rank_seed=42, max_new_tokens=500, max_context=8192,
                 temperature=0, equivalence_cases_per_rank=1,
                 historical_correct={"source100": 239, "epoch1": 203})
    for key, expected in fixed.items():
        if config[key] != expected:
            raise ValueError("Fixed native evaluation setting differs: " + key)
    if config.get("allow_fallback_18004", False):
        raise ValueError("This M4 plan has no automatic 18004 fallback")
    if config["retriever_url"] != "http://10.120.17.101:18005/retrieve":
        raise ValueError("M4 requires one admitted resident endpoint for all four versions")
    if set(config["adapters"]) != set(VERSIONS):
        raise ValueError("Exactly four fixed adapters are required")
    if config.get("minimum_free_mib", 20000) < 20000 or config.get("minimum_disk_bytes", 100 << 30) < 100 << 30:
        raise ValueError("Resource startup gate was weakened")
    for name, digest in PINNED.items():
        if config[name]["sha256"] != digest:
            raise ValueError("Historical native file identity differs: " + name)


def cohort(config):
    reference = config["validation"]
    path = Path(reference["path"])
    before = stat_identity(path)
    if file_sha(path) != reference["sha256"]:
        raise ValueError("Native 512 validation bytes changed")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Partial native validation input")
    lines = raw.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    validate_cohort_rows(rows)
    if before != stat_identity(path):
        raise ValueError("Validation changed during read")
    proof = dict(before, sha256=reference["sha256"], rows=512, unique_question_ids=512,
                 datasets=dict(Counter(row["data_source"] for row in rows)),
                 rank_indices=[list(range(rank, 512, 8)) for rank in range(8)],
                 row_sha256=[__import__("hashlib").sha256(line).hexdigest() for line in lines],
                 row_semantic_sha256=[object_sha(row) for row in rows])
    return rows, proof


def validate_cohort_rows(rows):
    if len(rows) != 512 or len({row["question_id"] for row in rows}) != 512 or len({row["question"] for row in rows}) != 512:
        raise ValueError("Native validation must contain 512 unique IDs and questions")
    if Counter(row["data_source"] for row in rows) != Counter(nq=256, hotpotqa=256):
        raise ValueError("Native cohort dataset balance changed")
    for row in rows:
        if set(row) != {"question_id", "question", "answers", "data_source"}:
            raise ValueError("Unexpected native validation row fields")
        if not row["answers"]:
            raise ValueError("Empty gold answers")


def native_files(config):
    values, proof = {}, {}
    for name in ("data_manifest", "native_config", "native_history", "generation_config"):
        values[name], proof[name] = checked_json(config[name])
    manifest = values["data_manifest"]
    if manifest["validation"] != 512 or manifest["train"] != 169103 or manifest["seed"] != 42:
        raise ValueError("Original heldout data manifest counts changed")
    native = values["native_config"]
    if Path(native["model"]).resolve() != Path(config["model"]).resolve() or Path(native["validation"]).resolve() != Path(config["validation"]["path"]).resolve():
        raise ValueError("Current source/cohort are not the original native config paths")
    if (native["world_size"], native["seed"], native["max_context"], native["rank"], native["alpha"]) != (8, 42, 8192, 16, 32):
        raise ValueError("Original native rank/seed/context/adapter settings changed")
    history = values["native_history"]
    if len(history) != 2 or history[0]["epoch"] != 0 or history[1]["epoch"] != 1:
        raise ValueError("Original native history is not baseline plus epoch1")
    for entry, version in zip(history, ("source100", "epoch1")):
        if entry["validation_em"] != config["historical_correct"][version] / 512:
            raise ValueError("Historical native denominator/count changed")
    generation = values["generation_config"]
    if generation["repetition_penalty"] != 1.05 or generation["eos_token_id"] != [151645, 151643]:
        raise ValueError("Inherited repetition penalty / EOS identity changed")
    proof["generation_defaults_policy"] = "Load actual model GenerationConfig; native run only supplies its original kwargs, including do_sample=False. Preserve repetition_penalty=1.05 and EOS list."
    return values, proof


def check_adapter_labels(weights):
    if len(weights["source_files"]) != 12 or weights["config_sha256"] != "5177b12444b69891bdb8e1cf8e6a037b25c293306ae283b8f5ac3886ef3ce903":
        raise ValueError("Fixed source100 twelve-file/config identity differs")
    for version, digest in ADAPTER_SHA.items():
        actual = weights["adapters"][version]
        if actual["adapter_model_sha256"] != digest or actual["adapter_config_sha256"] != ADAPTER_CONFIG_SHA:
            raise ValueError("Fixed checkpoint label mismatch: " + version)


def rank_slots(worker):
    return [rank for rank in range(8) if rank % 3 == worker]


def summarize_records(records, rows, versions):
    """No approximate denominators, duplicated questions, or missing-rank EM."""
    found = {}
    for record in records:
        version, index = record["version"], record["validation_index"]
        if version not in versions or not isinstance(index, int) or not 0 <= index < 512:
            raise ValueError("Unknown version/index in M4 records")
        key = version, index
        if key in found:
            raise ValueError("Duplicate model/question slot")
        if record["question_id"] != rows[index]["question_id"] or record["row_semantic_sha256"] != object_sha(rows[index]):
            raise ValueError("Record/native validation identity differs")
        if record["logical_rank"] != index % 8 or record["rank_slot"] != index // 8 or record["worker_index"] != (index % 8) % 3:
            raise ValueError("Original eight-rank slice or physical mapping changed")
        if record["rank_seed"] != 42 or record["protocol_em"] not in (0, 1) or record["status"] != "complete":
            raise ValueError("Incomplete/native invalid score record")
        found[key] = record
    expected = {(version, index) for version in versions for index in range(512)}
    if set(found) != expected:
        raise ValueError("Require every model's full 512 unique question slots")
    statistics = {}
    for version in versions:
        correct = sum(found[version, index]["protocol_em"] for index in range(512))
        statistics[version] = dict(correct=correct, denominator=512, em=correct / 512,
                                   logical_rank_counts=[sum(found[version, index]["protocol_em"] for index in range(rank, 512, 8)) for rank in range(8)])
    paired = {}
    if "source100" in versions:
        for version in versions:
            if version == "source100":
                continue
            counts = Counter((found["source100", index]["protocol_em"], found[version, index]["protocol_em"]) for index in range(512))
            paired[version] = {"correct_to_wrong": counts[1, 0], "wrong_to_correct": counts[0, 1],
                               "correct_to_correct": counts[1, 1], "wrong_to_wrong": counts[0, 0]}
    return dict(schema="native512-m4-counts-v1", rows=len(records), versions=statistics, paired_against_source100=paired,
                every_model_denominator=512, identical_question_slots=True,
                historical_question_correct_sets_unavailable=True)
