"""CPU-only preparation of explicit native512 config, after source publication.

Does not check HTTP/CUDA or start a process. Roots are intentionally restricted
to this authorized HKUST1 experiment; proof paths must already exist.
"""
import argparse
import json
from pathlib import Path

try:
    from .utils import atomic_json, file_sha
    from .identity import GPU_UUIDS, PINNED, validate_config
except ImportError:
    from utils import atomic_json, file_sha
    from identity import GPU_UUIDS, PINNED, validate_config

ORIGINAL = Path("/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007")
DEBUG = Path("/data/shaoshitong/search-e1-opsd-debug-20261008")


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-release-receipt", required=True)
    parser.add_argument("--publication-proof", required=True)
    parser.add_argument("--retriever-admission", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.parent != ORIGINAL.parent or not output_root.name.startswith("search-e1-opsd-native512-"):
        raise ValueError("Independent native512 output must remain under the authorized HKUST1 /data/shaoshitong scope")
    files = dict(validation=ORIGINAL / "data/validation.jsonl", data_manifest=ORIGINAL / "data/manifest.json",
                 native_config=ORIGINAL / "runs/opsd/config.json", native_history=ORIGINAL / "runs/opsd/history.json",
                 generation_config=ORIGINAL / "assets/archive100/generation_config.json")
    config = dict(schema="native512-m4-config-v1", uid=1005, original_root=str(ORIGINAL), debug_root=str(DEBUG),
                  output_root=str(output_root), python=str(ORIGINAL / "env/bin/python"),
                  diagnostic_source=str(DEBUG / "source/opsd_debug"), runtime=str(ORIGINAL / "runtime/opsd-preparation3-xformers"),
                  model=str(ORIGINAL / "assets/archive100"), weight_manifest=str(DEBUG / "receipts/weight-manifest.json"),
                  adapters={version: str(ORIGINAL / "runs/opsd" / directory) for version, directory in
                            dict(source100="best_adapter", update1="adapter_update_1", update5="adapter_update_5", epoch1="adapter_epoch_1").items()},
                  retriever_url="http://10.120.17.101:18005/retrieve", allow_fallback_18004=False,
                  retriever_admission=str(Path(args.retriever_admission).resolve()),
                  debug_source_manifest=reference(DEBUG / "receipts/gpu-source-manifest.json"),
                  helper_source_manifest=reference(Path(__file__).parent / "source-manifest.json"),
                  publication_proof=reference(args.publication_proof), m1_release_receipt=reference(args.m1_release_receipt),
                  physical_gpus=[4, 5, 6], gpu_uuids=GPU_UUIDS, logical_world_size=8, rank_seed=42,
                  max_new_tokens=500, max_context=8192, temperature=0, equivalence_cases_per_rank=1,
                  minimum_free_mib=20000, minimum_disk_bytes=100 << 30,
                  historical_correct={"source100": 239, "epoch1": 203}, no_training=True)
    config.update({name: reference(path) for name, path in files.items()})
    validate_config(config)
    atomic_json(args.output_config, config)
    print(json.dumps(dict(config=str(Path(args.output_config).resolve()), sha256=file_sha(args.output_config))))


if __name__ == "__main__":
    main()
