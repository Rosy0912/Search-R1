"""Read-only validation of root's fresh CPU weight-manifest attestation."""
import json
from pathlib import Path

if __package__:
    from .common import file_sha, stat_identity
else:
    from common import file_sha, stat_identity

VERSIONS = ("source100", "update1", "update5", "epoch1")


def check_weight_manifest(path, model_path, adapter_paths):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    source = manifest["source100"]
    model = Path(model_path).resolve()
    if Path(source["path"]).resolve() != model:
        raise ValueError("Fixed source100 model path mismatch")
    if file_sha(model / "config.json") != source["config_sha256"]:
        raise ValueError("Fixed source100 configuration changed")
    verified = []
    for entry in source["files"]:
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source inventory relative path")
        current = stat_identity(model / relative)
        for key in ("bytes", "inode", "mtime_ns", "ctime_ns", "device"):
            if current[key] != entry[key]:
                raise ValueError("Source100 file metadata changed after root's content hash: " + str(relative))
        if not entry.get("sha256") or not entry.get("sha256_origin"):
            raise ValueError("Missing source100 content hash origin")
        verified.append(dict(current, sha256=entry["sha256"], sha256_origin=entry["sha256_origin"],
                             worker_fresh_content_hash=False, worker_check="exact metadata matched supplied fresh CPU content-hash attestation"))
    if not verified:
        raise ValueError("Empty source100 file inventory")
    adapters = {}
    for version in VERSIONS:
        adapter = Path(adapter_paths[version]).resolve()
        expected = manifest["adapters"][version]
        if adapter != Path(expected["path"]).resolve():
            raise ValueError("Fixed adapter path mismatch: " + version)
        tensor_path, config_path = adapter / "adapter_model.safetensors", adapter / "adapter_config.json"
        before = [stat_identity(tensor_path), stat_identity(config_path)]
        tensor_sha, config_sha = file_sha(tensor_path), file_sha(config_path)
        if tensor_sha != expected["adapter_model_sha256"] or config_sha != expected["adapter_config_sha256"]:
            raise ValueError("Fixed adapter content mismatch: " + version)
        if before != [stat_identity(tensor_path), stat_identity(config_path)]:
            raise ValueError("Adapter changed during CPU hash")
        config = json.loads(config_path.read_text())
        if config.get("r") != 16 or config.get("lora_alpha") != 32 or config.get("lora_dropout") != 0.0:
            raise ValueError("Adapter configuration differs from actual fixed OPSD")
        adapters[version] = dict(path=str(adapter), adapter_model_sha256=tensor_sha,
                                 adapter_config_sha256=config_sha, fresh_cpu_hash=True, files=before)
    return dict(manifest=dict(path=str(path), sha256=file_sha(path)), model_path=str(model),
                config_sha256=source["config_sha256"], source_files=verified, adapters=adapters)
