"""CPU-only provenance helpers. Importing the controller never imports CUDA."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checked_json(reference):
    path = Path(reference["path"]).resolve()
    before = stat_identity(path)
    actual = file_sha(path)
    if actual != reference["sha256"]:
        raise ValueError("Receipt/source hash mismatch: " + str(path))
    value = json.loads(path.read_text())
    if before != stat_identity(path):
        raise ValueError("Receipt changed during read: " + str(path))
    return value, dict(before, sha256=actual)


def append_json(stream, value):
    stream.write(canonical(value).decode() + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def load_support(config):
    """Bind reused CPU identity helpers to the already published M1 source."""
    directory = Path(config["diagnostic_source"]).resolve()
    manifest, _ = checked_json(config["debug_source_manifest"])
    if not isinstance(manifest, dict) or len(manifest) != 15:
        raise ValueError("Require the original flat 15-file M1 source manifest")
    for name, digest in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
            raise ValueError("Unsafe diagnostic source manifest entry")
        if file_sha(directory / relative) != digest:
            raise ValueError("Published M1 source changed: " + name)
    sys.path.insert(0, str(directory))
    modules = {}
    for name in ("common", "weight_identity", "admission"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != directory / (name + ".py"):
            raise ValueError("A different support module was imported: " + name)
        modules[name] = module
    return modules


def proc_identity(pid):
    """Missing/reused PID is distinguished by the caller using starttick/UID."""
    directory = Path("/proc") / str(pid)
    raw = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
    return dict(pid=int(pid), uid=directory.stat().st_uid, ppid=int(raw[1]), pgid=int(raw[2]),
                starttick=int(raw[19]), state=raw[0],
                argv=(directory / "cmdline").read_bytes().decode(errors="replace").split("\0")[:-1])


def environment(pid):
    raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    return dict(item.decode(errors="replace").split("=", 1) for item in raw.split(b"\0") if b"=" in item)
