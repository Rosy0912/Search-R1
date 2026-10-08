"""CPU provenance and atomic output helpers for independent OPSD diagnostics."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

SCIENCE_HASHES = {
    "search_e1/core.py": "0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc",
    "search_e1/opsd.py": "61df9cc630bca7466f3edfb06da18ad92ddff1c713bf7c6c4a104ba88672c037",
    "search_e1/rollout.py": "a88506797ea7fe24e61e1ad53ba4487aef87bdd69ec395aef26c94973b3d629a",
    "upstream/Search-R1/search_r1/qa_protocol.py": "9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd",
}


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(stream, value):
    stream.write(canonical_bytes(value).decode() + "\n")
    stream.flush()


def science_identity(runtime):
    runtime = Path(runtime).resolve()
    result = {}
    for relative, expected in SCIENCE_HASHES.items():
        path = runtime / relative
        identity = stat_identity(path)
        actual = file_sha(path)
        if actual != expected or stat_identity(path) != identity:
            raise ValueError("Frozen scientific source mismatch: " + str(path))
        result[relative] = dict(identity, sha256=actual)
    return result


def load_science(runtime, with_torch=False):
    """Only the actual hashed frozen runtime is importable, never archive/core."""
    runtime = Path(runtime).resolve()
    identity = science_identity(runtime)
    sys.path.insert(0, str(runtime / "upstream/Search-R1"))
    sys.path.insert(0, str(runtime))
    core = importlib.import_module("search_e1.core")
    if Path(core.__file__).resolve() != runtime / "search_e1/core.py":
        raise RuntimeError("A different search_e1 module was already imported")
    if not with_torch:
        return core, identity
    opsd = importlib.import_module("search_e1.opsd")
    rollout = importlib.import_module("search_e1.rollout")
    for module, relative in [(opsd, "search_e1/opsd.py"), (rollout, "search_e1/rollout.py")]:
        if Path(module.__file__).resolve() != runtime / relative:
            raise RuntimeError("A different frozen scientific module was imported")
    return core, opsd, rollout, identity


def immutable_rows(path):
    """Yield exact row/byte identities and reject changes during the complete scan."""
    path = Path(path)
    before = stat_identity(path)
    digest = hashlib.sha256()
    offset = 0
    count = 0
    with path.open("rb") as stream:
        for count, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError("Immutable input contains a partial line: " + str(path))
            digest.update(line)
            row = json.loads(line)
            provenance = dict(line=count, byte_offset=offset, bytes=len(line),
                              row_sha256=hashlib.sha256(line).hexdigest(), semantic_sha256=object_sha(row))
            offset += len(line)
            yield row, provenance
    if stat_identity(path) != before:
        raise ValueError("Immutable input changed during scan: " + str(path))
    return dict(before, sha256=digest.hexdigest(), rows=count)


def scan(path, visit):
    rows = immutable_rows(path)
    while True:
        try:
            row, provenance = next(rows)
        except StopIteration as finished:
            return finished.value
        visit(row, provenance)
