"""Resolve actual split artifacts by their manifest and verified bytes, never derive path."""
import hashlib,json
from pathlib import Path
CONTENT="fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524"
TRAIN_SHA="e5e774fd1f5777a8be91103facaccb711cfd9593d7563e9d5afe04e6b1a52bea"
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def resolve_train(root):
    root=Path(root).resolve()
    logged=json.loads((root/"logs/common-data-manifest.json").read_text())
    if logged["content_identity"]["sha256"]!=CONTENT or logged["sha256"]["train.parquet"]!=TRAIN_SHA:raise ValueError("Unexpected trusted split identity/hash")
    candidates=[]
    for directory in (root/"outputs/common-data",root/"assets-root/common-data"):
        for manifest in directory.glob("*/manifest.json"):
            data=json.loads(manifest.read_text())
            if data!=logged:continue
            train=(manifest.parent/"train.parquet").resolve()
            if not train.is_relative_to(root):raise ValueError("Split artifact escapes own root")
            if sha(train)!=TRAIN_SHA:raise ValueError("Split train bytes differ")
            candidates.append((train,manifest))
    if len(candidates)!=1:raise ValueError(f"Need exactly one verified current split artifact, got {len(candidates)}")
    return candidates[0]
