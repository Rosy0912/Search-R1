set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo
RUN=$ROOT/runs/bapo-smoke2-2882
OUT=$ROOT/logs/bapo-smoke2-2882-protocol-cancelled.tar.gz
"$ROOT/env/bin/python" - <<'PY'
import tarfile, pathlib,hashlib
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo')
run=root/'runs/bapo-smoke2-2882'
out=root/'logs/bapo-smoke2-2882-protocol-cancelled.tar.gz'
with tarfile.open(out,'w:gz') as tar:
 for path in sorted(run.rglob('*')):
  rel=path.relative_to(run)
  if rel.parts[0] in ('runtime','output','cache') or path.is_symlink() or not path.is_file():continue
  tar.add(path,arcname=str(path.relative_to(run.parent)),recursive=False)
print(out, out.stat().st_size,hashlib.sha256(out.read_bytes()).hexdigest())
print((run/'own-process-cleanup.json').read_text())
PY
