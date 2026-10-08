set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
"$R/env/bin/python" - <<'PY'
import tarfile,pathlib,hashlib,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo')
run=root/'runs/bapo-smoke2-2917'
assert (run/'TRAIN_RETURNED_SUCCESSFULLY').is_file()
assert not json.loads((run/'own-process-cleanup.json').read_text())['remaining']
out=root/'logs/bapo-smoke2-2917-raw-evidence.tar.gz'
assert not out.exists()
with tarfile.open(out,'w:gz') as tar:
 for path in sorted(run.rglob('*')):
  rel=path.relative_to(run)
  if rel.parts[0] in ('runtime','output','cache') or path.is_symlink() or not path.is_file():continue
  tar.add(path,arcname=str(path.relative_to(run.parent)),recursive=False)
print(out,out.stat().st_size,hashlib.sha256(out.read_bytes()).hexdigest())
PY
