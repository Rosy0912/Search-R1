import json,time,hashlib,subprocess,sys,shlex
from pathlib import Path
root=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/topd-env-20261003')
record_path=root/'snapshot-manifest.json'
print('WAITING_VERIFIED_ENV_SNAPSHOT',flush=True)
while not record_path.exists():time.sleep(20)
record=json.loads(record_path.read_text());p=Path(record['archive']);h=hashlib.sha256()
with p.open('rb') as f:
 for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
assert h.hexdigest()==record['sha256']
target='/dataset/shaoshitong/search-r1-rl/topd/cache/pinned-site-packages.tar.gz'
subprocess.run([sys.executable,'work/transfer_file.py',str(p),target],check=True)
code=r'''set -euo pipefail
root=/dataset/shaoshitong/search-r1-rl/topd
for session in topd-bootstrap-v2-20261003 topd-bootstrap-20261002; do
 if tmux has-session -t "$session" 2>/dev/null; then tmux send-keys -t "$session" C-c; fi
done
exec 9>"$root/gpu-job.lock"
flock -w 60 9
export PATH="$root/tools:$PATH" UV_CACHE_DIR="$root/cache/uv" TMPDIR="$root/tmp"
/usr/bin/python3.10 - <<'CHECK'
import tarfile,posixpath
p='/dataset/shaoshitong/search-r1-rl/topd/cache/pinned-site-packages.tar.gz'
with tarfile.open(p,'r|gz') as f:
 for m in f:
  n=posixpath.normpath(m.name)
  assert not n.startswith('/') and n!='..' and not n.startswith('../'),m.name
  if m.issym() or m.islnk():
   link=posixpath.normpath(posixpath.join(posixpath.dirname(n) if m.issym() else '',m.linkname))
   assert not link.startswith('/') and link!='..' and not link.startswith('../'),(n,m.linkname)
print('ARCHIVE_PATHS_SAFE')
CHECK
if [ -f "$root/logs/environment.ready" ]; then mv "$root/logs/environment.ready" "$root/logs/environment.ready.before-cache-install"; fi
uv venv --python /usr/bin/python3.10 --clear "$root/env"
tar -xzf "$root/cache/pinned-site-packages.tar.gz" -C "$root/env/lib/python3.10/site-packages"
find "$root/env/lib/python3.10/site-packages" -maxdepth 1 \( -name '__editable__*verl*' -o -name 'verl.egg-link' -o -name 'verl*.dist-info' \) -exec rm -rf {} +
uv pip install --python "$root/env/bin/python" --no-build-isolation --no-deps --offline -e "$root/repo"
export CUDA_VISIBLE_DEVICES="" SEARCH_R1_BASE="$root" SEARCH_R1_REPO="$root/repo" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$root/env/bin/python" "$root/repo/reproduction/h20/check_environment.py" > "$root/logs/environment-acceptance.json"
uv pip freeze --python "$root/env/bin/python" > "$root/logs/environment-freeze.txt"
uv pip check --python "$root/env/bin/python" > "$root/logs/environment-dependency-check.txt" 2>&1 || true
git -C "$root/repo" rev-parse HEAD > "$root/logs/environment-source-commit.txt"
touch "$root/logs/environment.ready"
printf 'ENVIRONMENT_PINNED_IMPORTS_PASS\n'
'''
r=subprocess.run([sys.executable,'work/remote_topd.py'],input=code,text=True)
if r.returncode:raise SystemExit(r.returncode)
record['validated_target']=True;record['validation']='strict check_environment versions/imports/dependencies; no GPU integration yet'
record_path.write_text(json.dumps(record,indent=2));Path('outputs/environment-snapshot-manifest.json').write_text(json.dumps(record,indent=2))
p.unlink();print('ENV_TARGET_VALIDATED_LOCAL_RELAY_CLEANED',flush=True)
