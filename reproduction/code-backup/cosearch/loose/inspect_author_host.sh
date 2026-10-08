set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
hostname
squeue -h -j 3002 -o '%i %T %N'
ls -d "$R"/*
df -h "$R"
"$R/env/bin/python" - <<'PY'
import importlib.metadata as m,json
names=['torch','vllm','ray','transformers','flash-attn','tensordict','hydra-core','verl']
versions={}
for n in names:
 try:versions[n]=m.version(n)
 except m.PackageNotFoundError:versions[n]=None
print(json.dumps(versions))
PY
command -v uv || true
command -v nvcc || true
