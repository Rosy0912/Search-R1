set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
test -z "$(squeue -h --name=cosearch-ranker-probe -o %i)"
test -z "$(squeue -h --name=cosearch-local-five -o %i)"
base64 -d > "$R/tools/gpu-ranker-g.bundle" <<'BUNDLE'
IyB2MiBnaXQgYnVuZGxlCi0zZmM0Y2QwYjdlMTQxYzlkZjg3NjM1OGQwZDU1NWI3MTM4YTdkZGFkIEJpbmQgcmFua2VyIENVREEgYmVmb3JlIHJ1bnRpbWUgaW5pdGlhbGl6YXRpb24gYW5kIHJlc3RvcmUgYWN0b3IgR1BVIG1hc2tzCjczMThmOTY0YmE1ZTMzNjljMWE4MTg0ZWY2NzU4MmRhZGFmYWFjODMgSEVBRAoKUEFDSwAAAAIAAAAGkBJ4nJ3MS2oEIRCA4b2nqH0gaNs+GkLIGXKDaqtkJGo1jj2P209WOUB2P3zwz8EMuLqQdLB+3zmyTt4EkxBzomCYfDaGM27Log4c3CfYnNZEeg9sVpM2yjF46yJpcs7twdiIgQhJ4TkvMgC79GeT8woff/nFD2xH5ffSb1gLfYIJm9F68c7Cm45aqyStlTn53wP1zU1uDKVfeJTJBJ3nXcYPHEMeha+QhzSQe/+lKgkrSM61dIZx9lkaqxfmDF8g8QKpoWznqqNe/yavwlvzaUvUEaRntXicASEA3v/qBeoFsIEBFBbBU2NdauKqbKXnm0Vs2bYBCpaXs5UBVQH/2w6v/gGCMkxFTlH4+gu6Hx/kpYOxXR+1vnicAR4A4f/0AfQBkF4U7r2mZO9CkADhDbwu/Xbv/qbkhQaRcoL1YhA1/wGPloZOiYl/ot1IcpNy4ulHnTyq93ic28m6k3WCtQi7rqCz6/stsdWNIRO2al47+VhH02OjfxYTAL1hDHL+Abir18Qw/bAZBwMYPG5K1q98ZNNEeJwBHgDh/7ACsAKQlhRbqKiezsLQYZ37cRol/O1WZLi2BJGqhuusD8f0DSmI7BHIv3QMFu27hTEXciCO6k1SeJxbHfw5ZIMDa31afpFCQVF+RWV8dmqlQmaegoa6R0hIQHxAkH9EpLoOmBMM5zn6+MDZGSUlBfFgrVBOMZyXmJMDZWtacSkAQWpemV5BfoEG3CYdv/y8VE2uVHegRGlBSmJJqoafP8RoW3VDI3M9AyA01MnJT07MycgvLlHXycuHGIlderMda64KAN7iRg473xfHymSmu8q3kZKCsvo2tpoNjA==
BUNDLE
cd "$R/repo"
git fetch "$R/tools/gpu-ranker-g.bundle" HEAD
git merge --ff-only FETCH_HEAD
cp reproduction/baselines/agent6-20261002/local_gpu_five.py "$R/tools/five_iteration_job_g.py"
cp reproduction/baselines/agent6-20261002/native_gpu_ranker_probe.py "$R/tools/native_gpu_ranker_probe.py"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH="$R/tools/test-lib:$R/repo"
"$R/env/bin/python" -m pytest tests/baselines/test_cosearch_batch_parity.py tests/baselines/test_cosearch_fixes.py tests/baselines/test_cosearch_attention.py tests/baselines/test_runtime_audit.py tests/baselines/test_local_runtime.py -q > "$R/logs/gpu-ranker-g-native-tests.log" 2>&1
tail -2 "$R/logs/gpu-ranker-g-native-tests.log"
test ! -d "$R/runs/cosearch-local-gpu-five-20261003-g"
tmux new-session -d -s cosearch-local-gpu-five-g-20261003 "srun --immediate=60 -p gpu -w h100-16 -N1 --ntasks=1 --gres=gpu:h100:8 --exclusive --cpus-per-task=128 --mem=768G --job-name=cosearch-local-five env PYTHONPATH='$R/repo' COSEARCH_RUN_ID=cosearch-local-gpu-five-20261003-g '$R/env/bin/python' -m search_r1.baselines.preflight -- '$R/env/bin/python' '$R/tools/five_iteration_job_g.py' > '$R/logs/local-five-g.log' 2>&1"
