#!/bin/bash
set -eo pipefail
root=/data/user/jxu957/experiments/topd-20261008/environment
exec 9>"$root/build-fa3.lock"
flock -n 9 || exit 0
source /etc/profile
module load cuda/12.8 gcc/10.5
export CUDA_HOME=/data/apps/cuda/12.8
export FAHOPPER_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_OFFLINE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_OFFLINE_BUILD=TRUE
export PATH="$root/topd-py310/bin:$PATH" PIP_CACHE_DIR="$root/pip-cache" TMPDIR="$root/tmp" MAX_JOBS=${MAX_JOBS:-1} NVCC_THREADS=${NVCC_THREADS:-1}
printf '%s\n' "$$" > "$root/build-fa3.pid"
ninja --version
nvcc --version
src="$root/sources/flash_attn-2.7.4.post1/hopper/setup.py"
test -n "$src"
cd "$(dirname "$src")"
python -u setup.py bdist_wheel
wheel=$(ls -t dist/*.whl | head -1)
python -m pip install "$wheel" --no-deps --force-reinstall
export LD_PRELOAD=/data/apps/cuda/12.8/lib64/libcudart.so.12${LD_PRELOAD:+:$LD_PRELOAD}
cd "$root"
python -c 'import torch; import flash_attn_interface; import flash_attn_3_cuda; print(flash_attn_interface.__file__)'
echo FA3_IMPORT_PASS > "$root/fa3-complete.txt"
