#!/bin/bash
set -eo pipefail
r=/data/user/jxu957/experiments/topd-20261008/environment
exec 9>"$r/compat-build.lock"
flock -n 9 || exit 75
printf '%s\n' "$$" > "$r/compat-build.pid"
source /etc/profile
module load gcc/10.5
export PATH="$r/topd-py310/bin:$PATH" PIP_CACHE_DIR="$r/pip-cache" TMPDIR="$r/tmp"
export MAX_JOBS=8 NVCC_THREADS=1 FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_OFFLINE_BUILD=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE
if ! python -c 'import importlib.metadata as m;assert m.version("flash-attn-3")=="3.0.0b1"' 2>/dev/null; then
module load cuda/12.8 gcc/10.5
export CUDA_HOME=/data/apps/cuda/12.8
cd "$r/sources/flash_attn-2.7.4.post1/hopper"
python -u setup.py bdist_wheel
if python -c 'import importlib.metadata as m; assert m.version("flash-attn")=="3.0.0b1"' 2>/dev/null; then python -m pip uninstall -y flash-attn; fi
python -m pip install dist/flash_attn_3-3.0.0b1-cp310-cp310-linux_x86_64.whl --no-deps
fi
module unload cuda/12.8 2>/dev/null || true
module load cuda/12.1
export CUDA_HOME=/data/apps/cuda/12.1
cd "$r/tmp/pip-install-yhwfdmqi/flash-attn_d94d5cd94b97409ebe2abb803c8db75f"
python -u setup.py bdist_wheel
python -m pip install dist/flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl --no-deps
source "$r/activate-topd.sh"
cd "$r"
python -c 'import torch,flash_attn,flash_attn_interface,flash_attn_3_cuda;from flash_attn.bert_padding import pad_input,unpad_input;from flash_attn.layers.rotary import apply_rotary_emb; print("FA2_FA3_COMPAT_IMPORT_PASS")'
python -m pip check
printf '%s\n' FA2_FA3_COMPAT_IMPORT_PASS > "$r/compat-complete.txt"
