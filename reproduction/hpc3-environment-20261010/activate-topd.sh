# Source this file before using the TOP-D environment.
export PATH=/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/bin:$PATH
export CONDA_PREFIX=/data/user/jxu957/experiments/topd-20261008/environment/topd-py310
export LD_PRELOAD=/data/apps/cuda/12.8/lib64/libcudart.so.12${LD_PRELOAD:+:$LD_PRELOAD}
export PIP_INDEX_URL=http://harbor.internal.com:8081/repository/pypi-hkust/simple
export PIP_TRUSTED_HOST=harbor.internal.com
export PIP_CACHE_DIR=/data/user/jxu957/experiments/topd-20261008/environment/pip-cache
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/data/user/jxu957/experiments/topd-20261008/environment/cache/hf
export XDG_CACHE_HOME=/data/user/jxu957/experiments/topd-20261008/environment/cache/xdg
export TORCH_HOME=/data/user/jxu957/experiments/topd-20261008/environment/cache/torch
export TRITON_CACHE_DIR=/data/user/jxu957/experiments/topd-20261008/environment/cache/triton
export TORCH_EXTENSIONS_DIR=/data/user/jxu957/experiments/topd-20261008/environment/cache/torch-extensions
export TMPDIR=/data/user/jxu957/experiments/topd-20261008/environment/tmp
