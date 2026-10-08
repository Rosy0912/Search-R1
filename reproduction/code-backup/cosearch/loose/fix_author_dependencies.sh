set -euo pipefail
A=/dataset/shaoshitong/search-r1-rl/cosearch/author-native
export UV_CACHE_DIR="$A/cache/uv"
"$A/bootstrap/bin/uv" pip install --python "$A/env/bin/python" 'datasets==3.6.0'
"$A/env/bin/python" -c 'import torch;print("TORCH_ABI",torch._C._GLIBCXX_USE_CXX11_ABI)'
curl -fIL --connect-timeout 15 --max-time 45 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
