#!/usr/bin/env bash
set -euo pipefail
B=/home/shaoshitong/search-e1-migration-20261011
S=/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010
ENTRY=$S/source/hkust5-migration-v1/opsd_signal_weight_hkust5_migration_v1/entry.py
: "${E1_RUN_ID:?explicit migration supervisor marker required}"
exec bwrap --tmpfs / --ro-bind /usr /usr --ro-bind /etc /etc --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /bin /bin --ro-bind /sbin /sbin --ro-bind /var /var --bind /home /home --bind /tmp /tmp --bind /run /run --proc /proc --dev-bind /dev /dev --ro-bind /sys /sys --bind "$B/weka_3" /weka_3 --chdir "$S" /usr/bin/env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME="$S/cache/huggingface" XDG_CACHE_HOME="$S/cache/xdg" TORCH_EXTENSIONS_DIR="$S/cache/torch-extensions" /weka_3/shaoshitong/search-e1-config-study-20261004/env/bin/python "$ENTRY" --binding "$1" --binding-sha256 "$2" --mode driver
