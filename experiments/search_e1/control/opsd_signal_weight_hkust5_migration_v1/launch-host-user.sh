#!/usr/bin/env bash
set -euo pipefail
B=/home/shaoshitong/search-e1-migration-20261011
S=/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010
ENTRY=$S/source/hkust5-host-user-v3/opsd_signal_weight_hkust5_migration_v1/entry.py
: "${E1_RUN_ID:?explicit migration supervisor marker required}"
[[ $(docker info --format '{{.Name}}') == xie-lab-4 ]]
IMAGE=e1-signal-hostfs-minimal:20261011
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    tar -cf - --files-from /dev/null | docker import - "$IMAGE"
fi
# Empty image; all executables/environment/driver libraries are existing host bytes.
# No user namespace, no privilege/capability, only allocated0-3 CUDA device nodes.
exec docker run --rm --name "${E1_RUN_ID}" --user 1001:0 --userns host --pid host --uts host --ipc host --network host --cap-drop ALL \
 --mount type=bind,src=/usr,dst=/usr,readonly --mount type=bind,src=/etc,dst=/etc,readonly \
 --mount type=bind,src=/lib,dst=/lib,readonly --mount type=bind,src=/lib64,dst=/lib64,readonly \
 --mount type=bind,src=/bin,dst=/bin,readonly --mount type=bind,src=/sbin,dst=/sbin,readonly \
 --mount type=bind,src=/var,dst=/var,readonly --mount type=bind,src=/sys,dst=/sys,readonly \
 --mount type=bind,src=/home,dst=/home --mount type=bind,src=/tmp,dst=/tmp --mount type=bind,src=/run,dst=/run \
 --mount type=bind,src="$B/weka_3",dst=/weka_3 \
 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
 --device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3 \
 --env E1_RUN_ID --env CUDA_VISIBLE_DEVICES= --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
 --env TOKENIZERS_PARALLELISM=false --env HF_HOME="$S/cache/huggingface" --env XDG_CACHE_HOME="$S/cache/xdg" \
 --env TORCH_EXTENSIONS_DIR="$S/cache/torch-extensions" --workdir "$S" "$IMAGE" \
 /weka_3/shaoshitong/search-e1-config-study-20261004/env/bin/python "$ENTRY" --binding "$1" --binding-sha256 "$2" --mode driver
