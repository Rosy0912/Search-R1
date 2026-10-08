#!/bin/bash
set -euo pipefail
src=/dataset/shaoshitong/search-r1-rl/cosearch/assets-root/assets/hotpotqa/train.jsonl
dst=/dataset/shaoshitong/search-r1-rl/bapo/assets-root/assets/hotpotqa/train.jsonl
test "$(stat -c %s "$src")" = 569520788
cat "$src" | env -u LD_LIBRARY_PATH ssh -T -o BatchMode=yes -o ConnectTimeout=30 h100-5 'set -e; p=/dataset/shaoshitong/search-r1-rl/bapo/assets-root/assets/hotpotqa/train.jsonl; cat > "$p.cluster-copy.partial"; printf "%s  %s\n" a81274abafa899ec0ee073102edbe6bb694a8a1174201b4e43e2bc6c98964d1a "$p.cluster-copy.partial" | sha256sum -c -; mv "$p.cluster-copy.partial" "$p"; echo HOTPOT_CLUSTER_COPY_OFFICIAL_SHA_VERIFIED'
