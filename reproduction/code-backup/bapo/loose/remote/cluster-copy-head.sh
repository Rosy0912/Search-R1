#!/bin/bash
set -euo pipefail
SOURCE=/dataset/shaoshitong/search-r1-rl/cosearch/assets-root/assets/wiki
TARGET=/dataset/shaoshitong/search-r1-rl/bapo/assets-root/assets/wiki/cluster-incoming
sshargs=(-T -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)
env -u LD_LIBRARY_PATH ssh "${sshargs[@]}" h100-5 "mkdir -p '$TARGET'"
tar -cf - -C "$SOURCE" part_aa part_ab wiki-18.jsonl.gz | env -u LD_LIBRARY_PATH ssh "${sshargs[@]}" h100-5 "tar -xf - -C '$TARGET'"
env -u LD_LIBRARY_PATH ssh "${sshargs[@]}" h100-5 'ROOT=/dataset/shaoshitong/search-r1-rl/bapo; export SEARCH_R1_ASSETS="$ROOT/assets-root"; "$ROOT/env/bin/python" "$ROOT/tools/verify-cluster-copy.py"'
