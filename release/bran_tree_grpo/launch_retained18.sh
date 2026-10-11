#!/usr/bin/env bash
# Run on retained H10018. Creates one immutable code package and submits once.
set -euo pipefail
if [[ $(hostname -s) != h100-10 ]]; then
  echo 'Run this launcher on h100-10 (10.33.0.18).' >&2; exit 2
fi
: "${BRAN_TRAINING_INSTRUCTION:?Set the explicit fresh200 training instruction}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
python=/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python
root=/dataset/shaoshitong/search-r1-rl/tree-grpo
runtime="$repo/reproduction/tree-branpo-runtime-20261011"
commit=$(git rev-parse HEAD)
remote_sha=$(git ls-remote origin refs/heads/codex/bran-tree-grpo-release-20261011 | awk '{print $1}')
[[ "$remote_sha" == "$commit" ]] || { echo 'Checkout differs from published release branch.' >&2; exit 2; }
package="$root/instruct300/candidates/branpo-fresh200-$commit"
release="$root/instruct300/branpo-release-$commit.json"
script="$root/instruct300/branpo-$commit.sbatch"
# Fail before packaging if a Tree GPU job already exists. Never cancel it.
if squeue --local -h -o '%j|%b' | awk -F '|' '$1 ~ /^tree-/ && $2 !~ /^(gres\/gpu:0|gres\/gpu=0|gpu:0)$/ {found=1} END {exit !found}'; then
  echo 'Existing Tree job detected; refusing a duplicate launch.' >&2; exit 2
fi
"$python" "$runtime/build_package.py" --repo "$repo" --output "$package" --instruction "$BRAN_TRAINING_INSTRUCTION"
"$python" "$runtime/make_release.py" --package "$package" --review "$repo/release/bran_tree_grpo/CODE_REVIEW.md" --remote-source-sha "$remote_sha" --verified-at "$(date -u +%FT%TZ)" --output "$release"
manifest_sha=$("$python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$package/manifest.json")
release_sha=$("$python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$release")
"$python" "$package/render_dispatch.py" --package "$package" --manifest-sha "$manifest_sha" --release-path "$release" --release-sha "$release_sha" --wall-hours 168 --output "$script"
script_sha=$("$python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$script")
"$python" "$package/submit_once.py" --package "$package" --manifest-sha "$manifest_sha" --release-path "$release" --release-sha "$release_sha" --script "$script" --script-sha "$script_sha"
