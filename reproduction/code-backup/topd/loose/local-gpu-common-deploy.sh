set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd/repo
test -z "$(git status --porcelain)"
git bundle verify ../topd-local-gpu-common.bundle
git fetch ../topd-local-gpu-common.bundle reproduce/topd
git merge --ff-only FETCH_HEAD
test "$(git rev-parse HEAD)" = e2f104a85ee3880d79de10b4f25cacbf283cfbcf
record=../logs/local-gpu-common-native-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable ../tools/native-cpu-gate.sbatch > "$record"; cat "$record"; fi
