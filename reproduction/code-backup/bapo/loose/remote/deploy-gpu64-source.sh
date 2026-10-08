set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
export PATH="$R/env/bin:$PATH"
exec 8> "$R/gpu-job.lock"
flock -n 8
cd "$R/repo"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = b6ecbeb798039ff59a92a3d2f3a14a37fdb91c7a
git bundle verify "$R/bapo-gpu64.bundle"
git fetch "$R/bapo-gpu64.bundle" reproduce/bapo
git merge --ff-only FETCH_HEAD
test "$(git rev-parse HEAD)" = b20f371972bb1bc41b735c6cd350b9a779d8820d
test -z "$(git status --porcelain)"
