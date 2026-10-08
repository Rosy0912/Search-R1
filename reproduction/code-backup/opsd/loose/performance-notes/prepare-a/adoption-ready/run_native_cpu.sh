#!/usr/bin/env bash
# Run only after main agent applies/adopts A; this script never edits source.
set -euo pipefail
: "${OPSD_ADOPTED_COMMIT:?Main agent must supply the actual adopted source commit}"
python_bin=${1:?Pass the CPU Python executable}
repo_dir=${2:?Pass the actual repository directory}
cd "$repo_dir"
[[ "$(git rev-parse HEAD)" == "$OPSD_ADOPTED_COMMIT" ]]
[[ -z "$(git diff --name-only HEAD -- search_r1 verl tests)" ]]
"$python_bin" - <<'PY'
import hashlib
from pathlib import Path
path = Path('search_r1/baselines/distillation.py')
assert hashlib.sha256(path.read_bytes()).hexdigest() == '2131315b6770779f0c7579bf2a6139e742cf75de03b7ee9f44662f8c32a870cf'
assert Path('tests/baselines/test_opsd_empty_chunks.py').is_file()
assert Path('tests/baselines/fixtures/opsd_e43_loss_reference.py').is_file()
PY
"$python_bin" -m pytest tests/baselines -q
