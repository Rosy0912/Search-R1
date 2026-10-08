#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PIN=598e61bd1d36895726d28a8d06b3a15bed19f5d3
if [[ ! -d "$ROOT/upstream/Search-R1/.git" ]]; then
  git clone https://github.com/PeterGriffinJin/Search-R1.git "$ROOT/upstream/Search-R1"
fi
git -C "$ROOT/upstream/Search-R1" checkout "$PIN"
