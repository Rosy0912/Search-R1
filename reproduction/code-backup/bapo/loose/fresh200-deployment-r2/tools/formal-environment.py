"""Emit only verified fresh200 environment names, with shell-safe quoting."""
import json, shlex, sys
from pathlib import Path
if sys.flags.optimize: raise RuntimeError('optimized Python forbidden')
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r2-20261006')
value=json.loads((root/'logs/frozen-policy-safety-config.json').read_text())
for name,key in [('BAPO_FRESH200_CALIBRATION_MANIFEST','calibration_manifest'),('BAPO_FRESH200_CALIBRATION_SHA256','calibration_sha256'),('BAPO_FRESH200_REVIEW_SHA256','evidence_review_sha256'),('BAPO_FRESH200_REVIEW_NONCE','evidence_review_nonce')]:
 v=value.get(key)
 if not isinstance(v,str) or not v: raise RuntimeError('missing formal binding '+key)
 print('export '+name+'='+shlex.quote(v))
