"""Emit only verified fresh200 environment names, with shell-safe quoting."""
import json, shlex, sys, os
from pathlib import Path
if sys.flags.optimize: raise RuntimeError('optimized Python forbidden')
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r3-20261006')
assert sys.platform=='linux' and os.uname().nodename=='h100-5' and sys.prefix==str(root/'env')
assert root.resolve()==root
value=json.loads((root/'logs/frozen-policy-safety-config.json').read_text())
import search_r1.baselines.fresh200_contract as contract
assert Path(contract.__file__).resolve()==root/'repo/search_r1/baselines/fresh200_contract.py'
from search_r1.baselines.fresh200_contract import verify_fixed_policy_safety_artifact
verify_fixed_policy_safety_artifact(value)
for name,key in [('BAPO_FRESH200_FIXED_THRESHOLD_ARTIFACT','fixed_threshold_artifact'),('BAPO_FRESH200_FIXED_THRESHOLD_SHA256','fixed_threshold_artifact_sha256')]:
 v=value.get(key)
 if not isinstance(v,str) or not v: raise RuntimeError('missing formal binding '+key)
 print('export '+name+'='+shlex.quote(v))
