"""Allow explicit supervisor restart only after verifying frozen runtime bytes."""
import os,json,hashlib
from pathlib import Path
base=Path(os.environ['E1_BASE']);profile=os.environ['E1_PROFILE'];purpose=os.environ.get('E1_RUNTIME_PURPOSE','grpo');runtime=base/('runtime' if purpose=='grpo' else 'opsd-runtime')/profile
receipt_profile=profile if purpose=='grpo' else profile+'-opsd'
manifest=json.loads((base/'receipts'/f'{receipt_profile}-runtime.sha256.json').read_text())
for name,expected in manifest.items():
 p=runtime/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen runtime mismatch: '+name)
print('FROZEN_RUNTIME_VERIFIED',profile,len(manifest),flush=True)
