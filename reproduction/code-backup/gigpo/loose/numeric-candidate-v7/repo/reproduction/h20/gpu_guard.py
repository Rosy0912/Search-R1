"""Allow only the exact retriever identity launched by this project."""
import json
import subprocess
import sys
from process_identity import PIDFILE, owned_retriever

allowed=set()
try:
    record=json.loads(PIDFILE.read_text())
    if owned_retriever(record):
        allowed.add(str(record['pid']))
except (OSError,ValueError,TypeError,KeyError):
    pass
result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
other={p.strip() for p in result.stdout.splitlines() if p.strip()}-allowed
if other:
    print('GPU_CONFLICT: other compute process IDs',sorted(other),flush=True)
    sys.exit(3)
print('GPU_GUARD_OK',flush=True)
