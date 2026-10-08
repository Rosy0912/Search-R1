import base64,hashlib
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177/logs/user-stopped-raw-20261005.tar.gz')
b=p.read_bytes(); assert hashlib.sha256(b).hexdigest()=='2fa77787c94e6167a0fdf64eeff284107c5c43b2593b7c32648759f54354bfaa'; print(base64.b64encode(b).decode())
