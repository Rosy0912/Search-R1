"""Read the retained A scalar validation lines; never rerun any evaluation."""
import json
import re
import time
from pathlib import Path

b = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
paths = [b / 'logs/training.log', b / 'recovery-from100-20261013/logs/training.log']
records = []
for path in paths:
    if not path.is_file():
        continue
    with path.open(errors='replace') as stream:
        for line in stream:
            match = re.search(r'\bstep:(\d+)\b', line)
            if match and int(match.group(1)) in (5, 50, 100, 150) and 'val/em/all:' in line:
                records.append(dict(path=str(path), outer=int(match.group(1)), line=line.strip()))
print(json.dumps(dict(time=time.time(), rows=records, scope='Existing aggregate native logs only; no new prediction or paired rows')), flush=True)
