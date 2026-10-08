from pathlib import Path
import json,os
p=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006/logs/user-fixed-policy-thresholds.json')
print(json.dumps({'path':str(p),'absolute':p.is_absolute(),'file':p.is_file(),'symlink':p.is_symlink(),'resolve':str(p.resolve()),'exists':p.exists(),'parents':[{'path':str(q),'symlink':q.is_symlink(),'resolve':str(q.resolve())}for q in list(p.parents)[:5]],'stat':list(p.stat())if p.exists()else None}))
