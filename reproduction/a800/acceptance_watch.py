"""Read-only five-update verification while the same trainer keeps running."""
import json
from pathlib import Path
import re
import time
import traceback
from verify_continuation import verify

BASE=Path('/data/search-r1')
m=json.loads((BASE/'logs/a800-continuation-manifest.json').read_text())
stop=BASE/'logs'/('a800-monitor-'+m['run_id']+'.stop')
out=BASE/'logs/a800-five-resumed-updates-acceptance.json'
while not stop.exists():
    log=Path(m['train_log']).read_text(errors='replace')
    log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',log)
    pattern=r'\bstep:'+str(m['acceptance_step'])+r' - [^\n]*critic/rewards/mean:'
    if re.search(pattern,log):
        try:
            result=verify(m)
            temp=out.with_suffix('.json.partial');temp.write_text(json.dumps(result,indent=2));temp.replace(out)
            print('FIVE_ACTUAL_RESUMED_UPDATES_ACCEPTED '+m['run_id'],flush=True)
            break
        except Exception:
            traceback.print_exc()
    time.sleep(30)
