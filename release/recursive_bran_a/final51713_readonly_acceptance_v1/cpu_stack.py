"""Inspect only the current final run's CPU driver; never inspect foreign processes."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

B=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
F=B/'final51713-worker-recovery-v1'
plan=json.loads((F/'execution-plan.json').read_text())
tool=shutil.which('py-spy')
candidate=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/env/bin/py-spy')
if tool is None and candidate.exists():tool=str(candidate)
records=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if 'ray::run_final' not in argv:continue
        env=dict(item.split(b'=',1) for item in (p/'environ').read_bytes().split(b'\0') if b'=' in item)
        if env.get(b'E1_RUN_ID')!=plan['run_id'].encode():continue
        assert p.stat().st_uid==0
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        record=dict(pid=int(p.name),uid=0,tick=int(stat[19]),state=stat[0],
                    cpu_ticks=int(stat[11])+int(stat[12]),clock_ticks=os.sysconf('SC_CLK_TCK'))
        if tool:
            result=subprocess.run([tool,'dump','--pid',p.name],capture_output=True,text=True,timeout=15)
            record['stack']=dict(rc=result.returncode,stdout=result.stdout[-16000:],stderr=result.stderr[-2000:])
        records.append(record)
    except FileNotFoundError:pass
print(json.dumps(dict(time=time.time(),run_id=plan['run_id'],driver_records=records,
                     py_spy_available=tool is not None,scope='Own final Ray driver selected by exact E1_RUN_ID; no variables or credentials emitted')))
