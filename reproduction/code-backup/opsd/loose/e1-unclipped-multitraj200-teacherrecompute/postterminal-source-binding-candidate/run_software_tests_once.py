"""Local CPU software fixture runner; records real subprocess originals only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
H=HERE.parent
PYTHON='/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python'
formal=json.loads((H/'formal-71d2501d4512/manifest.json').read_bytes())
paths=[HERE/'collect_postterminal_source_bindings.py',HERE/'test_postterminal_source_bindings.py',Path(__file__).resolve(),H/'final200-runtime-bundle-f11580b6a7bc/common.py',H/'final200-runtime-bundle-f11580b6a7bc/consume_final200_logs_terminal.py']
paths+=[H/'formal-71d2501d4512'/name for name in formal['tools']]
paths+=[H.parent/'opsd-unclipped-multitraj200-teacherrecompute'/name for name in formal['source_files']]
def pins():return {str(path):dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size) for path in paths}
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out=HERE/'cpu-originals'/stamp;out.mkdir(parents=True)
before=pins();argv=[PYTHON,str(HERE/'test_postterminal_source_bindings.py')]
(out/'argv.original.json').write_text(json.dumps(dict(argv=argv,cwd=str(HERE),scope='LOCAL_CPU_SOFTWARE_FIXTURES_NOT_PRODUCTION_MAIN'),sort_keys=True)+'\n')
start=time.monotonic()
with (out/'stdout.original').open('xb') as stdout,(out/'stderr.original').open('xb') as stderr:
    result=subprocess.run(argv,cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stdout,stderr=stderr,check=False)
    for f in (stdout,stderr):f.flush();os.fsync(f.fileno())
after=pins()
status=dict(waited=True,returncode=result.returncode,elapsed_seconds=time.monotonic()-start,before=before,after=after,stable=before==after,scope='Actual local CPU test process only; no stored3304 terminal/source/native200/effect qualification')
(out/'wait.original.json').write_text(json.dumps(status,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(output=str(out),waited=True,returncode=result.returncode,stable=before==after),sort_keys=True))
sys.exit(result.returncode if before==after else 90)
