"""Retire only the unactivated v1 CPU candidate for phase-releasing implementation."""
import json,os,signal,subprocess,time
from pathlib import Path
p=Path('/proc/4071153');b=Path('/data/shaoshitong/recursive-bran-a-retriever-eight-20261012')
assert os.getuid()==1005 and p.stat().st_uid==1005 and not (b/'receipts/enable-gpu.json').exists()
argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
assert '/source-eight-02dbc2e/' in argv and str(b) in argv
apps=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split());assert '4071153' not in apps
r=dict(time=time.time(),pid=4071153,starttick=int((p/'stat').read_text().rsplit(') ',1)[1].split()[19]),reason='CPU-only candidate superseded to release CUDA context before HF; training untouched')
fd=os.pidfd_open(4071153);signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
for _ in range(20):
 if not p.exists():break
 time.sleep(.5)
assert not p.exists();r['actual_exit']=True
(b/'receipts/superseded-cpu-only.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
