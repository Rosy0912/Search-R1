/dataset/shaoshitong/search-r1-rl/cosearch/env/bin/python - <<'PYCODE'
import os,signal,json,time
from pathlib import Path
pid=1461484;p=Path('/proc')/str(pid)
if p.exists():
 command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 assert 'srun --jobid=2901 --overlap' in command and '/cosearch/tools/raw_logits_parity_probe.py' in command
 fd=os.pidfd_open(pid)
 signal.pidfd_send_signal(fd,signal.SIGTERM)
 os.close(fd)
 print(json.dumps(dict(action='Stopped only own pending diagnostic srun request',pid=pid,production_job=2901,at=time.time())))

PYCODE
squeue -h --name=cosearch-local-five -o "%i %T %R"
