"""Sample only this segment's assigned GPUs until its own stop marker exists."""
import csv
import datetime
import io
import json
from pathlib import Path
import subprocess
import time

BASE=Path('/data/search-r1')
m=json.loads((BASE/'logs/a800-continuation-manifest.json').read_text())
stop=BASE/'logs'/('a800-monitor-'+m['run_id']+'.stop')
output=BASE/'logs'/('a800-gpu-samples-'+m['run_id']+'.jsonl')
devices=set(m['training_physical_gpus']+m['retrieval_physical_gpus'])
with output.open('x') as writer:
    while not stop.exists():
        try:
            r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        except subprocess.TimeoutExpired:
            writer.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpus':[],'error':'nvidia-smi timed out'})+'\n');writer.flush()
            time.sleep(5)
            continue
        rows=[]
        if r.returncode==0:
            for row in csv.reader(io.StringIO(r.stdout)):
                if int(row[0]) in devices: rows.append(list(map(int,row)))
        writer.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpus':rows,'returncode':r.returncode})+'\n');writer.flush()
        time.sleep(5)
