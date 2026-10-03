"""Check selected GPU UUIDs, not utilization alone; never terminate outsiders."""
import subprocess,csv,os,json,sys
selected=[int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
def query(args):return list(csv.reader(subprocess.check_output(['nvidia-smi',*args],text=True).splitlines(),skipinitialspace=True))
gpus=query(['--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
rows=[g for g in gpus if int(g[0]) in selected]
if len(rows)!=len(selected):raise SystemExit('GPU inventory incomplete')
apps=query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
for g in rows:
 if g[4]=='[N/A]' or int(g[3])>128 or int(g[4])>1:raise SystemExit('Selected GPU busy/unknown: '+str(g))
 if any(p[0]==g[1] for p in apps):raise SystemExit('Selected GPU has compute processes: '+str(g))
print(json.dumps(dict(devices=rows,compute_apps=apps)),flush=True)
