import os,re,socket,subprocess,json
from pathlib import Path
def verify_previous_cleanup():
 job=os.environ['SLURM_JOB_ID'];node=socket.gethostname().split('.')[0]
 assert node=='h100-10' and os.environ.get('SLURM_JOB_NODELIST')=='h100-10'
 assert re.search(r'job_'+re.escape(job)+r'(?:/|$)',Path('/proc/self/cgroup').read_text())
 found=[]
 for p in Path('/proc').glob('[0-9]*/cgroup'):
  try:cg=p.read_text()
  except FileNotFoundError:continue
  if re.search(r'job_(?:2960|2962)(?:/|$)',cg):found.append({'pid':int(p.parent.name),'cgroup':cg})
 listeners=subprocess.check_output(['ss','-H','-ltnp'],text=True)
 ports=[line for line in listeners.splitlines() if re.search(r':(?:18084|16384)\s',line)]
 result={'job_id':job,'node_scope':node,'previous_jobs':['2960','2962'],'previous_job_processes':found,'previous_ports_listening':ports,'read_only_no_kill':True}
 assert not found and not ports, ('Previous jobs not cleaned up',result)
 return result

result=verify_previous_cleanup()
Path(os.environ['OPSD_EVIDENCE'],'previous-pilot-reload-cleanup.json').write_text(json.dumps(result,indent=2))
