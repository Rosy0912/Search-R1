"""One exact-UID CPU publisher for the additional human-authorized main."""
import importlib.util,json,os,sys,time
from pathlib import Path
from .contract import read_plan,sha,validate_tasks
from .runtime import save

def main(pp):
 p=read_plan(pp);root=Path(p['root']);save(root/'publisher-once.json',dict(at=time.time(),pid=os.getpid(),plan_sha256=sha(pp)))
 sub=json.loads((root/'submission.json').read_text());assert sub['job']==p['job'] and sub['plan_sha256']==sha(pp)
 spec=importlib.util.spec_from_file_location('pt','/plm-shared/zhangjunming/Workspace/mmsearch-hybrid-explore/runs/msjev-pt-probe-20260924/pt_client.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);client=m.get_cluster().client
 end=time.monotonic()+43200
 while True:
  j=client.get_training_job(p['job']);assert j['name']==p['job'] and j['uid']==sub['uid'];state=j['status']['state'];assert state in ('PENDING','RUNNING'),state
  paths=[root/f'node-{i}/entry-once.json' for i in range(2)]
  if state=='RUNNING' and all(v.exists() for v in paths):
   records=[json.loads(v.read_text()) for v in paths];assert {v['node'] for v in records}==set(range(2)) and len({v['hostname'] for v in records})==2 and len({(v['master'],v['master_port']) for v in records})==1 and all(v['plan_sha256']==sha(pp) for v in records)
   tasks=j['spec']['vc_job']['tasks'];validate_tasks(tasks)
   save(root/'scheduler-release-once.json',dict(at=time.time(),job=p['job'],uid=sub['uid'],plan_sha256=sha(pp),state=state,tasks=[{k:t[k] for k in ('role','replicas','resource_spec')} for t in tasks]));return
  assert time.monotonic()<end,'Bounded scheduler gate expired; inspect phase, never replay'
  time.sleep(10)
if __name__=='__main__':main(sys.argv[1])
