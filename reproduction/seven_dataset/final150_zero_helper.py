from pathlib import Path
import json,importlib.util,os,time
from reproduction.seven_dataset.ablation_pj import pj_job_name
s=importlib.util.spec_from_file_location('pt','/plm-shared/zhangjunming/Workspace/mmsearch-hybrid-explore/runs/msjev-pt-probe-20260924/pt_client.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);client=m.get_cluster().client
names=[pj_job_name('budget-5',i,CID) for i in (1,2,3)]+['zhangjunming-ttt-e-b5-150-r'+str(i)+'-'+CID[:8]+'-pj' for i in (1,2,3)]
found={}
for n in names:
 try:client.get_training_job(n)
 except Exception as e:assert getattr(getattr(e,'response',None),'status_code',None)==404;found[n]='EXACT_404'
 else:raise RuntimeError('Existing exact job; never replay')
b=Path(B);paths=[Path(DEST),b/'ablation-imports'/CID,b/'ablation-seven-evaluations'/CID]
assert all(not p.exists() for p in paths)
assert not list((b/'receipts').glob('ablation-pj-budget-5-150-r*.json'))
free=os.statvfs(b).f_bavail*os.statvfs(b).f_frsize;assert free>=450_000_000_000
print(json.dumps(dict(at=time.time(),exact_jobs=found,absent_paths=[str(p) for p in paths],free_bytes=free)))
