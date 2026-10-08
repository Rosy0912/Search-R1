import hashlib,json,importlib.util,tempfile,copy
from pathlib import Path
W=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
pins={}
for dirname,expected in [('strict-ray-marker-fix','89d819a4b3e7e17489184c33ef33df942ac17aef71aba55c006857ba6f39eb2e'),('first5-tools-service-marker-live-state-v2','74856b5f245cd7186ca7b40bcf2f949e3cfe26c59875ac279df46a43d5fffb58')]:
 d=W/dirname;p=d/('manifest.json' if dirname=='strict-ray-marker-fix' else 'candidate-manifest.json');assert H(p)==expected;pins[str(p)]=expected
 manifest=json.loads(p.read_text())
 for name,sha in manifest['files'].items():assert H(d/name)==sha;pins[str(d/name)]=sha
 for name,sha in manifest.get('external_files',{}).items():p=W.parents[1]/name;assert H(p)==sha;pins[str(p)]=sha
G=load('independent_gate',W/'strict-ray-marker-fix/checkpoint_gate_candidate.py')
R=load('independent_receipt',W/'first5-tools-service-marker-live-state-v2/receipt_payload.py')
rejected=[]
with tempfile.TemporaryDirectory() as td:
 root=Path(td).resolve();files={};rows=[]
 for rank in range(8):
  pt=root/f'rank_{rank}.pt';pt.write_bytes(b'CPU_SYNTHETIC_PARSER_ONLY_NOT_STATE');rec={'path':str(pt),'sha256':H(pt)};files[f'actor/training_state/rank_{rank}.pt']=rec
  row=dict(version=3,algorithm='pure_opsd',status='EXACT_FULL_STATE_RELOAD_OK',model_state_dict_type='SHARDED_STATE_DICT',model_dtensor_entries=1,path=rec['path'],sha256=rec['sha256'],state_sha256=rec['sha256'],rank=rank,world_size=8,step=5,pid=100+rank);rows.append(row)
  p=root/f'rank_{rank}.json';p.write_text(json.dumps(row));files[f'actor/training_state/rank_{rank}.json']={'path':str(p),'sha256':H(p)}
 m={'world_size':8,'files':files};assert G.manifest_worker_pids(m,root)==set(range(100,108))
 p=root/'rank_7.json'
 for key,value in [('status','FAILED'),('algorithm','bapo'),('version',2),('model_state_dict_type','LOCAL_STATE_DICT'),('model_dtensor_entries',0),('path',str(root/'other.pt')),('state_sha256','0'*64),('rank',0),('world_size',7),('step',4),('pid',100),('pid',True)]:
  row=dict(rows[7]);row[key]=value;p.write_text(json.dumps(row));files['actor/training_state/rank_7.json']['sha256']=H(p)
  try:G.manifest_worker_pids(m,root)
  except ValueError:rejected.append([key,value])
  else:raise AssertionError(('accepted',key,value))
 proc=root/'proc'/'200';proc.mkdir(parents=True);(proc/'stat').write_text('200 (service) '+' '.join(['R']+['0']*18+['4321']));(proc/'cgroup').write_text('0::/slurm/job_123/step_0\n')
 import os
 uid=os.getuid();manifest={'uid':uid,'job_id':'123'}
 for state in ['R','S','D','I']:
  (proc/'status').write_text(f'Uid: {uid} {uid} {uid} {uid}\nState: {state} (actual format)\n');assert R.actual_live_process(200,manifest,root/'proc')['start']=='4321'
 for cgroup in ['0::/slurm/job_1234/step_0\n','0::/slurm/job_0123/step_0\n','0::/foreign/job_999/step_0\n']:
  (proc/'cgroup').write_text(cgroup)
  try:R.actual_live_process(200,manifest,root/'proc')
  except ValueError:rejected.append(['cgroup',cgroup.strip()])
  else:raise AssertionError('foreign namespace accepted')
for name,sha in pins.items():assert H(Path(name))==sha
print(json.dumps({'scope':'CPU_STATIC_PARSER_AND_PROCESS_FIXTURES_NOT_ACTUAL_FIRST5','verified_files':pins,'additional_negatives_rejected':rejected,'passed':True},indent=2))
