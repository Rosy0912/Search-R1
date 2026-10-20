"""New allocation boundaries and observational quality evidence; never rescore."""
import hashlib,json,os,socket,time,uuid
from pathlib import Path
from urllib.parse import urlparse
from .contract import read_plan,sha,projected_config,validate_gate

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.pending')
 with temp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.link(temp,path);temp.unlink()
 fd=os.open(str(path.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def memory():
 base=Path('/sys/fs/cgroup')
 if (base/'memory.max').exists():
  def get(name):
   v=(base/name).read_text().strip();return int(v) if v!='max' else None
  return dict(current=get('memory.current'),peak=get('memory.peak'),limit=get('memory.max'))
 base=base/'memory'
 return dict(current=int((base/'memory.usage_in_bytes').read_text()),peak=int((base/'memory.max_usage_in_bytes').read_text()),limit=int((base/'memory.limit_in_bytes').read_text()))

def verify_node_records(plan):
 root=Path(plan['root']);sub=json.loads((root/'submission.json').read_text());gate=json.loads((root/'scheduler-release-once.json').read_text());validate_gate(plan,sub,gate)
 entries=[json.loads((root/f'node-{i}/entry-once.json').read_text()) for i in range(2)]
 hardware=[json.loads((root/f'node-{i}/hardware.json').read_text()) for i in range(2)]
 assert {e['node'] for e in entries}==set(range(2)) and len({e['hostname'] for e in entries})==2
 assert len({e['ip'] for e in entries})==2 and len({e['master'] for e in entries})==1
 assert all(e['plan_sha256']==sub['plan_sha256'] for e in entries)
 assert len({d[0].strip() for h in hardware for d in h['devices']})==16
 assert all(h['scheduler_uid']==sub['uid'] and h['no_prior_gpu_processes'] for h in hardware)
 return entries,hardware

def validate_cluster_boundary(config,ray_nodes=None):
 p=read_plan();projected_config(config);entries,_=verify_node_records(p)
 assert os.getuid()==0 and os.environ['SEARCH_R1_EXECUTION_BACKEND']=='pj_pt' and os.environ['SEARCH_R1_JOB_ID']==p['job'] and not os.environ.get('SLURM_JOB_ID')
 head=entries[0];assert config.baselines.local_runtime_host==head['hostname'].split('.')[0]
 parsed=urlparse(config.retriever.url)
 assert parsed.scheme=='http' and parsed.hostname==head['ip'] and parsed.port==p['proxy_port'] and parsed.path=='/retrieve'
 assert not parsed.username and not parsed.password
 assert os.environ['RAY_ADDRESS']==head['ip']+':'+str(p['ray_port'])
 assert not any(os.environ.get(k) for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'))
 if ray_nodes is not None:
  alive=[n for n in ray_nodes if n.get('Alive')];assert len(alive)==2
  assert {n['NodeManagerAddress'] for n in alive}=={e['ip'] for e in entries}
  assert all(n['Resources']['GPU']==8 and n['Resources']['CPU']>=48 for n in alive)

def admit_full_initialization(trainer):
 p=read_plan();root=Path(p['root']);entries,hardware=verify_node_records(p)
 workers=trainer.fresh_initialization_provenance['workers']
 assert trainer.actor_rollout_wg.world_size==16 and len(workers)==16 and {w['rank'] for w in workers}==set(range(16))
 assert all(w['world_size']==16 and w['optimizer_state_entries']==0 and w['optimizer_step_values']==[] and w['scheduler_step']==0 and w['optimizer_committed']==0 and not w['scheduler_advanced'] and w['checkpoint_recovery_environment_absent'] and w['configured_model_path']==w['expected_model_path']==p['model'] for w in workers)
 assert trainer.checkpoint_sampler.cursor==0 and trainer.actual_outer_updates==0
 proofs=[]
 for i in range(2):
  h=json.loads((root/f'node-{i}/memory.latest.json').read_text());assert time.time()-h['at']<45
  m=h['memory'];assert m['limit'] and m['current']<.85*m['limit'] and m['peak']<.90*m['limit']
  proofs.append(h)
 save(root/'initial16-admission.json',dict(at=time.time(),job=p['job'],uid=json.loads((root/'submission.json').read_text())['uid'],world_size=16,fresh=trainer.fresh_initialization_provenance,node_memory=proofs,hardware=hardware,scope='Actual initial model/empty Adam/scheduler0, not a completed optimizer round or measured training speedup'))

def raw_generation_record(item,tokenizer):
 from search_r1.baselines.tree_raw_tokens import FIELD
 if FIELD in item.non_tensor_batch:
  actions=[r['action']['native_emission']['raw_emitted_token_ids'] for r in item.non_tensor_batch[FIELD]['raw_ledger']]
 else:
  width=item.batch['responses'].shape[-1];mask=item.batch['attention_mask'][-width:].bool()
  if 'info_mask' in item.batch:mask &= item.batch['info_mask'][-width:].bool()
  actions=[item.batch['responses'][mask].detach().cpu().tolist()]
 texts=[tokenizer.decode(a,skip_special_tokens=False) for a in actions]
 return dict(long_im_start_repetition=any('<|im_start|>'*128 in s for s in texts),raw_emitted_token_ids=actions,raw_model_text=texts)

def observe_generation(data,tokenizer,rows):
 # The original scorer already validated every raw ledger and returned unchanged metrics.
 # Inspect raw emissions before special-token removal, where Tree degeneration was observed.
 output=[dict(**row,**raw_generation_record(data[i],tokenizer)) for i,row in enumerate(rows)]
 directory=Path(read_plan()['root'])/'generation-observations';identifier=str(time.time_ns())+'-'+uuid.uuid4().hex
 save(directory/(identifier+'.json'),dict(at=time.time(),rows=output,count=len(rows),same_existing_native_evaluation=True,metrics_unmodified=True))
 if len(rows)>=16 and sum(v['long_im_start_repetition'] for v in output)>=max(2,.01*len(rows)):
  raise RuntimeError('New16 run stopped: >=1% and >=2 model-authored long im_start repetition rows in an existing evaluation batch; preserve raw outputs')
