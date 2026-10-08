"""Independent actual a2 CPU replay; stdin execution, no remote writes or GPU/model loading."""
import os,sys,json,hashlib,subprocess,time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
RUN=ROOT/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
SOURCE='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
BASE=RUN/'logs/first5-evidence-v4-20261007-a2'
COLLECTION_SHA='a08423cda2f9f6b1baafebf6d0c586b60ff50fe9c6937ad2f2c19ef52d7bff09'
MANIFEST_SHA='a960351edd3bfb712ca3c89f2e278542ac67d5cfc9750256768853117d2d9f09'
TOOLS=ROOT/'first5-evidence-tools-3315-v4'
BINDINGS={'first5_helpers.py':'716f9410465c2f642b5cf9af7cbf6d7f503ff43c58aeb4e195f2c82deae525df',
'native_process_strict.py':'aa5dd74ae021363aecf1a3869bf74bfde50406c879fcfc1474938b326b9b8659',
'guard_snapshot.py':'a7124c76133c2785bc18f7d790b5cc3e2d486ce8cd0a5a51141736da1b35b621'}

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):
 path=Path(path);need(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink(),'canonical original required')
 return path.read_bytes()
def document(path,expected=None):
 raw=read(path)
 if expected is not None:need(sha(raw)==expected,'original bytes/hash changed')
 return json.loads(raw)
wall=time.perf_counter();cpu=time.process_time()
need(sys.platform=='linux' and sys.prefix==str(ROOT/'env') and not sys.flags.optimize,'own lexical Linux env')
need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','GPU hidden required')
need(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE,'actual source changed')
need(not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'source dirty')
for n,h in BINDINGS.items():need(sha(read(TOOLS/n))==h,'reviewed adapter/helper changed')
sys.path[:0]=[str(TOOLS),str(ROOT/'repo')]
from search_r1.baselines.fresh200_driver import Fresh200Progress
from search_r1.baselines.fresh200_contract import critical_source_hashes,verify_sampling_policy_binding
from search_r1.baselines.full_state import state_digest
from native_process_strict import verify_native_process,conservation_proof
from first5_helpers import marked,process_proof,verify_snapshot_prefix
from guard_snapshot import replay_snapshot
from verl import DataProto
import torch
need(not torch.cuda.is_initialized(),'CUDA unexpectedly initialized')
c=document(BASE/'collection.json',COLLECTION_SHA)
need(c['source_commit']==SOURCE and c['job_id']==3315 and c['run_root']==str(RUN),'collection identity')
m=document(RUN/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json',MANIFEST_SHA)
progress=Fresh200Progress()
for attempt in c['attempts']:
 need(progress.begin_attempt(attempt['candidate_outer'])==attempt['attempt_id'],'attempt order')
 complete=progress.record(attempt['rank_write_ledger'],attempt['candidate_outer'])
 need(complete is attempt['accepted_complete_outer'],'attempt accepted flag')
 need(document(attempt['ledger_path'])==attempt,'raw durable attempt ledger differs')
need(progress.completed_outers==5 and progress.state_dict()==c['attempt_accounting']==m['attempt_accounting'],'actual accepted accounting')
verify_snapshot_prefix(c['train_log_prefix']);text=read(c['train_log_prefix']['snapshot_path']).decode()
need(marked(text,'FRESH200_ATTEMPT_JSON=')==c['attempts'],'actual log/ledger order differs')
rankrows=marked(text,'ACTOR_RANK_UPDATE_JSON=');need(len(rankrows)==8*len(c['attempts']),'raw allrank numeric count')
# This actual run has five complete attempts with no skips; per-rank nth
# numerical record must match nth raw outcome. No general skip index invented.
for rank in range(8):
 rows=[r for r in rankrows if r['rank']==rank];need(len(rows)==5,'raw rank missing/duplicate')
 for i,r in enumerate(rows):
  e=next(e for e in c['attempts'][i]['rank_write_ledger']if e['rank']==rank)
  need(r['optimizer_steps']==e['optimizer_steps'] and r['scheduler_last_epoch']==i+1,'actual native write/clock mismatch')
  need(__import__('math').isclose(r['lr_before'],1e-6*(i+1)/57,rel_tol=1e-12) and __import__('math').isclose(r['lr_after'],1e-6*(i+2)/57,rel_tol=1e-12),'actual accepted LR mismatch')
  for key in ('actor/pg_loss','actor/entropy_loss','actor/ppo_kl','actor/pg_clipfrac','actor/grad_norm'):
   values=r['metrics'][key];need(len(values)==e['optimizer_steps'] and all(__import__('math').isfinite(v)for v in values),'actual numerical vector missing/nonfinite')
metrics=marked(text,'BAPO_TRAIN_METRICS_JSON=')
from search_r1.baselines.actor_metrics import merge_actor_metric_records
for i,metric in enumerate(metrics):
 merged=merge_actor_metric_records([next(r for r in [x for x in rankrows if x['rank']==rank] if r['scheduler_last_epoch']==i+1)['metrics']for rank in range(8)])
 for k,v in merged.items():
  expected=sum(v)/len(v) if isinstance(v,list)else v
  need(__import__('math').isclose(expected,metric['metrics'][k],rel_tol=1e-9,abs_tol=1e-12),'driver allrank numeric mean differs '+k)
member={Path(x['path']).name:x for x in m['files']}
rankdetails=[]
for item in c['native_roundtrip']:
 rank=item['rank'];verify_native_process(item['initial_process'],SOURCE)
 initial=document(member['native-initial-rank_%d.json'%rank]['path'],member['native-initial-rank_%d.json'%rank]['sha256'])
 need(initial['native_process']==item['initial_process'],'raw initial native metadata altered')
 report=item['roundtrip'];rawreport=document(Path(report['path']).with_suffix('.json'))
 need(rawreport==report and report['state_digest']==report['reloaded_state_digest'],'native roundtrip original changed')
 need(report['generation_rng_before_digest']==report['generation_rng_after_digest'],'native rng reload differs')
 live=process_proof(report['pid'],RUN);need(str(live['start_ticks'])==item['initial_process']['proc_start_ticks'],'live worker PID reused')
 need(report['latest_rank_event']==next(e for e in c['attempts'][-1]['rank_write_ledger']if e['rank']==rank),'latest native event drift')
 rankdetails.append(dict(rank=rank,pid=report['pid'],birth=live['start_ticks'],gpu_uuid=item['initial_process']['gpu_uuid'],scheduler_last_epoch=report['scheduler_last_epoch']))
need(len({r['gpu_uuid']for r in rankdetails})==8,'native UUID not distinct')
conserved=conservation_proof(ROOT/'repo/search_r1/baselines/fresh200_native_evidence.py')
need(conserved==document(c['native_process_conservation']['path'],c['native_process_conservation']['sha256']),'raw conservation witness changed')
rawpath=Path(c['sampling_proof']['path']);need(sha(read(rawpath))==c['sampling_proof']['sha256'],'first raw sampling immutable bytes')
payload=torch.load(rawpath,map_location='cpu',weights_only=False)
need(payload['source_sha256']==critical_source_hashes() and payload['driver_pid']==760122,'raw sampling actual source/driver')
data=DataProto.from_dict(tensors=payload['tensors'],non_tensors=payload['non_tensor_batch'],meta_info=payload['meta_info'])
cfg=document(member['resolved-config.json']['path'],member['resolved-config.json']['sha256'])
binding=verify_sampling_policy_binding(data,cfg['actor_rollout_ref']['actor']['policy_safety'])
need(binding==payload['sampling_binding_sha256']==c['sampling_proof']['sampling_binding_sha256'],'native probability raw replay differs')
mask=data.batch['sampling_mask'].bool();gap=float((data.batch['frozen_hf_log_probs'][mask].double()-data.batch['sampling_log_probs'][mask].double()).abs().max())
need(gap==payload['max_selected_logprob_abs_diff']==c['sampling_proof']['max_selected_logprob_abs_diff'],'HF separate diagnostic differs')
statepath=Path(member['driver.pt']['path']);need(sha(read(statepath))==member['driver.pt']['sha256'],'driver raw immutable hash')
state=torch.load(statepath,map_location='cpu',weights_only=False)
need(state_digest(state)==c['driver_state']['state_digest'] and state['global_step']==5 and state['next_global_step']==6 and state['progress']==progress.state_dict(),'driver RNG/sampler/BAPO progress changed')
resource=c['resources'];replayed=replay_snapshot(Path(resource['raw_path']),3315,str(RUN),SOURCE,ROOT/'tools/resource_watch.py',resource['snapshot_binding'])
need(replayed==resource,'original prefix resource replay differs')
driver=process_proof(760122,RUN);need(driver['start_ticks']==c['driver_process']['start_ticks'],'live original driver birth changed')
need(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
print(json.dumps(dict(status='PASS_READONLY_ACTUAL_A2_SMALL_EVIDENCE_CPU_REPLAY_NOT_FIRST5_RELEASE',source_commit=SOURCE,job_id=3315,collection_sha256=COLLECTION_SHA,manifest_sha256=MANIFEST_SHA,
 attempt_accounting=progress.state_dict(),native_rank_processes=rankdetails,source_files=len(critical_source_hashes()),native_sampler_binding_sha256=binding,selected_tokens=int(mask.sum()),hf_diagnostic_abs_gap=gap,
 conserved_nonallocation_statements=conserved['conserved_statements'],resource_prefix_sha256=resource['raw_sha256'],resource_prefix_counts={k:resource[k]for k in ('samples','recovery_samples','freeze_events','safe_resume_events')},
 driver_pid=760122,driver_birth=driver['start_ticks'],cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,
 big_native_model_files_hashed_or_loaded=False,actual_cpu_hf_examined=False,first5_accepted=False,release_authorized=False),allow_nan=False))
