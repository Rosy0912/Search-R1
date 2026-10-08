import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import torch,json,pathlib,hashlib,time,resource,math,gc
from collections import defaultdict
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-3268');S=R/'output/sampling-consistency';P=R/'output/probability-calibration';start=time.perf_counter();cpu=time.process_time();torch.set_num_threads(4);ready=json.loads((P/'ready.json').read_bytes());records=defaultdict(list);proof=[];dist_modes=[];all_delta=[];raw_row_bound=0
def loadsha(p,s):
 with p.open('rb') as f:
  before=os.fstat(f.fileno());h=hashlib.sha256()
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
  assert h.hexdigest()==s;f.seek(0);v=torch.load(f,map_location='cpu',weights_only=False);after=os.fstat(f.fileno())
  assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 return v
for rank,w in enumerate(sorted(ready['workers'],key=lambda x:x['rank'])):
 assert w['rank']==rank and w['job_id']=='3268' and w['code_commit']=='a632ba91840fb116aacfc0e770681a860647b295' and w['config_sha256']=='e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b' and w['scheduler_step']==110 and w['restored_optimizer_steps']==[2500] and w['optimizer_committed']==0 and w['scheduler_advanced'] is False
 for entry in w['diagnostic_entries']:
  p=pathlib.Path(entry['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['sha256'];j=json.loads(raw);assert j['rank']==rank and j['saved_rows']==j['local_rows'] and j['enforced'] is False
  raw_row_bound += j['saved_rows']*500;assert raw_row_bound<=20000000, 'Physical raw row bound exceeded'
  t=loadsha(pathlib.Path(entry['snapshot_path']),entry['snapshot_sha256']);sample=t['sampled_log_probs'];old=t['old_log_probs'];width=t['responses'].shape[-1];valid=t['attention_mask'][:,-width:].bool();assert sample.shape==old.shape==valid.shape and t['input_ids'].shape[0]==j['saved_rows'];assert torch.equal(t['input_ids'][:,-width:],t['responses'])
  delta=old[valid].double()-sample[valid].double();assert torch.isfinite(delta).all() and torch.isfinite(old[valid]).all() and torch.isfinite(sample[valid]).all();records[j['generation_call']].append((j,delta));all_delta.append(delta)
  proof.append(dict(path=str(p),sha256=entry['sha256'],snapshot_path=entry['snapshot_path'],snapshot_sha256=entry['snapshot_sha256'],rows=j['saved_rows'],response_token_occurrences=delta.numel()))
  dist_modes.append(t['distribution']);del t;gc.collect()
reports=[]
for call,rows in sorted(records.items()):
 assert len(rows)==8 and sorted(j['rank'] for j,d in rows)==list(range(8));j=rows[0][0];dec=j['decision'];assert all(x['decision']==dec for x,d in rows)
 d=torch.cat([x for j,x in rows]);maxabs=float(d.abs().max());absmean=float(d.abs().mean());rms=float(d.square().mean().sqrt());excess=int((d>30).sum());kl=None if excess else float((torch.expm1(d)-d).mean());assert dec['valid_tokens']==d.numel() and dec['invalid_tokens']==0 and dec['numeric_bound_exceeded']==excess
 for a,b in [(maxabs,dec['max_abs_delta']),(absmean,dec['mean_abs_delta']),(rms,dec['rms_delta'])]:assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(call,a,b)
 if kl is not None:assert math.isclose(kl,dec['sampled_k3_mean'],rel_tol=1e-10,abs_tol=1e-12)
 else:assert dec['sampled_k3_mean'] is None
 reports.append(dict(call=call,physical_rows=sum(j['local_rows'] for j,d in rows),valid_response_token_occurrences=d.numel(),max_abs_delta=maxabs,mean_abs_delta=absmean,rms_delta=rms,sampled_k3=kl,above_trial_dot01_occurrences=int((d.abs()>.01).sum()),signed_delta_quantiles=torch.quantile(d,torch.tensor([0.,.01,.5,.99,1.],dtype=torch.double)).tolist(),allowed=dec['allowed'],reasons=dec['reasons']))
all_d=torch.cat(all_delta);assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='RAW_SAME_TURN_PHYSICAL_TOKEN_STATS_INDEPENDENTLY_RECOMPUTED',job='3268',ready_sha256=hashlib.sha256((P/'ready.json').read_bytes()).hexdigest(),calls=reports,all_token_occurrences=all_d.numel(),all_max_abs=float(all_d.abs().max()),all_above_trial_dot01=int((all_d.abs()>.01).sum()),evidence=proof,scope='Raw sampler versus same-turn FSDP only; physical rows include repeats/padding duplicates; not final-leaf PPO sampling coverage or threshold acceptance',wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,threads=4,cuda_initialized=False,analysis_slurm_job_id=os.environ.get('SLURM_JOB_ID'),raw_max_row_token_bound=raw_row_bound),allow_nan=False))
