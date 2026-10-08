"""Independent CPU raw-capacity cross-check; does not accept Slurm terminal alone."""
import argparse, datetime, hashlib, json, math, pathlib, re
JOB='3192'; SOURCE='60cbea97d61ff95a8f7dfa58de7b79962e79bd10'
TOOL_SHA='495f3320314c6078ef0a25b7ca306a74cbc91bd7e333e9aef18172ec39a27d4d'
MODEL_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
ASSET_SHA='87b5e85df12cbf4635cb787b08161950ddc23e2e5b850e68fb7249668573cc87'
def unique(raw):
 def pairs(rows):
  out={}
  for k,v in rows:
   if k in out: raise ValueError('duplicate JSON key')
   out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def read(root,name): return unique((root/name).read_bytes())
def utc(s):
 d=datetime.datetime.fromisoformat(s)
 assert d.tzinfo and d.utcoffset()==datetime.timedelta(0)
 return d

def audit(root):
 root=pathlib.Path(root); alloc=read(root,'allocation.json'); summary=read(root,'capacity-summary.json'); cleanup=read(root,'cleanup.json')
 assert (root/'job-exit.txt').read_text().strip()=='0' and (root/'torchrun-exit.txt').read_text().strip()=='0'
 assert str(alloc['job_id'])==JOB and alloc['source']==SOURCE and alloc['hostname'].split('.')[0]=='h100-10'
 assert alloc['model_manifest_sha256']==MODEL_SHA and alloc['tool_assets_manifest_sha256']==ASSET_SHA
 assert alloc['tools_manifest']['source_commit']==SOURCE
 encoded=(json.dumps(alloc['tools_manifest'],sort_keys=True,indent=2)+'\n').encode(); assert hashlib.sha256(encoded).hexdigest()==TOOL_SHA
 assert summary['status']=='ACTUAL_OFFICIAL_3B_MICRO2_GPU_RETRIEVAL_CAPACITY_PASS' and summary['rl_outer_updates']==0 and str(summary['job_id'])==JOB and summary['source']==SOURCE
 assert cleanup['all_gpu_processes_zero'] is True and not cleanup['post_gpu_snapshot']['applications']
 uuidset={r['uuid'] for r in alloc['gpus']['inventory']}; assert len(uuidset)==8 and set(summary['uuids'])==uuidset
 workers=[read(root,f'worker-identity-{i}.json') for i in range(8)]
 assert {r['rank'] for r in workers}==set(range(8)) and len({r['pid'] for r in workers})==8 and {r['actual_cuda_uuid'] for r in workers}==uuidset
 for r in workers:
  assert str(r['job_id'])==JOB and r['source']==SOURCE and r['rank']==r['local_rank'] and r['world_size']==8
  assert set(map(int,r['uids']))=={alloc['uid']} and r['uid']==alloc['uid']
  assert re.search(r'(?:^|/)job_'+JOB+r'(?:/|$)',r['cgroup'],re.M)
 service=read(root,'service.json')['owner']; assert read(root,'http-top3-24.json')['service']['pid']==service['pid']
 http=read(root,'http-top3-24.json'); assert str(http['job_id'])==JOB and http['topk']==3 and len(http['queries'])==24 and len(http['response']['result'])==24
 assert math.isfinite(http['seconds']) and http['seconds']>0
 for result in http['response']['result']:
  assert len(result)==3
  for item in result: assert item['document']['contents'] and math.isfinite(float(item['score']))
 rows=[unique(line) for line in (root/'resources.ndjson').read_text().splitlines() if line]; prior=None; matched=[]; observed={u:0. for u in uuidset}
 for row in rows:
  t=utc(row['time']); assert prior is None or t>prior;prior=t
  assert str(row['job_id'])==JOB and 'query_error' not in row
  assert {r['uuid'] for r in row['gpus']}==uuidset
  for r in row['gpus']:
   assert all(math.isfinite(float(r[k])) for k in ('used_mib','total_mib','util'))
   assert 0<=r['used_mib']<=r['total_mib'] and 0<=r['util']<=100
   observed[r['uuid']]=max(observed[r['uuid']],r['used_mib'])
  apps=row.get('gpu_processes',[])
  for app in apps: assert not app.get('unavailable') and app['in_slurm_job'] and int(app['uid'])==alloc['uid']
  if all(any(a['pid']==w['pid'] and a['uuid']==w['actual_cuda_uuid'] and a['process_start_ticks']==w['start'] for a in apps) for w in workers) and any(a['pid']==service['pid'] for a in apps):matched.append(row['time'])
 assert len(matched)>=2
 sampler=read(root,'sampler-stop.json'); assert sampler['owned_stop'] is True and sampler['exit_code']==-15
 data=read(root,'micro-2.json'); assert data['passed'] is True and data['local_micro']==2 and len(data['ranks'])==8
 rankpeaks=[]
 for i,r in enumerate(sorted(data['ranks'],key=lambda v:v['rank'])):
  rankfile=read(root,f'micro-2-rank-{i}.json')
  assert r['rank']==i and all(rankfile[k]==v for k,v in r.items())
  assert r['passed'] is True and str(r['job_id'])==JOB and r['source_commit']==SOURCE and r['world_size']==8
  assert r['official_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1' and r['component_clip'] is None and r['teacher_prompt_mode']=='user_context'
  assert r['local_micro']==2 and r['input_width']==6144 and r['response_width']==4096 and r['optimizer_steps']==2
  assert [s['step'] for s in r['optimizer_step_records']]==[1,2]
  assert all(math.isfinite(s[k]) and s[k]>0 for s in r['optimizer_step_records'] for k in ('grad_norm','tracked_parameter_delta','seconds'))
  assert all(math.isfinite(s['loss']) for s in r['optimizer_step_records'])
  assert r['student_master_parameter_dtypes']==['torch.float32'] and r['student_mixed_precision_parameter_dtype']=='torch.bfloat16'
  assert r['teacher_active_runtime']['parameter_dtypes']==['torch.float32'] and r['teacher_active_runtime']['attention']=='eager' and r['teacher_returned_runtime']['active_device']=='cpu'
  assert r['synthetic_prompt_runtime']['actual_user_context_adapter_pass'] is True
  assert all(v==4 for v in r['decoder_forward_pre_hook_counts'].values())
  assert r['minimum_required_free_fraction']==.15 and r['estimated_peak_device_used_bytes']==r['external_baseline_used_bytes']+r['peak_reserved_bytes']+1024**3 and r['estimated_peak_device_used_bytes']<=.85*r['gpu_total_bytes']
  assert all(p['num_ooms']==0 for p in r['memory_phases'])
  rankpeaks.append(dict(rank=i,uuid=workers[i]['actual_cuda_uuid'],pytorch_peak_allocated_bytes=r['peak_allocated_bytes'],pytorch_peak_reserved_bytes=r['peak_reserved_bytes'],external_baseline_bytes=r['external_baseline_used_bytes'],nccl_slack_estimate_bytes=1024**3,estimated_peak_device_used_bytes=r['estimated_peak_device_used_bytes'],gpu_total_bytes=r['gpu_total_bytes']))
 return dict(status='RAW_CAPACITY_CROSSCHECK_PASS_NOT_FORMAL_RL_ACCEPTANCE',job_id=JOB,source=SOURCE,http24_seconds=http['seconds'],resource_rows=len(rows),matched_9app_times=matched,nvml_sampled_max_used_mib=observed,sampled_max_scope='2-second sampled total GPU usage, not continuous peak',rank_peaks=rankpeaks,scope='Actual synthetic micro2 twoAdam capacity only; requires separately reviewed real Slurm terminal/source/asset receipts; excludes vLLM/state/first5/RL effect')
def main():
 p=argparse.ArgumentParser();p.add_argument('evidence');p.add_argument('--output',required=True);a=p.parse_args(); result=audit(pathlib.Path(a.evidence))
 with open(a.output,'x') as f:json.dump(result,f,indent=2,sort_keys=True)
 print(json.dumps({k:v for k,v in result.items() if k not in ('rank_peaks','matched_9app_times')},sort_keys=True))
if __name__=='__main__':main()
