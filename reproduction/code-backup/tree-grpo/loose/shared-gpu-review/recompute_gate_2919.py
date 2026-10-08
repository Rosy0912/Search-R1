import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];p=ROOT/'outputs/evidence/shared-gate-2919'
r=json.loads((p/'retrieval-gate.json').read_text());b=json.loads((ROOT/'outputs/evidence/shared-original-gpu-2854.json').read_text());m=json.loads((p/'launch-manifest.json').read_text());c=json.loads((p/'retriever-coexistence.json').read_text());out=json.loads((p/'run-result.json').read_text());cost=json.loads((p/'cost.json').read_text())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
assert r['baseline_sha256']==sha(ROOT/'outputs/evidence/shared-original-gpu-2854.json')
assert r['source_sha256']==sha(ROOT/'work/probe_tree_shared_retrieval.py')
assert m['supervisor_source_sha256']==sha(ROOT/'work/tree_shared_supervisor.py') and m['owned_runtime_source_sha256']==sha(ROOT/'work/tree_shared_owned_runtime.py')
assert r['asset_identity_verified'] and r['actual_asset_identity']['index']==b['asset_identity']['index'] and r['actual_asset_identity']['corpus']==b['asset_identity']['corpus']
assert m['config_sha256']==sha(p/'resolved.yaml')
assert out['code_commit']==m['code_commit']=='87e4080fe94869c9f1978b9abf7b3931c0141f54'
assert out['status']=='REAL_LOCAL_GPU_RETRIEVAL_GATE_READY_FOR_REVIEW' and out['actual_training_accepted'] is False and cost['exit_code']==0
assert cost['allocated_gpus']==8 and cost['start_epoch']<cost['end_epoch']
assert '/job_2919/step_batch' in m['cgroup'] and m['cuda_visible_devices']=='0,1,2,3,4,5,6,7'
assert set(g['uuid'] for g in c['gpus'].values())==set(g['uuid'] for g in m['gpu_leases']) and len(m['gpu_leases'])==8
fixed={};dif=[];summaries=[]
for item in r['measurements']:
 n=item['queries'];k=item['k'];rows=item['results'];assert len(rows)==n and math.isfinite(item['wall_seconds']) and item['wall_seconds']>0
 for row in rows:
  assert len(row['corpus_ids'])==len(row['scores'])==len(row['contents_sha256'])==k
  assert len(set(row['corpus_ids']))==k and all(math.isfinite(s) for s in row['scores'])
 label=item.get('label','fixed14');key=(label,k)
 if item['repeat']==0:fixed[key]=rows
 else:assert rows==fixed[key]
 if label=='fixed14' and item['repeat']==0:
  for i,row in enumerate(rows):
   ref=b['results'][i][str(k)];L=dict(zip(ref['corpus_ids'],zip(ref['scores'],ref['contents_sha256'])));R=dict(zip(row['corpus_ids'],zip(row['scores'],row['contents_sha256'])));shared=L.keys()&R.keys()
   d={'query':i,'k':k,'exact_rank':row['corpus_ids']==ref['corpus_ids'],'overlap':len(shared),'content_shared':all(L[x][1]==R[x][1] for x in shared),'content_rank':row['contents_sha256']==ref['contents_sha256'],'max_score_delta':max(abs(L[x][0]-R[x][0]) for x in shared),'score_exact':row['scores']==ref['scores']};dif.append(d)
 elif label.startswith('repeated'):
  count=0
  for i,row in enumerate(rows):
   old=fixed[('fixed14',k)][i%14]
   assert row['corpus_ids']==old['corpus_ids'] and row['contents_sha256']==old['contents_sha256'];count+=1
  assert count==512
 if n==512:
  assert item['results_sha256']==hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
  assert item['concurrent_health_samples'] and all(x['instance_equal'] and math.isfinite(x['wall_seconds']) for x in item['concurrent_health_samples'])
 summaries.append(dict(label=label,k=k,repeat=item['repeat'],seconds=item['wall_seconds'],health_n=len(item.get('concurrent_health_samples',[])),health_max=max([v['wall_seconds'] for v in item.get('concurrent_health_samples',[])],default=None)))
for k in (3,50):
 calc=[d for d in dif if d['k']==k];print(k,'exact',sum(d['exact_rank'] for d in calc),'contents',sum(d['content_rank'] for d in calc),'delta',max(d['max_score_delta'] for d in calc))
assert r['over_512_rows']==513 and r['empty_query_seconds']>=0 and r['over_512_query_seconds']>0
samples=[json.loads(line) for line in (p/'resource-samples.jsonl').read_text().splitlines()]
for row in samples:
 assert row['job_id']=='2919' and 'event' not in row and len(row['gpus'])==8
 for g in row['gpus']:
  assert g['foreign_memory_fraction']<=.15 and g['foreign_sm_upper']<=15
  assert abs(g['foreign_memory_fraction']-sum(f['memory_bytes'] for f in g['foreign'])/g['total_memory_bytes'])<1e-12
assert all(g['memory_used']/g['memory_total']<=.15 and g['utilization']<=15 for g in c['gpus'].values())
assert str(c['retriever_pid']) in (p/'retriever.log').read_text()
assert m['initial_gpu']['processes']==[x for x in c['snapshot']['processes'] if str(c['retriever_pid']) not in x]
slurm=(ROOT/'outputs/evidence/shared-gate-2919-slurm.txt').read_text();assert 'COMPLETED' in slurm and 'ExitCode=0:0' in slurm and 'NumCPUs=192' in slurm and 'gres:gpu:8' in slurm
cleanup=(ROOT/'outputs/evidence/shared-gate-2919-cleanup.txt').read_text().splitlines();assert len(cleanup)==16 and all(line.endswith(', 1131, 0') for line in cleanup[:8]);assert sorted(int(x) for x in cleanup[8:])==sorted(int(x.split(', ')[1]) for x in m['initial_gpu']['processes'])
result={'status':'PASS_GATE_ONLY','source_commit':out['code_commit'],'query_differences':dif,'measurements':summaries,'guard_samples':len(samples),'sampled_total_gpu_peak_gib':max(g['total_used_memory_bytes'] for row in samples for g in row['gpus'])/2**30,'sampled_foreign_memory_fraction_peak':max(g['foreign_memory_fraction'] for row in samples for g in row['gpus']),'sampled_foreign_sm_upper_peak':max(g['foreign_sm_upper'] for row in samples for g in row['gpus']),'coexist_total_memory_fraction_peak':max(g['memory_used']/g['memory_total'] for g in c['gpus'].values()),'slurm_wall_seconds':140,'slurm_allocated_cpu':192,'slurm_requested_cpu':32,'slurm_gpu_hours':140*8/3600,'supervisor_wall_seconds':out['wall_seconds'],'empty_seconds':r['empty_query_seconds'],'over513_seconds':r['over_512_query_seconds'],'retriever_pid':c['retriever_pid'],'evidence_sha256':{x.name:sha(x) for x in p.iterdir() if x.is_file()}}
(ROOT/'work/shared-gpu-review/gate-2919-recomputed.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('query_differences','evidence_sha256','measurements')},indent=2))
