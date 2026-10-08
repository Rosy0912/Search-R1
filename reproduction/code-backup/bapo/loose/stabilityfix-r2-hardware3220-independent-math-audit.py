from pathlib import Path
import json,hashlib,math,zipfile,sys,importlib.util,torch
b=Path.cwd();cap=b/'work/evidence/stabilityfix-r2-hardware3220-raw';raw=cap/'raw';inv=json.loads((cap/'inventory.json').read_text());root=Path(inv['root']);run=raw/'runs/bapo-instruct300-hardware-3220';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(cap/'raw-evidence.tar.gz')=='b8de446639bcdafe702daa188608a33e0d9c86d2ff4676bc07673deefef074f6'
for x in inv['files']:
 p=raw/x['archive_path'];assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes']
assert inv['source_clean'] and inv['source_matches'] and not inv['frozen_tool_hash_mismatches']
pids=set();storages=0;norms=[];counts=0
for rank in range(8):
 c=json.loads((run/f'native-count/rank-{rank}.json').read_text());assert len(c['records'])==4
 for x,e in zip(c['records'],[(28,6.5,3.5),(0,6.5,.125),(27,4.375,3.375),(0,.125,.125)]):assert (x['global_tokens'],x['global_row_denominator'],x['token_denominator'])==e;counts+=1
 s=json.loads((run/f'native-state/rank-{rank}.json').read_text());assert len(s['records'])==2
 for x in s['records']:
  assert x['same_model_adam_scheduler_objects'] and x['next_update_full_state_exact_equal'];norms+=x['gradient_norms'];paths=[]
  for k in ['initial_snapshot','actual_worker_roundtrip','oracle_snapshot','reloaded_snapshot']:
   z=x[k];p=raw/Path(z['path']).relative_to(root);assert sha(p)==z['sha256'] and json.loads(p.with_suffix('.json').read_text())==z;assert z['model_state_dict_type']=='SHARDED_STATE_DICT' and z['model_dtensor_entries']>0;pids.add(z['pid']);paths.append(p)
  assert x['initial_snapshot']['state_digest']==x['actual_worker_roundtrip']['state_digest'] and x['oracle_snapshot']['state_digest']==x['reloaded_snapshot']['state_digest']
  for v in x['next_update_components'].values():assert v['exact_equal'] and v['no_roundtrip_sha256']==v['reloaded_sha256']
  for aa,cc in [paths[:2],paths[2:]]:
   with zipfile.ZipFile(aa)as a,zipfile.ZipFile(cc)as c:
    av={n.split('/',1)[1]:a.read(n)for n in a.namelist()if '/data/'in n};cv={n.split('/',1)[1]:c.read(n)for n in c.namelist()if '/data/'in n};assert av==cv;storages+=len(av)
assert len(pids)==8 and all(math.isfinite(n)for n in norms)
# Independent NumPy FP64 derivative; independently reviewed BF16 analytic VJP.
sys.path.insert(0,str(b/'work/bapo-e1-300-stabilityfix'))
from verl import DataProto
spec=importlib.util.spec_from_file_location('toy',raw/'tools/native-actor-fixture.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.DataProto=DataProto
spec=importlib.util.spec_from_file_location('analytic',b/'work/e1-300-countfix/independent-actor-oracle.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
cache={};checked=0;maxfp64=0.;maxbf16=0.
for rank in range(8):
 v=json.loads((run/f'native-actor/rank-{rank}.json').read_text());component=json.loads((run/f'native-actor/component-only-rank-{rank}.json').read_text());assert len(v['records'])==62 and component['records']==v['records']
 probe=json.loads((run/f'native-actor/production-forward-rank-{rank}.json').read_text());assert v['production_forward_precision']==probe
 for x in probe['branch_records']:assert x['logprob_dtype']=='torch.float32' and x['finite_forward_backward'] and x['nonzero_gradient'] and x['actual_device']==f'cuda:{rank}'
 for z in v['records']:
  n=z['global_rows'];local=n//8;steps=(local+15)//16;assert len(z['gradients'])==steps
  for st,grad in enumerate(z['gradients']):
   ids=[r*local+i for r in range(8)for i in range(st*16,min((st+1)*16,local))];key=(n,z.get('empty_rank',False),z['pg_on'],z['kl_on'],z['case']=='actual_actor_global_empty_step',z['production_bf16_fp32'],z['micro_per_rank'],st)
   if key not in cache:
    if z['production_bf16_fp32']:
     data=m.make_data(n);sub=m.select(data,ids);want=m.quantized_toy_reference_gradient(sub,z['micro_per_rank'],z['pg_on'],z['kl_on'],8)
    else:want=torch.tensor(a.reference(n,ids,z.get('empty_rank',False),z['pg_on'],z['kl_on'],z['case']=='actual_actor_global_empty_step'))
    cache[key]=want
   got=torch.tensor(grad,dtype=torch.float64);want=cache[key];tol=(.03,2e-4)if z['production_bf16_fp32']else(2e-5,2e-7);torch.testing.assert_close(got,want,rtol=tol[0],atol=tol[1]);err=float((got-want).abs().max());checked+=1
   if z['production_bf16_fp32']:maxbf16=max(maxbf16,err)
   else:maxfp64=max(maxfp64,err)
print(json.dumps({'verifiedfiles':len(inv['files']),'count_records':counts,'state_tensor_storages':storages,'state_finite_norms':len(norms),'actor_records':496,'actor_gradient_vectors':checked,'independent_fp64_maxerr':maxfp64,'reviewed_bf16_analytic_maxerr':maxbf16,'actual_probe_branches':16}))
