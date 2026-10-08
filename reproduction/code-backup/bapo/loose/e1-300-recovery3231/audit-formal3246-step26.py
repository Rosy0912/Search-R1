import json,math,hashlib
from pathlib import Path
base=Path('work/evidence/recovery3231-formal3246-fault-raw/runs/bapo-instruct300-formal300-3246');decoder=json.JSONDecoder();ranks=[];bindings=[]
for p in base.glob('runtime/ray/*/logs/worker-*.out'):
 for line in p.read_text(errors='replace').splitlines():
  if 'ACTOR_RANK_UPDATE_JSON='in line:
   x=decoder.raw_decode(line.split('ACTOR_RANK_UPDATE_JSON=',1)[1].lstrip())[0]
   if x['scheduler_last_epoch']==26:ranks.append(x);bindings.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(ranks)==8 and sorted(x['rank']for x in ranks)==list(range(8));assert all(x['optimizer_steps']==20 for x in ranks)
metrics=json.loads(Path('work/evidence/recovery3231-formal3246-live-progress5.json').read_text())['last_driver_metrics']['metrics'];out={}
for k in ['actor/pg_loss','actor/kl_loss','actor/pg_clipfrac','actor/grad_norm','actor/entropy_loss','actor/ppo_kl']:
 values=[v for x in ranks for v in x['metrics'][k]];assert len(values)==160 and all(math.isfinite(v)for v in values);mean=sum(values)/len(values);assert math.isclose(mean,metrics[k],rel_tol=1e-12,abs_tol=1e-12);out[k]={'records':len(values),'mean':mean,'driver':metrics[k],'min':min(values),'max':max(values)}
assert all(x['metrics']['bapo/old_logprob_nonfinite']==0 and x['metrics']['bapo/new_logprob_nonfinite']==0 for x in ranks)
assert all(math.isclose(x['lr_before'],1e-6*25/114,rel_tol=1e-12)and math.isclose(x['lr_after'],1e-6*26/114,rel_tol=1e-12)for x in ranks)
record={'status':'PASS_ACTUAL_STEP26_ALL8RANK_AGGREGATION_FINITE_AND_WARMUP114','job_id':3246,'step':26,'ranks':sorted(x['rank']for x in ranks),'optimizer_updates_this_outer':20,'scheduler_epoch':26,'allrank_metrics':out,'raw_bindings':bindings,'scope':'Actual failed3246 branch26 only; no checkpoint26 or final300 acceptance'}
Path('work/evidence/recovery3231-formal3246-step26-allrank-audit.json').write_text(json.dumps(record,indent=2)+'\n');print(record['status'])
