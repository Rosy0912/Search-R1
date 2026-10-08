"""Read-only runtime arithmetic audit; no checkpoint gate, training or mutation."""
import json,hashlib,math,statistics,time,subprocess
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
RUN=ROOT/'runs/bapo-instruct300-formal300-3231'
assert (RUN/'source-commit.txt').read_text().strip()=='39cdcdd1404213b897bb0de99cc52d72707682f0'
raw=(RUN/'train.log').read_bytes();decoder=json.JSONDecoder()
lines=raw.splitlines(keepends=True)
end=max(i for i,line in enumerate(lines) if b'BAPO_TRAIN_METRICS_JSON='in line)
prefix=b''.join(lines[:end+1]);text=prefix.decode()
def events(marker):return [decoder.raw_decode(line.split(marker,1)[1])[0]for line in text.splitlines()if marker in line]
steps=events('BAPO_TRAIN_METRICS_JSON=');ranks=events('ACTOR_RANK_UPDATE_JSON=');issues=[];alerts=[];rows=[]
def check(ok,message):
 if not ok:issues.append(message)
def close(x,y):return math.isfinite(float(x))and math.isfinite(float(y))and math.isclose(float(x),float(y),rel_tol=1e-10,abs_tol=1e-10)
check([e['step']for e in steps]==list(range(1,len(steps)+1)),'Outer sequence gap or duplicate')
for e in steps:
 i=e['step'];m=e['metrics'];rr=[r for r in ranks if r['scheduler_last_epoch']==i]
 check(sorted(r['rank']for r in rr)==list(range(8)),f'Outer{i} rank coverage')
 if len(rr)!=8:continue
 for key in ['actor/pg_loss','actor/kl_loss','actor/entropy_loss','actor/ppo_kl','actor/pg_clipfrac']:
  vals=[r['metrics'][key]for r in rr];check(all(len(v)==20 for v in vals),f'Outer{i}/{key} mini count')
  v=statistics.mean(statistics.mean(values[j]for values in vals)for j in range(20))
  check(close(v,m[key]),f'Outer{i}/{key} allrank mean differs')
 for r in rr:
  check(r['optimizer_steps']==20 and close(r['lr_before'],1e-6*min(i-1,114)/114)and close(r['lr_after'],1e-6*min(i,114)/114),f'Outer{i}/rank{r["rank"]} optimizer/LR')
 norms=rr[0]['metrics']['actor/grad_norm'];check(all(len(r['metrics']['actor/grad_norm'])==20 and all(close(a,b)for a,b in zip(norms,r['metrics']['actor/grad_norm']))for r in rr),'Global norm replicas differ at'+str(i))
 check(close(statistics.mean(norms),m['actor/grad_norm'])and close(max(norms),m['bapo/grad_norm_max']),'Norm summary differs at'+str(i))
 check(m['actor/optimizer_steps']==20 and m['actor/scheduler_outer_steps']==i and m['bapo/actual_outer_updates']==i,'Driver clock differs at'+str(i))
 for key in ['bapo/old_logprob_nonfinite','bapo/new_logprob_nonfinite','bapo/mask_binary_errors','bapo/mask_on_observation_tokens','bapo/mask_on_padding_tokens','bapo/model_tokens_missing_loss','rollout/context_mismatches']:
  check(m[key]==0,f'Outer{i}/{key} nonzero')
 check(m['bapo/stage_evaluated_count']==64 and 2560<=m['rollout/trajectories']<=7680,'Stage/rollout budget at'+str(i))
 check(all(not isinstance(v,(int,float))or math.isfinite(v)for v in m.values()),'Nonfinite driver metric at'+str(i))
 if abs(m['actor/pg_loss'])>=1 or m['bapo/grad_norm_max']>=1000:alerts.append({'step':i,'kind':'Large finite PG/norm diagnostic; not automatic cancellation'})
 rows.append({k:m[k]for k in ['critic/rewards/mean','actor/pg_loss','actor/kl_loss','actor/grad_norm','bapo/grad_norm_max','actor/lr','actor/optimizer_steps','actor/scheduler_outer_steps','bapo/early_stage','bapo/stalled','bapo/stage_validation','timing_s/step','timing_s/gen','timing_s/update_actor','rollout/trajectories']})
 rows[-1]['step']=i
out={'scope':'READ_ONLY_RUNTIME_RAW_RANK_MEAN_AND_CLOCK_AUDIT_NOT_FINAL_ACCEPTANCE','job_id':3231,'run':str(RUN),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'actual_completed_outer':len(steps),'raw_rank_records':len(ranks),'train_prefix_bytes':len(prefix),'train_prefix_sha256':hashlib.sha256(prefix).hexdigest(),'issues':issues,'alerts':alerts,'steps':rows,'source':subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()}
print(json.dumps(out,indent=2,allow_nan=False));raise SystemExit(2 if issues else 0)
