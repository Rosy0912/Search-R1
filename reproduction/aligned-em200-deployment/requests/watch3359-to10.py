"""Read-only short observation of the existing 3359 run through complete outer10."""
import json,time,re,subprocess
from pathlib import Path
run=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/runs/formal-aligned-em200-16c570e111f947e2b0026983df2f7938-3359')
last=None;deadline=time.monotonic()+1500
while time.monotonic()<deadline:
 rows=[]
 for line in(run/'train.log').read_text().splitlines():
  marker='FRESH200_ATTEMPT_JSON='
  if marker not in line:continue
  try:d=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
  except (ValueError,TypeError):continue
  if d.get('candidate_outer',0)>=6:
   m=d.get('reward_metrics',{});r=d.get('rank_write_ledger',[])
   rows.append({'outer':d['candidate_outer'],'attempt':d['attempt_id'],'complete':d.get('accepted_complete_outer'),'status':d.get('status'),'ranks':[{'rank':x.get('rank'),'writes':x.get('optimizer_steps'),'skip':x.get('skipped_opportunities'),'complete':x.get('update_complete'),'scheduler_after':x.get('scheduler_after')}for x in r],'em':m.get('train_em/initial/em'),'any5':m.get('train_em/initial/any_of_5_em'),'retained_em':m.get('train_em/retained/em'),'stage64':m.get('bapo/stage_validation'),'pg_loss':m.get('actor/pg_loss'),'grad_min':m.get('bapo/grad_norm_min'),'grad_max':m.get('bapo/grad_norm_max'),'mask_errors':{k:m.get(k)for k in ['bapo/mask_binary_errors','bapo/mask_on_observation_tokens','bapo/model_tokens_missing_loss','bapo/mask_on_padding_tokens','bapo/new_logprob_nonfinite','bapo/old_logprob_nonfinite']}})
 signature=json.dumps(rows,sort_keys=True)
 if signature!=last:
  print(json.dumps({'job':3359,'rows':rows,'observed_epoch':time.time()}),flush=True);last=signature
 if any(x['outer']>=10 and x['complete'] is True for x in rows):break
 if not Path('/proc/1598631').exists():
  print(json.dumps({'driver_missing':True,'job':subprocess.run(['squeue','-j','3359','-h','-o','%i %T %N'],capture_output=True,text=True).stdout}),flush=True);break
 time.sleep(20)
else:print(json.dumps({'observation_window_expired':True,'not_training_terminal':True}),flush=True)
