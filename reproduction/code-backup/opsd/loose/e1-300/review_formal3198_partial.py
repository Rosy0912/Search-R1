"""Read-only local original partial-run review; not an acceptance producer."""
from pathlib import Path
import base64,json,hashlib,sys,math,re,csv,io
H=Path(__file__).resolve().parent;D=H/'formal3198-first5-observed-20261004T133909Z';R=D/'run';SOURCE='89beac273f783d8e48d7d3e34da040f753514d90';REMOTE='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004';RUN=REMOTE+'/runs/formal-3198'
sys.path.insert(0,str(H.parent/'opsd-e1-300'));from search_r1.baselines.checkpoint_gate import validate_outer,validate_evaluation
sys.path.insert(0,str(H/'formal-tools-89beac2'));import formal_supervisor as f
c=f.unique((D/'collection-original.json').read_bytes());assert c['source_head']==SOURCE and not c['source_status'] and c['job_id']=='3198' and c['host']=='h100-10' and c['uid']==0;f.slurm(c['slurm'],'3198',0)
for path,v in c['files'].items():
 raw=base64.b64decode(v['base64']);assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256'] and v['uid']==0
 if path.startswith(RUN+'/'):assert (R/path.removeprefix(RUN+'/')).read_bytes()==raw
log=(R/'train.log').read_text();rows=[f.unique(l.split('OPSD_E1_OUTER_JSON=',1)[1]) for l in log.splitlines() if 'OPSD_E1_OUTER_JSON=' in l];assert [r['actual_outer_updates'] for r in rows]==[1,2,3,4]
assert not any(m in log for m in ('OPSD_E1_STEP5_PAUSED_JSON=','OPSD_E1_STEP5_RELEASED_JSON='))
configs={r['config_sha256'] for r in rows};assert len(configs)==1
summary=[]
for row in rows:
 step=row['actual_outer_updates'];assert row['global_step']==step and row['source_commit']==SOURCE and row['driver_pid']==469564 and row['job_id']=='3198';m=row['metrics'];validate_outer(m)
 assert all(type(v) in (int,float) and math.isfinite(v) for v in m.values())
 for rank in range(8):
  s='/rank_'+str(rank)
  for k,v in {'actor/optimizer_steps':20,'actor/samples':320,'actor/skipped_empty_minibatches':0,'actor/old_log_probs_present':1,'actor/old_log_probs_finite':1,'teacher/parameters_per_rank':3085938688,'teacher/update_device_is_cuda_per_rank':1,'teacher/cuda_update_entries_cumulative_per_rank':step}.items():assert m[k+s]==v
  assert m['actor/supervised_tokens'+s]>0 and m['actor/opsd_loss_mean'+s]>=0
  assert m['teacher/scored_tokens_cumulative_per_rank'+s]>0
 assert m['actor/lr']==1e-6*(step/57)
 summary.append({'outer':step,'reported_optimizer_steps_per_rank':20,'reported_samples_per_rank':320,'reported_after_scheduler_lr':m['actor/lr'],'rank0_loss':m['actor/opsd_loss_mean/rank_0'],'step_seconds':m['timing_s/step'],'generation_search_seconds':m['timing_s/gen'],'actor_update_seconds':m['timing_s/update_actor']})
assert len(c['processes'])==1;p=c['processes']['469564'];assert p['start_ticks']=='1817999209' and re.search(r'/job_3198(?:/|$)',p['cgroup'],re.M);assert set(map(int,re.search(r'^Uid:\s+(.*)',p['status'],re.M).group(1).split()))=={0}
evals=[f.unique(l.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]) for l in log.splitlines() if 'BASELINE_EVALUATION_METRICS_JSON=' in l];assert len(evals)==1;validate_evaluation(evals[0],512);assert evals[0]['val/evaluated_count/nq']==evals[0]['val/evaluated_count/hotpotqa']==256
profile=f.unique((R/'evidence/pre-gpu-profile.json').read_bytes());from search_r1.baselines.e1_profile import validate_profile;from omegaconf import OmegaConf;validate_profile(OmegaConf.create(profile))
assert profile['baselines']['distillation']['teacher_prompt_mode']=='user_context' and profile['baselines']['distillation']['component_clip'] is None
assert profile['actor_rollout_ref']['model']['path']==profile['baselines']['distillation']['teacher_path']==REMOTE+'/assets/qwen2.5-3b-instruct-aa8e725'
env=f.unique((R/'evidence/actual-training-wrapper-environment.json').read_bytes())['values'];assert env['PYTHONPATH']==env['SEARCH_R1_REPO']==REMOTE+'/repo-89beac273f78';assert env['OPSD_E1_BINDING_MANIFEST_SHA256']=='a2237f40627281b8b70e58ab78dfe02f97d81bba3602981a4e8c134251fbc13c';assert env['OPSD_E1_RETRIEVER_URL']=='http://127.0.0.1:18084/retrieve'
service=f.unique((R/'evidence/service.json').read_bytes());owner=service['owner'];assert owner['pid']==468085 and '/job_3198/' in owner['cgroup'] and owner['uid']==0
cmd=service['command'];assert '--faiss_gpu' in cmd and cmd[cmd.index('--device')+1]=='cuda' and cmd[cmd.index('--host')+1]=='127.0.0.1' and cmd[cmd.index('--topk')+1]=='3'
http=f.unique((R/'evidence/http-top3-24.json').read_bytes());assert http['job_id']=='3198' and all(http['service'][k]==owner[k] for k in ('pid','start','uid','cgroup')) and len(http['queries'])==len(http['response']['result'])==24
for q in http['response']['result']:
 assert len(q)==3
 for x in q:assert x['document']['contents'] and math.isfinite(float(x['score']))
apps=list(csv.reader(io.StringIO(c['gpu_compute'])));gpu_uuids={r[0].strip() for r in apps};pids={int(r[1]) for r in apps};assert len(gpu_uuids)==8 and len(pids)==9 and len([r for r in apps if int(r[1])==owner['pid']])==8;assert all(len([r for r in apps if int(r[1])==p])==1 for p in pids-{owner['pid']})
for rank in range(8):
 hw=f.unique((R/f'evidence/hardware-rank-{rank}.json').read_bytes());assert hw['rank']==rank and hw['world_size']==8 and hw['source_commit']==SOURCE and hw['job_id']=='3198' and hw['actual_cuda_uuid']==hw['uuid'] and hw['uuid'] in gpu_uuids and hw['flash_finite'] is True and hw['allreduce_value']==36
result={'status':'ACTUAL_FIRST4_OBSERVATIONS_VALID_FIRST5_ACCEPTANCE_PENDING','source':SOURCE,'job_id':'3198','driver_pid':469564,'driver_start_ticks':p['start_ticks'],'actual_complete_rows':4,'config_sha256':next(iter(configs)),'outer_summaries':summary,'initial512':evals[0],'actual_nvml_gpu_uuid_count':8,'actual_nvml_pid_count':9,'actual_nvml_service_pid':owner['pid'],'actual_nvml_one_gpu_worker_pids':sorted(pids-{owner['pid']}),'resource_scope':'One current NVML snapshot, no complete sampler prefix or native rank-PID join in this collection','missing':['fifth completed outer','stage64 evaluation','unique step5 paused manifest','full native/HF/driver payload inspections','two sampled rankPID/start/UUID plus current live join','independent first5 review and external signature'],'collection_sha256':hashlib.sha256((D/'collection-original.json').read_bytes()).hexdigest(),'scope':'Actual four original rows and original runtime/config/current driver observations only; not first5 PASS/continuation/effect'}
(H/'formal3198-first4-independent-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
