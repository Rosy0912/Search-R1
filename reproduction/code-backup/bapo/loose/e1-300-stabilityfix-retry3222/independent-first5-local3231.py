from pathlib import Path
import json,hashlib,math,tarfile,re,statistics
B=Path('work/evidence/first5-3231-actual/first5-extraction-3231-20261005T0435');raw=B/'raw'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
EP=B/'evidence.json';assert sha(EP)=='7587fe09a066a63d34161771e2b47cf24c989c36700f42c8db5d9958a7f69803'
e=json.loads(EP.read_text());RUN=e['run_root'];ROOT=RUN.rsplit('/runs/',1)[0]
assert e['job_id']==3231 and e['expected_driver_pid']==40564 and e['source_expected']=='39cdcdd1404213b897bb0de99cc52d72707682f0'
assert e['extraction_complete']is True and e['issues']==[] and e['actual_training_accepted']is False
assert e['tool']['sha256']=='a27674a26f4958f9514c19be6f9f5ba5fab9bfd640c586b5bf293266da29ee74'
archive=Path('work/evidence/first5-3231-actual.tar.gz');assert sha(archive)=='d4a87925e160f2e71fcd35244f36a5b46bf0b989a140f608aaf21bd182f20bd1'
with tarfile.open(archive)as tf:
 names=set();files=0
 for m in tf.getmembers():
  p=Path(m.name);assert not p.is_absolute()and '..'not in p.parts and m.name not in names;names.add(m.name)
  assert m.isdir()or m.isfile()
  if m.isfile():
   q=B.parent/m.name;data=tf.extractfile(m).read();assert q.is_file()and q.read_bytes()==data and len(data)==m.size;files+=1
 assert files==27
for snapshot in e['raw_snapshots'].values():
 q=raw/Path(snapshot['snapshot_path']).name;assert sha(q)==snapshot['sha256']and q.stat().st_size==snapshot['bytes']
prefix=e['train_log_prefix'];assert sha(raw/'step5-prefix.log')==prefix['sha256']and (raw/'step5-prefix.log').stat().st_size==prefix['bytes']
assert prefix['live_path']==RUN+'/train.log'
text=(raw/'step5-prefix.log').read_text()
def events(marker):return [json.loads(x.split(marker,1)[1])for x in text.splitlines()if marker in x]
train=events('BAPO_TRAIN_METRICS_JSON=');ranks=events('ACTOR_RANK_UPDATE_JSON=');paused=events('E1_STEP5_PAUSED_JSON=')
assert train==e['train_metrics']and ranks==e['rank_updates']and len(train)==5 and len(ranks)==40
assert [r['step']for r in train]==list(range(1,6))and {(r['rank'],r['scheduler_last_epoch'])for r in ranks}=={(r,s)for r in range(8)for s in range(1,6)}
assert len(paused)==1 and paused[0]['pid']==40564 and paused[0]['manifest_sha256']=='c208fcdb73f808fc3cd720e5321a3fb4820602fffd348352b2d6671f7136f821'
assert not events('E1_STEP5_RELEASED_JSON=')
manifest=json.loads((raw/'checkpoint-manifest.json').read_text());assert sha(raw/'checkpoint-manifest.json')==paused[0]['manifest_sha256']
for k,v in e['release_identity'].items():
 if k!='checkpoint_manifest_sha256':assert manifest[k]==v
assert manifest['warmup_steps']==114 and manifest['horizon']==300 and manifest['global_step']==manifest['actual_outer_updates']==5
assert manifest['run_root']==RUN and manifest['source_commit']==e['source_expected']
bindings={r['path']:r['sha256']for r in e['release_evidence_bindings']};assert len(bindings)==60
assert RUN+'/train.log'not in bindings and bindings[RUN+'/source-commit.txt']==sha(raw/'source-commit.txt')
assert bindings[prefix['snapshot_path']]==prefix['sha256']
assert {r['path']:r['sha256']for r in manifest['files']}=={r['path']:r['sha256']for r in e['checkpoint_members']}
for r in e['checkpoint_members']:assert bindings[r['path']]==r['sha256']
mean=('actor/pg_loss','actor/kl_loss','actor/entropy_loss','actor/ppo_kl','actor/pg_clipfrac')
sumkeys=('bapo/loss_tokens','bapo/old_logprob_nonfinite','bapo/new_logprob_nonfinite')
replica=('actor/grad_norm','actor/optimizer_steps','actor/scheduler_outer_steps','actor/lr','actor/lr_before_outer','actor/kl_coef','actor/skipped_empty_minibatches','bapo/grad_norm_min','bapo/grad_norm_max')
best=-math.inf;stall=0;early=True;summary=[]
for event in train:
 outer=event['step'];m=event['metrics'];rr=[r for r in ranks if r['scheduler_last_epoch']==outer];records=[r['metrics']for r in rr]
 assert m['bapo/actual_outer_updates']==outer and m['bapo/stage_evaluated_count']==m['bapo/stage_cost/trajectories']==64
 for key in mean:
  assert all(len(r[key])==20 and all(math.isfinite(x)for x in r[key])for r in records)
  want=math.fsum(math.fsum(r[key][i]for r in records)/8 for i in range(20))/20
  assert math.isclose(m[key],want,rel_tol=1e-10,abs_tol=1e-13),(outer,key)
 for key in sumkeys:assert m[key]==sum(r[key]for r in records)
 for key in replica:
  for record in records:assert record[key]==records[0][key],(outer,key)
  want=statistics.mean(records[0][key])if isinstance(records[0][key],list)else records[0][key]
  assert math.isclose(m[key],want,rel_tol=1e-10,abs_tol=1e-13),(outer,key)
 assert math.isclose(m['mfu/actor'],statistics.mean(r['mfu/actor']for r in records),rel_tol=1e-10)
 assert m['actor/optimizer_steps']==20 and m['actor/scheduler_outer_steps']==outer and m['actor/skipped_empty_minibatches']==0
 assert math.isclose(m['actor/lr_before_outer'],1e-6*(outer-1)/114,rel_tol=1e-12,abs_tol=1e-18)
 assert math.isclose(m['actor/lr'],1e-6*outer/114,rel_tol=1e-12,abs_tol=1e-18)
 for r in rr:assert r['lr_before']==r['metrics']['actor/lr_before_outer']and r['lr_after']==r['metrics']['actor/lr']and r['optimizer_steps']==20
 assert m['bapo/grad_norm_min']==min(records[0]['actor/grad_norm'])and m['bapo/grad_norm_max']==max(records[0]['actor/grad_norm'])
 audits=('bapo/mask_binary_errors','bapo/mask_on_observation_tokens','bapo/mask_on_padding_tokens','bapo/model_tokens_missing_loss','bapo/new_logprob_nonfinite','bapo/old_logprob_nonfinite')
 assert all(m[x]==0 for x in audits)
 score=m['bapo/stage_validation'];assert 0<=score<=1 and math.isclose(score*64,round(score*64),abs_tol=1e-9)
 if score>best:best=score;stall=0
 else:stall+=1
 if stall>=5:early=False
 assert m['bapo/stalled']==stall and m['bapo/early_stage']==int(early)
 summary.append(dict(outer=outer,stage_correct=round(score*64),stage_questions=64,pg=m['actor/pg_loss'],kl=m['actor/kl_loss'],grad_mean=m['actor/grad_norm'],grad_min=m['bapo/grad_norm_min'],grad_max=m['bapo/grad_norm_max'],step_seconds=m['timing_s/step'],stage_seconds=m['bapo/stage_wall_seconds'] if 'bapo/stage_wall_seconds'in m else m.get('timing_s/stage_validation')))
assert e['expected_bapo']==dict(best=best,stalled=stall,early=early,patience=5)==e['driver_state']['bapo']==e['algorithm_state']['bapo']
full=events('FULL_STATE_RANK_JSON=');assert len(full)==8 and {r['rank']for r in full}==set(range(8))
reports=e['native_state']['reports'];assert {r['rank']for r in reports}==set(range(8))
for report in reports:
 assert report in full and report['status']=='EXACT_FULL_STATE_RELOAD_OK'and report['world_size']==8 and report['step']==report['scheduler_last_epoch']==5
 assert report['model_state_dict_type']=='SHARDED_STATE_DICT'and report['model_dtensor_entries']==435 and report['optimizer_state_entries']==37
 assert report['sha256']==bindings[report['path']]
assert len({r['pid']for r in reports})==8 and 40564 not in {r['pid']for r in reports}
for before,after in zip(e['actual_processes'],e['actual_processes_after_cpu']):
 assert before['pid']==after['pid']and before['start_ticks']==after['start_ticks']and after['state']not in ('Z','X')
 assert after['selected_env']['SLURM_JOB_ID']=='3231'and after['selected_env']['BAPO_RUN_ID']==RUN and after['selected_env']['RAY_DEDUP_LOGS']=='0'
uuids=set(e['resources']['allocated_uuids']);assert len(uuids)==8 and set(e['resources_after_cpu']['allocated_uuids'])==uuids
for lease in (e['actual_uuid_leases'],e['actual_uuid_leases_after_cpu']):
 assert {r['uuid']for r in lease['leases']}==uuids and len(lease['leases'])==8
 pid=lease['guard']['pid'];assert all('WRITE '+str(pid)+' 'in r['exclusive_lock_record']for r in lease['leases'])
watch=[json.loads(x)for x in (raw/'resource-watch-after-cpu.jsonl').read_text().splitlines()];assert len(watch)==211 and all(not r['pause']for r in watch)
foreign={(p['gpu_uuid'],p['pid'],p['memory_mib'])for p in watch[0]['external_processes']}
for row in watch:assert {(p['gpu_uuid'],p['pid'],p['memory_mib'])for p in row['external_processes']}==foreign and {g['uuid']for g in row['allocated_gpus'].values()}==uuids
assert sha(raw/'hardware-acceptance.json')=='f762e8c9d5b6af33e0c0f1cb2883b4d25a78dcd05d905b6780fa7911fb1816db'
assert sha(raw/'slurm-current.txt')=='4e5d28cd7dbad0840c62f883c6b91ad7e05c1035b0320bd021b7b262e53a98cd'
slurm=(raw/'slurm-current.txt').read_text();assert 'JobId=3231 'in slurm and 'JobState=RUNNING'in slurm and 'Restarts=0'in slurm and 'NodeList=h100-5'in slurm
out=dict(status='PASS_INDEPENDENT_FIRST5_RAW_MATH_AND_IDENTITY',job=3231,outer=5,rank_records=40,mini_per_rank=20,summary=summary,bapo_replay=dict(best=best,stalled=stall,early=early,patience=5),native_reports=8,checkpoint_members=len(e['checkpoint_members']),immutable_release_bindings=len(bindings),resource_samples=len(watch),resource_pause_samples=0,foreign_stable_pids=sorted(p[1]for p in foreign),full_archive_regular_files=27,scope='FIVE_OUTER_ENGINEERING_ACCEPTANCE_ONLY_NOT_EFFECT_SUPERIORITY')
Path('work/evidence/first5-3231-independent-local-review.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
