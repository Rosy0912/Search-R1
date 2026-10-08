"""Read-only actual job3231 training/evaluation and pause identity; no submission or GPU operations."""
import pathlib,json,subprocess,collections,datetime,re
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
job=3231;run=root/'runs'/f'bapo-instruct300-formal300-{job}'
out={'job_id':job,'run':str(run),'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=subprocess.run(['scontrol','show','job',str(job)],capture_output=True,text=True);out['slurm']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
if run.exists():
 assert not run.is_symlink() and run.resolve()==run
 if (run/'source-commit.txt').exists():
  out['source']=(run/'source-commit.txt').read_text().strip();assert out['source']=='39cdcdd1404213b897bb0de99cc52d72707682f0'
 lines=(run/'train.log').read_text(errors='replace').splitlines() if (run/'train.log').exists() else []
 def events(marker):
  found=[]
  for line in lines:
   if marker in line:found.append(json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0])
  return found
 outer=events('BAPO_TRAIN_METRICS_JSON=');out['actual_completed_outer']=max([r['step']for r in outer],default=0);out['latest3_outer']=outer[-3:]
 out['evaluation_events']=events('BASELINE_EVALUATION_METRICS_JSON=')
 initial_lines=[]
 for line in lines:
  if 'BASELINE_EVALUATION_METRICS_JSON=' in line:break
  initial_lines.append(line)
 out['initial_generation_batch_markers']=sum('ACTIVE_TRAJ_NUM:' in line for line in initial_lines)
 out['initial_generation_progress_scope']='Generation-loop completion markers only; not scored questions, training updates or evaluation acceptance'
 out['paused_events']=events('E1_STEP5_PAUSED_JSON=')
 out['released_events']=events('E1_STEP5_RELEASED_JSON=')
 epochs=re.findall(r'epoch (\d+), step (\d+)', '\n'.join(lines))
 out['latest_epoch_step']=[int(x)for x in epochs[-1]]if epochs else None
 ranks=events('ACTOR_RANK_UPDATE_JSON=');out['rank_update_records']=len(ranks);out['rank_record_counts']=dict(collections.Counter(str(r.get('rank'))for r in ranks))
 out['train_tail']=[x[:1000]for x in lines[-8:]]
 health=run/'retriever-health.json';out['retriever_health_file_exists']=health.is_file();out['retriever_health_bytes']=health.stat().st_size if health.is_file() else 0
 watch=run/'resource-watch.jsonl'
 if watch.exists():
  rows=[json.loads(x)for x in watch.read_text().splitlines() if x];out['resource_samples']=len(rows);out['resource_pauses']=sum(bool(x.get('pause'))for x in rows);out['latest_watch']=rows[-1]if rows else None
print(json.dumps(out,ensure_ascii=False,indent=2))
