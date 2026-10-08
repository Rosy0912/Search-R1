import pathlib,json,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');r=root/'runs/bapo-instruct300-formal300-3246';out={'job_id':3246,'time_unix':time.time(),'run_files':sorted(x.name for x in r.iterdir()),'checkpoint_files':[str(x.relative_to(r))for x in r.rglob('*')if x.is_file()and any('checkpoint' in v or 'global_step' in v for v in x.relative_to(r).parts)]}
for p in [r/'phase.log',root/'logs/slurm-formal300-3246.out',root/'logs/slurm-formal300-3246.err',root/'scheduler-formal-routefix/formal300-3246-terminal.txt']:
 if p.is_file():out[str(p)]={'bytes':p.stat().st_size,'text_tail':p.read_text(errors='replace')[-14000:]}
p=r/'resource-watch.jsonl';rows=[json.loads(x)for x in p.read_text().splitlines()];out['resource_rows']=len(rows);out['events']=[x for x in rows if x.get('event')];out['unsafe_samples']=[x for x in rows if x.get('reason')not in ['within_threshold',None]][-5:];print(json.dumps(out,allow_nan=False))
