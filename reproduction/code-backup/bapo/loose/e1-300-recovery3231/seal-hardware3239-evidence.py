import pathlib,json,hashlib,sys,importlib.util,tarfile,time,resource,re
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');run=root/'runs/bapo-instruct300-hardware-3239';source='7de6494bff08f35ffeefbf23ed9a9e1b73e5612e'
start=time.monotonic();terminal=sys.stdin.buffer.read()
for key,val in [('JobId','3239'),('JobState','COMPLETED'),('ExitCode','0:0'),('DerivedExitCode','0:0'),('Restarts','0'),('NodeList','h100-5')]:assert re.search(rb'(?:^|\s)'+key.encode()+b'='+val.encode()+rb'(?:\s|$)',terminal)
assert run==run.resolve() and (run/'source-commit.txt').read_text().strip()==source
assert json.loads((run/'own-process-cleanup.json').read_text())['remaining']==[]
assert (run/'HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY').exists()
with (run/'slurm-terminal.txt').open('xb')as f:f.write(terminal)
spec=importlib.util.spec_from_file_location('actual_bound_guard_journal',root/'tools/guard_journal.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
summary=helper.replay(run/'resource-watch.jsonl',3239,str(run),source,root/'tools/resource_watch.py')
with (run/'guard-summary.json').open('x')as f:json.dump(summary,f,indent=2,allow_nan=False)
paths=[]
for p in run.rglob('*'):
 if any(part in ('runtime','cache','tmp')for part in p.relative_to(run).parts):continue
 assert not p.is_symlink(),str(p)
 if p.is_file():paths.append(p)
paths += [root/'source-expected.txt',root/'common-data/manifest.json',root/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json']
paths += list((root/'tools').glob('*'))
paths += [root/'logs'/n for n in ['launch-approval.json','tools-manifest.json','cpu-gate.json','cpu-gate.log','cpu-gate-source.txt','recovery-source-independent-code-approval.json','RECOVERY25-FINAL-INDEPENDENT-SOURCE-REVIEW.md','PRODUCTION27-LAUNCH-TOOLS-INDEPENDENT-REVIEW.md','slurm-hardware-3239.out','slurm-hardware-3239.err']]
paths += [root/'repo'/n for n in ['search_r1/baselines/distillation.py','search_r1/baselines/objectives.py','search_r1/baselines/recovery.py','search_r1/baselines/actor_metrics.py','verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py','verl/trainer/config/ppo_bapo_e1_300.yaml','verl/workers/sharding_manager/fsdp_vllm.py']]
items=[];small=[]
for p in sorted(set(paths)):
 assert p.is_file()and not p.is_symlink()and p==p.resolve()and root in p.parents,str(p)
 before=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(p)
 items.append({'path':str(p),'sha256':h.hexdigest(),'bytes':after.st_size})
 if after.st_size<=20*1024*1024:small.append(p)
record={'status':'ACTUAL_COMPLETED_HARDWARE_FULL_HASH_INVENTORY_NOT_INDEPENDENT_ACCEPTANCE','job_id':3239,'source_commit':source,'run_root':str(run),'evidence':items,'wall_seconds':time.monotonic()-start,'process_cpu_seconds':time.process_time(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'slurm_reserved_wall_seconds':1236,'slurm_reserved_gpu_hours':1236*8/3600,'guard_summary':summary,'large_files_preserved_remotely':len(items)-len(small)}
inv=run/'actual-fullhash-inventory.json'
with inv.open('x')as f:json.dump(record,f,indent=2,allow_nan=False)
archive=root/'logs/hardware3239-raw-evidence.tar.gz'
with tarfile.open(archive,'x:gz')as t:
 for p in small+[inv]:t.add(p,arcname=str(p.relative_to(root)),recursive=False)
h=hashlib.sha256()
with archive.open('rb')as f:
 for block in iter(lambda:f.read(1048576),b''):h.update(block)
print(json.dumps({'status':record['status'],'archive':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest(),'files':len(items),'large_files':record['large_files_preserved_remotely'],'fullhash_wall_seconds':record['wall_seconds'],'guard_summary':summary}),flush=True)
