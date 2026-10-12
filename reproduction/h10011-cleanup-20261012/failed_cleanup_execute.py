import pathlib,json,os,time,shutil,subprocess
PLAN = {'epoch': 1791782863.6802583, 'scope': 'BAPO only', 'required_hf_status': 'NOT_REQUIRED_FOR_FAILED_SUPERSEDED_GROUP', 'preserved': ['outer200 complete recovery', 'all source checkouts', 'necessary logs and metrics', 'TOPD and all shared models/retrieval'], 'delete': [{'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v7-20261009/cache', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 4096}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v10-20261009/cache', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 28161833933}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v10-20261009/runs/formal-aligned-em200-ecbddc0404f041babd8fef3b3a9d7a1e-3358/diagnostics/worker-events', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 25937213369}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v10-20261009/runs/formal-aligned-em200-ecbddc0404f041babd8fef3b3a9d7a1e-3358/checkpoints', 'reason': 'Completed failed/superseded BAPO experiment checkpoint', 'symlink': False, 'bytes': 51909322692}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009/cache', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 28161833933}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009/runs/formal-aligned-em200-ef141268a5694dabace114936d380acc-3356/diagnostics/worker-events', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 25937319658}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009/runs/formal-aligned-em200-ef141268a5694dabace114936d380acc-3356/training-em-audit', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 9314492}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009/runs/formal-aligned-em200-ef141268a5694dabace114936d380acc-3356/checkpoints', 'reason': 'Completed failed/superseded BAPO experiment checkpoint', 'symlink': False, 'bytes': 38279206635}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v4-20261009/cache', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 28161833933}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v4-20261009/runs/formal-aligned-em200-671b1d8a4e004a89a35afb232272ed0c-3354/diagnostics/worker-events', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 25937211601}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v9-20261009/cache', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 4096}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v5-20261009/cache', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 28161833933}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v5-20261009/runs/formal-aligned-em200-d30bb88d6fb64a5ba005a3fad7f03ca8-3355/diagnostics/worker-events', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 25937210275}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v5-20261009/runs/formal-aligned-em200-d30bb88d6fb64a5ba005a3fad7f03ca8-3355/training-em-audit', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 9314492}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v8-20261009/cache', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 28161833933}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v8-20261009/runs/formal-aligned-em200-ae20107aeead4a848b43e450d0c1ca11-3357/diagnostics/worker-events', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 25938759478}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v8-20261009/runs/formal-aligned-em200-ae20107aeead4a848b43e450d0c1ca11-3357/training-em-audit', 'reason': 'Idle BAPO disposable cache or verbose diagnostics', 'symlink': False, 'bytes': 37513120}, {'path': '/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v8-20261009/runs/formal-aligned-em200-ae20107aeead4a848b43e450d0c1ca11-3357/checkpoints', 'reason': 'Completed failed/superseded BAPO experiment checkpoint', 'symlink': False, 'bytes': 51910213510}], 'total_bytes': 412651777179}
HF_RECEIPT = {'status':'FAILED_GROUP_SOURCE_AND_SUMMARIES_GITHUB_VERIFIED'}
GITHUB_SHA = '368b9df5791501eeca9194d2453fe782bf65b1f2'
assert HF_RECEIPT['status']=='FAILED_GROUP_SOURCE_AND_SUMMARIES_GITHUB_VERIFIED'
base=pathlib.Path('/dataset/shaoshitong/search-r1-rl')
paths=[pathlib.Path(x['path']) for x in PLAN['delete']]
def under(path,root):
 try:path.relative_to(root);return True
 except ValueError:return False
# Abort if any process has a BAPO command, cwd, or open file in candidate paths.
roots=set(str(pathlib.Path(x['path']).parts[len(base.parts)]) for x in PLAN['delete'])
ancestors={str(os.getpid())}
pid=os.getpid()
while pid>1:
 try:pid=int(pathlib.Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[1]);ancestors.add(str(pid))
 except (OSError,ValueError):break
refs=[]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  if proc.name in ancestors:continue
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if any('/'+root+'/' in cmd for root in roots) and ('python' in cmd or 'ray' in cmd or 'train' in cmd):
   if 'cleanup_execute' not in cmd and not 'required_hf_status' in cmd:refs.append(dict(pid=proc.name,kind='cmdline',detail=cmd[:300]))
  links=[proc/'cwd']+list((proc/'fd').glob('*'))
  for link in links:
   try:dest=pathlib.Path(os.readlink(link))
   except OSError:continue
   if any(under(dest,d) for d in paths):refs.append(dict(pid=proc.name,kind='open_path',detail=str(dest)))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not refs, json.dumps(refs)
receipt={'epoch_start':time.time(),'github_sha':GITHUB_SHA,'hf_receipt':HF_RECEIPT,'before':subprocess.check_output(['df','-B1','/dataset'],text=True),'deleted':[],'preserved_latest_200':True}
for d in paths:
 assert d.is_absolute() and under(d,base)
 assert d.parts[len(base.parts)].startswith('bapo-aligned-cohost-node11-v')
 assert 'topd' not in str(d) and 'global_step_200' not in d.parts and 'outer_200_COMPLETE' not in d.parts
 if d.exists():
  if d.is_symlink():d.unlink()
  else:shutil.rmtree(d)
  receipt['deleted'].append(str(d))
receipt['after']=subprocess.check_output(['df','-B1','/dataset'],text=True)
receipt['queue']=subprocess.check_output(['squeue','-h','-o','%i|%j|%T|%N'],text=True)
receipt['epoch_end']=time.time()
(base/'bapo-aligned-cohost-node11-v11-20261009/failed-cleanup-receipt-20261012.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
