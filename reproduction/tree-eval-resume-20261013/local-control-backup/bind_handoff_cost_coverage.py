"""Bind omitted existing artifacts without rerunning any measured computation."""
import importlib.util,json,shlex
from pathlib import Path
spec=importlib.util.spec_from_file_location('ssh',Path(__file__).with_name('ssh_retained18.py'))
ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
code='''import hashlib,json,os,stat
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=r/'runs/tree-branpo-fresh200-3360';out=run/'output/eval-handoff-hf-acceptance'
def sha(p):
 assert stat.S_ISREG(p.lstat().st_mode)
 h=hashlib.sha256()
 with p.open('rb') as f:
  before=os.fstat(f.fileno())
  for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
  after=os.fstat(f.fileno())
 assert (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns)
 return h.hexdigest()
v1=out/'cost-summary.json';assert sha(v1)=='58680c1a0736f4ff10baa5b17da3696a5d4fdb4576fb4021ec5ae57e3de6e67f'
cost=json.loads(v1.read_text());bound=cost['input_sha256']
composite=out/'identity-bound-handoff-hf.json';assert sha(composite)==cost['composite_receipt_sha256'];c=json.loads(composite.read_text())
def bind(p,expected=None):
 p=Path(p);assert p.is_absolute() and p.resolve().is_relative_to(r.resolve());h=sha(p);assert expected is None or h==expected
 assert str(p) not in bound or bound[str(p)]==h
 bound[str(p)]=h;return dict(path=str(p),sha256=h)
bind(v1)
for name,item in c['input_hashes_before'].items():
 if name in ('replacement_terminal','replacement_allocation','final_metrics','final_metrics_done') or name.startswith(('batch/','batch_done/','original_partial_cost/')):bind(item['path'],item['sha256'])
bind(out/'tokenizer-cold-reload.json',c['tokenizer_report_sha256']);bind(out/'actual-terminal-observation.json',c['actual_terminal_observation_sha256'])
shared=r/'runs/sst-tree-eval-shared-0e00998'
if (shared/'cost.json').exists():bind(shared/'cost.json')
for p in sorted((run/'output/eval-resume').glob('shared*dispatch-once.json')):
 journal=json.loads(p.read_text());unit=journal.get('unit')
 if unit in cost['failed_own_startup_inputs']:
  item=bind(p)
  if item not in cost['failed_own_startup_inputs'][unit]:cost['failed_own_startup_inputs'][unit].append(item)
assert all(len(items)>=2 for items in cost['failed_own_startup_inputs'].values()),cost['failed_own_startup_inputs']
cost['coverage_revision']=2;cost['coverage_revision_scope']='Bind existing completion/source/cost journals; reuse unchanged captured computation and totals; preserve immutable v1'
p=out/'cost-summary-v2.json'
with p.open('x') as f:json.dump(cost,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(path=str(p),sha256=sha(p),bound_inputs=len(bound),failed_own_startup_inputs=cost['failed_own_startup_inputs'])))
'''
r=ssh.run('python3 -c '+shlex.quote(code),timeout=120)
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
receipt=json.loads(r.stdout)
Path('outputs/tree-eval-resume-20261013/COMPOSITE_COST_COVERAGE_V2.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
