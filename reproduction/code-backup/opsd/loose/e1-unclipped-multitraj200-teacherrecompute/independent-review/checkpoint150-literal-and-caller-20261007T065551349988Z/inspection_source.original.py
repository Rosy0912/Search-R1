from pathlib import Path
import ast,hashlib,json,types
H=Path(__file__).resolve().parents[2];C=H/'checkpoint150-observation-candidate'
def pin(p,expected):
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected;return raw
old=pin(H/'checkpoint50-observation-candidate/collect_checkpoints50_100_v2.py','61368d54b0e6732c9fef424b18df1903cce3bd50ed2ba3582c8c122b8426deba')
new=pin(C/'collect_checkpoints50_100_150.py','56d096390abc92cfb1f58b06f49f2acb2eaafb75b5f0bad41fcf39edbd5c7e1f')
ready_bytes=pin(C/'SOFTWARE_READINESS.json','6269845b66b02ca3f470c47536fc12c547e0bfb8f12f304e26ecd1446a6a201c');ready=json.loads(ready_bytes)
text=old.decode()
for change in ready['exact_literal_changes']:
 assert text.count(change['old'])==change['count'];text=text.replace(change['old'],change['new'])
assert text.encode()==new
caller=pin(H/'checkpoint150-root-once-caller-v1.py','415a9afd7115f736f294e5278d6865c71a3b4f94c965f2d3c2a424495d13479e')
oldcaller=(H/'checkpoint50_100-collection-once-v2-original/caller-source.original.py').read_bytes();text=oldcaller.decode()
changes=[('checkpoint50-observation-candidate/collect_checkpoints50_100_v2.py','checkpoint150-observation-candidate/collect_checkpoints50_100_150.py'),('61368d54b0e6732c9fef424b18df1903cce3bd50ed2ba3582c8c122b8426deba','56d096390abc92cfb1f58b06f49f2acb2eaafb75b5f0bad41fcf39edbd5c7e1f'),('checkpoint50_100-collection-once-v2-original','checkpoint150-collection-once-v1-original'),('sys.argv=["collect_checkpoints50_100_v2.py","--steps","50","100"]','sys.argv=["collect_checkpoints50_100_150.py","--steps","150"]'),("'requested_steps':[50,100]","'requested_steps':[150]")]
for a,b in changes:assert text.count(a)==1;text=text.replace(a,b)
assert text.encode()==caller;compile(caller,str(H/'checkpoint150-root-once-caller-v1.py'),'exec')
c=types.ModuleType('checkpoint150_independent_literal');c.__file__=str(C/'collect_checkpoints50_100_150.py');exec(compile(new,c.__file__,'exec'),c.__dict__)
# Only guards before Linux/proc/file collection: accepted150 stops on platform, mistyped requests stop first.
for steps in ([150.0],[True],[149],[151],[150,150],[150,100]):
 try:c.collect({},steps)
 except ValueError as exc:assert str(exc)=='Exact ordered unique steps50/100/150 required'
 else:raise AssertionError('Typed selector failed')
try:c.collect({},[150])
except ValueError as exc:assert str(exc)=='Actual same-node UID0 collector only'
else:raise AssertionError('This CPU fixture unexpectedly reached remote environment')
originals=ready['cpu_originals']
for r in originals:assert len(pin(Path(r['path']),r['sha256']))==r['bytes']
ex=json.loads(Path(originals[0]['path']).read_bytes());assert ex['waited'] is True and ex['returncode']==0 and ex['inputs_unchanged'] is True and ready['actual_tests']==3
assert b'Ran 3 tests' in Path(originals[-1]['path']).read_bytes() and b'OK' in Path(originals[-1]['path']).read_bytes()
repo=H.parent/'opsd-unclipped-multitraj200-teacherrecompute';source=pin(repo/'verl/trainer/ppo/ray_trainer.py',c.SOURCE_FILES['verl/trainer/ppo/ray_trainer.py']).decode()
for line,literal in [(987,'val_metrics: dict = self._validate()'),(994,'self.baseline_controller.stage_validation(self.global_steps)'),(1005,'self._save_checkpoint(metrics)'),(1012,'logger.log(data=metrics, step=self.global_steps)'),(1015,"print('OPSD_E1_OUTER_JSON='")]:assert literal in source.splitlines()[line-1]
for p,expected in [(C/'collect_checkpoints50_100_150.py',ready['runtime']['sha256']),(C/'SOFTWARE_READINESS.json','6269845b66b02ca3f470c47536fc12c547e0bfb8f12f304e26ecd1446a6a201c'),(H/'checkpoint150-root-once-caller-v1.py','415a9afd7115f736f294e5278d6865c71a3b4f94c965f2d3c2a424495d13479e')]:pin(p,expected)
print(json.dumps({'status':'PASS_EXACT_LITERAL_EXTENSION_AND_ROOT_CALLER_ONLY','collector_runtime_sha256':ready['runtime']['sha256'],'collector_literal_count':sum(x['count'] for x in ready['exact_literal_changes']),'root_caller_literal_count':len(changes),'original3_cpu_wait0_audited_not_rerun':True,'independent_typed_guard_boundaries':True,'source150_write_order_line_basis':True,'all_immutable_reader_logic_preserved_by_exact_transform':True,'actual150':None,'remote':False,'production_main':False,'caller_executed':False,'gpu':False,'pt_or_hf_read':False},sort_keys=True))
