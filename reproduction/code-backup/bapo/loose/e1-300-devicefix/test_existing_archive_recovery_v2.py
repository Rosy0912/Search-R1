"""Local synthetic identity/tar fixture, never actual3156 or GPU proof."""
import importlib.util,subprocess,tarfile,tempfile,json,hashlib,unittest,base64,ast,re,contextlib,io
from pathlib import Path
from unittest.mock import patch
base=Path(__file__).parent
spec=importlib.util.spec_from_file_location('recovery',base/'recover3156-existing-archive-worker-v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RecoveryV2(unittest.TestCase):
 def setup(self,tmp):
  root=tmp.resolve()/'owned';logs=root/'logs';logs.mkdir(parents=True);run=root/'runs/bapo-instruct300-hardware-3156';run.mkdir(parents=True);(root/'tools').mkdir();repo=root/'repo';repo.mkdir()
  for name in m.TOOLS:(root/'tools'/name).write_text('SYNTHETIC TOOL')
  for name in m.SOURCES:
   p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('SYNTHETIC SOURCE')
  subprocess.run(['git','init','-q',str(repo)],check=True);subprocess.run(['git','-C',str(repo),'add','.'],check=True);subprocess.run(['git','-C',str(repo),'-c','user.name=fixture','-c','user.email=fixture@local','commit','-qm','fixture'],check=True);source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
  (root/'source-expected.txt').write_text(source);(run/'source-commit.txt').write_text(source)
  manifest=dict(files={n:m.sha(root/'tools'/n)for n in m.TOOLS},source_commit=source,root=str(root));(logs/'tools-manifest.json').write_text(json.dumps(manifest))
  (logs/'cpu-gate-source.txt').write_text(source);(logs/'cpu-gate.log').write_text('SYNTHETIC CPU LOG NOT TARGET')
  cpu=dict(source_commit=source,source_marker_sha256=m.sha(logs/'cpu-gate-source.txt'),passed=256,skipped=0,failed=0,exit_code=0,scope='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE',sys_prefix=str(root/'env'),test_launcher_sha256=m.sha(root/'tools/cpu-tests.py'),log_sha256=m.sha(logs/'cpu-gate.log'));(logs/'cpu-gate.json').write_text(json.dumps(cpu))
  approval=dict(source_commit=source,root=str(root),status='PASS_INDEPENDENT_E1_LAUNCH_REVIEW',tools_sha256=manifest['files'],tools_manifest_sha256=m.sha(logs/'tools-manifest.json'),cpu_gate_record_sha256=m.sha(logs/'cpu-gate.json'));(logs/'launch-approval.json').write_text(json.dumps(approval))
  for name in ('tools-manifest.json','launch-approval.json','cpu-gate.json','cpu-gate-source.txt'):(run/name).write_bytes((logs/name).read_bytes())
  (run/'slurm-terminal.txt').write_text('JobId=3156 JobName=bapo-instruct300-hardware NodeList=h100-5 JobState=FAILED ExitCode=1:0')
  (run/'own-process-cleanup.json').write_text('{"remaining":[]}');(run/'hardware.log').write_text('SYNTHETIC');(run/'resource-watch.jsonl').write_text('SYNTHETIC')
  with patch.object(m,'SOURCE',source):required,_=m.validate_bindings(root)
  for p in required:
   if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'SYNTHETIC STRUCTURE NOT NATIVE')
  archive=logs/'hardware-3156-raw-capture-fixture.tar.gz'
  with tarfile.open(archive,'x:gz')as tf:
   for p in sorted(required):tf.add(p,arcname=str(p.relative_to(root)),recursive=False)
  return root,source,archive
 def test_full_required_inventory_reuses_archive_bytes(self):
  with tempfile.TemporaryDirectory()as d:
   root,source,a=self.setup(Path(d));before=a.read_bytes()
   with patch.object(m,'ROOT',root),patch.object(m,'SOURCE',source),patch.object(m,'archive_readers_or_writers',lambda logs:[]),contextlib.redirect_stdout(io.StringIO()):m.main()
   out=json.loads((root/'logs/hardware-3156-existing-archive-recovery-v2-result.json').read_text());self.assertEqual(out['status'],m.SUCCESS,out);self.assertEqual(a.read_bytes(),before);self.assertEqual(out['required_missing'],[])
 def test_binding_missing_extra_symlink_and_sha_refusal(self):
  for kind in ('missing','extra','symlink','sha','marker','slurm','cleanup'):
   with tempfile.TemporaryDirectory()as d:
    root,source,a=self.setup(Path(d));p=root/'tools/cpu-tests.py'
    if kind=='missing':p.unlink()
    if kind=='extra':(root/'tools/unapproved.py').write_text('extra')
    if kind=='symlink':other=root/'other';p.rename(other);p.symlink_to(other)
    if kind=='sha':p.write_text('changed')
    if kind=='marker':(root/'logs/cpu-gate-source.txt').write_text('wrong')
    if kind=='slurm':(root/'runs/bapo-instruct300-hardware-3156/slurm-terminal.txt').write_text('JobId=3157')
    if kind=='cleanup':(root/'runs/bapo-instruct300-hardware-3156/own-process-cleanup.json').write_text('{"remaining":[123]}')
    with patch.object(m,'SOURCE',source):
     with self.assertRaises((AssertionError,FileNotFoundError),msg=kind):m.validate_bindings(root)
 def test_active_any_fd_owner_refused(self):
  with tempfile.TemporaryDirectory()as d:
   root,source,a=self.setup(Path(d))
   with patch.object(m,'ROOT',root),patch.object(m,'SOURCE',source),patch.object(m,'archive_readers_or_writers',lambda logs:[{'pid':1,'target':str(a)}]),contextlib.redirect_stdout(io.StringIO()):m.main()
   self.assertEqual(json.loads((root/'logs/hardware-3156-existing-archive-recovery-v2-result.json').read_text())['status'],'RECOVERY_REFUSED_NO_ARCHIVE_OR_IDENTITY_CHANGED')
 def test_launcher_embedded_worker_and_regex(self):
  tree=ast.parse((base/'start3156-background-recovery-v2.py').read_text());arg=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='b64decode').args[0].value
  self.assertEqual(base64.b64decode(arg),(base/'recover3156-existing-archive-worker-v2.py').read_bytes())
  tree=ast.parse((base/'download3156-recovered-archive-v2.py').read_text());part=next(n for n in ast.walk(tree)if isinstance(n,ast.Constant)and isinstance(n.value,str)and n.value.startswith('/logs/hardware-3156-raw-capture-'))
  self.assertTrue(re.fullmatch(part.value,'/logs/hardware-3156-raw-capture-20261004T151800Z-123.tar.gz'));self.assertIsNone(re.fullmatch(part.value,'/logs/hardware-3156-raw-capture-../outside.tar.gz'))
 def test_unknown_intent_returns_without_relaunch_or_overwrite(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve()/'owned';logs=root/'logs';logs.mkdir(parents=True)
   intent=logs/'hardware-3156-existing-archive-recovery-v2-intent.json';intent.write_text('{"status":"UNKNOWN_PRIOR_OUTCOME"}')
   pid=logs/'hardware-3156-existing-archive-recovery-v2-pid.json';pid.write_text('original PID');result=logs/'hardware-3156-existing-archive-recovery-v2-result.json';result.write_text('original result')
   code=(base/'start3156-background-recovery-v2.py').read_text().replace("Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-countfix-20261004')",'Path('+repr(str(root))+')')
   script=root/'launcher.py';script.write_text(code);subprocess.run(['python3',str(script)],check=True,capture_output=True)
   self.assertEqual(pid.read_text(),'original PID');self.assertEqual(result.read_text(),'original result');self.assertEqual(intent.read_text(),'{"status":"UNKNOWN_PRIOR_OUTCOME"}')
 def test_required_snapshot_absent_from_tar_is_refused(self):
  with tempfile.TemporaryDirectory()as d:
   root,source,a=self.setup(Path(d));a.unlink()
   with patch.object(m,'SOURCE',source):required,_=m.validate_bindings(root)
   missing=root/'runs/bapo-instruct300-hardware-3156/native-state/baseline-fp32/initial-step5/rank_0.pt'
   with tarfile.open(a,'x:gz')as tf:
    for p in sorted(required-{missing}):tf.add(p,arcname=str(p.relative_to(root)),recursive=False)
   with patch.object(m,'ROOT',root),patch.object(m,'SOURCE',source),patch.object(m,'archive_readers_or_writers',lambda logs:[]),contextlib.redirect_stdout(io.StringIO()):m.main()
   self.assertEqual(json.loads((root/'logs/hardware-3156-existing-archive-recovery-v2-result.json').read_text())['status'],'RECOVERY_REFUSED_NO_ARCHIVE_OR_IDENTITY_CHANGED')
if __name__=='__main__':unittest.main()
