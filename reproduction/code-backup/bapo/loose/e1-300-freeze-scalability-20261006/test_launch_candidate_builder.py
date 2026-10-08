import importlib.util
from pathlib import Path
import json
import shutil
import tempfile
import unittest

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate_builder',BASE/'build-final-launch-candidate.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
CPU=BASE.parent/'evidence/freeze-scalability-b711-independent-cpu-raw/logs__cpu-gate.json'
CONFIG=BASE.parent/'bapo-e1-300-freeze-scalability/verl/trainer/config/ppo_bapo_e1_300.yaml'

class BuilderTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.dir=Path(self.tmp.name);self.tools=self.dir/'tools'
  shutil.copytree(BASE/'deployment-tools',self.tools,ignore=shutil.ignore_patterns('__pycache__'))
  shutil.copyfile(BASE/'freeze_scalability_gpu_fixture.py',self.tools/'freeze_scalability_gpu_fixture.py')
 def build(self,**kw):
  return b.build(self.tools,kw.pop('cpu',CPU),kw.pop('config',CONFIG),**kw)
 def test_genuine_cpu_raw_and_unsigned_only(self):
  m,a,h=self.build()
  self.assertEqual(b.sha(CPU),b.CPU_SHA)
  self.assertEqual(m['files'],b.inventory(self.tools))
  self.assertEqual(a['tools_manifest_sha256'],b.hashlib.sha256(b.encode(m)).hexdigest())
  self.assertNotEqual(a['status'],'PASS_INDEPENDENT_E1_LAUNCH_REVIEW')
  self.assertFalse(a['production_training_permission_signed'])
  self.assertIsNone(a['independent_review']['path'])
  self.assertEqual(h,{name:m['files'][name]for name in ('submit.sh','hardware.sbatch','formal300.sbatch')})
 def test_mutated_cpu_bytes_rejected(self):
  p=self.dir/'bad-cpu.json';obj=json.loads(CPU.read_text());obj['passed']=394;p.write_text(json.dumps(obj))
  with self.assertRaisesRegex(ValueError,'CPU record SHA'):self.build(cpu=p)
 def test_changed_cpu_launcher_rejected(self):
  with (self.tools/'cpu-tests.py').open('a')as f:f.write('\n# changed after actual CPU tests\n')
  with self.assertRaisesRegex(ValueError,'CPU launcher changed'):self.build()
 def test_tools_symlink_rejected(self):
  (self.tools/'shadow.py').symlink_to(self.tools/'cpu-tests.py')
  with self.assertRaisesRegex(ValueError,'Symlink'):self.build()
 def test_configuration_mutation_rejected(self):
  p=self.dir/'cfg.yaml';p.write_bytes(CONFIG.read_bytes()+b'\n# changed\n')
  with self.assertRaisesRegex(ValueError,'Configuration differs'):self.build(config=p)
 def test_report_pair_and_root_boundary(self):
  p=self.dir/'report.md';p.write_text('Independent actual report supplied by reviewer')
  with self.assertRaisesRegex(ValueError,'both report'):self.build(report_local=p)
  with self.assertRaisesRegex(ValueError,'inside the new root'):self.build(report_local=p,report_remote=b.ROOT+'-other/logs/report.md')
  _,a,_=self.build(report_local=p,report_remote=b.ROOT+'/logs/report.md')
  self.assertEqual(a['independent_review']['sha256'],b.sha(p));self.assertTrue(a['independent_review']['pending_independent_review'])

if __name__=='__main__':unittest.main()
