"""Local unsigned-builder trust boundary tests; no remote side effects."""
import importlib.util,pathlib,tempfile,shutil,unittest,json,copy
BASE=pathlib.Path(__file__).parent;spec=importlib.util.spec_from_file_location('builder',BASE/'build-unsigned-launch-candidate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
TOOLS=BASE/'deployment-tools';CPU=BASE/'cpu-gate-durable-derived-record-candidate.json';RAW=pathlib.Path('work/evidence/ray-titlefix-target-cpu-independent-raw');CONFIG=pathlib.Path('work/bapo-e1-300-ray-titlefix/verl/trainer/config/ppo_bapo_e1_300.yaml')
class Test(unittest.TestCase):
 def test_unsigned_status_rejects_launch_and_preserves_no_permission(self):
  manifest,approval,head=m.build(TOOLS,CPU,RAW,CONFIG);self.assertEqual(approval['status'],'CANDIDATE_REQUIRES_INDEPENDENT_E1_LAUNCH_REVIEW');self.assertFalse(approval['actual_gpu_accepted']);self.assertFalse(approval['production_training_permission_signed']);self.assertFalse(approval['launch_authorized']);self.assertIsNone(approval['independent_review']['sha256']);self.assertEqual(len(manifest['files']),35);self.assertEqual(approval['tools_manifest_sha256'],__import__('hashlib').sha256(m.encode(manifest)).hexdigest())
 def test_rebound_or_edited_cpu_original_refused(self):
  with tempfile.TemporaryDirectory()as d:
   p=pathlib.Path(d)/'cpu.json';record=json.loads(CPU.read_text());record['exit_code']=1;p.write_text(json.dumps(record))
   with self.assertRaises(ValueError):m.build(TOOLS,p,RAW,CONFIG)
 def test_unreviewed_tool_replacement_or_extra_refused(self):
  with tempfile.TemporaryDirectory()as d:
   tools=pathlib.Path(d)/'tools';shutil.copytree(TOOLS,tools);(tools/'hardware.sh').write_text('unexpected hardware')
   with self.assertRaises(ValueError):m.build(tools,CPU,RAW,CONFIG)
  with tempfile.TemporaryDirectory()as d:
   tools=pathlib.Path(d)/'tools';shutil.copytree(TOOLS,tools);(tools/'EXTRA').touch()
   with self.assertRaises(ValueError):m.build(tools,CPU,RAW,CONFIG)
 def test_modified_runtime_or_exit_refused(self):
  for name in ('logs__runtime-executable.json','logs__cpu-gate-durable-exit.json'):
   with self.subTest(name=name),tempfile.TemporaryDirectory()as d:
    raw=pathlib.Path(d)/'raw';shutil.copytree(RAW,raw);record=json.loads((raw/name).read_text());record['source_commit']='old';(raw/name).write_text(json.dumps(record))
    with self.assertRaises(ValueError):m.build(TOOLS,CPU,raw,CONFIG)
 def test_originals_copied_as_exact_bytes_and_first_retained(self):
  out=BASE/'unsigned-launch-durable-r1';self.assertEqual((out/'cpu-gate.json').read_bytes(),CPU.read_bytes());self.assertEqual((out/'runtime-executable.json').read_bytes(),(RAW/'logs__runtime-executable.json').read_bytes());self.assertEqual(m.sha(out/'tools-manifest.DEPLOYED35-BEFORE-DURABLE-GATE.json'),'de0ed9fc782dbedf1f0e8681b56574188d28c2a5e04a723ba98fa5e2405801b2');self.assertTrue((RAW/'logs__cpu-gate.log').is_file())
if __name__=='__main__':unittest.main(verbosity=2)
