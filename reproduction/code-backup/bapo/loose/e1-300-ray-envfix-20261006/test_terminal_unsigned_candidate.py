"""Local candidate schema tests only, no actual hardware terminal evidence."""
import ast,hashlib,importlib.util,json,pathlib,tempfile,unittest
D=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('terminal_unsigned',D/'build-hardware-final-receipt.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Candidate(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.archive=pathlib.Path(self.tmp.name)/'raw';self.job=999;self.run=m.ROOT+'/runs/bapo-instruct300-hardware-999';r=self.archive/'runs/bapo-instruct300-hardware-999';r.mkdir(parents=True);self.inventory=r/'actual-fullhash-inventory.json';names=['recovery-manifest.json','slurm-terminal.txt','source-commit.txt','resolved-config.yaml','tools-manifest.json','cpu-gate.json','cpu-gate-source.txt','launch-approval.json','guard-summary.json','resource-watch.jsonl','own-process-cleanup.json','train.log']+[f'recovery-witness/rank_{i}.json'for i in range(8)]+['recovery-witness/driver.json']+[f'recovery-native-v3-roundtrip/rank_{i}.{s}'for i in range(8)for s in ('pt','json')];ev=[dict(path=self.run+'/'+n,sha256='a'*64,bytes=1)for n in names]+[dict(path=m.ROOT+'/'+n,sha256='b'*64,bytes=1)for n in ('logs/cpu-gate-durable.log','tools/guard_journal.py','tools/resource_watch.py')];self.i=dict(job_id=999,source_commit=m.SOURCE,run_root=self.run,evidence=ev);self.inventory.write_text(json.dumps(self.i));self.check=pathlib.Path(self.tmp.name)/'check.json';self.c=dict(status='PASS_LOCAL_RAW_RECOMPUTATION_NOT_GPU_ACCEPTANCE_OR_PERMISSION',job_id=999,source_commit=m.SOURCE,run_root=self.run,inventory_sha256=m.sha(self.inventory),actual_gpu_accepted=False,permission_signed=False,inventory_count=len(ev),reserved_wall_seconds=101,reserved_gpu_hours=101*8/3600,fullhash_wall_seconds=3,fullhash_cpu_seconds=2,guard=dict(cost_wall_seconds=1),scale_fixture_wall_seconds=7,scale_freeze_seconds=1.2)
 def tearDown(self):self.tmp.cleanup()
 def build(self):self.check.write_text(json.dumps(self.c));return m.build(self.archive,self.check,self.job)
 def test_unsigned_only_and_dynamic_cost(self):
  r=self.build();self.assertTrue(r['status'].startswith('CANDIDATE'));self.assertFalse(r['actual_gpu_accepted']);self.assertFalse(r['production_training_permission_signed']);self.assertTrue(r['independent_review']['pending_independent_review']);self.assertEqual(r['cost']['slurm_reserved_wall_seconds'],101);self.assertEqual(len(r['gpu_validation']['raw_artifacts']),14)
 def test_unsigned_count_not_old372(self):self.assertEqual(len(self.build()['evidence']),len(self.i['evidence'])+1)
 def test_wrong_source_reject(self):
  self.c['source_commit']='old'
  with self.assertRaises(AssertionError):self.build()
 def test_gpu_claim_reject(self):
  self.c['actual_gpu_accepted']=True
  with self.assertRaises(AssertionError):self.build()
 def test_stale_inventory_reject(self):
  self.c['inventory_sha256']='0'*64
  with self.assertRaises(AssertionError):self.build()
 def test_duplicate_binding_reject(self):
  self.i['evidence'].append(self.i['evidence'][0]);self.inventory.write_text(json.dumps(self.i));self.c.update(inventory_sha256=m.sha(self.inventory),inventory_count=len(self.i['evidence']))
  with self.assertRaises(AssertionError):self.build()
 def test_partial_no_post_fake_terminal_rejected(self):
  spec=importlib.util.spec_from_file_location('actual_check',D/'check-hardware-final.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
  with self.assertRaises((AssertionError,KeyError)):c.check(self.archive,self.job)
 def test_check_import_no_run_or_signal(self):
  tree=ast.parse((D/'check-hardware-final.py').read_text());ops={'kill','pidfd_send_signal','cuda','init_process_group','write_text','unlink','chmod','mkdir'};self.assertFalse(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in ops for n in ast.walk(tree)))
 def test_mandatory_original_parsers_and_no_old_cost(self):
  s=(D/'check-hardware-final.py').read_text();self.assertIn('validate_identity_linux_fixture',s);self.assertIn('validate_production_ray_identity',s);self.assertNotIn('==372',s);self.assertNotIn('==1395',s);self.assertNotIn("samples']==69",s)
if __name__=='__main__':unittest.main()
