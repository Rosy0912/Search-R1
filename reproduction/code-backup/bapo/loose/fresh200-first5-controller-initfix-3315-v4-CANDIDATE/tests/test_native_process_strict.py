import ast,copy,hashlib,json,sys,unittest
from pathlib import Path
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
import native_process_strict as v
RAW_SHA='665399e2f93d1ba8c5bfb4643f22bcec9f0b7a687ad6cd638d9d7a064220925a'
def actual():
 p=B/'reference/native-allocation-failure.actual.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==RAW_SHA
 return json.loads(p.read_bytes())
def original():
 p=B/'reference/fresh200_native_evidence.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==v.ORIGINAL_SOURCE_SHA
 t=ast.parse(p.read_text());node=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='verify_native_process');ns={'require':v.require}
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns);return ns['verify_native_process']
class NativeStrict(unittest.TestCase):
 def setUp(self):self.raw=actual();self.p=self.raw['native_initial_documents'][0]['document']['native_process']
 def parse(self,raw,uid=0,host='h100-5'):return v.strict_allocation(raw,'3315',host,v.RUN,v.SOURCE,uid)
 def test_actual_all_eight_unmodified_process_format_replay_and_original_rejection(self):
  ps=[x['document']['native_process']for x in self.raw['native_initial_documents']];self.assertEqual(len(ps),8);self.assertEqual(len({p['gpu_uuid']for p in ps}),8)
  for p in ps:
   v.verify_native_process(p,v.SOURCE,expected_uid=0)
   with self.assertRaisesRegex(ValueError,'eight-GPU allocation not established'):original()(p,v.SOURCE)
 def test_exact_8_formats_not80_or_unrelated_substrings(self):
  raw=self.p['allocation_raw']
  for claim in ('gres:gpu:8','gres:gpu:h100:8'):self.assertEqual(self.parse(raw.replace('gres:gpu:h100:8',claim))['gpu_count'],8)
  self.parse(raw.replace('TresPerNode=gres:gpu:h100:8','TRESPerNode=gres/gpu:h100:8'))
  self.parse(raw.replace('TresPerNode=gres:gpu:h100:8','TRESPerNode=gres/gpu=8'))
  for claim in ('gres:gpu:h100:80','gres:gpu:18','gres:gpu:08','gres:gpu:1','cpu:8','gres:gpu:h100:8junk'):
   with self.assertRaises(ValueError):self.parse(raw.replace('gres:gpu:h100:8',claim))
 def test_duplicate_and_conflicting_fields(self):
  raw=self.p['allocation_raw']
  for extra in (' JobId=3315',' TresPerNode=gres:gpu:h100:8',' TRESPerNode=gres:gpu:h100:1',' TRESPerNode=gres:gpu:a100:8',' AllocTRES=gres/gpu=1',' AllocTRES=gres/gpu:a100=8'):
   with self.assertRaises(ValueError):self.parse(raw+extra)
  for claim in ('gres:gpu:h100:8,gres:gpu:h100:8','gres:gpu:h100:8,gres:gpu:a100:8'):
   with self.assertRaises(ValueError):self.parse(raw.replace('gres:gpu:h100:8',claim))
 def test_missing_gpu_and_node_fields(self):
  raw=self.p['allocation_raw']
  for old,new in [('TresPerNode=gres:gpu:h100:8',''),('NodeList=h100-5','NodeList=h100-6'),('BatchHost=h100-5','BatchHost=h100-6'),('NumNodes=1','NumNodes=2')]:
   with self.assertRaises(ValueError):self.parse(raw.replace(old,new))
  with self.assertRaises(ValueError):self.parse(raw,host='h100-16')
 def test_owner_current_target_uid(self):
  raw=self.p['allocation_raw'];self.assertEqual(self.parse(raw)['allocation_uid'],0)
  for uid in (1,True,None):
   with self.assertRaises(ValueError):self.parse(raw,uid=uid)
  for user in ('UserId=foreign(99)','UserId=0',''):
   with self.assertRaises(ValueError):self.parse(raw.replace('UserId=root(0)',user))
 def test_fixed_source_job_run_command_restart_scope(self):
  raw=self.p['allocation_raw']
  for old,new in [('JobId=3315','JobId=3312'),('JobState=RUNNING','JobState=COMPLETED'),('Restarts=0','Restarts=1'),(v.NONCE,'old'),(v.ROOT+'/tools/formal200.sbatch','/old/formal200.sbatch')]:
   with self.assertRaises(ValueError):self.parse(raw.replace(old,new))
  p=copy.deepcopy(self.p);p['run_root']='old'
  with self.assertRaises(ValueError):v.verify_native_process(p,v.SOURCE,expected_uid=0)
 def test_original_pid_cuda_uuid_guard_cgroup_mapping_checks_retained(self):
  variants=[('source_head','old'),('proc_start_ticks','0'),('proc_stat_raw','0 (fake)'),('proc_maps_raw','noCUDA'),('gpu_leases',[]),('gpu_uuid','GPU-foreign'),('resource_guard_stat_raw','broken'),('resource_guard_cmdline','foreign'),('proc_cgroup_raw','unrelated'),('cuda_visible_devices',[]),('gpu_pci_bus_id','wrong'),('nvidia_smi_raw','')]
  for key,value in variants:
   p=copy.deepcopy(self.p);p[key]=value
   with self.assertRaises((ValueError,IndexError)):v.verify_native_process(p,v.SOURCE,expected_uid=0)
 def test_ast_preserves_every_original_statement_except_two_allocation_requires(self):
  orig=ast.parse((B/'reference/fresh200_native_evidence.py').read_text());new=ast.parse((B/'native_process_strict.py').read_text())
  a=next(n for n in orig.body if isinstance(n,ast.FunctionDef)and n.name=='verify_native_process').body
  b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='verify_native_process').body
  index=next(i for i,n in enumerate(a)if isinstance(n,ast.Expr)and 'actual running Slurm allocation mismatch'in ast.unparse(n))
  self.assertIn('actual eight-GPU allocation not established',ast.unparse(a[index+1]));self.assertIn('strict_allocation',ast.unparse(b[index]))
  self.assertEqual([ast.dump(n,include_attributes=False)for n in a[:index]+a[index+2:]],[ast.dump(n,include_attributes=False)for n in b[:index]+b[index+1:]])
  proof=v.conservation_proof(B/'reference/fresh200_native_evidence.py');self.assertEqual(proof['original_conserved_ast_sha256'],proof['adapter_conserved_ast_sha256']);self.assertFalse(proof['native_acceptance'])
  collector=(B/'first5-extract.py').read_text();self.assertIn("result['native_process_conservation']",collector);self.assertIn("result['native_process_adapter'],result['original_native_verifier']",collector)
if __name__=='__main__':unittest.main()
