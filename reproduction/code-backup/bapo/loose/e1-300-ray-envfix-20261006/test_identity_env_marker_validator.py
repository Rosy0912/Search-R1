import ast,base64,copy,importlib.util,json,pathlib,unittest
B=pathlib.Path(__file__).parent;P=B/'deployment-tools/identity_evidence_validator.py';spec=importlib.util.spec_from_file_location('identity',P);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
run=pathlib.Path('/owned/runs/bapo-instruct300-hardware-1');uuids=['GPU-'+str(i)for i in range(8)];ns=dict(need=v.need,parse=v.parse,base64=base64,run=run,job=1,uuids=uuids)
outer=next(n for n in ast.parse(P.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='validate_identity_linux_fixture');nodes=[n for n in outer.body if isinstance(n,ast.FunctionDef)and n.name in('marker_proof','initial_markers')];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(P),'exec'),ns)
def marker(values=None):
 values=values or dict(BAPO_RUN_ID=str(run),SLURM_JOB_ID='1',SEARCH_R1_GPU_LEASES=json.dumps([{'uuid':x}for x in uuids]),SPT_NOENV='1');return dict(values=values,original_selected_environ_base64=base64.b64encode(b'\0'.join((k+'='+x).encode()for k,x in values.items())+b'\0').decode())
class Tests(unittest.TestCase):
 def test_actual_allocation_marker_schema(self):ns['marker_proof'](marker())
 def test_missing_clobbered_raw_rejected(self):
  for value in (dict(values={},original_selected_environ_base64=base64.b64encode(b'\0'*200).decode()),marker(dict(BAPO_RUN_ID=str(run),SLURM_JOB_ID='1',SPT_NOENV='1'))):
   with self.assertRaises(ValueError):ns['marker_proof'](value)
 def test_marker_json_cannot_override_original(self):
  x=marker();x['values']['SLURM_JOB_ID']='2'
  with self.assertRaises(ValueError):ns['marker_proof'](x)
 def test_duplicate_original_marker_rejected(self):
  x=marker();raw=base64.b64decode(x['original_selected_environ_base64'])+b'SLURM_JOB_ID=1\0';x['original_selected_environ_base64']=base64.b64encode(raw).decode()
  with self.assertRaises(ValueError):ns['marker_proof'](x)
 def test_foreign_job_run_lease_or_noenv_policy_rejected(self):
  for key,value in [('BAPO_RUN_ID','/foreign'),('SLURM_JOB_ID','2'),('SEARCH_R1_GPU_LEASES',json.dumps([{'uuid':'GPU-foreign'}]*8)),('SPT_NOENV','0')]:
   x=marker();values=dict(x['values'],**{key:value})
   with self.subTest(key=key),self.assertRaises(ValueError):ns['marker_proof'](marker(values))
 def test_ready_marker_equality_capacity_contract(self):
  x=marker();ready=dict(process_markers_before=x,process_markers_after=copy.deepcopy(x),argv_capacity_bytes=512);ns['initial_markers'](ready)
  ready['argv_capacity_bytes']=True
  with self.assertRaises(ValueError):ns['initial_markers'](ready)
 def test_full_original_marker_bytes_must_remain_same(self):
  x=marker();y=copy.deepcopy(x);raw=base64.b64decode(y['original_selected_environ_base64']);y['original_selected_environ_base64']=base64.b64encode(b'\0'.join(reversed(raw.rstrip(b'\0').split(b'\0')))+b'\0').decode()
  with self.assertRaises(ValueError):ns['initial_markers'](dict(process_markers_before=x,process_markers_after=y,argv_capacity_bytes=512))
if __name__=='__main__':unittest.main(verbosity=2)
