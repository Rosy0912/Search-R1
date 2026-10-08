"""Structural refusal tests; .pt bytes are synthetic, never actual FSDP acceptance."""
import ast,json,hashlib,re,copy,tempfile,unittest
from pathlib import Path
path=Path(__file__).parent/'tools/verify_hardware_receipt.py';tree=ast.parse(path.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='validate_state_snapshots')
ns=dict(json=json,re=re,sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest());exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns);gate=ns['validate_state_snapshots']
class SnapshotGate(unittest.TestCase):
 def fixture(self,root):
  run=root.resolve();x=dict(profile='production-nested-bf16-manual');bound={}
  for key,directory,step,status,digest in [('initial_snapshot','initial-step5',5,'FULL_STATE_SERIALIZED_OK','1'*64),('actual_worker_roundtrip','roundtrip-step5',5,'EXACT_FULL_STATE_RELOAD_OK','1'*64),('oracle_snapshot','no-roundtrip-step6',6,'FULL_STATE_SERIALIZED_OK','2'*64),('reloaded_snapshot','reloaded-step6',6,'FULL_STATE_SERIALIZED_OK','2'*64)]:
   p=run/'native-state'/x['profile']/directory/'rank_0.pt';p.parent.mkdir(parents=True);p.write_bytes(b'SYNTHETIC-STRUCTURE-NOT-NATIVE')
   report=dict(path=str(p),sha256=ns['sha'](p),rank=0,world_size=8,step=step,status=status,model_state_dict_type='SHARDED_STATE_DICT',model_dtensor_entries=4,scheduler_last_epoch=step,optimizer_state_entries=1,pid=123,state_digest=digest)
   p.with_suffix('.json').write_text(json.dumps(report));bound[str(p)]=ns['sha'](p);bound[str(p.with_suffix('.json'))]=ns['sha'](p.with_suffix('.json'));x[key]=report
  x['next_update_components']={k:dict(no_roundtrip_sha256='a'*64,reloaded_sha256='a'*64,exact_equal=True)for k in ('model','optimizer','scheduler','rng')}
  return x,run,bound
 def test_all_four_bound_snapshots(self):
  with tempfile.TemporaryDirectory()as d:
   x,r,b=self.fixture(Path(d));self.assertEqual(gate(x,0,r,b),123)
 def test_refuse_each_missing_pt_or_json_binding(self):
  with tempfile.TemporaryDirectory()as d:
   x,r,b=self.fixture(Path(d))
   for key in x:
    if not isinstance(x[key],dict)or 'path'not in x[key]:continue
    for suffix in ('pt','json'):
     bad=b.copy();bad.pop(str(Path(x[key]['path']).with_suffix('.'+suffix)))
     with self.assertRaises(AssertionError):gate(x,0,r,bad)
 def test_refuse_source_schema_counterexamples(self):
  with tempfile.TemporaryDirectory()as d:
   x,r,b=self.fixture(Path(d))
   for key,val in [('model_state_dict_type','LOCAL_STATE_DICT'),('model_dtensor_entries',0),('world_size',1),('step',6),('pid',124),('state_digest','3'*64),('scheduler_last_epoch',6),('optimizer_state_entries',0),('rank',1)]:
    bad=copy.deepcopy(x);bad['actual_worker_roundtrip'][key]=val
    # Rebind actual JSON bytes to avoid proving only hash mismatch.
    p=Path(bad['actual_worker_roundtrip']['path']).with_suffix('.json');p.write_text(json.dumps(bad['actual_worker_roundtrip']));bb=b.copy();bb[str(p)]=ns['sha'](p)
    with self.assertRaises(AssertionError,msg=key):gate(bad,0,r,bb)
   Path(x['actual_worker_roundtrip']['path']).with_suffix('.json').write_text(json.dumps(x['actual_worker_roundtrip']))
 def test_component_set_or_digest_mismatch_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   x,r,b=self.fixture(Path(d))
   for kind in ('missing','mismatch','notexact'):
    bad=copy.deepcopy(x)
    if kind=='missing':bad['next_update_components'].pop('rng')
    if kind=='mismatch':bad['next_update_components']['rng']['reloaded_sha256']='b'*64
    if kind=='notexact':bad['next_update_components']['rng']['exact_equal']=False
    with self.assertRaises(AssertionError):gate(bad,0,r,b)
if __name__=='__main__':unittest.main()
