FIX="def __setstate__(self, data):\n    import io\n    (batch_deserialized_bytes, non_tensor_batch, meta_info) = data\n    batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)\n    load_context = torch._utils._thread_local_state\n    missing = object()\n    outer_map = getattr(load_context, 'map_location', missing)\n    mapping = outer_map if outer_map is not missing else 'cpu' if not torch.cuda.is_available() else None\n    try:\n        batch = torch.load(batch_deserialized, weights_only=False, map_location=mapping)\n    finally:\n        if outer_map is not missing:\n            load_context.map_location = outer_map\n        elif hasattr(load_context, 'map_location'):\n            del load_context.map_location\n    self.batch = batch\n    self.non_tensor_batch = non_tensor_batch\n    self.meta_info = meta_info"
TESTS='"""Regression for PyTorch 2.4 nested DataProto checkpoint deserialization."""\nimport io\nimport unittest\nfrom unittest.mock import patch\nimport numpy as np\nimport torch\nfrom tensordict import TensorDict\nfrom verl.protocol import DataProto\n\n\nclass NestedDataProtoLoadTest(unittest.TestCase):\n    def proto(self):\n        return DataProto(batch=TensorDict({\'tokens\':torch.arange(6).reshape(2,3)},batch_size=[2]),\n                         non_tensor_batch={\'id\':np.array([\'a\',\'b\'],dtype=object)},meta_info={\'scope\':\'fixture\'})\n\n    def test_nested_archive_honors_explicit_cpu_and_remaining_tensor(self):\n        value={\'pending_source\':self.proto(),\'after\':torch.tensor([7.0]),\'second\':self.proto()}\n        stream=io.BytesIO();torch.save(value,stream);stream.seek(0)\n        with patch(\'torch.cuda.is_available\',return_value=True):\n            restored=torch.load(stream,map_location=\'cpu\',weights_only=False)\n        self.assertTrue(torch.equal(restored[\'pending_source\'].batch[\'tokens\'],value[\'pending_source\'].batch[\'tokens\']))\n        self.assertTrue(torch.equal(restored[\'after\'],value[\'after\']))\n        self.assertEqual(restored[\'second\'].batch[\'tokens\'].device.type,\'cpu\')\n        self.assertFalse(hasattr(torch._utils._thread_local_state,\'map_location\'))\n\n    def test_direct_pickle_roundtrip(self):\n        import pickle\n        value=self.proto();restored=pickle.loads(pickle.dumps(value))\n        self.assertTrue(torch.equal(restored.batch[\'tokens\'],value.batch[\'tokens\']))\n        self.assertEqual(restored.non_tensor_batch[\'id\'].tolist(),[\'a\',\'b\'])\n        self.assertFalse(hasattr(torch._utils._thread_local_state,\'map_location\'))\n\n    def test_inner_error_propagates_and_preserves_outer_mapping(self):\n        context=torch._utils._thread_local_state;context.map_location=\'cpu\'\n        try:\n            with self.assertRaises(Exception):\n                self.proto().__setstate__((b\'not a torch archive\',{},{}))\n            self.assertEqual(context.map_location,\'cpu\')\n        finally:\n            del context.map_location\n\n\nif __name__==\'__main__\':unittest.main()\n'
import pathlib,json,datetime,hashlib,os,traceback,io,unittest,sys,gc
import torch,numpy as np
from safetensors import safe_open
from verl.protocol import DataProto
import verl.protocol as proto
ROOT=pathlib.Path(__file__).parent
BASE=pathlib.Path('/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011')
CP=BASE/'checkpoints/search-g1-common8/actor/global_step_5'
RUN='15bb15365346401da1d919e9cde8f813'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,value):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,default=str)+'\n');os.replace(t,p)
result={'utc_start':utc(),'run_id':RUN,'step':5,'semantic_gpu_reload_performed':False,'source_modified':False}
try:
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
 # Reproduce the precise nested-loader failure using a tiny fixture first.
 from tensordict import TensorDict
 fixture=DataProto(batch=TensorDict({'x':torch.arange(2)},batch_size=[2]))
 buf=io.BytesIO();torch.save({'nested':fixture,'after':torch.tensor([3.0])},buf);buf.seek(0)
 try:torch.load(buf,map_location='cpu',weights_only=False)
 except AttributeError as e:assert str(e)=='map_location';result['unmodified_nested_reproduction']='AttributeError: map_location'
 else:raise AssertionError('Frozen PyTorch2.4 fixture did not reproduce the actual error')
 exec(FIX,proto.__dict__);DataProto.__setstate__=proto.__dict__['__setstate__']
 ns={'__name__':'nested_load_regression'};exec(TESTS,ns)
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(ns['NestedDataProtoLoadTest']);testresult=unittest.TextTestRunner(verbosity=2).run(suite);assert testresult.wasSuccessful();result['cpu_regression_tests']=testresult.testsRun
 run=json.loads((BASE/'logs/run-manifest.json').read_text());manifest=json.loads((CP/'training_state/manifest.json').read_text());assert run['run_id']==RUN and manifest['completed_outer']==5
 facts=[]
 for p in sorted(CP.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink();before=p.stat();h=hashlib.sha256();n=0
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block);n+=len(block)
  after=p.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
  assert n==before.st_size;facts.append({'path':str(p),'relative_path':str(p.relative_to(CP)),'bytes':n,'inode':before.st_ino,'mtime_ns':before.st_mtime_ns,'full_read_sha256':h.hexdigest()});write('PROGRESS.json',{'utc':utc(),'phase':'full_source_file_read','completed_files':len(facts),'completed_bytes':sum(x['bytes'] for x in facts),'last_file':str(p)})
 lookup={x['path']:x for x in facts};hf=json.loads((CP/'complete.json').read_text());assert hf['run_id']==RUN and hf['step']==5
 for name,sha in hf['files_sha256'].items():assert lookup[str(CP/name)]['full_read_sha256']==sha
 rankfacts=[]
 for proof in manifest['worker_files']:
  p=pathlib.Path(proof['path']);assert lookup[str(p)]['full_read_sha256']==proof['sha256'] and lookup[str(p)]['bytes']==proof['bytes']
  st=torch.load(p,map_location='cpu',weights_only=False);assert st['run_id']==RUN and st['rank']==proof['rank'] and st['world_size']==8 and st['actual_optimizer_writes']==100 and st['scheduler']['last_epoch']==5
  assert st['optimizer']['state'];states=st['optimizer']['state']
  for val in states.values():
   assert int(val['step'])==100
   for k in ['step','exp_avg','exp_avg_sq']:assert torch.isfinite(val[k]).all()
  rng=st['rng'];assert rng['vllm_version']=='0.6.3' and rng['boundary']=='idle synchronous generation engine'
  assert {'torch_cpu','torch_cuda','numpy','python','actor_cuda_saved','rollout_cuda_saved','request_generators'}<=set(rng)
  rankfacts.append({'rank':st['rank'],'actual_optimizer_writes':100,'scheduler_last_epoch':5,'optimizer_tensors_finite':True,'rng_keys':sorted(rng),'optimizer_parameter_count':len(states),'learning_rates':[g['lr'] for g in st['optimizer']['param_groups']]});del st;gc.collect();write('PROGRESS.json',{'utc':utc(),'phase':'actual_rank_state_cpu_reload','ranks':rankfacts})
 proof=manifest['driver_file'];assert lookup[proof['path']]['full_read_sha256']==proof['sha256']
 driver=torch.load(proof['path'],map_location='cpu',weights_only=False);assert driver['completed_outer']==5
 stream=driver['candidate_stream'];assert stream['consumed_source_rows']==10240 and stream['candidate_cursor']==10240
 assert len(stream['candidate_order'])==169028 and len(set(stream['candidate_order']))==169028
 result['driver_state']={'completed_outer':5,'source_consumed':stream['consumed_source_rows'],'candidate_cursor':stream['candidate_cursor'],'epoch':stream['epoch'],'pending_offset':stream['pending_offset'],'pending_length':len(stream['pending_source']) if stream['pending_source'] is not None else None,'probe_version':driver['probe_version'],'rng_keys':sorted(driver['rng'])}
 weightmap=json.loads((CP/'model.safetensors.index.json').read_text())['weight_map'];seen=set()
 for shard in sorted(set(weightmap.values())):
  with safe_open(str(CP/shard),framework='pt',device='cpu') as f:
   for name in f.keys():
    assert weightmap[name]==shard and name not in seen;v=f.get_tensor(name);assert torch.isfinite(v).all();seen.add(name);del v
 assert seen==set(weightmap) and len(seen)==hf['validated_tensor_count'];assert not torch.cuda.is_initialized()
 result.update(status='SOURCE_CHECKPOINT5_FULL_SHA_AND_CPU_STATES_PASS',utc_end=utc(),files=facts,ranks=rankfacts,hf_saved_tensor_count=len(seen),full_state_gpu_restore='PENDING',bytes=sum(x['bytes'] for x in facts));write('RESULT.json',result)
except BaseException as e:
 result.update(status='FAILED_PRESERVE_ALL_SOURCE',utc_end=utc(),error=repr(e),traceback=traceback.format_exc());write('RESULT.json',result);raise
