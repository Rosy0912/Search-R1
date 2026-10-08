"""Structural receipt counterexamples, not actual GPU records."""
import ast,json,hashlib,unittest,tempfile,copy,pathlib
from pathlib import Path
base=Path(__file__).parent;source='733097feeae371b4efd8fd48940c46ff36da839b';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tree=ast.parse((base/'tools/verify_hardware_receipt.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='validate_production_forward_probe');ns=dict(json=json,pathlib=pathlib,sha=sha);exec(compile(ast.Module(body=[fn],type_ignores=[]),'probe-gate','exec'),ns);gate=ns[fn.name]
class Gate(unittest.TestCase):
 def test_production_branch_current_bytes_exact_bindings_and_counterexamples(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();run=root/'runs/bapo-instruct300-hardware-1';actor=run/'native-actor';actor.mkdir(parents=True);(root/'tools').mkdir();(root/'tools/native-actor-fixture.py').write_bytes((base/'tools/native-actor-fixture.py').read_bytes());repo=base.parent/'bapo-e1-300-devicefix'
   names=('verl/workers/actor/dp_actor.py','search_r1/baselines/objectives.py','search_r1/baselines/gradient_norm.py','verl/trainer/ppo/core_algos.py','verl/utils/fsdp_utils.py','verl/workers/fsdp_workers.py','verl/protocol.py','verl/utils/torch_functional.py','search_r1/baselines/distillation.py','verl/trainer/config/ppo_bapo_e1_300.yaml')
   for name in names:p=root/'repo'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((repo/name).read_bytes())
   kernel=root/'repo/verl/utils/torch_functional.py';owned=actor/'production-logprob-kernel-source-rank-0.py';owned.write_bytes(kernel.read_bytes());p=actor/'production-forward-rank-0.json'
   record=dict(status='PASS_REAL_ACTOR_FORWARD_PRECISION_PROBE',rank=0,world=8,source_commit=source,actor_source_sha256=sha(root/'repo/verl/workers/actor/dp_actor.py'),fixture_sha256=sha(root/'tools/native-actor-fixture.py'),input_device_before_forward='cpu',device='cuda:0',config_path=str(root/'repo/verl/trainer/config/ppo_bapo_e1_300.yaml'),config_sha256=sha(root/'repo/verl/trainer/config/ppo_bapo_e1_300.yaml'),torch='2.4.0',sys_prefix=str(root/'env'),kernel_owned_source_path=str(owned),kernel_source_path=str(kernel),kernel_source_sha256=sha(kernel),kernel_branch='naive_log_softmax',kernel_module='verl.utils.torch_functional',branch_records=[dict(use_remove_padding=padding,production_branch=padding,use_ulysses_sp=False,ulysses_sequence_parallel_size=1,input_device_before_forward='cpu',actual_device='cuda:0',actual_parameter_devices=['cuda:0'],actual_tensor_devices=['cuda:0'],logits_dtype='torch.bfloat16',logprob_dtype='torch.float32',entropy_dtype='torch.float32',gradient_dtypes=['torch.float32'],finite_forward_backward=True,nonzero_gradient=True,max_logprob_fp64_error=1e-6)for padding in (False,True)])
   v=dict(production_forward_precision=record,source_files_sha256={name:sha(root/'repo'/name)for name in names[:-1]});p.write_text(json.dumps(record));bound={str(p):sha(p),str(owned):sha(owned)};gate(v,0,run,root,source,bound)
   for kind in ('missing-rmpad','bf16-logprob','wrong-device','old-source','missing-binding','old-kernel','kernel-hash','config-hash','nan'):
    q=copy.deepcopy(v);b=bound.copy();x=q['production_forward_precision']
    if kind=='missing-rmpad':x['branch_records']=x['branch_records'][:1]
    if kind=='bf16-logprob':x['branch_records'][1]['logprob_dtype']='torch.bfloat16'
    if kind=='wrong-device':x['branch_records'][1]['actual_device']='cuda:1'
    if kind=='old-source':x['source_commit']='0'*40
    if kind=='missing-binding':b.pop(str(owned))
    if kind=='old-kernel':x['kernel_source_path']=str(repo/'verl/utils/torch_functional.py')
    if kind=='kernel-hash':x['kernel_source_sha256']='0'*64
    if kind=='config-hash':x['config_sha256']='0'*64
    if kind=='nan':x['branch_records'][1]['max_logprob_fp64_error']=float('nan')
    p.write_text(json.dumps(x));b[str(p)]=sha(p)
    with self.assertRaises((AssertionError,KeyError),msg=kind):gate(q,0,run,root,source,b)
 def test_producer_calls_real_constructor_both_paths_and_collector_requires_all(self):
  s=(base/'tools/native-actor-fixture.py').read_text();self.assertIn('instance = DataParallelPPOActor(config, model)',s);self.assertIn('for remove_padding in (False, True)',s);self.assertIn('model.use_remove_padding is True',s)
  c=(base/'capture-hardware-devicefix.py').read_text();self.assertIn('production-forward-rank-',c);self.assertIn('production-logprob-kernel-source-rank-',c)
if __name__=='__main__':unittest.main()
