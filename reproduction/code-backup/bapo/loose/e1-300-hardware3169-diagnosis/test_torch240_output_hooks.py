"""Exact Torch2.4 traversal/prehook code on real CPU autograd; no native GPU pass."""
import ast,dataclasses,collections,functools,typing,json,hashlib,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn.utils.rnn import PackedSequence
from torch.nn.parallel.scatter_gather import _is_namedtuple
from transformers.modeling_outputs import CausalLMOutputWithPast
r=Path(__file__).parent

def actual_torch240_functions():
 ns=dict(torch=torch,dataclasses=dataclasses,OrderedDict=collections.OrderedDict,PackedSequence=PackedSequence,_is_namedtuple=_is_namedtuple,functools=functools,no_type_check=typing.no_type_check)
 for filename,name in [('torch240-distributed-utils.py','_apply_to_tensors'),('torch240-fsdp-runtime-utils.py','_register_pre_backward_hooks')]:
  tree=ast.parse((r/filename).read_text());node=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name][-1];node.decorator_list=[];node.returns=None
  for arg in node.args.args:arg.annotation=None
  exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),ns)
 return ns
class OutputHooks(unittest.TestCase):
 def test_actual_torch240_simple_namespace_is_unseen_but_HF_ModelOutput_hooks_run(self):
  for kind in ('namespace','HF'):
   ns=actual_torch240_functions();state=SimpleNamespace(_is_root=True,training='IDLE');handle=SimpleNamespace()
   calls=[]
   def prehook(state,module,handle,grad):state.training='FORWARD_BACKWARD';calls.append('real_output_hook');return grad
   ns['_pre_backward_hook']=prehook;parameter=torch.tensor([1.,2.],requires_grad=True);logits=parameter*2
   output=SimpleNamespace(logits=logits)if kind=='namespace'else CausalLMOutputWithPast(logits=logits)
   hooked=ns['_register_pre_backward_hooks'](state,None,output,handle)
   if kind=='namespace':
    self.assertEqual(len(logits._backward_hooks or {}),0);self.assertFalse(handle._needs_pre_backward_unshard)
    hooked.logits.sum().backward();self.assertEqual(calls,[]);self.assertEqual(state.training,'IDLE')
   else:
    self.assertEqual(len(logits._backward_hooks),1);self.assertTrue(handle._needs_pre_backward_unshard)
    hooked.logits.sum().backward();self.assertEqual(calls,['real_output_hook']);self.assertEqual(state.training,'FORWARD_BACKWARD')
    self.assertTrue(dataclasses.is_dataclass(hooked));self.assertIsInstance(hooked,collections.OrderedDict);self.assertTrue(torch.equal(parameter.grad,torch.tensor([2.,2.])))
 def test_tensor_nested_model_output_not_detached_or_replaced(self):
  ns=actual_torch240_functions();x=torch.tensor([3.],requires_grad=True);output=CausalLMOutputWithPast(logits=x);seen=[]
  y=ns['_apply_to_tensors'](lambda t:seen.append(t)or t,output);self.assertEqual(len(seen),1);self.assertIs(y.logits,x);self.assertTrue(y.logits.requires_grad)
if __name__=='__main__':unittest.main()
