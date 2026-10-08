import ast, hashlib, os
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get("R1_ENTROPY_SOURCE", HERE.parent/"search-r1-32b-grpo100-source-v5"))
OLD=HERE.parent/"search-r1-32b-grpo100-source-v4"
OPTIONS={key:False for key in ("triton.autotune_pointwise", "max_autotune", "max_autotune_pointwise", "coordinate_descent_tuning", "dynamic_scale_rblock")}
def compile_call():
 tree=ast.parse((SOURCE/"verl/workers/actor/dp_actor.py").read_text())
 return next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="compile")
def test_actual_compile_call_retains_original_entropy_and_dynamic():
 call=compile_call();seen={};sentinel=object()
 class Torch:
  def compile(self, fn, **kwargs):seen.update(kwargs);assert fn is sentinel;return sentinel
 class F:entropy_from_logits=sentinel
 assert eval(compile(ast.Expression(call),"actual-actor-compile", "eval"),{"torch":Torch(),"verl_F":F()}) is sentinel
 assert seen=={"dynamic":True,"options":OPTIONS}
def test_entropy_original_function_bytes_unchanged():
 source=SOURCE/"verl/utils/torch_functional.py"
 assert hashlib.sha256(source.read_bytes()).hexdigest()=="e6f0578b0f52b530d52e9fc69968108d67badeb8374f18f39258e3550ac35454"
def test_only_compile_expression_changes_actor_ast():
 old=ast.parse((HERE/"prior-dp-actor.py").read_text());new=ast.parse((SOURCE/"verl/workers/actor/dp_actor.py").read_text())
 class Normalize(ast.NodeTransformer):
  def visit_Call(self,node):
   if isinstance(node.func,ast.Attribute) and node.func.attr=="compile":return ast.Constant("LOCAL_ENTROPY_COMPILE")
   return self.generic_visit(node)
 assert ast.dump(Normalize().visit(old))==ast.dump(Normalize().visit(new))
def test_actual_native_torch24_accepts_all_local_option_keys_without_cuda():
 import torch
 if torch.__version__.split("+")[0]!="2.4.0":pytest.skip("Requires pinned actual native Torch2.4")
 import torch._inductor.config as config
 available=config._config
 assert set(OPTIONS)<=set(available)
 before={k:available[k] for k in OPTIONS}
 compiled=torch.compile(lambda x:x, dynamic=True,options=OPTIONS)
 assert callable(compiled) and not torch.cuda.is_initialized()
 assert {k:available[k] for k in OPTIONS}==before
