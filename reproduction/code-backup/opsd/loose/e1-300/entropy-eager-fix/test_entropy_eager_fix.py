import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
import pytest
D=Path(__file__).resolve().parent;R=D.parents[1]/'opsd-e1-300'
@pytest.mark.parametrize('leg',['legacy','candidate'])
def test_real_inherited_constructor_entropy_and_strict_code_guard(tmp_path,leg):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');output=tmp_path/(leg+'.json')
 p=subprocess.run([sys.executable,str(D/'cpu_entropy_probe.py'),'--repo',str(R),'--candidate',str(D/'actor_candidate.py'),'--leg',leg,'--output',str(output)],env=env,capture_output=True,text=True,timeout=60)
 assert p.returncode==0,p.stdout+'\n'+p.stderr
 x=json.loads(output.read_text());assert x['cuda_is_initialized'] is False and x['entropy_values_exact'] and x['entropy_gradients_exact']
 assert bool(x['strict_guard_error'])==(leg=='legacy');assert x['eager_selection']==(leg=='candidate')
 if leg=='candidate':assert x['tamper_rejected'] and not x['generated_functions']
def test_minimal_source_only_constructor_selector_no_update_math_diff():
 old=ast.parse((D/'actor_89be_original.py').read_text());new=ast.parse((D/'actor_candidate.py').read_text())
 # Remove only exact added import and __init__ assignment; all remaining AST exact.
 imports=[n for n in new.body if isinstance(n,ast.Import) and any(x.name=='verl.utils.torch_functional' and x.asname=='verl_F' for x in n.names)];assert len(imports)==1;new.body.remove(imports[0])
 cls=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='DistillationActor');init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
 select=init.body[1];assert ast.dump(select)==ast.dump(ast.parse('self.compute_entropy_from_logits = verl_F.entropy_from_logits').body[0]);init.body.pop(1)
 assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
