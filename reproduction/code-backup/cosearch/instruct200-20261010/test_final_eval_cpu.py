"""Execute the actual final fit branch with stand-ins; entering PPO must fail."""
import ast,json,sys,tempfile,types
from pathlib import Path
source=ast.parse(Path(__file__).with_name('launch_training.py').read_text())
trainer=next(n for n in ast.walk(source) if isinstance(n,ast.ClassDef) and n.name=='Trainer')
fit=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
class Sized:
    def __init__(self,n):self.n=n
    def __len__(self):return self.n
class Base:
    total_training_steps=200
    train_dataset=Sized(169028);val_dataset=Sized(51713)
    def fit(self):raise AssertionError('PPO fit must never run during final evaluation')
    def _validate(self):return {'final/score':1.0}
fix=types.ModuleType('runtime_fixes');fix.install_controller=lambda *a:None
sys.modules['runtime_fixes']=fix
with tempfile.TemporaryDirectory() as d:
    namespace={'original':Base,'args':types.SimpleNamespace(phase='final_eval'),'root':Path(d),'json':json}
    node=ast.ClassDef(name='Trainer',bases=[ast.Name(id='original',ctx=ast.Load())],keywords=[],body=[fit],decorator_list=[])
    tree=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));exec(compile(tree,'<actual-final-branch>','exec'),namespace)
    cls=namespace['Trainer']
    def loaded(instance):instance.global_steps=200
    cls._load_checkpoint=loaded
    instance=cls();instance.fit()
    assert instance.global_steps==200
    assert json.loads((Path(d)/'final-evaluation-metrics.json').read_text())['final/score']==1.0
print('PASS: actual final-evaluation branch bypasses PPO and preserves step200')
