"""Generated runtime uses a real task-disk directory and a short UNIX path."""
import ast,os,socket,subprocess,tempfile,unittest,uuid
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
class RayPathTests(unittest.TestCase):
 def test_actual_generated_init_and_unix_bind(self):
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);(base/'source').symlink_to(ROOT)
   subprocess.run(['python3',str(ROOT/'experiments/search_e1/control/build_runtime.py'),'--base',str(base),'--profile','archive'],check=True,stdout=subprocess.DEVNULL)
   source=(base/'runtime/archive/search_e1/grpo_main.py').read_text();tree=ast.parse(source)
   node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='allocated_init')
   namespace=dict(os=os,Path=Path,_original_init=lambda *args,**kwargs:kwargs)
   exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-generated-init>','exec'),namespace)
   job=uuid.uuid4().hex
   with patch.dict(os.environ,{'E1_BASE':str(base),'E1_RUN_ID':job}):kwargs=namespace['allocated_init']()
   short=Path(kwargs['_temp_dir'])
   try:
    self.assertEqual(short.resolve(),(base/'ray'/job).resolve());self.assertEqual(kwargs['address'],'local')
    name=short/'session_2026-10-04_03-32-23_997339_1184409/sockets/plasma_store'
    self.assertLess(len(str(name).encode()),104);name.parent.mkdir(parents=True)
    with socket.socket(socket.AF_UNIX) as channel:channel.bind(str(name))
   finally:short.unlink()
if __name__=='__main__':unittest.main()
