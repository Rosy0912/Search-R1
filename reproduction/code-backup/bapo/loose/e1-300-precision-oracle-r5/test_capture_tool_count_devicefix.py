"""Explicit evidence collector manifest-count regression."""
import ast,importlib.util,unittest
from pathlib import Path
p=Path(__file__).parent/'capture-hardware-devicefix.py';spec=importlib.util.spec_from_file_location('collector',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ToolCount(unittest.TestCase):
 def test_generated_worker_requires21_regular_tool_bindings(self):
  code=m.worker_code(3156,'JobId=3156 JobName=bapo-instruct300-hardware NodeList=h100-5 JobState=FAILED ExitCode=1:0 ',{})
  tree=ast.parse(code);assertion=next(n for n in tree.body if isinstance(n,ast.Assert)and isinstance(n.test,ast.Compare)and ast.unparse(n.test.left)=="len(manifest['files'])")
  fragment=compile(ast.Module(body=[assertion],type_ignores=[]),'manifest-count','exec')
  exec(fragment,dict(manifest={'files':{str(i):'sha'for i in range(21)}}))
  for n in (20,22):
   with self.assertRaises(AssertionError):exec(fragment,dict(manifest={'files':{str(i):'sha'for i in range(n)}}))
 def test_unrelated20_values_are_optimizer_calls(self):
  source=Path(__file__).parents[1]/'e1-300-precision-oracle-r5';stale=[]
  for file in list((source/'tools').glob('*'))+[source/'step5-extract.py',p]:
   if file.is_file():
    for line in file.read_text().splitlines():
     if '==20'in line and ('manifest'in line or "files']"in line):stale.append(str(file))
  self.assertEqual(stale,[])
if __name__=='__main__':unittest.main()
