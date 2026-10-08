"""Actual-files CPU marker refusal tests; no target CPU proof is generated."""
import ast,hashlib,tempfile,unittest
from pathlib import Path
p=Path(__file__).parent/'tools/verify_launch.py';tree=ast.parse(p.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='validate_cpu_source_marker');ns=dict(sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest());exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns);gate=ns['validate_cpu_source_marker'];SOURCE='733097feeae371b4efd8fd48940c46ff36da839b'
class MarkerGate(unittest.TestCase):
 def setup_marker(self,root):
  (root/'logs').mkdir();p=root/'logs/cpu-gate-source.txt';p.write_text(SOURCE+'\n');return p,dict(source_commit=SOURCE,source_marker_sha256=ns['sha'](p))
 def test_actual_regular_marker_and_bound_record(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();p,r=self.setup_marker(root);self.assertEqual(gate(root,SOURCE,r),ns['sha'](p))
 def test_missing_source_marker_fails_before_submit(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();p,r=self.setup_marker(root);p.unlink()
   with self.assertRaises(AssertionError):gate(root,SOURCE,r)
 def test_wrong_source_marker_even_rebound_hash_fails(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();p,r=self.setup_marker(root);p.write_text('6a7bcec66d6d7a5da527d5e703bec83819dab183\n');r['source_marker_sha256']=ns['sha'](p)
   with self.assertRaises(AssertionError):gate(root,SOURCE,r)
 def test_stale_marker_hash_or_wrong_record_source_fails(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();p,r=self.setup_marker(root)
   for key,value in [('source_marker_sha256','0'*64),('source_commit','b09736d9d1b487a5c134e7a358761e04613cea32')]:
    bad=dict(r);bad[key]=value
    with self.assertRaises(AssertionError):gate(root,SOURCE,bad)
 def test_symlink_marker_refused(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();p,r=self.setup_marker(root);other=root/'other';p.rename(other);p.symlink_to(other)
   with self.assertRaises(AssertionError):gate(root,SOURCE,r)
 def test_launch_gate_precedes_intents_and_sbatch(self):
  s=(Path(__file__).parent/'tools/submit.sh').read_text();a=s.index('$ROOT/tools/verify_launch.py $phase')
  self.assertLess(a,s.index('printf \'%s\\n\' "$phase"'));self.assertLess(a,s.index('id=$(sbatch'))
if __name__=='__main__':unittest.main()
