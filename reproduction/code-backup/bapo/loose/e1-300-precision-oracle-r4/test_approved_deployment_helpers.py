"""Deployment decoder boundary tests, no remote mutations/submission."""
import ast,pathlib,tarfile,tempfile,io,hashlib,unittest,json
from pathlib import Path
r=Path(__file__).parent
class Deploy(unittest.TestCase):
 def decoder(self,name):
  tree=ast.parse((r/name).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='decode_bundle');ns=dict(tarfile=tarfile,pathlib=pathlib,hashlib=hashlib);exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-bundle-decoder','exec'),ns);return ns[node.name]
 def test_actual49_worker_and5_head_archives_exact_hash_maps(self):
  v=json.loads((r/'approved-deployment-packages.json').read_text())
  for kind,script,count in [('worker','deploy-approved-tools.py',49),('head','deploy-approved-head.py',5)]:
   spec=v[kind];p=r/spec['archive'];self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),spec['sha256']);payload=self.decoder(script)(p,spec['files']);self.assertEqual(len(payload),count)
   for name,(b,mode)in payload.items():self.assertEqual(hashlib.sha256(b).hexdigest(),spec['files'][name])
 def test_escape_symlink_duplicates_wrong_hash_missing_extra_refused(self):
  for script in ('deploy-approved-tools.py','deploy-approved-head.py'):
   gate=self.decoder(script)
   for case in ('escape','absolute','symlink','duplicate','hash','missing','extra'):
    with tempfile.TemporaryDirectory()as d:
     p=Path(d)/'unit.tar.gz';b=b'unit';name={'escape':'../unit','absolute':'/unit'}.get(case,'unit');expected={name:hashlib.sha256(b).hexdigest()};rows=[] if case=='missing'else[(name,b)]
     if case=='duplicate':rows*=2
     if case=='extra':rows.append(('extra',b))
     with tarfile.open(p,'w:gz')as t:
      for n,bytes_ in rows:
       m=tarfile.TarInfo(n);m.size=len(bytes_)
       if case=='symlink':m.type=tarfile.SYMTYPE;m.linkname='/outside';m.size=0
       t.addfile(m,io.BytesIO(bytes_))
     if case=='hash':expected[name]='0'*64
     with self.assertRaises(AssertionError,msg=case):gate(p,expected)
 def test_no_write_before_full_prevalidation_and_unchanged_blocked_placeholders(self):
  s=(r/'deploy-approved-tools.py').read_text();self.assertLess(s.index('payload=decode_bundle'),s.index("dst.open('xb')"));self.assertLess(s.index("record['sys_prefix']"),s.index("dst.open('xb')"));self.assertIn("not any((ROOT/'logs').glob('submission-intent-*.json'))",s)
  head=(r/'deploy-approved-head.py').read_text();self.assertLess(head.index('payload=decode_bundle'),head.index('d.mkdir()'));self.assertIn('assert not d.exists()',head)
  d=json.loads((r/'launch-tools-draft-hashes.json').read_text())
  for name in ('deploy-frozen-tools.py','deploy-frozen-head.py'):self.assertEqual(hashlib.sha256((r/name).read_bytes()).hexdigest(),d['extra_helpers_sha256'][name]);self.assertIn('BLOCKED',(r/name).read_text())
if __name__=='__main__':unittest.main()
