import hashlib,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'control'))
from assemble_assets import assemble
class AssembleTests(unittest.TestCase):
 def test_exact_bytes_with_prefix_and_two_sources(self):
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);specs=[]
   for name,blocks in [('aa',[b'prefix',b'part1',b'part2']),('ab',[b'last'])]:
    paths=[]
    for i,b in enumerate(blocks):
     p=base/f'{name}-{i}';p.write_bytes(b);paths.append(p)
    raw=b''.join(blocks);specs.append(dict(name=name,size=len(raw),sha=hashlib.sha256(raw).hexdigest(),prefix=str(paths[0]) if name=='aa' else None,ranges=[(0,0,str(p)) for p in (paths[1:] if name=='aa' else paths)]))
   result=assemble(specs,base/'index');expected=b'prefixpart1part2last'
   self.assertEqual((base/'index').read_bytes(),expected);self.assertEqual(result['sha256'],hashlib.sha256(expected).hexdigest())
 def test_corrupt_source_never_publishes_destination(self):
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);(base/'part').write_bytes(b'corrupt')
   with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):
    assemble([dict(name='aa',size=7,sha=hashlib.sha256(b'correct').hexdigest(),prefix=str(base/'part'),ranges=[])],base/'index')
   self.assertFalse((base/'index').exists())
if __name__=='__main__':unittest.main()
