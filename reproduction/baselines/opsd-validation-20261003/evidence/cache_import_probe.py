import ast,hashlib,json,os,shutil,sys,tempfile
from pathlib import Path
from unittest.mock import patch
SOURCE=Path(__file__).resolve().parents[5]/'runtime-20261003/import_cache.py'
results={}
for case in ['correct','wrong_sha','source_mutation','existing_candidate']:
 with tempfile.TemporaryDirectory() as td:
  folder=Path(td);root=folder/'target';source_dir=folder/'source';source_dir.mkdir();payload=b'fixed asset bytes'
  source=source_dir/'part_aa';source.write_bytes(payload)
  (root/'logs').mkdir(parents=True);(root/'assets-root/assets/wiki').mkdir(parents=True)
  expected=hashlib.sha256(payload).hexdigest() if case!='wrong_sha' else 'f'*64
  (root/'assets-root/assets.json').write_text(json.dumps([{'path':'wiki/part_aa','size':len(payload),'sha256':expected}]))
  tree=ast.parse(SOURCE.read_text())
  for node in tree.body:
   if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
    if node.targets[0].id=='root':node.value=ast.parse('Path('+repr(str(root))+')',mode='eval').body
    if node.targets[0].id=='source':node.value=ast.parse('Path('+repr(str(source_dir))+')/name',mode='eval').body
  ast.fix_missing_locations(tree);code=compile(tree,str(SOURCE),'exec')
  dest=root/'assets-root/assets/wiki/part_aa';partial=dest.with_name('part_aa.cache-import.partial')
  if case=='existing_candidate':partial.write_bytes(b'preserve')
  original_copy=shutil.copyfile
  def copy(src,dst):
   result=original_copy(src,dst)
   if case=='source_mutation':
    s=source.stat();os.utime(source,ns=(s.st_atime_ns,s.st_mtime_ns+1000000000))
   return result
  try:
   with patch.object(sys,'argv',['import_cache.py','part_aa']),patch.object(shutil,'copyfile',side_effect=copy):exec(code,{})
   accepted=True
  except AssertionError:accepted=False
  assert accepted==(case=='correct')
  assert dest.exists()==accepted
  if case=='existing_candidate':assert partial.read_bytes()==b'preserve'
  results[case]={'accepted':accepted,'dest_exists':dest.exists(),'partial_preserved':partial.exists()}
print('PROBE_RESULT='+json.dumps(results,sort_keys=True))
