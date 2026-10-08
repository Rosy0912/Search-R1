from pathlib import Path
import tempfile,json,hashlib,gzip,contextlib,io
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2')
source=(root/'work/prepare_derived_assets.py').read_text();records=[]
for case in ('valid','legacy_partial_json','corrupt_part','corrupt_compressed'):
 with tempfile.TemporaryDirectory() as temporary:
  task=Path(temporary);assets=task/'assets-root';wiki=assets/'assets/wiki'
  wiki.mkdir(parents=True);(assets/'logs').mkdir();(task/'logs').mkdir()
  data={'part_aa':b'index-A','part_ab':b'index-B','wiki-18.jsonl.gz':gzip.compress(b'{"id":"toy","contents":"fixed"}\n',mtime=0)}
  for name,value in data.items():(wiki/name).write_bytes(value)
  (assets/'assets.json').write_text(json.dumps([{'path':'wiki/'+name,'sha256':hashlib.sha256(value).hexdigest()} for name,value in data.items()]))
  status=assets/'logs/direct-copy-status.json';status.write_text('' if case=='legacy_partial_json' else '{"complete":true}')
  (wiki/'e5_Flat.index').write_bytes(b'previous-index')
  if case=='corrupt_part':(wiki/'part_ab').write_bytes(b'corrupt-index')
  if case=='corrupt_compressed':(wiki/'wiki-18.jsonl.gz').write_bytes(gzip.compress(b'altered',mtime=0))
  script=source.replace("pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')",'pathlib.Path('+repr(str(task))+')')
  # Replace only the synthetic polling delay with publication of legacy status.
  if case=='legacy_partial_json':script=script.replace('time.sleep(30)',"status.write_text('{\"complete\":true}')")
  failed=False
  try:
   with contextlib.redirect_stdout(io.StringIO()):exec(compile(script,'derived-assets-toy','exec'),{})
  except AssertionError:failed=True
  ready=(task/'logs/derived-assets.ready').exists()
  if case in ('valid','legacy_partial_json'):
   assert not failed and ready
   assert (wiki/'e5_Flat.index').read_bytes()==b'index-Aindex-B'
   assert (wiki/'wiki-18.jsonl').read_bytes()==b'{"id":"toy","contents":"fixed"}\n'
   report=json.loads((task/'logs/derived-assets.json').read_text())
   assert all(hashlib.sha256((wiki/name).read_bytes()).hexdigest()==record['sha256'] for name,record in report['outputs'].items())
  else:
   assert failed and not ready
   if case=='corrupt_part':assert (wiki/'e5_Flat.index').read_bytes()==b'previous-index'
  records.append({'case':case,'source_hash_rejection':failed,'ready_marker':ready,'passed':True})
print(json.dumps({'simulation_only':True,'changes':'temporary root substitution; legacy scenario substitutes poll delay with a valid status publication','cases':records},indent=2))
