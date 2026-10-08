import hashlib,json,pathlib,gzip,time,shutil
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');assets=root/'assets-root';status=assets/'logs/direct-copy-status.json';exitfile=root/'logs/direct-copy-supervised-20261003.exit'
print('WAITING_FOR_ALL_IMMUTABLE_ASSET_HASHES',flush=True)
while True:
 try:
  if status.exists() and json.loads(status.read_text()).get('complete'):break
 except json.JSONDecodeError:
  pass  # Previous downloader versions did not publish status atomically.
 if exitfile.exists():raise RuntimeError('Asset downloader finished without full verified assets; retain partials and repair')
 time.sleep(30)
manifest={a['path']:a for a in json.loads((assets/'assets.json').read_text())};wiki=assets/'assets/wiki'
report={'source_manifest_sha256':hashlib.sha256((assets/'assets.json').read_bytes()).hexdigest(),'outputs':{}}
def produce(name,inputs,compressed=False):
 dest=wiki/name;temp=dest.with_name(name+'.building.'+str(time.time_ns()));h=hashlib.sha256();size=0
 with temp.open('wb') as out:
  for p in inputs:
   source_h=hashlib.sha256()
   if compressed:
    with p.open('rb') as source:
     for chunk in iter(lambda:source.read(8*1024**2),b''):source_h.update(chunk)
    assert source_h.hexdigest()==manifest['wiki/'+p.name]['sha256'], 'Compressed source SHA mismatch'
   with (gzip.open(p,'rb') if compressed else p.open('rb')) as f:
    for chunk in iter(lambda:f.read(8*1024**2),b''):
     out.write(chunk);h.update(chunk);size+=len(chunk)
     if not compressed:source_h.update(chunk)
   assert source_h.hexdigest()==manifest['wiki/'+p.name]['sha256'], 'Source SHA mismatch'
 if not compressed:assert size==sum(p.stat().st_size for p in inputs)
 if dest.exists():dest.rename(dest.with_name(dest.name+'.previous.'+str(time.time_ns())))
 temp.replace(dest)
 report['outputs'][name]={'sha256':h.hexdigest(),'size':size,'source_sha256':[manifest['wiki/'+p.name]['sha256'] for p in inputs]}
 print('DERIVED_ASSET_SHA_RECORDED',name,size,h.hexdigest(),flush=True)
produce('e5_Flat.index',[wiki/'part_aa',wiki/'part_ab'])
produce('wiki-18.jsonl',[wiki/'wiki-18.jsonl.gz'],True)
(root/'logs/derived-assets.json').write_text(json.dumps(report,indent=2));(root/'logs/derived-assets.ready').write_text('Verified source concatenation and gzip decoding; actual FAISS/corpus integration still required.\n')
