import tarfile,pathlib,hashlib,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo');run=root/'runs/bapo-formal500-2948'
assert (run/'TRAIN_RETURNED_SUCCESSFULLY').is_file()
cleanup=json.loads((run/'own-process-cleanup.json').read_text());assert not cleanup['remaining']
out=root/'logs/bapo-formal500-2948-raw-evidence.tar.gz'
if not out.exists():
 with tarfile.open(out,'w:gz') as tar:
  for path in sorted(run.rglob('*')):
   rel=path.relative_to(run)
   if rel.parts[0] in ('runtime','output','cache') or path.is_symlink() or not path.is_file():continue
   tar.add(path,arcname=str(path.relative_to(run.parent)),recursive=False)
h=hashlib.sha256()
with out.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':h.hexdigest(),'cleanup':cleanup}))
