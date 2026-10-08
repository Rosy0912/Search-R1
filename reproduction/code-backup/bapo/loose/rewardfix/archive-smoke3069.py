import pathlib,json,tarfile,hashlib
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097');run=root/'runs/bapo-smoke2-3069'
assert (run/'TRAIN_RETURNED_SUCCESSFULLY').exists()
assert json.loads((run/'own-process-cleanup.json').read_text())['remaining']==[]
archive=root/'logs/bapo-rewardfix-smoke3069-raw-measurements.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(run.iterdir()):
  if p.is_file():
   assert not p.is_symlink()
   t.add(p,arcname='bapo-smoke2-3069/'+p.name,recursive=False)
record=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),scope='B097_REAL_SMOKE2_RAW_LOG_RESOURCE_FILES_CHECKPOINT_PROOF_SEPARATE',excluded_regenerable=['cache','runtime','tmp'],checkpoint='output checkpoint complete hash and actual CPU HF verified separately')
(root/'logs/smoke3069-archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
