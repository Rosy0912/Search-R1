from pathlib import Path
old='/dataset/shaoshitong/search-r1-rl/bapo'
new=old+'-rewardfix-b097'
r=Path('work/rewardfix/remote')
for name in ['stage.sh','stage-gpu-inner.sh','hardware.sh','monitor.sh','resource_watch.py','owned_processes.py','ray_cli_compat.py','flash_preflight.py','parity-gpu64.sbatch']:
 s=(Path('work/remote')/name).read_text().replace(old,new).replace('b20f371972bb1bc41b735c6cd350b9a779d8820d','b09736d9d1b487a5c134e7a358761e04613cea32')
 if name=='stage.sh':
  s=s.replace('export OMP_NUM_THREADS=1','export PYTHONDONTWRITEBYTECODE=1\nexport OMP_NUM_THREADS=1')
  s=s.replace('test -f "$ROOT/logs/environment.ready"','test -f "$ROOT/logs/environment.ready"\npython "$ROOT/tools/verify_shared_inputs.py"')
 if name=='stage-gpu-inner.sh':
  s=s.replace("real=p.resolve();assert root in real.parents", "real=p.resolve(); allowed=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo/assets-root/assets').resolve(); assert real==allowed or allowed in real.parents")
  s=s.replace('cd "$ROOT/repo"','cd "$ROOT/repo"\npython "$ROOT/tools/verify_shared_inputs.py"')
 if name=='parity-gpu64.sbatch':s=s.replace('bapo-hw-gpu64','bapo-rfix-hw-b097')
 (r/name).write_text(s)
(r/'bapo-retrieval-parity.py').write_bytes(Path('outputs/bapo-retrieval-parity.py').read_bytes())
# This is a fresh derivation against copied inputs; its path-specific manifest is independent.
s=Path('work/remote/prepare-stage64-proof.sh').read_text().replace(old,new).replace('stage64-b20f371','stage64-b097')
s=s.replace("old=json.loads((root/'logs/common-split-manifest.json').read_text())",'')
s=s.replace('new=prepare_common_holdout(c);assert new==old',"new=prepare_common_holdout(c); assert new['content_identity']['sha256']=='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'\n(root/'logs/common-split-manifest.json').write_text(json.dumps(new,indent=2)+'\\n')")
s=s.replace("assert len(record['question_identities'])==64", "assert len(record['question_identities'])==64\nassert file_hash(p)=='82a5ed0d04b261031d4ee03532f0c35fc688dd2fb11c7cdaf1cc87ad0e46c34e'\nassert len(train)==166222 and file_hash(Path(c.data.train_files))=='e5e774fd1f5777a8be91103facaccb711cfd9593d7563e9d5afe04e6b1a52bea'")
s=s.replace('export PATH=', 'export PYTHONDONTWRITEBYTECODE=1\nexport PATH=')
(r/'prepare-stage64-proof.sh').write_text(s)
