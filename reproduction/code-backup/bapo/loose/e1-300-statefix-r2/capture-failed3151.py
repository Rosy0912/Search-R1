import pathlib,tarfile,sys
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-statefix-20261004')
run=root/'runs/bapo-instruct300-hardware-3151'
files=[p for p in run.rglob('*')if p.is_file()and not p.is_symlink()]
files += [root/'logs/slurm-hardware-3151.out',root/'logs/slurm-hardware-3151.err']
assert len(files)>=8
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz')as tar:
 for p in files:
  assert p.is_file()and not p.is_symlink()and root in p.resolve().parents
  tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
