"""Deploy only a reviewed, checksum-bound source bundle to the fresh ROOT."""
import pathlib,hashlib,subprocess,json,sys
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005')
source,bundle_sha=sys.argv[1:];assert len(source)==40 and len(bundle_sha)==64
assert r.is_dir()and not r.is_symlink()and r==r.resolve()
p=r/'source-verified.bundle';assert p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()==bundle_sha
assert not (r/'repo').exists()and not (r/'source-expected.txt').exists()and not list((r/'runs').iterdir())
subprocess.run(['git','clone','--no-checkout',str(p),str(r/'repo')],check=True)
subprocess.run(['git','-C',str(r/'repo'),'checkout','--detach',source],check=True)
assert subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True)
with (r/'source-expected.txt').open('x')as f:f.write(source+'\n')
print(json.dumps(dict(status='FRESH_REVIEWED_SOURCE_DEPLOYED_NOT_GPU_ACCEPTANCE',root=str(r),source_commit=source,bundle_sha256=bundle_sha)))
