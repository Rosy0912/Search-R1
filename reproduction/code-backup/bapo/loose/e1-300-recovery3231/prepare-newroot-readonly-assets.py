"""Create only fresh recovery root infrastructure; no source deployment/job/GPU action."""
import pathlib,os,json,hashlib,subprocess
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');OLD=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005');BASE=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo')
assert os.uname().nodename=='h100-5'
assert subprocess.check_output(['git','-C',str(OLD/'repo'),'rev-parse','HEAD'],text=True).strip()=='39cdcdd1404213b897bb0de99cc52d72707682f0'
assert not subprocess.check_output(['git','-C',str(OLD/'repo'),'status','--porcelain'],text=True).strip()
assert ROOT.is_dir()and not ROOT.is_symlink()and not (ROOT/'repo').exists()
expected={'tools','logs','cache','runs','assets','common-data','assets-root','recovery-inputs','scheduler-head','env','retriever-gpu-lib','lib'}
assert {p.name for p in ROOT.iterdir()}==expected
for name in ['tools','logs','runs','recovery-inputs','scheduler-head']:assert not list((ROOT/name).iterdir())
for name in ['env','retriever-gpu-lib','lib']:assert (ROOT/name).is_symlink()and (ROOT/name).resolve()==(BASE/name).resolve()
assert (ROOT/'assets-root/assets').is_symlink()and (ROOT/'assets-root/assets').resolve()==(BASE/'assets-root/assets').resolve()
# Fresh directory with verified read-only shared file hardlinks; no content updates.
records=[]
for src,dst in [(OLD/'common-data',ROOT/'common-data'),(OLD/'assets/qwen2.5-3b-instruct-aa8e725',ROOT/'assets/qwen2.5-3b-instruct-aa8e725')]:
 if not dst.exists():dst.mkdir()
 for p in sorted(src.iterdir()):
  if p.name=='raw'and p.is_dir()and not p.is_symlink():continue # not a prebuilt training input
  assert p.is_file()and not p.is_symlink()
  q=dst/p.name
  if not q.exists():os.link(p,q)
  assert q.is_file()and not q.is_symlink()and p.stat().st_ino==q.stat().st_ino
  records.append(dict(source=str(p),target=str(q),bytes=q.stat().st_size,read_only_input=True))
p=BASE/'assets-root/assets.json';q=ROOT/'assets-root/assets.json';assert not q.exists();os.link(p,q)
print(json.dumps(dict(status='FRESH_RECOVERY_INFRASTRUCTURE_ONLY_NOT_SOURCE_DEPLOYED_OR_TRAINING',root=str(ROOT),files=records,old_source_unchanged=True,job_submitted=False,gpu_initialized=False),indent=2))
