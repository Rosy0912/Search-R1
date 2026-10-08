"""Own isolated directories/read-only hardlinks only; no GPU/source/job launch."""
import pathlib,os,json,subprocess
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006');O=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006');B=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo');assert os.uname().nodename=='h100-5'and R.resolve()==R and R.is_dir();assert not (R/'repo').exists()and not (R/'source-expected.txt').exists();assert subprocess.check_output(['git','-C',str(O/'repo'),'rev-parse','HEAD'],text=True).strip()=='b7119791b86951727cea36a1881e8a525c932494';assert not subprocess.check_output(['git','-C',str(O/'repo'),'status','--porcelain'],text=True).strip()
for n in ['tools','logs','cache','runs','assets','common-data','assets-root','recovery-inputs','scheduler-head']:(R/n).mkdir(exist_ok=False)
for n in ['env','retriever-gpu-lib','lib']:(R/n).symlink_to(B/n,target_is_directory=True)
(R/'assets-root/assets').symlink_to(B/'assets-root/assets',target_is_directory=True);os.link(B/'assets-root/assets.json',R/'assets-root/assets.json');count=0;size=0
for src,dst in [(O/'common-data',R/'common-data'),(O/'assets/qwen2.5-3b-instruct-aa8e725',R/'assets/qwen2.5-3b-instruct-aa8e725')]:
 dst.mkdir(exist_ok=True)
 for p in sorted(src.rglob('*')):
  assert not p.is_symlink();q=dst/p.relative_to(src)
  if p.is_dir():q.mkdir(exist_ok=True)
  else:
   assert p.is_file();os.link(p,q);assert q.stat().st_ino==p.stat().st_ino;count+=1;size+=p.stat().st_size
out=dict(status='ISOLATED_INFRASTRUCTURE_INPUT_HARDLINKS_ONLY_NOT_SOURCE_OR_TRAINING',root=str(R),hardlinked_files=count,hardlinked_bytes=size,includes_original_raw=True,gpu_initialized=False,job_submitted=False);(R/'logs/input-preparation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
