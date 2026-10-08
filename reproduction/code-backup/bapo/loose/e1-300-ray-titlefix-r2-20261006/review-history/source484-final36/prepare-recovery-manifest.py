"""Build allocated-run recovery inputs; never issues an acceptance permission."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-r2-20261006')
OLD=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
ORUN=OLD/'runs/bapo-instruct300-formal300-3231'
def binding(p):
 p=Path(p);assert p.is_absolute()and p==p.resolve()and p.is_file()and not p.is_symlink()
 before=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 return dict(path=str(p),sha256=h.hexdigest(),bytes=after.st_size)
a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--phase',choices=('hardware','formal300'),required=True);args=a.parse_args()
run=Path(args.run);job=int(os.environ['SLURM_JOB_ID']);source=(ROOT/'source-expected.txt').read_text().strip()
assert run==ROOT/'runs'/f'bapo-instruct300-{args.phase}-{job}'and run==run.resolve()and os.environ['BAPO_RUN_ID']==str(run)
assert (run/'source-commit.txt').read_text().strip()==source==subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True)
from omegaconf import OmegaConf
cfg=OmegaConf.to_container(OmegaConf.load(ORUN/'resolved-config.yaml'),resolve=True)
shared=set()
for value in (cfg['data']['train_files'],cfg['data']['val_files'],cfg['data']['final_val_files'],cfg['baselines']['bapo']['stage_validation_files']):shared.update(value if isinstance(value,list)else[value])
shared.add(cfg['baselines']['holdout']['manifest_path'])
model=Path(cfg['actor_rollout_ref']['model']['path']);shared.add(str(model/'VERIFIED-MODEL-MANIFEST.json'))
for item in json.loads((model/'VERIFIED-MODEL-MANIFEST.json').read_text())['files']:shared.add(str(model/item['file']))
actor=ORUN/'checkpoints/actor/global_step_25';driver=ORUN/'checkpoints/full_state/global_step_25';prefix=ROOT/'recovery-inputs/train-prefix1-25.log'
pb=binding(prefix)
m=dict(version=1,old_source_commit='39cdcdd1404213b897bb0de99cc52d72707682f0',old_run_root=str(ORUN),step=25,next_step=26,actual_outer_updates=25,world_size=8,target_source_commit=source,target_run_root=str(run),target_job_id=job,
 ranks=[dict(rank=i,state=binding(actor/'training_state'/f'rank_{i}.pt'),report=binding(actor/'training_state'/f'rank_{i}.json'))for i in range(8)],
 driver=binding(driver/'driver.pt'),driver_report=binding(driver/'driver-reload.json'),algorithm=binding(ORUN/'checkpoints/algorithm/global_step_25/state.json'),resolved_config=binding(ORUN/'resolved-config.yaml'),
 shared_inputs=[binding(p)for p in sorted(shared)],hf_files=[binding(p)for p in sorted(actor.iterdir())if p.is_file()],historical_first5_receipt=binding(ORUN/'logs/step5-independent-release.json'),train_prefix=pb,train_prefix_live=dict(path=str(ORUN/'train.log'),bytes=pb['bytes'],sha256=pb['sha256']))
path=run/'recovery-manifest.json'
with path.open('x')as f:json.dump(m,f,indent=2,allow_nan=False)
print(json.dumps(dict(status='ALLOCATED_RECOVERY_INPUT_MANIFEST_NOT_PERMISSION',phase=args.phase,job_id=job,run_root=str(run),source_commit=source,manifest=binding(path)),indent=2),flush=True)
