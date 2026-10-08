"""Check a separately published actual-allocation permission before GPU services."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005')
a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--phase',choices=('hardware','formal300'),required=True);args=a.parse_args()
run=Path(args.run);source=(ROOT/'source-expected.txt').read_text().strip();job=int(os.environ['SLURM_JOB_ID'])
assert run==ROOT/'runs'/f'bapo-instruct300-{args.phase}-{job}'and os.environ['BAPO_RUN_ID']==str(run)and run==run.resolve()
verify_only=args.phase=='hardware'
command=[sys.executable,'-m','search_r1.baselines.main','--config-name','ppo_bapo_e1_300',f'baselines.algorithm_source_commit={source}',f'trainer.resume_full_state={run}/recovery-manifest.json',f'trainer.resume_permission={run}/recovery-permission.json',f'trainer.resume_witness_dir={run}/recovery-witness',f'trainer.resume_verify_only={str(verify_only).lower()}','--cfg','job','--resolve']
resolved=subprocess.check_output(command,cwd=ROOT/'repo',text=True)
with (run/'permission-resolved-config.yaml').open('x')as f:f.write(resolved)
from omegaconf import OmegaConf
from search_r1.baselines.recovery import validate_manifest
config=OmegaConf.to_container(OmegaConf.create(resolved),resolve=True)
m=validate_manifest(run/'recovery-manifest.json',config,source,str(run),run/'recovery-permission.json')
print(json.dumps(dict(status='ACTUAL_ALLOCATION_RECOVERY_PERMISSION_VALIDATED_NOT_GPU_PROOF',source_commit=source,job_id=job,run_root=str(run),verify_only=verify_only,step=m['step'],next_step=m['next_step'])),flush=True)
