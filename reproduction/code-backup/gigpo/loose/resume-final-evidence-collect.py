"""Read-only freeze of the completed original200->300 run, excluding model/data files."""
from pathlib import Path
import json,subprocess,hashlib,base64,gzip,datetime
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo');RUN=ROOT/'runs/gigpo-resume200to300-job3130-1791086878335123421'
result=json.loads((RUN/'run-result.json').read_text());assert result['status']=='CHECKPOINT_RELOADED_UNREVIEWED' and result['cleanup_residual_states']=={}
paths=set(RUN.glob('*.json'))|set(RUN.glob('*.jsonl'))|set((RUN/'audit').rglob('*.jsonl'))|set(RUN.glob('*.yaml'))
paths.update(RUN/name for name in ['training.log','guard.log','checkpoint205-cpu-verification.log','checkpoint-cpu-verification.log','ray.log'])
for step in (205,225,250,275,300):
 paths.add(RUN/'checkpoints/actor'/f'global_step_{step}'/'training_state/manifest.json')
 paths.add(RUN/'checkpoints/algorithm'/f'global_step_{step}'/'state.json')
 paths.add(ROOT/'logs'/f'resume3130-checkpoint{step}-cpu-state-verification-v1.json') if step!=205 else None
paths.update(ROOT/'logs'/name for name in ['resume3130-205-independent-review.json','resume-original200-submission-v2.json','deployment-binding-resume-60a472b-v2.json','resume-header-deployment-independent-review-v2.json','resume-checkpoint-inspector-independent-review-v3.json','resume-original200-3130.log','original2992-checkpoint200-resume-inventory.json','source-activation-resume-60a472b-v1.json'])
for p in (RUN/'ray').rglob('worker-*.out'):
 if b'SEARCH_R1_RECOVERY_COMPLETE_JSON=' in p.read_bytes():paths.add(p)
raw={};missing=[]
for p in sorted(paths):
 if not p.is_file():missing.append(str(p));continue
 b=p.read_bytes();raw[str(p)]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
def command(args):
 p=subprocess.run(args,capture_output=True,text=True);return dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
obj=dict(captured_at=datetime.datetime.now().isoformat(),run=str(RUN),job_id='3130',source=str(ROOT/'runs/gigpo-formal500-job2992-1791016340597739432/checkpoints/actor/global_step_200'),raw_files=raw,missing_optional_paths=missing,live_scheduler=command(['scontrol','show','job','3130']),live_queue=command(['squeue','-j','3130','-o','%.18i %.9T %.10M %.20R']),gpu=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader']),code_head=command(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD']),code_status=command(['git','-C',str(ROOT/'repo'),'status','--porcelain']))
p=ROOT/'logs/resume3130-final-evidence-v1.json.gz';assert not p.exists(),'Existing final evidence must be reused'
blob=gzip.compress(json.dumps(obj,indent=2).encode(),mtime=0);p.write_bytes(blob);print(json.dumps(dict(path=str(p),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),raw_count=len(raw),missing_optional_paths=missing)))
