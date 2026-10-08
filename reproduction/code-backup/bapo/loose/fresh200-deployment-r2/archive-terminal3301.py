from pathlib import Path
import hashlib,json,os,subprocess,tarfile
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r2-20261006');run=R/'runs/diagnostic-discovery128-5f63f5380e8a45ee98e4feeb129bad63-3301';source='8e01320bd4ba06f5b64cb6a3164373e6f25f35f7'
assert os.uname().nodename=='h100-5' and (R/'source-expected.txt').read_text().strip()==source and run.is_dir()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
slurm=subprocess.check_output(['scontrol','show','job','3301'],text=True);assert 'JobState=FAILED' in slurm and 'ExitCode=42:0' in slurm
cleanup=json.loads((run/'own-process-cleanup.json').read_text());assert cleanup['remaining']==[] and cleanup['own_marker']==str(run)
record=R/'logs/diagnostic3301-terminal-record.json';assert not record.exists()
accounting=subprocess.run(['sacct','-j','3301','--format=JobID,State,ExitCode,Elapsed,Start,End,NodeList','-P'],text=True,capture_output=True)
record.write_text(json.dumps({'scontrol':slurm,'sacct':{'exit_code':accounting.returncode,'stdout':accounting.stdout,'stderr':accounting.stderr},'source_commit':source,'run_root':str(run),'formal_outer':0,'actual_adam_writes_per_rank':0,'reserved_gpu_seconds':454*8,'reserved_gpu_hours':454*8/3600,'cleanup':cleanup},indent=2))
files=[p for p in run.iterdir() if p.is_file() and not p.is_symlink()]
files += [p for p in (run/'diagnostics').rglob('*') if p.is_file() and not p.is_symlink() and p.suffix!='.pt'] if (run/'diagnostics').exists() else []
files += [R/'logs'/n for n in ['slurm-diagnostic-discovery128-3301.out','slurm-diagnostic-discovery128-3301.err','launch-approval.json','fresh200-diagnostic-code-review.json','tools-manifest.json','target-cpu-gate.json','context-budget-audit.json']]
files.append(record);assert all(p.is_file() and not p.is_symlink() for p in files)
tensor_hashes=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in (run/'diagnostics').rglob('*.pt')] if (run/'diagnostics').exists() else []
manifest=R/'logs/diagnostic3301-terminal-archive-manifest.json';assert not manifest.exists();manifest.write_text(json.dumps({'files':{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files},'remote_retained_initial_tensors':tensor_hashes,'runtime_executable_file_exists':(R/'logs/runtime-executable.json').exists(),'scope':'RAW_FAILED_RESOURCE_GUARD_ARCHIVE_NOT_NATIVE_TRAINING_ACCEPTANCE'},indent=2))
archive=R/'logs/diagnostic3301-terminal-raw.tar.gz';assert not archive.exists()
with tarfile.open(archive,'x:gz') as t:
 for p in files+[manifest]:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files)+1,'retained_tensor_hashes':len(tensor_hashes)}))
