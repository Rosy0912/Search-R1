from pathlib import Path
import hashlib,json,os,subprocess,tarfile
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006');run=R/'runs/formal200-6a29642035ff40b2b7787de206f886c0-3307';source='aec5e057093b4c491bfb0a47abd7605d98981b56'
assert os.uname().nodename=='h100-5' and (R/'source-expected.txt').read_text().strip()==source and run.is_dir()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
slurm=subprocess.check_output(['scontrol','show','job','3307'],text=True);assert 'JobState=FAILED' in slurm and 'ExitCode=1:0' in slurm
cleanup=json.loads((run/'own-process-cleanup.json').read_text());assert cleanup['remaining']==[] and cleanup['own_marker']==str(run)
record=R/'logs/diagnostic3307-terminal-record.json';assert not record.exists()
accounting=subprocess.run(['sacct','-j','3307','--format=JobID,State,ExitCode,Elapsed,Start,End,NodeList','-P'],text=True,capture_output=True)
record.write_text(json.dumps({'scontrol':slurm,'sacct':{'exit_code':accounting.returncode,'stdout':accounting.stdout,'stderr':accounting.stderr},'source_commit':source,'run_root':str(run),'formal_outer':0,'actual_adam_writes_per_rank':0,'reserved_gpu_seconds':1001*8,'reserved_gpu_hours':1001*8/3600,'cleanup':cleanup},indent=2))
files=[p for p in run.iterdir() if p.is_file() and not p.is_symlink()]
files += [p for p in (run/'diagnostics').rglob('*') if p.is_file() and not p.is_symlink() and p.suffix!='.pt'] if (run/'diagnostics').exists() else []
files += [R/'logs'/n for n in ['slurm-formal200-3307.out','slurm-formal200-3307.err','launch-approval.json','tools-manifest.json','target-cpu-gate.json','context-budget-audit.json']]
files += [R/'logs/runtime-tests.actual.json',R/'logs/runtime-tests.actual.log',R/'logs/runtime-executable.json',R/'logs/frozen-policy-safety-config.json',R/'logs/user-fixed-policy-thresholds.json'];files.append(record);assert all(p.is_file() and not p.is_symlink() for p in files)
files += [p for p in (run/'checkpoints/diagnostics').rglob('*') if p.is_file() and not p.is_symlink() and p.suffix!='.pt']
tensor_hashes=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in (run/'diagnostics').rglob('*.pt')] if (run/'diagnostics').exists() else []
tensor_hashes += [{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in (run/'checkpoints/diagnostics').rglob('*.pt')]
manifest=R/'logs/diagnostic3307-terminal-archive-manifest.json';assert not manifest.exists();manifest.write_text(json.dumps({'files':{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files},'remote_retained_initial_tensors':tensor_hashes,'runtime_executable_file_exists':(R/'logs/runtime-executable.json').exists(),'scope':'RAW_FAILED_SAMPLING_PROBABILITY_ARCHIVE_NOT_NATIVE_TRAINING_ACCEPTANCE'},indent=2))
archive=R/'logs/diagnostic3307-terminal-raw.tar.gz';assert not archive.exists()
with tarfile.open(archive,'x:gz') as t:
 for p in files+[manifest]:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files)+1,'retained_tensor_hashes':len(tensor_hashes)}))
