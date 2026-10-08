"""Failed3156 existing archive preservation only; no archive creation or signals."""
from pathlib import Path
import os,json,hashlib,tarfile,time,re,subprocess
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-countfix-20261004');JOB=3156
SOURCE='f0e37bf68d61ec926ac225f5e1ee1f606cb09bf9'
TOOLS={'bapo-retrieval-parity.py','cpu-tests.py','flash_preflight.py','formal300.sbatch','hardware.sbatch','hardware.sh','monitor.sh','native-actor-fixture.py','native-count-fixture.py','native-state-fixture.py','numerics-gpu-fixture.py','owned_processes.py','ray_cli_compat.py','resource_watch.py','stage-gpu-inner.sh','stage.sh','submission-worker-intent.py','submit.sh','verify_hardware_receipt.py','verify_launch.py','verify_shared_inputs.py'}
SOURCES=('verl/workers/actor/dp_actor.py','search_r1/baselines/objectives.py','search_r1/baselines/gradient_norm.py','verl/trainer/ppo/core_algos.py','verl/utils/fsdp_utils.py','verl/workers/fsdp_workers.py','verl/protocol.py','verl/utils/torch_functional.py','search_r1/baselines/distillation.py','search_r1/baselines/full_state.py','search_r1/baselines/checkpoint_gate.py')
SUCCESS='FAILED_JOB_EXISTING_RAW_ARCHIVE_REQUIRED_FILES_VERIFIED_REQUIRES_INDEPENDENT_REVIEW'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def own_path(p,root):
 assert p.resolve()==p and p!=root and root in p.parents and p.is_file()and not p.is_symlink()and p.stat().st_uid==os.getuid(),str(p)
def archive_readers_or_writers(logs):
 found=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit()or int(proc.name)==os.getpid():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:
     target=os.readlink(fd)
     if target.startswith(str(logs)+'/hardware-3156-raw-capture-'):
      stat=(proc/'stat').read_text().rsplit(')',1)[1].split();found.append(dict(pid=int(proc.name),start_ticks=int(stat[19]),target=target))
    except OSError:pass
  except OSError:pass
 return found
def validate_bindings(root):
 logs=root/'logs';run=root/'runs/bapo-instruct300-hardware-3156'
 for p in (root,logs,run,root/'tools',root/'repo'):
  assert p.resolve()==p and p.is_dir()and not p.is_symlink()and p.stat().st_uid==os.getuid(),str(p)
 assert (root/'source-expected.txt').read_text().strip()==SOURCE==(run/'source-commit.txt').read_text().strip()
 manifest=json.loads((logs/'tools-manifest.json').read_text());approval=json.loads((logs/'launch-approval.json').read_text())
 assert set(manifest['files'])==TOOLS and len(manifest['files'])==21 and manifest['source_commit']==SOURCE and manifest['root']==str(root)
 actual={p.name for p in (root/'tools').iterdir()if p.is_file()};assert actual==TOOLS
 assert approval['source_commit']==SOURCE and approval['root']==str(root)and approval['status']=='PASS_INDEPENDENT_E1_LAUNCH_REVIEW'
 assert approval['tools_sha256']==manifest['files']and approval['tools_manifest_sha256']==sha(logs/'tools-manifest.json')
 for name,digest in manifest['files'].items():
  p=root/'tools'/name;own_path(p,root);assert sha(p)==digest
 assert sha(logs/'cpu-gate.json')==approval['cpu_gate_record_sha256']
 cpu=json.loads((logs/'cpu-gate.json').read_text());marker=logs/'cpu-gate-source.txt';own_path(marker,root)
 assert marker.read_text().strip()==SOURCE and cpu['source_commit']==SOURCE and cpu['source_marker_sha256']==sha(marker)
 assert cpu['passed']==256 and cpu['skipped']==cpu['failed']==cpu['exit_code']==0 and cpu['scope']=='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE'
 assert cpu['sys_prefix']==str(root/'env')and cpu['test_launcher_sha256']==sha(root/'tools/cpu-tests.py')and cpu['log_sha256']==sha(logs/'cpu-gate.log')
 for name in ('tools-manifest.json','launch-approval.json','cpu-gate.json','cpu-gate-source.txt'):assert sha(logs/name)==sha(run/name)
 terminal=(run/'slurm-terminal.txt').read_text()
 for field,value in [('JobId','3156'),('JobName','bapo-instruct300-hardware'),('NodeList','h100-5'),('JobState','FAILED'),('ExitCode','1:0')]:
  assert re.findall(r'(?:^|\s)'+field+r'=(\S+)',terminal)==[value],field
 assert json.loads((run/'own-process-cleanup.json').read_text())['remaining']==[]
 required={root/'source-expected.txt',run/'source-commit.txt',run/'slurm-terminal.txt',run/'own-process-cleanup.json',run/'hardware.log',run/'resource-watch.jsonl'}
 required.update(logs/n for n in ('tools-manifest.json','launch-approval.json','cpu-gate.json','cpu-gate.log','cpu-gate-source.txt'))
 required.update(run/n for n in ('tools-manifest.json','launch-approval.json','cpu-gate.json','cpu-gate-source.txt'))
 required.update(root/'tools'/n for n in TOOLS);required.update(root/'repo'/n for n in SOURCES)
 for rank in range(8):
  for kind in ('native-count','native-state'):required.add(run/kind/f'rank-{rank}.json')
  for profile in ('baseline-fp32','production-nested-bf16-manual'):
   for directory in ('initial-step5','roundtrip-step5','no-roundtrip-step6','reloaded-step6'):
    for suffix in ('pt','json'):required.add(run/'native-state'/profile/directory/f'rank_{rank}.{suffix}')
 expected_missing_after_failed_actor={run/k/f'rank-{rank}.json'for k in ('native-actor','numeric-fixture','flash-native')for rank in range(8)}|{run/n for n in ('model-gpu.json','retrieval-cpu.json','retrieval-gpu.json','retrieval-comparison.json','HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY')}
 return required,sorted(str(p)for p in expected_missing_after_failed_actor if not p.is_file())
def main():
 logs=ROOT/'logs';output=logs/'hardware-3156-existing-archive-recovery-v2-result.json'
 for directory in (ROOT,logs):assert directory.resolve()==directory and directory.is_dir()and not directory.is_symlink()and directory.stat().st_uid==os.getuid()
 try:
  assert not output.exists(),'Never replace prior recovery'
  required,expected_missing=validate_bindings(ROOT)
  active=archive_readers_or_writers(logs);assert not active,repr(active)
  archives=sorted(logs.glob('hardware-3156-raw-capture-*.tar.gz'));assert len(archives)==1,'Ambiguous/missing original archive; do not recreate'
  archive=archives[0];own_path(archive,ROOT);assert archive.parent==logs;before=archive.stat();records=[];seen=set()
  with tarfile.open(archive,'r:gz')as tf:
   for member in tf:
    path=Path(member.name);assert member.isfile()and not path.is_absolute()and '..'not in path.parts and member.name not in seen;seen.add(member.name)
    current=ROOT/path;own_path(current,ROOT);h=hashlib.sha256();count=0
    with tf.extractfile(member)as f:
     for b in iter(lambda:f.read(1048576),b''):h.update(b);count+=len(b)
    assert count==member.size==current.stat().st_size and h.hexdigest()==sha(current)
    records.append(dict(path=str(current),archive_path=member.name,bytes=count,sha256=h.hexdigest()))
  missing=sorted(str(p)for p in required if str(p.relative_to(ROOT))not in seen);assert not missing,repr(missing)
  assert not archive_readers_or_writers(logs),'Archive is in use'
  after=archive.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
  assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE
  assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip()
  validate_bindings(ROOT)
  result=dict(status=SUCCESS,root=str(ROOT),job_id=JOB,source_commit=SOURCE,files=records,required_missing=[],expected_unproduced_after_actor_failure=expected_missing,archive=dict(path=str(archive),bytes=after.st_size,sha256=sha(archive)),original_remote_collector_exit_cause='UNKNOWN_NO_REMOTE_STDERR_OBSERVED',local_old_collector_known_stale20_assert=True,no_signals=True,no_new_archive=True,actual_hardware_accepted=False,time=time.time())
 except Exception as e:
  result=dict(status='RECOVERY_REFUSED_NO_ARCHIVE_OR_IDENTITY_CHANGED',job_id=JOB,error=type(e).__name__+': '+str(e),actual_hardware_accepted=False,time=time.time())
 with output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
