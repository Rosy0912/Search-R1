"""Publish a new immutable resume launch on HKUST1; no process signals or hot edits."""
import argparse, hashlib, json, os, socket, subprocess, time, uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--source',required=True);a=p.parse_args()
b=Path(a.base).resolve();source=Path(a.source).resolve()
assert socket.gethostname().split('.')[0]=='xielab1' and os.getuid()==1005
assert source.parent==b and source.name.startswith('source-timed-guard-')
assert not Path('/proc/3620055').exists(),'Previous failed driver remains present'
old=subprocess.check_output(['tmux','display-message','-pt','recursive-bran-a-mu-meta-20261012','#{pane_dead}:#{pane_dead_status}'],text=True).strip()
assert old=='1:1',old
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert set(apps.split())<={'2239529'},apps
oracle=Path('/proc/2239529');assert oracle.stat().st_uid==1005 and oracle.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19]=='914664099'
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert len(free.split())==8 and min(map(int,free.split()))>=20000,free
assert os.statvfs(b).f_bavail*os.statvfs(b).f_frsize>=150*1024**3
commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
r=json.loads((b/'receipts/sparse-schedule/mu-only-meta-configuration.json').read_text())
run=uuid.uuid4().hex;r.update(run_id=run,source=str(source),source_commit=commit,
    output=str(b/('output-timed-guard-resume4-'+run)),purpose='human_resume_from4_per_action_trajectory_live_timing_guard_round_progress')
receipt=b/'receipts'/('timed-guard-config-'+run+'.json')
with receipt.open('x') as f:json.dump(r,f,indent=2)
log=b/'logs'/('timed-guard-resume4-'+run+'.log')
python='/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python'
env=os.environ.copy();env.update(RECURSIVE_BRAN_BASE=str(b),RECURSIVE_A_CHILD_CONFIG_RECEIPT=str(receipt),
 RECURSIVE_A_CONTINUATION_BINDING=str(b/'receipts/sparse-schedule/continuation-binding.json'),
 RECURSIVE_A_CONTINUATION_SHA256='da0b23050532fe7b7f77b02c9241bb39ee8aec5462aae6914b7dc40fb7b6ef9e',
 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONPATH=str(source),PYTHONUNBUFFERED='1',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',RAY_DEDUP_LOGS='0')
# Capture every inherited execution setting explicitly in the persistent launch command.
import shlex
exports={k:env[k] for k in ('RECURSIVE_BRAN_BASE','RECURSIVE_A_CHILD_CONFIG_RECEIPT','RECURSIVE_A_CONTINUATION_BINDING',
 'RECURSIVE_A_CONTINUATION_SHA256','CUDA_VISIBLE_DEVICES','PYTHONPATH','PYTHONUNBUFFERED','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','RAY_DEDUP_LOGS')}
command='cd '+shlex.quote(str(source))+'\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in exports.items())
command+='exec '+shlex.join([python,'-u','-m','search_r1.baselines.preflight','--',python,'-u',str(source/'release/recursive_bran_a/continue_timed_guard.py')])+' >'+shlex.quote(str(log))+' 2>&1\n'
launch=b/'receipts'/('timed-guard-launch-'+run+'.sh');launch.write_text(command);launch.chmod(0o700)
session='rba-timed-guard-'+run[:8]
subprocess.run(['tmux','new-session','-d','-s',session,'bash '+shlex.quote(str(launch))],check=True)
subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
pane=subprocess.check_output(['tmux','display-message','-pt',session,'#{session_id}:#{pane_id}:#{pane_pid}'],text=True).strip()
proof=dict(time=time.time(),run_id=run,source_commit=commit,source=str(source),config=str(receipt),
 config_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),launch=str(launch),
 launch_sha256=hashlib.sha256(launch.read_bytes()).hexdigest(),log=str(log),session=session,pane=pane,
 status='LAUNCHED_NOT_ACCEPTED',parent_completed_step=4,parent_actual_adam=80)
(b/'receipts'/('timed-guard-launch-'+run+'.json')).write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
