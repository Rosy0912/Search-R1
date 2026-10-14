"""Publish a code-only isolated repair for the exact zero-Adam namespace fault."""
import argparse
import base64
import gzip
import json
import subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
fixed=['release/recursive_bran_a/ablation150_v1/entry.py',
       'release/recursive_bran_a/ablation150_v1/study.py',
       'release/recursive_bran_a/ablation150_v1/bootstrap.py']
new=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,
    'release/recursive_bran_a/ablation150_guard_recovery_v1',
    'release/recursive_bran_a/ablation150_zero_write_recovery_v1'],text=True).splitlines()
paths=fixed+new
files={name:base64.b64encode(subprocess.check_output(['git','show',f'{commit}:{name}'])).decode() for name in paths}
encoded=base64.b64encode(gzip.compress(json.dumps(dict(commit=commit,files=files)).encode())).decode()
code='''import base64,gzip,hashlib,json,os,shlex,shutil,subprocess,time
from pathlib import Path
r=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)));b=Path('/dataset/shaoshitong/recursive-ablation150-20261014')
ticket=b/('zero-write-recovery-launch-'+r['commit'][:8]+'.json')
if ticket.exists():print(ticket.read_text());raise SystemExit(0)
g=b/'C1-root-only';old=json.loads((g/'receipts/current-entry.json').read_text())
wait=json.loads((g/'receipts/actual-parent-wait.json').read_text());terminal=json.loads((Path(old['output'])/'entry-exit.json').read_text())
assert old['pid']==1590507 and old['uid']==0 and old['tick']==676184748
assert wait['pid']==old['pid'] and wait['rc']==1 and terminal['pid']==old['pid'] and terminal['rc']==1
assert not Path('/proc',str(old['pid'])).exists()
manifest=g/'checkpoints/training_state/partial_attempt_000001_outer_000001/manifest.json'
raw=manifest.read_bytes();m=json.loads(raw);manifest_sha=hashlib.sha256(raw).hexdigest()
assert m['status']=='PARTIAL_OUTER_STATE_INCOMPLETE_CONTROLLED_STOP' and m['attempted_outer']==1 and m['completed_outer_updates']==0
assert m['identity']['run_id']==old['run_id'] and m['identity']['config_sha256']==old['config_sha256']
workers=m['worker_outcome']['worker_states'];assert len(workers)==8 and {x['rank'] for x in workers}==set(range(8))
assert all(x['committed_optimizer_calls']==0 and x['scheduler_step_before_actual']==0 and x['scheduler_step_after_actual']==0 and x['path'] is None and 'Partial state directory must remain in this own run' in x['error'] for x in workers)
job=subprocess.check_output(['scontrol','show','job','3393'],text=True);assert 'JobState=RUNNING' in job and 'NodeList=h100-11' in job
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True);assert not apps.strip()
source=b/('source-zero-write-repair-'+r['commit'][:8]);assert not source.exists()
shutil.copytree(b/'source-entry-repair-72a09951',source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
hashes={}
for name,text in r['files'].items():
 path=source/name;path.parent.mkdir(parents=True,exist_ok=True);data=base64.b64decode(text);path.write_bytes(data)
 hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();assert hashes[name]==hashlib.sha256(data).hexdigest()
preserved=b/('preserved-zero-write-namespace-failure-'+str(old['pid']));preserved.mkdir()
for path in [g/'receipts/current-entry.json',g/'receipts/actual-parent-wait.json',b/'study-start.json',b/'study-progress.json',b/'active-study.json']:
 shutil.copy2(path,preserved/path.name)
request=dict(kind='ZERO_WRITE_SAVED_BATCH',previous_pid=old['pid'],previous_tick=old['tick'],previous_uid=old['uid'],manifest=str(manifest),sha256=manifest_sha,source_commit=r['commit'],actual_prior_wait_rc=wait['rc'],scope='Official initial actor/empty Adam restart after proved zero writes, saved prepared batch reused once; not a lost worker-RNG restoration claim')
request_path=g/'receipts/explicit-recovery.json';assert not request_path.exists();request_path.write_text(json.dumps(request,indent=2))
script=b/('study-wrapper-zero-write-repair-'+r['commit'][:8]+'.sh');exitfile=b/('study-actual-exit-zero-write-repair-'+r['commit'][:8]+'.json')
exit_command='import json,time;from pathlib import Path;Path('+repr(str(exitfile))+').write_text(json.dumps(dict(time=time.time(),rc=int(__import__("sys").argv[1]))))'
script.write_text('#!/bin/bash\\nset +e\\nexport RBA_ABLATION_BASE='+str(b)+'\\nexport RBA_ABLATION_SOURCE_COMMIT='+old['source_commit']+'\\nexport RBA_ABLATION_ENTRY_COMMIT='+r['commit']+'\\nexport RBA_H100_ALLOCATION_ID="$SLURM_JOB_ID"\\nbash '+str(source/'release/recursive_bran_a/ablation150_infra/study_run.sh')+'\\nrc=$?\\npython3 -c '+shlex.quote(exit_command)+' "$rc"\\nexit "$rc"\\n')
with (b/'logs'/('study-srun-zero-write-repair-'+r['commit'][:8]+'.log')).open('ab',buffering=0) as log:
 child=subprocess.Popen(['srun','--jobid=3393','--overlap','--nodes=1','--ntasks=1','--gres=gpu:h100:8','--cpus-per-task=96','--mem=768G','bash',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time=time.time(),pid=child.pid,uid=os.getuid(),tick=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]),source=str(source),entry_source_commit=r['commit'],scientific_source_commit=old['source_commit'],manifest=hashes,actual_exit_path=str(exitfile),previous_owner=old,old_failure_preserved=str(preserved),explicit_recovery=request,scope='Exact zero-write prepared-batch operational recovery; independent source, frozen science/config, allocation retained')
ticket.write_text(json.dumps(record,indent=2));(b/'active-study.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
'''.replace('PAYLOAD',repr(encoded))
Path(a.output).write_text(code)
print(json.dumps(dict(commit=commit,request_bytes=len(code.encode()),files=len(files))))
