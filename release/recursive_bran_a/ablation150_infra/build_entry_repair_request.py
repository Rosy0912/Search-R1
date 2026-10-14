"""Build an isolated repair for the verified pre-model AF_UNIX failure."""
import argparse,base64,gzip,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
paths=['release/recursive_bran_a/ablation150_v1/entry.py','release/recursive_bran_a/ablation150_v1/study.py','release/recursive_bran_a/ablation150_v1/unstarted_retry.py']
files={name:base64.b64encode(subprocess.check_output(['git','show',f'{commit}:{name}'])).decode() for name in paths}
encoded=base64.b64encode(gzip.compress(json.dumps(dict(commit=commit,files=files)).encode())).decode()
code='''import base64,gzip,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
r=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)));b=Path('/dataset/shaoshitong/recursive-ablation150-20261014')
ticket=b/('repair-launch-'+r['commit'][:8]+'.json')
if ticket.exists():print(ticket.read_text());raise SystemExit(0)
g=b/'C1-root-only';old=json.loads((g/'receipts/current-entry.json').read_text())
wait=json.loads((g/'receipts/actual-parent-wait.json').read_text())
assert wait['pid']==old['pid'] and wait['rc']!=0
assert not Path('/proc',str(old['pid'])).exists()
assert not Path(old['output'],'fresh-initialization.json').exists()
assert not list((g/'checkpoints').rglob('*.pt')) and not list((g/'checkpoints').rglob('*.bin'))
assert 'AF_UNIX path length cannot exceed 107 bytes' in (g/'logs/training.log').read_text()
job=subprocess.check_output(['scontrol','show','job','3393'],text=True)
assert 'JobState=RUNNING' in job and 'NodeList=h100-11' in job
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True);assert not apps.strip()
source=b/('source-entry-repair-'+r['commit'][:8]);assert not source.exists()
shutil.copytree(b/'source',source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
manifest={}
for name,text in r['files'].items():
 raw=base64.b64decode(text);(source/name).write_bytes(raw);manifest[name]=hashlib.sha256((source/name).read_bytes()).hexdigest()
preserved=b/('preserved-pre-model-failure-'+str(old['pid']));preserved.mkdir()
for name in ['study-start.json','study-progress.json','study-actual-exit.json']:
 if (b/name).is_file():shutil.copy2(b/name,preserved/name)
script=b/('study-wrapper-entry-repair-'+r['commit'][:8]+'.sh')
exitfile=b/('study-actual-exit-repair-'+r['commit'][:8]+'.json')
script.write_text('#!/bin/bash\\nset +e\\nexport RBA_ABLATION_BASE='+str(b)+'\\nexport RBA_ABLATION_SOURCE_COMMIT='+old['source_commit']+'\\nexport RBA_ABLATION_ENTRY_COMMIT='+r['commit']+'\\nexport RBA_H100_ALLOCATION_ID="$SLURM_JOB_ID"\\nbash '+str(source/'release/recursive_bran_a/ablation150_infra/study_run.sh')+'\\nrc=$?\\npython3 -c '+__import__('shlex').quote('import json,time;from pathlib import Path;Path('+repr(str(exitfile))+').write_text(json.dumps(dict(time=time.time(),rc=int(__import__("sys").argv[1]))))')+' "$rc"\\nexit "$rc"\\n')
with (b/'logs'/('study-srun-entry-repair-'+r['commit'][:8]+'.log')).open('ab',buffering=0) as log:
 child=subprocess.Popen(['srun','--jobid=3393','--overlap','--nodes=1','--ntasks=1','--gres=gpu:h100:8','--cpus-per-task=96','--mem=768G','bash',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time=time.time(),pid=child.pid,uid=os.getuid(),tick=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]),
 source=str(source),entry_source_commit=r['commit'],scientific_source_commit=old['source_commit'],manifest=manifest,
 actual_exit_path=str(exitfile),previous_owner=old,old_failure_preserved=str(preserved),scope='Pre-model zero-Adam fresh retry, same immutable run/config, shortened Ray socket path')
ticket.write_text(json.dumps(record,indent=2));(b/'active-study.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
'''.replace('PAYLOAD',repr(encoded))
Path(a.output).write_text(code)
print(json.dumps(dict(commit=commit,request_bytes=len(code.encode()))))
