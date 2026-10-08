from pathlib import Path
import hashlib,json,tarfile,shutil
W=Path('work');D=W/'standard-identity-v3-native-cpu';D.mkdir(exist_ok=False)
names=['qwen25-32b-trained-eval64-candidate-v2/eval64.py','qwen25-32b-terminal-barrier-candidate-v2/eval64.py','topd-32b-pipeline-ipc-candidate-v2/topd32b_shared_pipeline/backend.py','topd-32b-pipeline-ipc-candidate-v2/topd32b_shared_pipeline/protocol.py','qwen25-32b-window-safety-candidate-v4/startup_guard.py','qwen25-32b-window-safety-candidate-v4/runtime_safety.py','qwen25-32b-terminal-fix-refreeze/eval64.py','topd-32b-native-score-caller-v1-readonly-review/caller_contracts.py']
for n in names:
 p=D/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(W/n,p)
(D/'native-cpu.sbatch').write_text('''#!/bin/bash
#SBATCH --job-name=standard-r1-teacher-identity-v3-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/standard-r1-teacher-identity-v3-cpu-%j.log
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
PACKAGE="$TASK_ROOT/tools/standard-search-r1-teacher-identity-v3"
RUN="$TASK_ROOT/logs/standard-r1-teacher-identity-v3-cpu-$SLURM_JOB_ID"
mkdir "$RUN"
export AGENT_ROOT="$TASK_ROOT" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$TASK_ROOT/test-tools:$TASK_ROOT/repo-shared32b"
cd "$PACKAGE"
"$TASK_ROOT/env/bin/python" "$PACKAGE/native_cpu_runner.py" "$PACKAGE" "$RUN"
''')
(D/'native_cpu_runner.py').write_text('''"""CPU contracts only; no model load/GPU lifecycle or submission."""
import sys,pathlib,subprocess,json,hashlib,os,re
P,R=map(pathlib.Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for manifest in ('standard-search-r1-teacher-identity-v3-files.json','native-files.json'):
 for name,v in json.loads((P/manifest).read_text())['files'].items():
  p=P/name
  assert p.is_file() and not p.is_symlink() and p.stat().st_size==v['size'] and sha(p)==v['sha256'],name
import torch
assert not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']==''
print('NATIVE_CPU_CONTRACTS_NO_MODEL_NO_GPU',sys.version,torch.__version__,flush=True)
suites={
'chain':(34,['test_standard_teacher_identity_v3_chain.py','qwen25-32b-trained-eval64-r1-v3/test_training_identity.py','qwen25-32b-trained-eval64-r1-v3/test_trained_assets.py']),
'score':(49,['topd-32b-native-score-gate-r1-v3/test_score_gate.py','topd-32b-native-score-gate-r1-v3/test_caller_contracts.py']),
'owner':(26,['topd-32b-shared-owner-r1-v3/test_owner.py','topd-32b-shared-owner-r1-v3/test_context_policy.py']),
'ipc':(18,['topd-32b-pipeline-ipc-r1-v3/test_backend.py']),
'supervisor':(62,['topd-shared32b-supervisor-r1-v3/test_contracts.py'])}
rows=[]
for name,(count,paths) in suites.items():
 log=R/(name+'.log')
 with log.open('x') as f:
  r=subprocess.run([sys.executable,'-m','pytest','-q',*[str(P/p) for p in paths],'--basetemp',str(R/(name+'-tmp')),'-o','cache_dir='+str(R/(name+'-cache'))],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  f.flush();os.fsync(f.fileno())
 text=log.read_text();print(name,r.returncode,text,flush=True)
 good=r.returncode==0 and len(re.findall(r'(?m)^'+str(count)+r' passed(?:,.*)? in ',text))==1 and not re.search(r'\\b[1-9][0-9]* skipped\\b',text)
 rows.append(dict(suite=name,passed=count if good else None,returncode=r.returncode,path=str(log),sha256=sha(log),verified=good))
 if not good:break
report=dict(scope='NATIVE_SOURCE_CONTRACT_ONLY_NO_MODEL_NO_GPU_NOT_TEACHER_APPROVAL',status='PASS' if len(rows)==5 and all(x['verified'] for x in rows) else 'FAIL',suites=rows,passed=189 if len(rows)==5 and all(x['verified'] for x in rows) else None,parent_cuda_initialized=torch.cuda.is_initialized(),source_manifest_sha256=sha(P/'standard-search-r1-teacher-identity-v3-files.json'),entry_manifest_sha256=sha(P/'native-files.json'))
with (R/'result.json').open('x') as f:json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
assert report['status']=='PASS',report
''')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={str(p.relative_to(D)):{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file()}
(D/'native-files.json').write_text(json.dumps({'scope':'SOURCE_DEPENDENCY_SNAPSHOTS_AND_CPU_ENTRY_ONLY','files':files},indent=2)+'\n')
archive=W/'standard-identity-v3-native-cpu.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
 for p in sorted(D.rglob('*')):
  if p.is_file():tf.add(p,arcname=str(p.relative_to(D)),recursive=False)
with tarfile.open(archive) as tf:
 for m in tf.getmembers():assert m.isfile() and tf.extractfile(m).read()==(D/m.name).read_bytes()
print('ARCHIVE',sha(archive),'MANIFEST',sha(D/'native-files.json'))
