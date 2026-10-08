# Operational reuse of the published stdlib fileproof/metadata functions.
# Read-only; no Torch, GPU, HTTP, process signals, model reload, or Node writes.
import hashlib,json,pathlib,runpy,sys,time
H=pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py')
assert hashlib.sha256(H.read_bytes()).hexdigest()=='d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b=runpy.run_path(str(H));require=b['require'];fileproof=b['fileproof'];metadata=b['metadata'];stable=b['stable']
S=pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851/zero/runs/bf16_zero')
b['checked_path'](str(S),directory=True)
phase,pf=b['checked_json'](S/'phase-complete.json','09d430b301fdf453cdb9ac9a0e5d371a40640bbb85a1261cddcc2520025bc94f')
terminal,tf=b['checked_json'](S/'actual-terminal.json','086ab8c00c684c2651140594d584b90b37ee2fc388825109ab9a1609e42572e4')
require(phase['accepted'] is True and phase['release']['accepted'] is True and phase['version']=='bf16_zero' and phase['run_id']=='d6fb70ec8f374ff8b8a841b2e6f15450','closed phase identity differs')
require(phase['actual_worker_count']==phase['expected_worker_count']==8 and len(phase['child_exit_evidence'])==8 and all(x['actual_wait_returncode']==0 for x in phase['child_exit_evidence']),'phase has no exact eight actual rc0')
require(terminal['all_actual_wait_zero'] is True and terminal['child_exit_evidence']==phase['child_exit_evidence'],'phase terminal differs')
paths=[]
for p in sorted(S.rglob('*')):
 require(not p.is_symlink(),'symlink refused')
 if p.is_dir():continue
 b['checked_path'](str(p));b['safe_name'](p.relative_to(S).as_posix());paths.append(p)
files=[dict(relative=p.relative_to(S).as_posix(),**fileproof(p)) for p in paths]
for f in files:stable(f)
print(json.dumps(dict(schema='closed-bf16-zero-phase-file-inventory-v1',source=str(S),run_id=phase['run_id'],version='bf16_zero',phase=pf,actual_terminal=tf,actual_worker_wait_codes=[x['actual_wait_returncode'] for x in phase['child_exit_evidence']],parent_controller_may_still_be_running=True,primary_slots=512,strict_hits=239,files=files,bytes=sum(x['bytes'] for x in files),observed_unix=time.time()),sort_keys=True))
