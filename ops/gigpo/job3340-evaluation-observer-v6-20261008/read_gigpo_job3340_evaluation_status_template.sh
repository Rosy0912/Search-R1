/usr/bin/python3 - <<'PY'
import ast,sys,subprocess,shlex,json,datetime
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
module=ast.parse(Path('work/read_gigpo_job3332_actual.py').read_text())
observer=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
observer=observer.replace('3332','3340')
observer=observer.replace('print(json.dumps(out))',"if run:\n    for name in ['reused-user-authorized-baseline.json','actual-vllm-backend-identity.json','actual-bf16-accumulation-immediately-before-first-adam.json']:\n        pp=run/name\n        if pp.is_file():out['metadata'][name]=json.loads(pp.read_text())\n    out['completed_attempt_receipts']=[]\n    for pp in sorted((run/'audit/partial-updates').glob('attempt_*/attempt-result.json'))[-3:]:\n        out['completed_attempt_receipts'].append({'path':str(pp),'record':json.loads(pp.read_text())})\n    out['last_completed_attempt_metrics']=[]\n    import hashlib\n    for pp in sorted(run.glob('calibration-attempt*.json'),key=lambda x:int(x.stem.replace('calibration-attempt','')))[-3:]:\n        raw=pp.read_bytes();rr=json.loads(raw)\n        out['last_completed_attempt_metrics'].append({'path':str(pp),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'attempt_id':rr['attempt_id'],'counted_complete':rr['counted_complete'],'metrics':rr['metrics'],'timings':rr['timings'],'actual_lr_before_update':rr['actual_lr_before_update'],'data_cursor':rr['data_cursor']})\n    out['actual_partial_policy_last_by_rank']=[]\n    for rank in range(8):\n        pp=run/('audit/policy-safety/rank-'+str(rank)+'.jsonl')\n        if not pp.exists():continue\n        with pp.open('rb') as f:\n            f.seek(max(0,pp.stat().st_size-32768));raw=f.read()\n        valid=[]\n        for line in raw.splitlines():\n            try:valid.append(json.loads(line))\n            except (ValueError,UnicodeDecodeError):pass\n        if valid:\n            rr=valid[-1];out['actual_partial_policy_last_by_rank'].append({'rank':rank,'path':str(pp),'phase':rr.get('phase'),'attempt_id':rr.get('attempt_id'),'actual_optimizer_writes':rr.get('actual_optimizer_writes'),'allowed':rr.get('allowed'),'rejection_reason':rr.get('rejection_reason')})\nprint(json.dumps(out))")
observer=observer.replace('print(json.dumps(out))',"""if run:
    for name in ['fresh200-initialization.json','actual-official-zero-initialization.json','resolved-config-C0.json','local-http-retrieval-verification.json','human-nonblocking-diagnostics.json']:
        p=run/name
        if p.is_file():out['metadata'][name]=json.loads(p.read_text())
    originals=[]
    for p in (run/'ray').glob('session_*/logs/worker-*.out'):
        if p.parent.parent.name == 'session_latest':continue
        with p.open('rb') as f:
            head=f.read(2048)
            if b'main_task' not in head:continue
            f.seek(max(0,p.stat().st_size-16000))
            tail=f.read().decode(errors='replace')
        originals.append({'path':str(p),'dev':p.stat().st_dev,'inode':p.stat().st_ino,'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'tail':tail})
    out['fixed_original_main_worker_logs']=originals
    out['actual_xformers_original_worker_logs']=[]
    for pp in (run/'ray').glob('session_*/logs/worker-*.out'):
        if pp.parent.parent.name == 'session_latest':continue
        with pp.open('rb') as ff:
            ff.seek(max(0,pp.stat().st_size-131072));raw=ff.read().decode(errors='replace')
        lines=[line for line in raw.splitlines() if 'Using XFormers backend' in line or 'CUDA error:' in line or 'illegal memory access' in line]
        if lines:out['actual_xformers_original_worker_logs'].append({'path':str(pp),'dev':pp.stat().st_dev,'inode':pp.stat().st_ino,'bytes':pp.stat().st_size,'lines':lines[-12:]})

    if len(originals)==1:out['logs']['original-main_task']=originals[0]['tail']
print(json.dumps(out))""")
observer=observer.replace('print(json.dumps(out))',"if run:\n    out['first5_checkpoint_observation']={}\n    for name in ['first5-human-nonblocking-diagnostics.json','human-nonblocking-step5-state-roundtrip.json','instruct5-ready.json']:\n        pp=run/name\n        if pp.is_file():\n            raw=pp.read_bytes()\n            out['first5_checkpoint_observation'][name]={'path':str(pp),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'record':json.loads(raw)}\n    root=run/'checkpoints/actor/global_step_5'\n    out['first5_checkpoint_files']=[{'path':str(pp),'bytes':pp.stat().st_size} for pp in sorted(root.rglob('*')) if pp.is_file()] if root.is_dir() else []\nprint(json.dumps(out))")
observer=observer.replace('print(json.dumps(out))',"if run:\n    pp=run/'audit/evaluations.jsonl'\n    out['actual_evaluation_records']=[]\n    if pp.is_file():\n        with pp.open('rb') as ff:\n            ff.seek(max(0,pp.stat().st_size-65536));raw=ff.read()\n        for line in raw.splitlines():\n            try:record=json.loads(line)\n            except (ValueError,UnicodeDecodeError):continue\n            out['actual_evaluation_records'].append(record)\n        out['actual_evaluation_records']=out['actual_evaluation_records'][-8:]\nprint(json.dumps(out))")
worker_cmd='env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c '+shlex.quote(observer)
mgmt=r'''import json,shlex,subprocess,datetime
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-8';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
args=prefix.copy()
if ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(keys)])]
r=subprocess.run(args+[user+'@'+ip,WORKER_CMD],capture_output=True,text=True,timeout=65)
print(json.dumps({'route':'retained18 SSH-W h100-8','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
'''.replace('WORKER_CMD',repr(worker_cmd))
r=subprocess.run(['env','-u','LD_LIBRARY_PATH','/usr/bin/ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host']+['env -u LD_LIBRARY_PATH /usr/bin/python3 -c '+shlex.quote(mgmt)],capture_output=True,text=True,timeout=85)
p=Path('outputs/gigpo-e1-instruct200/RETAINED18-JOB3340-ACTUAL-STATUS-OBSERVATION_TAG.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':True},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
o=json.loads(m['stdout'])
import re
m=o.get('metadata',{});baseline=m.get('baseline-evaluation.json',{}).get('full512',{})
print(json.dumps({'utc':o['utc'],'hostname':o['hostname'],'job_state':next((w.split('=',1)[1] for w in o['scheduler']['stdout'].split() if w.startswith('JobState=')),None) if o['scheduler']['returncode']==0 else o['scheduler'],'run':o['run'],'original':{k:v for k,v in o.get('actual_original_eval_summary',{}).items() if k!='metric_lines'},'metadata_present':list(m),'probability_json_count':len(o.get('actual_probability_files',[])),'baseline_f1':baseline.get('val/f1/all'),'baseline_em':baseline.get('val/em/all'),'live_progress':{k:(m.get('live-progress.json') or {}).get(k) for k in ['completed_outer_updates','actual_adam_total','config_epoch','config_sha256']},'partial_by_rank':o.get('actual_partial_policy_last_by_rank',[]),'probability_ready_status':m.get('probability-identity-ready.json',{}).get('status'),'probability_admitted_status':m.get('probability-identity-admitted.json',{}).get('status'),'instruct5_ready_status':m.get('instruct5-ready.json',{}).get('status'),'run_result_status':(m.get('run-result.json') or {}).get('status'),'driver_result':m.get('driver-result.json')},ensure_ascii=True))
print(o.get('logs',{}).get('original-main_task','')[-300:])
PY
