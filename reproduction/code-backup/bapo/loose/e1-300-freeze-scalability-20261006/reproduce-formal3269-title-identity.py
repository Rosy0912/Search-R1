"""Pure local fake-proc test of exact bound reader; no signal or GPU calls."""
from pathlib import Path
import json,types,copy,tempfile,os,hashlib
B=Path(__file__).resolve().parent;E=B.parent/'evidence';p=E/'freeze-scalability-formal3269-fault-resource-watch.jsonl';rows=[json.loads(x)for x in p.read_text().splitlines()];certs=[x['certificate']for x in rows if x.get('event')=='owned_freeze_signal'and x.get('pid')==319794]
assert len(certs)==9
commands={x['command'][0]for x in certs};assert len(commands)>=2
watch=B/'resource_watch.py';assert hashlib.sha256(watch.read_bytes()).hexdigest()=='e56781fefcd42b3d97f06af824312ac69fd318eb480093a66c4194d1f0fcaf74';m=types.ModuleType('frozenwatch');m.__file__=str(watch);exec(compile(watch.read_bytes(),str(watch),'exec'),m.__dict__)
proof=copy.deepcopy(certs[0]);proof['uid']=os.getuid();alt=next(x['command']for x in certs if x['command'][0]!=proof['command'][0])
def stat(pid,ppid,tick,state):
 f=['0']*20;f[0]=state;f[1]=str(ppid);f[19]=str(tick);return str(pid)+' (fixture) '+' '.join(f)
with tempfile.TemporaryDirectory(prefix='bapo-readonly-title-recheck-')as tmp:
 proc=Path(tmp);child=proc/str(proof['pid']);parent=proc/str(proof['ppid']);child.mkdir();parent.mkdir();(child/'stat').write_text(stat(proof['pid'],proof['ppid'],proof['start_ticks'],'T'));(parent/'stat').write_text(stat(proof['ppid'],1,proof['parent_start_ticks'],'T'))
 env={'BAPO_RUN_ID':proof['run'],'SLURM_JOB_ID':proof['slurm_job_id'],'SEARCH_R1_GPU_LEASES':json.dumps([{'uuid':x}for x in proof['allocated_uuid_set']])};(child/'environ').write_bytes(b'\0'.join((k+'='+v).encode()for k,v in env.items())+b'\0')
 def put(cmd):(child/'cmdline').write_bytes('\0'.join(cmd).encode())
 put(proof['command']);now=m.local_freeze_recheck(proof,{},proc_root=proc);assert now['state']=='T'and all(now[k]==proof[k]for k in m.FREEZE_IDENTITY_KEYS)
 put(alt)
 try:m.local_freeze_recheck(proof,{},proc_root=proc)
 except ValueError as e:assert str(e)=='PID epoch/identity changed before STOP: 319794';failure=str(e)
 else:raise AssertionError('Mutable Ray command should reproduce original exception')
print(json.dumps({'status':'PROVEN_EXACT_READER_MUTABLE_TITLE_COUNTEREXAMPLE_NOT_ACTUAL_FAILURE_DELTA','watch_sha256':hashlib.sha256(watch.read_bytes()).hexdigest(),'actual_pid':319794,'actual_prior_certificates':9,'same_actual_start_ticks':sorted({x['start_ticks']for x in certs}),'actual_titles':sorted(commands),'state_T_same_command_passed':True,'same_PID_epoch_env_UID_parent_UUID_with_different_prior_actual_Ray_title_rejected':True,'exception':failure,'actual_final_changed_field_not_logged':True,'signal_calls':0,'gpu_calls':0}))
