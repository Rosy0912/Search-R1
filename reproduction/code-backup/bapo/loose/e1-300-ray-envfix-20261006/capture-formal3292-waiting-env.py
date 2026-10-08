import pathlib,subprocess,shlex,json
B=pathlib.Path(__file__).resolve().parent.parent
code="""import pathlib,os,base64,hashlib,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');run=str(root/'runs/bapo-instruct300-formal300-3292');source='a2568d0df306ead6e059760e28b5bd53ee3f6876';keys={b'SPT_NOENV',b'BAPO_RUN_ID',b'BAPO_E1_RUN',b'SLURM_JOB_ID',b'SOURCE',b'CUDA_VISIBLE_DEVICES'}
def snap(pid):
 p=pathlib.Path('/proc')/str(pid);stat=(p/'stat').read_bytes();f=stat[stat.rfind(b')')+2:].split();cmd=(p/'cmdline').read_bytes();env=(p/'environ').read_bytes();selected=[v for v in env.split(b'\\0')if v.split(b'=',1)[0]in keys];assert len(selected)==len({v.split(b'=',1)[0]for v in selected});e=p/'exe';st=e.stat();uid=p.stat().st_uid
 return dict(pid=pid,ppid=int(f[1]),start_ticks=int(f[19]),uid=uid,command_base64=base64.b64encode(cmd).decode(),command=[v.decode()for v in cmd.split(b'\\0')if v],selected_environment_base64=base64.b64encode(b'\\0'.join(selected)+b'\\0').decode(),environment={v.split(b'=',1)[0].decode():v.split(b'=',1)[1].decode()for v in selected},executable=dict(resolved_path=str(e.resolve()),device=st.st_dev,inode=st.st_ino))
found=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  child=snap(int(p.name))
  if child['command']!=['sleep','5']or child['environment'].get('BAPO_RUN_ID')!=run:continue
  assert child['environment']==dict(SPT_NOENV='1',BAPO_RUN_ID=run,BAPO_E1_RUN=run,SLURM_JOB_ID='3292',SOURCE=source,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'),child['environment']
  parent=snap(child['ppid']);assert parent['command']==['bash',str(root/'tools/stage.sh'),'formal300']or parent['command']==['/bin/bash',str(root/'tools/stage.sh'),'formal300']or parent['command']==['/usr/bin/bash',str(root/'tools/stage.sh'),'formal300'],parent['command'];assert child['uid']==parent['uid']==0
  assert snap(child['pid'])==child and snap(parent['pid'])==parent;found.append(dict(child_before=child,child_after=snap(child['pid']),parent_before=parent,parent_after=snap(parent['pid'])))
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
assert len(found)==1,found
assert (pathlib.Path(run)/'WAITING_FOR_INDEPENDENT_RECOVERY_PERMISSION').is_file()and not(pathlib.Path(run)/'recovery-permission.json').exists()
stage=root/'tools/stage.sh';print(json.dumps(dict(status='ACTUAL_READ_ONLY_EXISTING_WAIT_SLEEP_CHILD_ENVIRONMENT_NOT_PERMISSION',job_id=3292,source_commit=source,root=str(root),run_root=run,stage_path=str(stage),stage_sha256=hashlib.sha256(stage.read_bytes()).hexdigest(),proofs=found,processes_spawned=0,signals_sent=0,gpu_executed=False)))
"""
cmd='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/env/bin/python -B -c '+shlex.quote(code)
r=subprocess.run(['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B',str(B/'h100_exec.py'),'--socks','--timeout','60',cmd],capture_output=True,timeout=75)
(B/'evidence/ray-envfix-formal3292-existing-wait-child-env.json').write_bytes(r.stdout);(B/'evidence/ray-envfix-formal3292-existing-wait-child-env.stderr').write_bytes(r.stderr);print(json.dumps(dict(ssh_exit_code=r.returncode,bytes=len(r.stdout),stderr=r.stderr.decode())));assert r.returncode==0 and not r.stderr
