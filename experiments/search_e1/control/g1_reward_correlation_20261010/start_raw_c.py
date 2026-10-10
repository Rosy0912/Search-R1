"""Launch the already deployed, published strict-C reconstruction once."""
import datetime, hashlib, json, pathlib, shlex, subprocess, sys
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control

cfg=json.loads((HERE/'raw-C-run-config.json').read_bytes())
dep=json.loads((HERE/'raw-C-deployment.stdout.json').read_bytes())
assert dep['deployed'] and dep['remote_base']==cfg['remote_base']
p=cfg['remote_base']; base=cfg['base']
args=['env','CUDA_VISIBLE_DEVICES=','USE_TORCH=0','USE_TF=0','USE_FLAX=0',
      'PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
      base+'/env/bin/python','-B',p+'/source/reconstruct_train_raw_c_preflight_v2.py',
      '--base',base,'--diagnostic-base',p+'/raw-C',
      '--expected-run-id',cfg['expected_run_id'],'--run-manifest-sha256',cfg['run_manifest_sha256'],
      '--resolved-config-sha256',cfg['resolved_config_sha256'],
      '--gold-audit',p+'/receipts/gold-audit.json','--gold-audit-sha256',cfg['gold_audit_sha256'],
      '--helper-sha256',cfg['helper_sha256'],'--expected-node',cfg['expected_node'],
      '--start-step',str(cfg['start_step']),'--end-step',str(cfg['end_step'])]
session='g1-full-raw-C-'+cfg['diagnostic_run_id'][:12]
foreground='exec '+shlex.join(args)+' > '+shlex.quote(p+'/logs/raw-C.stdout.json')+' 2> '+shlex.quote(p+'/logs/raw-C.stderr.log')
remote='''import subprocess,json,os,socket,pathlib,hashlib
p=pathlib.Path(P_REMOTE); assert not (p/'raw-C').exists()
assert socket.gethostname()==P_NODE
assert not subprocess.run(['tmux','has-session','-t',P_SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
subprocess.run(['tmux','new-session','-d','-s',P_SESSION,P_FOREGROUND],check=True)
subprocess.run(['tmux','set-window-option','-t',P_SESSION,'remain-on-exit','on'],check=True)
r=subprocess.run(['tmux','list-panes','-t',P_SESSION,'-F','#{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True,check=True)
print(json.dumps(dict(session=P_SESSION,pane=r.stdout.strip(),uid=os.getuid(),node=socket.gethostname(),foreground_sha256=hashlib.sha256(P_FOREGROUND.encode()).hexdigest(),started=True,not_completed=True)))
'''
for tag,value in [('P_REMOTE',p),('P_NODE',cfg['expected_node']),('P_SESSION',session),('P_FOREGROUND',foreground)]:
    remote=remote.replace(tag,repr(value))
r=subprocess.run(control.route('baai-a800',shlex.join(['python3','-c',remote])),capture_output=True,timeout=45)
(HERE/'raw-C-launch.stdout.json').write_bytes(r.stdout); (HERE/'raw-C-launch.stderr.txt').write_bytes(r.stderr)
(HERE/'raw-C-launch.execution.json').write_text(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rc=r.returncode,foreground_sha256=hashlib.sha256(foreground.encode()).hexdigest()),indent=2)+'\n')
assert r.returncode==0, 'launch failed; inspect actual session before retry'
print(r.stdout.decode())
