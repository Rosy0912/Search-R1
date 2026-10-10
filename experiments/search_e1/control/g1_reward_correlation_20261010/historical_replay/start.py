"""Start only the independently published R replay, in its own session/GPU5."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
c=json.loads((HERE/'run-config.json').read_bytes());d=json.loads((HERE/'deployment.stdout.json').read_bytes())
assert d['CPU_fixture_rc']==0 and d['GPU_started'] is False
source=d['source'];parent=str(pathlib.Path(c['output']).parent);rid=c['diagnostic_run_id'][:12]
config_sha=hashlib.sha256((HERE/'run-config.json').read_bytes()).hexdigest();manifest_sha=hashlib.sha256((HERE/'source-manifest.json').read_bytes()).hexdigest()
numeric=json.loads((HERE/'probe-numeric-env-live-whitelist.json').read_bytes());assert numeric['rc']==0
numeric=json.loads(numeric['stdout'])['numeric_environment_whitelist'];assert numeric['OMP_NUM_THREADS']=='8'
unset=['MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','PYTHONHASHSEED']
assert all(numeric[k] is None for k in unset)
args=['env']
for key in unset+['USE_TORCH','USE_TF','USE_FLAX']:args+=['-u',key]
args+=['CUDA_VISIBLE_DEVICES=5','OMP_NUM_THREADS=8','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
       'setsid','--wait',c['base']+'/env/bin/python','-B',source+'/replay_gpu.py','--config',source+'/run-config.json','--config-sha256',config_sha,
       '--source-manifest',source+'/source-manifest.json','--source-manifest-sha256',manifest_sha]
foreground='tmux set-window-option -t "$TMUX_PANE" remain-on-exit on && exec '+shlex.join(args)+' > '+shlex.quote(parent+'/logs/R-'+rid+'.stdout.jsonl')+' 2> '+shlex.quote(parent+'/logs/R-'+rid+'.stderr.log')
session='g1-full-pair-R-'+rid
remote=r'''
import datetime,hashlib,json,os,pathlib,socket,subprocess
assert socket.gethostname()==P_NODE and not pathlib.Path(P_OUT).exists()
assert subprocess.run(['tmux','has-session','-t',P_SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
subprocess.run(['tmux','new-session','-d','-s',P_SESSION,P_FOREGROUND],check=True)
subprocess.run(['tmux','set-window-option','-t',P_SESSION,'remain-on-exit','on'],check=True)
r=subprocess.run(['tmux','list-panes','-t',P_SESSION,'-F','#{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True,check=True)
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),session=P_SESSION,pane=r.stdout.strip(),node=socket.gethostname(),uid=os.getuid(),foreground_sha256=hashlib.sha256(P_FOREGROUND.encode()).hexdigest(),started=True,scope_accepted=False)))
'''
for tag,val in [('P_NODE',c['expected_node']),('P_OUT',c['output']),('P_SESSION',session),('P_FOREGROUND',foreground)]:remote=remote.replace(tag,repr(val))
r=subprocess.run(control.route('baai-a800',shlex.join(['python3','-c',remote])),capture_output=True,timeout=45)
(HERE/'launch.stdout.json').write_bytes(r.stdout);(HERE/'launch.stderr.txt').write_bytes(r.stderr)
(HERE/'launch.execution.json').write_text(json.dumps(dict(rc=r.returncode,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),foreground_sha256=hashlib.sha256(foreground.encode()).hexdigest()),indent=2)+'\n')
assert r.returncode==0,'inspect actual session; do not double-launch'
print(r.stdout.decode())
