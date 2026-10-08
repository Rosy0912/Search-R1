"""Own native CPU historical-log replay; no production source, GPU or receipt."""
import base64,hashlib,json,pathlib,shlex,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
D=HERE/'strict-ray-marker-fix';gate=D/'checkpoint_gate_candidate.py';replay=D/'native_marker_replay.py'
GATE_SHA='74c3c13929c4f5cb784b90b7fd568ee80dfe1d37c8efaa28f97aae60b3492632'
REPLAY_SHA='ebd7ac5869c8deb71695f22862a2e1aea754dd424f6dbe5e7ba172836ef61445'
assert hashlib.sha256(gate.read_bytes()).hexdigest()==GATE_SHA and hashlib.sha256(replay.read_bytes()).hexdigest()==REPLAY_SHA
log=HERE/'formal3198-first5-observed-20261004T134348Z/run/train.log';LOG_SHA=hashlib.sha256(log.read_bytes()).hexdigest()
payload={p.name:{'bytes_b64':base64.b64encode(p.read_bytes()).decode(),'sha256':pin} for p,pin in [(gate,GATE_SHA),(replay,REPLAY_SHA)]}
remote='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004';own=remote+'/tools/strict-marker-74c3-independent-cpu'
code='''import pathlib,json,base64,hashlib,os,subprocess,socket
assert os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
data=json.loads(DATAJSON);directory=pathlib.Path(OWNPATH);directory.mkdir(exist_ok=False)
for name,record in data.items():
 assert pathlib.Path(name).name==name;raw=base64.b64decode(record['bytes_b64']);assert hashlib.sha256(raw).hexdigest()==record['sha256']
 with (directory/name).open('xb') as f:f.write(raw)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
command=COMMANDLIST
result=subprocess.run(command,env=env,capture_output=True)
for name,raw in [('stdout.txt',result.stdout),('stderr.txt',result.stderr),('exit.txt',(str(result.returncode)+'\\n').encode())]:
 with (directory/name).open('xb') as f:f.write(raw)
print(json.dumps({'scope':'OWN_NATIVE_CPU_HISTORICAL_MARKER_REPLAY_NOT_PRODUCTION_ACCEPTANCE','node':socket.gethostname(),'uid':os.getuid(),'wrapper_pid':os.getpid(),'returncode':result.returncode,'own_directory':str(directory),'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()},'stdout_b64':base64.b64encode(result.stdout).decode(),'stderr_b64':base64.b64encode(result.stderr).decode()},sort_keys=True))
'''.replace('DATAJSON',repr(json.dumps(payload))).replace('OWNPATH',repr(own)).replace('COMMANDLIST',repr(['/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python',own+'/native_marker_replay.py','--repo',remote+'/repo-89beac273f78','--candidate',own+'/checkpoint_gate_candidate.py','--candidate-sha256',GATE_SHA,'--log',remote+'/runs/formal-3198/train.log','--log-sha256',LOG_SHA,'--output',own+'/native-replay.json']))
result=subprocess.run([sys.executable,str(HERE.parent/'remote.py'),'python3 -c '+shlex.quote(code)],capture_output=True,timeout=80)
(HERE/'strict-marker-native-independent-original.txt').write_bytes(result.stdout);(HERE/'strict-marker-native-independent-stderr.txt').write_bytes(result.stderr)
print('remote wrapper exit',result.returncode)
if result.returncode:sys.exit(result.returncode)
v=json.loads(result.stdout);print('native replay exit',v['returncode']);print(base64.b64decode(v['stdout_b64']).decode());print(base64.b64decode(v['stderr_b64']).decode())
if v['returncode']:sys.exit(v['returncode'])
