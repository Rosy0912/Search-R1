"""Deploy published independent R sources; run CPU tests only on the original Node."""
import base64,datetime,hashlib,json,pathlib,shlex,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
config_raw=(HERE/'run-config.json').read_bytes(); c=json.loads(config_raw)
pub=json.loads((HERE/'publication-verified.json').read_bytes())
byname={r['path']:r['sha256'] for r in pub['files']}
manifest=json.loads((HERE/'source-manifest.json').read_bytes())
parent=pathlib.Path(c['output']).parent
source=parent/('R-source-'+c['diagnostic_run_id'][:12])
files=[]
for name in list(manifest['source_sha256'])+['source-manifest.json','run-config.json']:
    raw=(HERE/name).read_bytes(); sha=hashlib.sha256(raw).hexdigest()
    assert byname['experiments/search_e1/control/g1_reward_correlation_20261010/historical_replay/'+name]==sha
    files.append(dict(path=str(source/name),sha256=sha,data=base64.b64encode(raw).decode()))
for local,remote in [('preflight-actual.stdout.json',c['encoder_preflight_path']),('reward-parameter-live-whitelist.json',c['reward_parameter_proof_path'])]:
    raw=(HERE/local).read_bytes(); sha=hashlib.sha256(raw).hexdigest()
    files.append(dict(path=remote,sha256=sha,data=base64.b64encode(raw).decode()))
raw=(HERE/'probe-numeric-env-live-whitelist.json').read_bytes()
files.append(dict(path=str(source/'numeric-env-proof.json'),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))
remote=r'''
import base64,hashlib,json,os,pathlib,socket,subprocess,sys,datetime
j=json.load(sys.stdin);c=j['config'];p=pathlib.Path(j['source'])
assert sys.platform.startswith('linux') and socket.gethostname()==c['expected_node'] and p.parent.is_dir()
assert p.resolve()==p and str(p).startswith('/data/shaoshitong/Search-G1-reward-reconstruction-20261010-') and not p.exists()
assert p.parent.stat().st_uid==os.getuid();p.mkdir(mode=0o700);(p/'fixture-tmp').mkdir()
rows=[]
for item in j['files']:
    q=pathlib.Path(item['path']);assert q.is_relative_to(p.parent) and q.resolve()==q and not q.exists()
    raw=base64.b64decode(item['data']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    with q.open('xb') as f:f.write(raw)
    assert hashlib.sha256(q.read_bytes()).hexdigest()==item['sha256']
    rows.append(dict(path=str(q),sha256=item['sha256'],bytes=len(raw)))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',G1_REPLAY_FIXTURE_REPO=c['base']+'/repo',TMPDIR=str(p/'fixture-tmp'))
r=subprocess.run([c['base']+'/env/bin/python','-B','-m','unittest','discover','-s',str(p),'-p','test_*.py'],env=env,capture_output=True,timeout=60)
(p/'fixture.stdout.txt').write_bytes(r.stdout);(p/'fixture.stderr.txt').write_bytes(r.stderr)
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(p),deployed=True,files=rows,node=socket.gethostname(),uid=os.getuid(),CPU_fixture_rc=r.returncode,CPU_fixture_summary=r.stderr.decode()[-1000:],GPU_started=False)))
'''
r=subprocess.run(control.route('baai-a800',shlex.join(['python3','-c',remote])),input=json.dumps(dict(config=c,source=str(source),files=files)).encode(),capture_output=True,timeout=100)
(HERE/'deployment.stdout.json').write_bytes(r.stdout);(HERE/'deployment.stderr.txt').write_bytes(r.stderr)
(HERE/'deployment.execution.json').write_text(json.dumps(dict(rc=r.returncode,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=pub['source_commit']),indent=2)+'\n')
assert r.returncode==0, 'preserve failed deployment; do not retry over its directory'
d=json.loads(r.stdout);assert d['CPU_fixture_rc']==0,'CPU fixtures failed; preserve logs'
print(json.dumps(d))
