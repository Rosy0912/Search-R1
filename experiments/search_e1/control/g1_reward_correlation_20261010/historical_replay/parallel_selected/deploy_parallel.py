"""Root-invoked CPU-only deployment of a published selected-replay cohort.

Transfers only code/config/compact proof. No GPU/model/training execution.
"""
import base64,datetime,hashlib,json,pathlib,shlex,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
TARGET='experiments/search_e1/control/g1_reward_correlation_20261010/historical_replay/parallel_selected/'

def main():
    sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
    cohort=json.loads((HERE/'cohort-config.json').read_bytes())
    pub=json.loads((HERE/'publication-verified.json').read_bytes());byname={r['path']:r['sha256'] for r in pub['files']}
    manifest=json.loads((HERE/'source-manifest.json').read_bytes());source=cohort['remote_source']
    entries=[]
    def add(local,relative):
        raw=pathlib.Path(local).read_bytes();sha=hashlib.sha256(raw).hexdigest()
        assert byname[TARGET+relative]==sha,'local bytes not published and remotely verified'
        entries.append(dict(path=source+'/'+relative,sha256=sha,data=base64.b64encode(raw).decode()))
    for name in list(manifest['source_sha256'])+['source-manifest.json','cohort-config.json']:
        add(HERE/name,name)
    configs=[]
    for entry in cohort['entries']:
        p=pathlib.Path(entry['local_config']);c=json.loads(p.read_bytes())
        assert c['gpu_index']==entry['gpu'] and c['selected_steps']==entry['steps'] and c['cohort_id']==cohort['cohort_id']
        add(p,'configs/gpu'+str(entry['gpu'])+'.json');configs.append(c)
    add(HERE.parent/'original-memory-peaks.json','original-memory-peaks.json')
    add(HERE.parent/'probe-numeric-env-live-whitelist.json','numeric-env-proof.json')
    remote=r'''
import base64,datetime,hashlib,json,os,pathlib,socket,subprocess,sys
j=json.load(sys.stdin);co=j['cohort'];p=pathlib.Path(co['remote_source']);parent=pathlib.Path(co['remote_parent'])
assert sys.platform.startswith('linux') and socket.gethostname()==co['expected_node']
assert p.resolve()==p and parent.resolve()==parent and p.parent.is_dir() and not p.exists() and not parent.exists()
assert str(p).startswith('/data/shaoshitong/Search-G1-reward-reconstruction-') and p.parent.stat().st_uid==os.getuid()
p.mkdir(mode=0o700);parent.mkdir(mode=0o700);(parent/'logs').mkdir();(p/'configs').mkdir();(p/'fixture-tmp').mkdir()
rows=[]
for item in j['files']:
    q=pathlib.Path(item['path']);assert q.is_relative_to(p) and q.resolve()==q and not q.exists()
    raw=base64.b64decode(item['data']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    with q.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    assert hashlib.sha256(q.read_bytes()).hexdigest()==item['sha256']
    rows.append(dict(path=str(q),sha256=item['sha256'],bytes=len(raw)))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',G1_REPLAY_FIXTURE_REPO=co['base']+'/repo',TMPDIR=str(p/'fixture-tmp'))
r=subprocess.run([co['base']+'/env/bin/python','-B','-m','unittest','discover','-s',str(p),'-p','test_*.py'],env=env,capture_output=True,timeout=60)
(p/'fixture.stdout.txt').write_bytes(r.stdout);(p/'fixture.stderr.txt').write_bytes(r.stderr)
print(json.dumps(dict(source=str(p),remote_parent=str(parent),observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=socket.gethostname(),uid=os.getuid(),files=rows,CPU_fixture_rc=r.returncode,CPU_fixture_summary=r.stderr.decode()[-1200:],GPU_started=False)))
'''
    r=subprocess.run(control.route('baai-a800',shlex.join(['python3','-c',remote])),input=json.dumps(dict(cohort=cohort,files=entries)).encode(),capture_output=True,timeout=100)
    (HERE/'deployment.stdout.json').write_bytes(r.stdout);(HERE/'deployment.stderr.txt').write_bytes(r.stderr)
    (HERE/'deployment.execution.json').write_text(json.dumps(dict(rc=r.returncode,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=pub['source_commit']),indent=2)+'\n')
    assert r.returncode==0,'preserve failed new deployment; no overwrite/retry'
    result=json.loads(r.stdout);assert result['CPU_fixture_rc']==0,'preserve Node fixture failure; no GPU launch'
    print(json.dumps(result))

if __name__=='__main__':main()
