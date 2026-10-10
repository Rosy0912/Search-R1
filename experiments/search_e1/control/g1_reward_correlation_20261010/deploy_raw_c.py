"""Deploy published CPU-only code and bounded audits to a fresh remote directory.

No raw trace, model or per-trajectory output is transferred to Mac.  Source bytes
must already have been pushed and independently read back from FETCH_HEAD.
"""
import base64, datetime, hashlib, json, pathlib, shlex, subprocess, sys


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

REMOTE = r'''
import base64, hashlib, json, os, pathlib, socket, sys
j=json.load(sys.stdin); c=j['config']; p=pathlib.Path(c['remote_base'])
assert sys.platform.startswith('linux') and socket.gethostname()==c['expected_node']
assert not p.exists() and p.parent.is_dir() and p.resolve()==p
assert str(p).startswith('/data/shaoshitong/Search-G1-reward-reconstruction-20261010-')
assert p.parent.stat().st_uid==os.getuid()
p.mkdir(mode=0o700); (p/'source').mkdir(); (p/'logs').mkdir(); (p/'receipts').mkdir()
rows=[]
for row in j['files']:
    rel=pathlib.PurePosixPath(row['relative'])
    assert not rel.is_absolute() and '..' not in rel.parts
    raw=base64.b64decode(row['data']); assert hashlib.sha256(raw).hexdigest()==row['sha256']
    q=p/str(rel); q.parent.mkdir(parents=True,exist_ok=True)
    with q.open('xb') as f: f.write(raw)
    assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
    rows.append(dict(path=str(q),sha256=row['sha256'],bytes=len(raw)))
print(json.dumps(dict(remote_base=str(p),files=rows,node=socket.gethostname(),uid=os.getuid(),deployed=True,GPU_started=False)))
'''


def main():
    cfg_raw=(HERE/'raw-C-run-config.json').read_bytes(); cfg=json.loads(cfg_raw)
    pub=json.loads((HERE/'raw-C-source-publication-verified.json').read_bytes())
    assert pub['checkout_preserved'] and pub['no_force_push']
    published={r['path']:r['sha256'] for r in pub['files']}
    items=[('reconstruct_train_raw_c_preflight_v2.py','source/reconstruct_train_raw_c_preflight_v2.py'),
           ('test_reconstruct_train_raw_c_preflight_v2.py','source/test_reconstruct_train_raw_c_preflight_v2.py'),
           ('raw-C-run-config.json','receipts/config.json')]
    payload=[]
    for name,rel in items:
        raw=(HERE/name).read_bytes(); sha=hashlib.sha256(raw).hexdigest()
        target='experiments/search_e1/control/g1_reward_correlation_20261010/'+name
        assert published[target]==sha
        payload.append(dict(relative=rel,sha256=sha,data=base64.b64encode(raw).decode()))
    audit=(HERE.parent/'g1-probe-evolution-20261010/historical-train-input-audit-v2-default.json').read_bytes()
    assert hashlib.sha256(audit).hexdigest()==cfg['gold_audit_sha256']
    payload.append(dict(relative='receipts/gold-audit.json',sha256=cfg['gold_audit_sha256'],data=base64.b64encode(audit).decode()))
    publication=(HERE/'raw-C-source-publication-verified.json').read_bytes()
    payload.append(dict(relative='receipts/publication.json',sha256=hashlib.sha256(publication).hexdigest(),data=base64.b64encode(publication).decode()))
    command=shlex.join(['python3','-c',REMOTE])
    r=subprocess.run(control.route('baai-a800',command),input=json.dumps(dict(config=cfg,files=payload)).encode(),capture_output=True,timeout=60)
    (HERE/'raw-C-deployment.stdout.json').write_bytes(r.stdout)
    (HERE/'raw-C-deployment.stderr.txt').write_bytes(r.stderr)
    proof=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rc=r.returncode,stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),source_commit=pub['source_commit'])
    (HERE/'raw-C-deployment.execution.json').write_text(json.dumps(proof,indent=2)+'\n')
    assert r.returncode==0, 'deployment failed; preserve partial remote directory'
    d=json.loads(r.stdout); assert d['deployed'] and d['remote_base']==cfg['remote_base']
    print(json.dumps(d))


if __name__=='__main__': main()
