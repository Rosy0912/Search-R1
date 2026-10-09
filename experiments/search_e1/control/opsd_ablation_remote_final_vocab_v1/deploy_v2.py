"""Publish only CPU source and small captured receipts to a fresh HKUST directory."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE = '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008'
SOURCES = {
    'prepare_inputs.py': 'opsd_ablation_remote_final_vocab_v1/prepare_inputs.py',
    'close_remote.py': 'opsd_ablation_remote_final_vocab_v1/close_remote.py',
    'precision_history.py': 'opsd_ablation_recorded_precision_history_remote_v2/derive.py',
    'collect.py': 'opsd_ablation_final_vocab_collection_v1/collect.py',
    'close_ledger.py': 'opsd_ablation_final_vocab_ledger_v1/close_ledger.py',
    'aggregate_clipmass.py': 'opsd_ablation_recorded_clipmass_v1/aggregate_clipmass.py',
}
FROZEN = {
    'collect.py': '4dac0a6d607e4e62e31018ae63c85e7e9c8285cd238657c5edc60d66e8a7bff0',
    'close_ledger.py': '0123b06b13c6abb0554d0a825bb1a28565f192e0b046ef878feda4af35f8a2a0',
    'aggregate_clipmass.py': 'b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a',
}
RECEIPTS = {
    'six-deployment.json': 'remote-raw-six-deployment-v2.json',
    'six-launch.json': 'remote-raw-six-launch-v2.json',
    'six-wait.json': 'remote-raw-six-execution-v2.json',
    'J-launch.json': 'remote-J-copy-launch.json',
    'J-wait.json': 'remote-J-copy-actual-wait.json',
}


def main():
    p = argparse.ArgumentParser()
    for name in ('repo', 'evidence', 'remote-root', 'receipt'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args(); repo, evidence = Path(a.repo), Path(a.evidence)
    remote = Path(a.remote_root)
    if not remote.is_absolute() or Path(BASE) / 'analysis' not in remote.parents or '..' in remote.parts:
        raise ValueError('New explicit original HKUST study directory required')
    files = {}
    for name, rel in SOURCES.items():
        rel = 'experiments/search_e1/control/' + rel
        raw = (repo / rel).read_bytes()
        if raw != subprocess.check_output(['git', '-C', str(repo), 'show', 'FETCH_HEAD:' + rel]):
            raise ValueError('Current source differs from actually fetched publication: ' + rel)
        if name in FROZEN and hashlib.sha256(raw).hexdigest() != FROZEN[name]:
            raise ValueError('Frozen full-seven source pin differs before deployment: ' + name)
        files['source/' + name] = raw
    for name, rel in RECEIPTS.items():
        raw = (evidence / rel).read_bytes(); json.loads(raw)
        files['evidence/' + name] = raw
    packet = dict(root=str(remote), files={k:dict(data=base64.b64encode(v).decode(),
        sha256=hashlib.sha256(v).hexdigest(), bytes=len(v)) for k,v in files.items()})
    # Only new independent control files are written; no original output/source is changed.
    code = ('import json,base64,hashlib\nfrom pathlib import Path\np=' + repr(packet) + '\n'
        'root=Path(p["root"]); assert not root.exists(); root.mkdir(parents=True)\nrefs={}\n'
        'for name,f in p["files"].items():\n'
        ' raw=base64.b64decode(f["data"]); assert hashlib.sha256(raw).hexdigest()==f["sha256"]\n'
        ' q=root/name; q.parent.mkdir(exist_ok=True); q.write_bytes(raw); assert q.read_bytes()==raw\n'
        ' refs[name]={"path":str(q),"sha256":f["sha256"],"bytes":len(raw)}\n'
        'print(json.dumps({"accepted":True,"root":str(root),"files":refs,"no_GPU":True}))\n')
    argv = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=10', '-oServerAliveInterval=5',
            '-oServerAliveCountMax=2', 'hkust1', 'python3 -']
    started = time.time(); r = subprocess.run(argv, input=code.encode(), capture_output=True)
    receipt = dict(actual_ssh_returncode=r.returncode, argv=argv, started_unix=started,
        ended_unix=time.time(), source_payload_only=True, root=str(remote),
        source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','FETCH_HEAD'], text=True).strip(),
        stderr=r.stderr.decode(errors='replace'), files={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in files.items()})
    if r.returncode == 0:
        receipt['actual_deployment'] = json.loads(r.stdout)
    Path(a.receipt).write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(actual_ssh_returncode=r.returncode, accepted=r.returncode==0,
                         receipt=a.receipt, original_tasks_changed=False)))
    return r.returncode


if __name__ == '__main__':
    raise SystemExit(main())
