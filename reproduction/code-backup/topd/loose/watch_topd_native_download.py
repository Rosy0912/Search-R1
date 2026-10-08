"""Mirror the single authorized native download; never submit or restart jobs."""
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'outputs'
lock = (OUT / 'qwen25-32b-native-download-watch.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
remote = """python3 - <<'INNER'
import pathlib,json,hashlib,base64,time,subprocess
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
p=root/'logs/qwen25-32b-download-h16proxy-state.json'
b=p.read_bytes()
target=root/'assets-root/assets/qwen2.5-32b'
print(json.dumps(dict(raw_b64=base64.b64encode(b).decode(),sha256=hashlib.sha256(b).hexdigest(),epoch=time.time(),partials={x.name:x.stat().st_size for x in target.glob('*.h16proxy.partial')},job=subprocess.check_output(['squeue','-h','-j','3016','-o','%i %T %M'],text=True).strip())))
INNER"""
deadline = time.monotonic() + 24 * 3600
while time.monotonic() < deadline:
    try:
        result = subprocess.run(['python3', str(BASE / 'work/remote_topd.py')],
                                input=remote, text=True, capture_output=True, timeout=80)
        if result.returncode:
            raise RuntimeError(result.stderr[-1000:])
        response = json.loads(result.stdout)
        raw = base64.b64decode(response.pop('raw_b64'))
        assert hashlib.sha256(raw).hexdigest() == response['sha256']
        state = json.loads(raw)
        assert state['scope'] == 'OFFICIAL_NATIVE_H16PROXY_TRANSPORT_NO_TRAINING'
        assert state['revision'] == '1818d35814b8319459f4bd55ed1ac8709630f003'
        assert state['metadata_sha256'] == 'e546ba207f643dbe1a098323e28d45ff07942a6390bc73b8391c0acfc20d84d1'
        assert state['host'] == 'h100-11'
        assert state['script_sha256'] == 'cedc4a5e7dd5f920646a5fde63960c3b49793e18c8fef6af6fb3e06509e7df9a'
        destination = OUT / 'qwen25-32b-download-h16proxy-state.json'
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_bytes(raw)
        temporary.replace(destination)
        response.update(status=state['status'], complete=state.get('complete', False),
                        active=state['active'], native_pid=state['pid'],
                        verified_shards=sum(n.endswith('.safetensors') for n in state['files']),
                        watcher_scope='READ_ONLY_NO_SUBMISSION_NO_RESTART_NO_GPU')
        (OUT / 'qwen25-32b-native-download-watch-status.json').write_text(
            json.dumps(response, indent=2) + '\n')
        if response['complete'] or state['errors']:
            print(json.dumps(response), flush=True)
            break
    except Exception as error:
        (OUT / 'qwen25-32b-native-download-watch-status.json').write_text(
            json.dumps(dict(status='READ_RETRY_NO_JOB_RESTART', error=str(error),
                            epoch=time.time()), indent=2) + '\n')
    time.sleep(60)
