#!/usr/bin/env python3
"""Read only the own backup's small status, PID identity and receipt."""
import importlib.util
import json
from pathlib import Path
import shlex
import time

HERE = Path(__file__).resolve().parent
REQUEST = 'retained18-native-once-20261008-r2'


def main():
    spec = importlib.util.spec_from_file_location('ssh_route', HERE.parent / 'tree-speed-repair-20261008/ssh_retained18.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    code = '''import json,os,time
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/.shared-output-backup-20261008')
request=REQUEST
status=json.loads((root/(request+'.status.json')).read_text())
journal=json.loads((root/(request+'.launch.json')).read_text())
pid=status['pid']; proc=Path('/proc')/str(pid)
alive=proc.exists()
start=proc.joinpath('stat').read_text().split()[21] if alive else None
result={'observed_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':status,'journal':journal,'journal_path':str(root/(request+'.launch.json')),'status_path':str(root/(request+'.status.json')),'pid_alive':alive,'pid_start_ticks_match':start==status['proc_start_ticks'] if alive else False}
partial=Path(status['partial']); final=Path(status['final'])
asset=final if final.exists() else partial
result['destination_exists']=asset.exists()
result['actual_native_destination_bytes']=sum(p.stat().st_size for p in asset.glob('*.pt')) if asset.exists() else 0
receipt=final/'.complete'
result['complete_receipt']=json.loads(receipt.read_text()) if receipt.exists() else None
supplement=asset/'external-provenance-receipt.json'
result['external_provenance_receipt']=json.loads(supplement.read_text()) if supplement.exists() else None
print(json.dumps(result,sort_keys=True))
'''.replace('REQUEST', repr(REQUEST))
    result = helper.run('python3 -c ' + shlex.quote(code), timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    value = json.loads(result.stdout)
    (HERE / 'metadata').mkdir(exist_ok=True)
    (HERE / 'metadata/progress.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
