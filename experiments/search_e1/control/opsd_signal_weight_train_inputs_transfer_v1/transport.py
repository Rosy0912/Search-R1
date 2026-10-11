"""Published-code-only direct HKUST transport; only compact receipts on Mac."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ProxyCommand=none', '-o', 'ProxyJump=none']
S = '/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'
T = '/home/shaoshitong/search-e1-migration-20261011'
DEPLOY = '''import base64,hashlib,json,os,pathlib,sys,time
p=json.load(sys.stdin);assert os.getuid()==1001
s=pathlib.Path(p['study']);h=s/'source'/('train-inputs-'+p['commit'][:7])/'opsd_signal_weight_train_inputs_transfer_v1'
h.mkdir(parents=True,exist_ok=True)
for name,data in p['files'].items():
 raw=base64.b64decode(data);f=h/name
 if f.exists():assert f.read_bytes()==raw,'Existing published source differs'
 else:
  with f.open('xb') as out:out.write(raw)
m=json.loads((h/'source-manifest.json').read_text())
for name,r in m['files'].items():assert hashlib.sha256((h/name).read_bytes()).hexdigest()==r['sha256']
print(json.dumps(dict(accepted=True,helper=str(h),commit=p['commit'],manifest_sha256=hashlib.sha256((h/'source-manifest.json').read_bytes()).hexdigest(),cpu_only=True)))
'''


def checked_call(host, command, output, input_bytes=None, forward_agent=False):
    args = SSH[:1] + (['-A'] if forward_agent else []) + SSH[1:] + [host, command]
    result = subprocess.run(args, input=input_bytes, capture_output=True)
    Path(str(output)+'.stdout').write_bytes(result.stdout)
    Path(str(output)+'.stderr').write_bytes(result.stderr)
    proof = dict(actual_wait_returncode=result.returncode, checked_unix=time.time(), host=host,
                 stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stderr_bytes=len(result.stderr))
    Path(str(output)+'.json').write_text(json.dumps(proof, indent=2)+'\n')
    if result.returncode != 0:
        raise RuntimeError('Remote action failed; preserved output: '+str(output))
    return result.stdout


def main():
    p = argparse.ArgumentParser(); p.add_argument('--commit', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); root = Path(__file__).resolve().parent
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    local = Path(a.output).resolve(); local.mkdir(parents=True, exist_ok=False)
    source = {}
    for f in sorted(root.iterdir()):
        if not f.is_file(): continue
        raw = f.read_bytes()
        assert raw == subprocess.check_output(['git', 'show', a.commit+':'+str(f.relative_to(repo))])
        source[f.name] = base64.b64encode(raw).decode()
    helpers = {}
    for host, study in [('hkust4', S), ('hkust5', T+S)]:
        payload=json.dumps(dict(commit=a.commit, study=study, files=source)).encode()
        out=checked_call(host, 'python3 -B -c '+shlex.quote(DEPLOY), local/(host+'-deployment'), payload)
        helpers[host] = json.loads(out)['helper']
        checked_call(host, shlex.join(['/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
                     'python3','-B','-m','unittest','discover','-s',helpers[host],'-v']),local/(host+'-tests'))
    stamp=str(int(time.time()))
    inventory=S+'/analysis/train-inputs-G-'+stamp+'/inventory.json'
    raw=checked_call('hkust4',shlex.join(['/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
        'python3','-B',helpers['hkust4']+'/train_inputs.py','inventory','--output',inventory]),local/'inventory')
    summary=json.loads(raw);assert summary['accepted']
    stage=T+'/train-inputs-g-stage-'+stamp
    setup='''import os,pathlib,sys
p=pathlib.Path(sys.argv[1]);n=int(sys.argv[2]);assert os.getuid()==1001
assert os.statvfs(p.parent).f_bavail*os.statvfs(p.parent).f_frsize>n+5*1024**3
p.mkdir(exist_ok=False)
'''
    checked_call('hkust5','python3 -B -c '+shlex.quote(setup)+' '+shlex.quote(stage)+' '+str(summary['member_bytes']),local/'stage')
    tar=shlex.join(['tar','-C','/','--verbatim-files-from','--files-from',inventory.replace('.json','.members'),'-cf','-'])
    receiver=shlex.join(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12',
                        'shaoshitong@10.123.4.12',shlex.join(['tar','-xf','-','-C',stage])])
    launch=dict(started_unix=time.time(),inventory=summary,stage=stage,commit=a.commit,
                payload_route='HKUST4 direct SSH to HKUST5 using existing forwarded agent; no key copy or Mac payload file')
    (local/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    checked_call('hkust4','bash -c '+shlex.quote('set -o pipefail; '+tar+' | '+receiver),local/'direct-transfer',forward_agent=True)
    target_inventory=stage+'/'+inventory.lstrip('/')
    target_receipt=T+S+'/receipts/g-training-inputs-transfer-'+stamp+'.json'
    accepted=json.loads(checked_call('hkust5',shlex.join(['/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
        'python3','-B',helpers['hkust5']+'/train_inputs.py','install','--output',target_receipt,
        '--inventory',target_inventory,'--inventory-sha256',summary['sha256'],'--stage',stage]),local/'install'))
    assert accepted['accepted']
    (local/'complete.json').write_text(json.dumps(dict(accepted=True,started=launch,installation=accepted,
        source_originals_preserved=True,large_local_files=0,scope=summary['scope']),indent=2)+'\n')
    print(json.dumps(accepted))


if __name__ == '__main__': main()
