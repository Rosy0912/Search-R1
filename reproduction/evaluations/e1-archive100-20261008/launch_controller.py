"""Copy same-cluster weights through memory pipes, then launch isolated evaluation."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = '/home/shaoshitong/search-e1-archive100-eval-20261008'
SOURCE = '/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/assets/archive100'


def ssh(host, command):
    return ['ssh','-T','-o','ProxyCommand=none','-o','BatchMode=yes','-o','ConnectTimeout=10',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=12',host,command]


def save(phase, **fields):
    row = dict(phase=phase, observed_epoch=time.time(), **fields)
    temp = HERE / 'controller-status.json.tmp'
    temp.write_text(json.dumps(row, indent=2)+'\n')
    temp.replace(HERE / 'controller-status.json')
    print(json.dumps(row), flush=True)


def main():
    lock = (HERE / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((HERE / 'plan.json').read_text())
    receipt = json.loads((HERE.parent / 'search-e1-archive100-hf-verified-receipt-20261007.json').read_text())
    names = [row['name'] for row in receipt['files']]
    files = {'plan.json':json.dumps(plan,ensure_ascii=False,indent=2),
        'model-manifest.json':json.dumps(receipt,indent=2),
        'evaluate_worker.py':(HERE/'evaluate_worker.py').read_text()}
    for relative in plan['strict_reward_source_manifest']:
        files['code/'+relative] = (HERE/relative).read_text()
    bootstrap = '''import json,sys,os\nfrom pathlib import Path\npayload=json.load(sys.stdin)\nb=Path(payload['base'])\nos.umask(0o077)\nassert not b.exists(), 'Task directory already exists; check current worker instead of duplicating'\nb.mkdir(mode=0o700)\n(b/'model.partial').mkdir()\nfor name,text in payload['files'].items():\n p=b/name;assert b in p.parents\n p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)\nprint('TASK_CREATED')\n'''
    save('preparing_destination', weight_bytes_written_on_mac=0)
    r = subprocess.run(ssh('hkust5','python3 -c '+shlex.quote(bootstrap)),
        input=json.dumps(dict(base=BASE, files=files)), capture_output=True,text=True,timeout=35)
    assert r.returncode == 0 and 'TASK_CREATED' in r.stdout, (r.stderr+r.stdout)[-1000:]
    source_check = '''import json,sys\nfrom pathlib import Path\nb=Path(sys.argv[1]);spec=json.load(sys.stdin)\nfor row in spec['files']:\n p=b/row['name'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes']\nprint('SOURCE_SIZES_OK')\n'''
    r = subprocess.run(ssh('hkust1','python3 -c '+shlex.quote(source_check)+' '+shlex.quote(SOURCE)),
        input=json.dumps(receipt),capture_output=True,text=True,timeout=35)
    assert r.returncode == 0 and 'SOURCE_SIZES_OK' in r.stdout
    source_command = shlex.join(['tar','-C',SOURCE,'-cf','-',*names])
    target_command = shlex.join(['tar','-C',BASE+'/model.partial','-xf','-'])
    save('copying_weights', route='HKUST1 -> existing SSH -> Mac memory pipe -> existing SSH -> HKUST5',
         expected_bytes=receipt['checkpoint_bytes'], weight_bytes_written_on_mac=0)
    with (HERE/'source-transfer.log').open('ab') as src_log, (HERE/'destination-transfer.log').open('ab') as dst_log:
        source = subprocess.Popen(ssh('hkust1',source_command),stdout=subprocess.PIPE,stderr=src_log)
        destination = subprocess.Popen(ssh('hkust5',target_command),stdin=source.stdout,stdout=dst_log,stderr=dst_log)
        source.stdout.close()
        target_exit = destination.wait()
        source_exit = source.wait()
    assert target_exit == source_exit == 0, 'Weight stream interrupted; retain partial destination'
    launch = '''import json,subprocess,os,time\nfrom pathlib import Path\nb=Path(__BASE__)\nassert not (b/'worker-launch.json').exists()\nlog=(b/'worker.log').open('ab')\np=subprocess.Popen(['python3','-B',str(b/'evaluate_worker.py')],cwd=str(b),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nr={'pid':p.pid,'launched_epoch':time.time(),'script':str(b/'evaluate_worker.py')}\n(b/'worker-launch.json').write_text(json.dumps(r,indent=2)+'\\n')\nprint(json.dumps(r))\n'''.replace('__BASE__',repr(BASE))
    r = subprocess.run(ssh('hkust5','python3 -c '+shlex.quote(launch)),capture_output=True,text=True,timeout=35)
    assert r.returncode == 0, r.stderr[-1000:]
    remote = json.loads(r.stdout.strip().splitlines()[-1])
    (HERE/'remote-worker-launch.json').write_text(json.dumps(remote,indent=2)+'\n')
    save('remote_verification_started', remote_worker=remote, weight_bytes_written_on_mac=0)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('failed', error_type=type(error).__name__, error=str(error)[:1000])
        raise
