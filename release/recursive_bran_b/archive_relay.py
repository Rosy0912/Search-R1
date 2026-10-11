"""Local control and small receipts only; large payloads stay in an SSH pipe."""
import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import time


def ssh(host, helper, mode, data=None, options=()):
    command = shlex.join(['python3', '-B', helper, mode, *options])
    p = subprocess.run(['ssh', '-oBatchMode=yes', host, command], input=data,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError(f'{host} {mode} exit={p.returncode}: ' + p.stderr.decode()[-3000:])
    return json.loads(p.stdout)


def save(path, obj):
    temp = path.with_suffix('.new')
    temp.write_text(json.dumps(obj, indent=2) + '\n'); temp.replace(path)


def transfer(source_helper, target_helper, inv, proof):
    pack_command = shlex.join(['python3', '-B', source_helper, 'pack'])
    recv_command = shlex.join(['python3', '-B', target_helper, 'receive'])
    # Binary producer stdout is inherited directly by the consumer: no .tar/local spool.
    with (proof/'source.stderr').open('ab') as e1, (proof/'target.stderr').open('ab') as e2:
        source = subprocess.Popen(['ssh','-oBatchMode=yes','hkust5',pack_command],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=e1)
        target = subprocess.Popen(['ssh','-oBatchMode=yes','hkust4',recv_command],
                                  stdin=source.stdout, stdout=subprocess.PIPE, stderr=e2)
        source.stdout.close()
        source.stdin.write(json.dumps(inv).encode()); source.stdin.close()
        save(proof/'transfer-live.json', dict(time=time.time(),source_pid=source.pid,target_pid=target.pid,
                                             mode='Mac memory-pipe relay; no local large files'))
        out = target.communicate()[0]
        target_rc = target.returncode; source_rc = source.wait()
    save(proof/'transfer-exit.json',dict(time=time.time(),source_rc=source_rc,target_rc=target_rc))
    if source_rc or target_rc:
        raise RuntimeError('Transfer did not return two actual exit0 results; retain source and partial')
    receipt = json.loads(out)
    save(proof/'target-accepted.json',receipt)
    return receipt


def run_once(a, proofdir):
    scan = ssh('hkust5',a.source_helper,'scan')
    save(proofdir/'scan-latest.json',dict(time=time.time(),**scan))
    for item in scan['candidates']:
        d = proofdir/f'{item["kind"]}-{item["step"]:06d}'
        d.mkdir(exist_ok=True)
        if (d/'pruned.json').exists():
            continue
        inv = ssh('hkust5',a.source_helper,'inventory',options=['--kind',item['kind'],'--step',str(item['step'])])
        # Inventory contains only paths/bytes/hashes, no tensors or raw trajectories.
        save(d/'inventory.json',inv)
        target = ssh('hkust4',a.target_helper,'verify',json.dumps(inv).encode())
        if target['exists']:
            receipt = target['receipt']
            save(d/'target-fresh-reverified.json',receipt)
        else:
            receipt = transfer(a.source_helper,a.target_helper,inv,d)
        result = ssh('hkust5',a.source_helper,'prune',json.dumps(dict(inventory=inv,receipt=receipt)).encode())
        save(d/'pruned.json',result)
    return scan


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-helper',required=True)
    p.add_argument('--target-helper',required=True);p.add_argument('--proof-dir',required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=600)
    a=p.parse_args();d=pathlib.Path(a.proof_dir);d.mkdir(parents=True,exist_ok=True)
    # Single local archiver. flock is absent on macOS; lockf works on both hosts.
    import fcntl
    lock=(d/'control.lock').open('a');fcntl.lockf(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            scan=run_once(a,d)
            save(d/'status.json',dict(time=time.time(),status='HEALTHY',completed=scan['completed']))
            if not a.watch or scan['completed'] >= 200:
                return
        except Exception as e:
            save(d/'status.json',dict(time=time.time(),status='ARCHIVE_HOLD_SOURCE_RETAINED',error=str(e)))
            if not a.watch:
                raise
        time.sleep(max(30,a.interval))


if __name__=='__main__':
    main()
