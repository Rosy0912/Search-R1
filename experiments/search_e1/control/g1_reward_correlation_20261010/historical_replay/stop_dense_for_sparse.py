"""Human-requested scope replacement: signal only the exact owned dense replay.

Original training, probe, retriever and the independent C reader are preserved.
Every accepted and partial output is preserved on the original remote node.
"""
import datetime
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

REMOTE = r'''
import datetime,hashlib,json,os,pathlib,signal,socket,subprocess,time
D=pathlib.Path('/data/shaoshitong/Search-G1-reward-reconstruction-20261010-1ec36152611a')
R=D/'R-d0e94bdf64d8'
assert socket.gethostname()=='job-20e13974-c420-4074-a464-68e93738cfc0-master-0'
assert os.getuid()==0
inputs=json.loads((R/'inputs.json').read_bytes())
assert inputs['config_sha256']=='3e8b03f02d95225bf8afdfc6886fd281568942237e2c1e024262af5ae017bc13'
assert inputs['source_manifest_sha256']=='de2464fd4d419a5f17e3aec5fe056edaf56803b11c3e4fc23d1201cdee8bd7a5'
expected=dict(pid=4157484,sid=4157484,start_ticks=568763464,uid=0,command_sha256='18284c190daec639c0830a13b2eea91f7d86eb35f3112920a7babf879490ae12')
assert all(inputs['own_identity'][k]==v for k,v in expected.items())
def identity(pid):
    p=pathlib.Path('/proc')/str(pid)
    try:
        raw=(p/'stat').read_text();a=raw[raw.rfind(')')+2:].split()
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        return dict(pid=pid,sid=int(a[3]),start_ticks=int(a[19]),uid=p.stat().st_uid,state=a[0],command_sha256=hashlib.sha256(cmd.encode()).hexdigest())
    except FileNotFoundError:return dict(pid=pid,absent=True)
def match(actual,wanted):return all(actual.get(k)==v for k,v in wanted.items())
protected=[dict(pid=1338978,start_ticks=552409911,uid=0),dict(pid=1336445,start_ticks=552398244,uid=0),dict(pid=1328893,start_ticks=552365460,uid=0)]
before=[identity(p['pid']) for p in protected]
assert all(match(a,b) for a,b in zip(before,protected)), 'protected process identity changed'
own=identity(expected['pid']);assert match(own,expected),'dense replay identity changed; no signal'
fd=os.pidfd_open(expected['pid'],0)
try:
    assert match(identity(expected['pid']),expected),'dense replay identity raced; no signal'
    signal.pidfd_send_signal(fd,signal.SIGTERM)
finally:os.close(fd)
for _ in range(100):
    if identity(expected['pid']).get('absent'):break
    time.sleep(.1)
after=[identity(p['pid']) for p in protected]
assert all(match(a,b) for a,b in zip(after,protected)), 'protected identity changed after own signal'
pane=subprocess.run(['tmux','list-panes','-t','g1-full-pair-R-d0e94bdf64d8','-F','#{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True,check=True).stdout.strip()
gpu=subprocess.run(['nvidia-smi','-i','5','--query-gpu=index,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
compute=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
uuid='GPU-21508d9d-bd43-ff94-7bee-1227ec7b007b'
compute5=[x for x in compute.splitlines() if x.split(',')[0].strip()==uuid]
assert identity(expected['pid']).get('absent'), 'own replay not yet released'
assert all(int(x.split(',')[1])==1328893 for x in compute5),'unexpected GPU5 process after stop'
summaries=[]
for p in R.glob('TRAIN-*-summary.json'):
    raw=p.read_bytes();row=json.loads(raw)
    assert row['status']=='ACCEPTED_RECONSTRUCTED_R_STEP' and row['trajectory_count']==2560
    summaries.append(dict(step=row['step'],summary_sha256=hashlib.sha256(raw).hexdigest(),output_sha256=row['output_sha256']))
summaries.sort(key=lambda x:x['step'])
result=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='Latest human explicitly replaces dense reconstruction with [1,20,40,60,80,100,120,140].',selected_steps=[1,20,40,60,80,100,120,140],signalled_only_bound_owned_pid=expected,own_after=identity(expected['pid']),protected_before=before,protected_after=after,retained_pane=pane,direct_gpu5=gpu,gpu5_compute=compute5,preserved_accepted_steps=summaries,partial_preserved=True,training_or_production_signals_sent=False,raw_data_not_exported=True)
proof=D/'receipts/dense-stop-for-sparse8.json'
with proof.open('xb') as f:f.write(json.dumps(result,sort_keys=True,separators=(',',':')).encode())
print(json.dumps(result))
'''

if __name__ == '__main__':
    r = subprocess.run(control.route('baai-a800', shlex.join(['python3', '-c', REMOTE])),
                       capture_output=True, timeout=45)
    for suffix, raw in [('stdout.json', r.stdout), ('stderr.txt', r.stderr)]:
        (HERE / ('dense-stop-for-sparse8.' + suffix)).write_bytes(raw)
    (HERE / 'dense-stop-for-sparse8.execution.json').write_text(json.dumps(dict(
        rc=r.returncode, observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        helper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    assert r.returncode == 0, 'inspect actual own-stop evidence; never signal a replacement PID'
    print(r.stdout.decode())
