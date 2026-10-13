"""CPU-only watcher and owned terminal-final handoff. Never interrupts training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import uuid

from final_eval import validate_checkpoint

BASE = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
OWNER = dict(pid=3714909, uid=0, tick=658086351)
RUN = '1c806d15752d44a7ac99687f0737d7e3'
CONFIG_SHA = '653193a4eb06bd28667f78ac14a66a2d1c97200ca8199b8521dcc2d3de35b319'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(record, indent=2))
    temp.replace(path)
    return dict(path=str(path), sha256=sha(path))


def process(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), uid=path.stat().st_uid, tick=int(fields[19]),
            ppid=int(fields[1]), state=fields[0],
            argv=(path/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
    except FileNotFoundError:
        return None


def matches(record, identity):
    return record is not None and all(record[k] == identity[k] for k in ('pid', 'uid', 'tick'))


def owned_tree():
    assert matches(process(OWNER['pid']), OWNER), 'Do not signal reused or foreign PID'
    processes = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            record = process(int(p.name))
            if record: processes[record['pid']] = record
    tree = {OWNER['pid']: processes[OWNER['pid']]}
    while True:
        added = {pid: p for pid, p in processes.items() if p['ppid'] in tree and pid not in tree}
        if not added: break
        tree.update(added)
    assert all(p['uid'] == OWNER['uid'] for p in tree.values())
    return list(tree.values())


def terminal_condition(manifest, config_sha, step200_logged, final_metrics):
    if final_metrics is not None:
        assert final_metrics['val/qa_count/all'] == 51713
        return 'NATIVE_FINAL_COMPLETE'
    if manifest is None or not step200_logged:
        return 'WAIT_TRAINING'
    validate_checkpoint(manifest, config_sha)
    assert manifest.get('boundary_phase') == 'post-evaluation'
    return 'HANDOFF_FINAL_ONLY'


def terminate_owned(records, folder):
    # Capture pidfds before any signal; no UID-based broad kill, Ray stop, pkill,
    # foreign parent cancellation or signals to the persistent CPU retriever.
    handles = []
    for record in records:
        if not matches(process(record['pid']), record): continue
        fd = os.pidfd_open(record['pid'])
        if not matches(process(record['pid']), record):
            os.close(fd)
            raise RuntimeError('Process identity raced while opening pidfd')
        handles.append((record, fd))
    write(folder/'owned-before-stop.json', dict(time=time.time(), records=records,
        reason='200 training and scheduled dev512 complete; replace terminal low-concurrency final only'))
    try:
        for record, fd in reversed(handles):
            try: signal.pidfd_send_signal(fd, signal.SIGTERM)
            except ProcessLookupError: pass
        for _ in range(30):
            if all(not matches(process(p['pid']), p) or process(p['pid'])['state'] == 'Z' for p in records): break
            time.sleep(1)
        for record, fd in handles:
            p = process(record['pid'])
            if matches(p, record) and p['state'] != 'Z':
                signal.pidfd_send_signal(fd, signal.SIGKILL)
        for _ in range(30):
            remaining = [p for p in records if matches(process(p['pid']), p) and process(p['pid'])['state'] != 'Z']
            if not remaining: break
            time.sleep(1)
        assert not remaining, 'Owned process release incomplete'
        return write(folder/'owned-terminal-release.json', dict(time=time.time(),
            cause='INTENTIONAL_FINAL_ONLY_HANDOFF_AFTER_COMPLETE200', natural_exit0=False,
            records=[dict(original=p, current=process(p['pid'])) for p in records], released=True))
    finally:
        for _, fd in handles: os.close(fd)


def resource_receipt(folder):
    def query(fields):
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu='+fields,
            '--format=csv,noheader,nounits'], text=True)
        return [line.strip().split(', ') for line in raw.strip().splitlines()]
    cards = query('index,uuid,memory.free')
    assert len(cards) == 8 and sorted(int(x[0]) for x in cards) == list(range(8))
    assert all(int(x[2]) >= 20480 for x in cards), 'Insufficient free VRAM; keep watcher alive and retry admission'
    assert shutil.disk_usage(BASE).free >= 100*1024**3
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    lease = subprocess.check_output(['scontrol', 'show', 'job', '3389'], text=True)
    assert 'JobState=RUNNING' in lease and 'h100-11' in lease
    # Same authorized overlapping Slurm allocation. Preserve any cohost process;
    # only the captured original task tree is ever signalled.
    return write(folder/'resource-admission.json', dict(time=time.time(), cards=cards,
        compute_processes=active, allocation=lease, disk_free=shutil.disk_usage(BASE).free))


def evaluator_command(plan_path, payload):
    runtime='/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009'
    source=str(BASE/'source-71d4e69')
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        PYTHONPATH=':'.join([str(payload),source,source+'/release/recursive_bran_a/h100_migration',runtime+'/retriever-gpu-lib']),
        CPATH='/root/miniconda3/pkgs/python-3.10.21-h17756b0_0/include/python3.10',
        TRITON_LIBCUDA_PATH=runtime+'/lib/driver',LIBRARY_PATH=runtime+'/lib/driver',
        OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        VLLM_ATTENTION_BACKEND='XFORMERS',NVIDIA_TF32_OVERRIDE='0',WANDB_MODE='offline',
        RECURSIVE_A_H100_FRESH_PROFILE='1', RECURSIVE_A_GUARD_ROUND_PROGRESS='1',
        RECURSIVE_A_OMIT_FULL_BETA='1', TREE_LIGHTWEIGHT_CANDIDATE_REWARD='1',
        RECURSIVE_A_SPARSE_DETAILS='1', XDG_CACHE_HOME=str(BASE/'cache'), HF_HOME=str(BASE/'cache/hf'),
        RAY_TMPDIR='/tmp/rba-final-'+json.loads(Path(plan_path).read_text())['run_id'])
    for key in ('LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        env.pop(key, None)
    env['E1_SEGMENTED_PROBE_TOKEN']=(BASE/'retriever/receipts/private-probe-token').read_text().strip()
    command=['srun','--jobid=3389','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=48',
        '--mem=256G','--gres=gpu:h100:8','--nodelist=h100-11',
        runtime+'/env/bin/python','-B','-u',str(payload/'final_eval.py'),'--plan',str(plan_path)]
    return command, env


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',required=True)
    args=parser.parse_args();folder=Path(args.control);folder.mkdir(parents=True,exist_ok=True)
    # One watcher per task; this is not a GPU training/evaluation lock.
    import fcntl
    lock=(folder/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=json.loads((BASE/'receipts/configuration.json').read_text())
    assert receipt['config_sha256']==CONFIG_SHA and sha(receipt['config'])==CONFIG_SHA
    assert json.loads((BASE/'receipts/train-start.json').read_text())['run_id']==RUN
    log=BASE/'logs/training.log';offset=max(0,log.stat().st_size-16*1024**2)
    step200=False;final_metrics=None;pending=''
    cp=BASE/'checkpoints/training_state/global_step_200/post-evaluation/manifest.json'
    while True:
        with log.open(errors='replace') as stream:
            stream.seek(offset);pending+=stream.read();offset=stream.tell()
        lines=pending.split('\n');pending=lines.pop()
        for line in lines:
            if re.search(r'\bstep:200\b',line) and 'training/actual_outer_updates:200' in line.replace(' ',''):
                step200=True
            if 'SEARCH_R1_FINAL_METRICS_JSON=' in line:
                final_metrics=json.loads(line.split('SEARCH_R1_FINAL_METRICS_JSON=',1)[1])
        manifest=json.loads(cp.read_text()) if cp.exists() else None
        condition=terminal_condition(manifest,CONFIG_SHA,step200,final_metrics)
        write(folder/'status.json',dict(time=time.time(),pid=os.getpid(),condition=condition,
            original_owner=process(OWNER['pid']), step200_logged=step200,
            final_manifest_present=manifest is not None))
        if condition=='NATIVE_FINAL_COMPLETE':
            write(folder/'native-final-complete.json',dict(metrics=final_metrics,time=time.time()))
            return
        if condition=='HANDOFF_FINAL_ONLY': break
        if not matches(process(OWNER['pid']),OWNER):
            raise RuntimeError('Original owner terminal before complete200: repair needed, never blind restart')
        time.sleep(30)
    completion=write(folder/'complete200-boundary.json',dict(time=time.time(),
        manifest_path=str(cp),manifest_sha256=sha(cp),manifest=manifest,
        training_run=RUN,step200_logged=step200,scheduled_dev512_finished=True))
    release=terminate_owned(owned_tree(),folder)
    # The scanner is owned by the persistent CPU retriever, not the training
    # process tree. Release only its stage GPU child via its private endpoint.
    import urllib.request
    token=(BASE/'retriever/receipts/private-probe-token').read_text().strip()
    request=urllib.request.Request('http://127.0.0.1:18086/release-gpu',method='POST',
        headers={'X-E1-Probe-Token':token})
    with urllib.request.urlopen(request,timeout=120) as response:
        scanner=json.load(response)
    assert scanner['released'] is True and scanner['gpu_worker_pid'] is None
    write(folder/'retriever-stage-release.json',dict(time=time.time(),response=scanner))
    # Preserve the final result fragments in the original log. Original runtime
    # source, configuration, checkpoints and predictions are never changed.
    while True:
        try: resource=resource_receipt(folder);break
        except (AssertionError,subprocess.CalledProcessError) as error:
            write(folder/'admission-wait.json',dict(time=time.time(),error=str(error)))
            time.sleep(30)
    retriever=json.loads((BASE/'retriever/receipts/source-identity.json').read_text())
    assert retriever['pid']==3713706
    assert process(3713706)['tick']==658076829
    run=uuid.uuid4().hex
    plan=dict(ready=True,training_complete_outer=200,owned_training_released=True,
        native_final_complete=False,config=receipt['config'],config_sha256=CONFIG_SHA,
        configuration_receipt=str(BASE/'receipts/configuration.json'),
        checkpoint_manifest=str(cp),checkpoint_manifest_sha256=sha(cp),
        training_start=str(BASE/'receipts/train-start.json'),training_start_sha256=sha(BASE/'receipts/train-start.json'),
        training_run=RUN,run_id=run,output=str(BASE/'final51713'/run),
        admission_receipts=[completion,release,resource])
    plan_path=folder/'execution-plan.json';write(plan_path,plan)
    command,env=evaluator_command(plan_path,Path(__file__).resolve().parent)
    with (folder/'evaluation.log').open('ab') as output:
        child=subprocess.Popen(command,env=env,stdout=output,stderr=output)
        write(folder/'evaluation-launch.json',dict(time=time.time(),pid=child.pid,command=command,run_id=run))
        with (folder/'gpu-device-observer.jsonl').open('a') as memory:
            while child.poll() is None:
                sample=subprocess.run(['nvidia-smi',
                    '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
                memory.write(json.dumps(dict(time=time.time(),rc=sample.returncode,
                    device_sample=sample.stdout,scope='Whole-device sample; may include preserved cohost tasks'))+'\n')
                memory.flush()
                time.sleep(10)
        rc=child.wait()
    write(folder/'evaluation-actual-exit.json',dict(time=time.time(),pid=child.pid,rc=rc,run_id=run))
    if rc: raise RuntimeError('Final evaluator exited with rc='+str(rc)+'; preserve output for targeted recovery')


if __name__=='__main__': main()
