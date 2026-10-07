"""Root-reviewed zero-OPSD preparation migration; no foreign process stops."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from prepare_preparation2 import (DEFAULT_BASE, DEFAULT_CANDIDATE, CANDIDATE_PID,
    assert_opsd_unstarted, read_json, verify_candidate, write_new_json)

OLD_PID = 2196101
OLD_GROUP = 2196099
OLD_RUN = '8f0bb94e2bad41c4bfaf893acc244803'


def proc_snapshot(pid, include_env=True):
    proc = Path('/proc') / str(pid)
    values = (proc/'stat').read_text().rsplit(')',1)[1].split()
    command = (proc/'cmdline').read_bytes()
    env = (dict(item.split(b'=',1) for item in (proc/'environ').read_bytes().split(b'\0') if b'=' in item)
           if include_env else {})
    return dict(pid=pid, uid=proc.stat().st_uid, state=values[0], ppid=int(values[1]),
        pgid=int(values[2]), start_ticks=int(values[19]), command=command, env=env)


def actual_gpu_pids():
    rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    return {int(row.strip()) for row in rows if row.strip().isdigit()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    base=DEFAULT_BASE
    gate=verify_candidate(DEFAULT_CANDIDATE,require_ready=True)
    zero=assert_opsd_unstarted(base)
    config=read_json(base/'receipts/preparation-2-config.json')
    assert config['prepared'] and config['opsd_zero_evidence']['actual_opsd_updates']==0
    old=read_json(base/'receipts/new-run.json')
    assert old['pid']==OLD_PID and old['run_id']==OLD_RUN
    root=proc_snapshot(OLD_PID)
    assert root['pgid']==OLD_GROUP and root['uid']==os.getuid()
    assert str(base/'source/pipeline.py').encode() in root['command']
    stage_path=base/'receipts/pipeline-stage.json'
    stage_bytes=stage_path.read_bytes();stage=json.loads(stage_bytes)
    assert stage['controller_pid']==OLD_PID and stage['phase']=='OPSD100_FULL_FRESH_TRAJECTORIES'
    snapshots=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:snapshot=proc_snapshot(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if snapshot['pgid']!=OLD_GROUP:continue
        assert snapshot['uid']==os.getuid(),'Foreign UID in group: never stop group'
        if snapshot['pid'] not in [OLD_PID,OLD_GROUP]:
            assert snapshot['env'].get(b'E1_BASE')==str(base).encode()
            assert snapshot['env'].get(b'E1_RUN_ID')==OLD_RUN.encode()
            assert b'search_e1.rollout' in snapshot['command'], 'Unexpected group process; inspect before stop'
        else:
            assert str(base/'source/pipeline.py').encode() in snapshot['command']
        snapshots.append(snapshot)
    owned={item['pid'] for item in snapshots}
    assert OLD_PID in owned and OLD_GROUP in owned and CANDIDATE_PID not in owned
    gpu_before=actual_gpu_pids()
    assert gpu_before-owned=={CANDIDATE_PID},'Unexpected GPU process: never stop it'
    rows=[]
    for part in sorted((base/'runs/rollouts.jsonl.parts').glob('rollouts-*.jsonl')):
        with part.open('rb') as stream:count=sum(block.count(b'\n') for block in iter(lambda:stream.read(8*1024*1024),b''))
        rows.append(dict(path=str(part),rows=count,bytes=part.stat().st_size))
    audit=[{k:(v.decode(errors='replace').replace('\0',' ') if k=='command' else v)
        for k,v in item.items() if k!='env'} for item in snapshots]
    proof=dict(time=time.time(),stopped=False,controller_pid=OLD_PID,controller_pgid=OLD_GROUP,
        run_id=OLD_RUN,opsd_updates=0,scope='preparation-1 only; OPSD updates 0',
        owned_pids=sorted(owned),process_identity=audit,partial_rows_before_stop=rows,
        source_runtime_unchanged=True,old_trajectory_artifacts_preserved_in_place=True,
        scientific_optimizer_restarted=False,candidate=gate,zero=zero,
        reason='Use fully verified exact segmented retrieval for new complete preparation with original seeds and cache; never merge interrupted partials')
    if not args.execute:
        print(json.dumps(dict(dry_run=True,owned_pids=sorted(owned),trajectory_rows=sum(x['rows'] for x in rows),opsd_updates=0)))
        return
    path=base/'receipts/preparation-1-stopped-for-preparation-2.json'
    pending=base/'receipts/preparation-1-stop-pending.json'
    assert not path.exists() and not pending.exists()
    write_new_json(pending,proof)
    saved_stage=base/'receipts/preparation-1-final-pipeline-stage.json'
    with saved_stage.open('xb') as stream:stream.write(stage_bytes)
    proof['preserved_stage_path']=str(saved_stage)
    proof['preserved_stage_sha256']=hashlib.sha256(stage_bytes).hexdigest()
    proof['final_pipeline_stage_sha256']=proof['preserved_stage_sha256']
    # Recheck each PID/start/ownership immediately before signalling. Do not
    # send a group signal that could hit an unreviewed new child.
    for item in sorted(snapshots,key=lambda x:x['pid']!=OLD_PID):
        try:current=proc_snapshot(item['pid'],include_env=False)
        except (FileNotFoundError,ProcessLookupError):continue
        assert (current['start_ticks'],current['uid'],current['pgid'])==(item['start_ticks'],item['uid'],item['pgid'])
        os.kill(item['pid'],signal.SIGTERM)
    deadline=time.monotonic()+60
    survivors=[]
    while True:
        survivors=[]
        for item in snapshots:
            try:current=proc_snapshot(item['pid'],include_env=False)
            except (FileNotFoundError,ProcessLookupError):continue
            assert current['start_ticks']==item['start_ticks'],'PID reused: never signal reused PID'
            if current['state']!='Z':survivors.append(item)
        if not survivors:break
        if time.monotonic()>=deadline:break
        time.sleep(1)
    for item in survivors:
        current=proc_snapshot(item['pid'],include_env=False)
        assert (current['start_ticks'],current['uid'],current['pgid'])==(item['start_ticks'],item['uid'],item['pgid'])
        os.kill(item['pid'],signal.SIGKILL)
    deadline=time.monotonic()+30
    while actual_gpu_pids()-{CANDIDATE_PID}:
        assert actual_gpu_pids()<=owned|{CANDIDATE_PID},'Unexpected GPU process after stop; never stop it'
        assert time.monotonic()<deadline,'Owned GPU memory not released'
        time.sleep(1)
    final=[]
    for item in snapshots:
        try:current=proc_snapshot(item['pid'],include_env=False)
        except (FileNotFoundError,ProcessLookupError):continue
        assert current['start_ticks']==item['start_ticks'] and current['state']=='Z','Owned process still running'
        final.append(dict(pid=item['pid'],state='Z'))
    assert_opsd_unstarted(base)
    proof.update(stopped=True,finished_at=time.time(),remaining_owned_zombies=final,
        actual_gpu_pids_after=sorted(actual_gpu_pids()),escalated_owned_pids=[x['pid'] for x in survivors])
    write_new_json(path,proof)
    print(json.dumps(dict(stopped=True,owned_pids=sorted(owned),opsd_updates=0,receipt=str(path))))


if __name__=='__main__':main()
