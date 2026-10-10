"""Pure CPU original-run shared-GPU identity snapshots; never allocates GPU.

Only current human-authorized original devices0–3/5–6. Original trainer,
probe-server and retriever ledgers remain unmodified. No signals or HTTP.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess


def proc_identity(pid):
    p=Path('/proc')/str(pid);text=(p/'stat').read_text();tail=text[text.rfind(')')+2:].split()
    raw=(p/'cmdline').read_bytes()
    return dict(pid=int(pid),start_ticks=int(tail[19]),sid=int(tail[3]),uid=p.stat().st_uid,
        state=tail[0],command_sha256=hashlib.sha256(raw.replace(b'\0',b' ').decode(errors='replace').encode()).hexdigest())


def same_process(actual, recorded):
    return all(actual[k]==recorded[k] for k in ['pid','start_ticks','sid','uid']) and actual.get('state')!='Z'


def candidate_allowed(run, candidate):
    return candidate in [0,1,2,3,5,6] and candidate in run['training_physical_indices']+run['auxiliary_physical_indices'] and candidate not in run['excluded_physical_indices']


def parse_csv(text):return [r for r in csv.reader(io.StringIO(text)) if r]


def resource_snapshot(base, run, candidate):
    assert candidate_allowed(run,candidate),'Device is outside latest human shared-run authorization'
    path=base/'logs/process-ledger.json';raw=path.read_bytes();ledger=json.loads(raw)
    assert ledger['run_id']==run['run_id']
    allowed_roles=['trainer'] if candidate in [0,1,2,3] else ['retriever'] if candidate==5 else ['retriever','probe-server']
    records={}
    for role in allowed_roles:
        assert role in ledger['jobs']
        for key,row in ledger['jobs'][role]['ledger'].items():
            pid=int(key)
            assert pid not in records or records[pid][1]==row
            records[pid]=(role,row)
    cmd=['nvidia-smi','-i',str(candidate),'--query-gpu=index,uuid,memory.total,memory.used,memory.free,name','--format=csv,noheader,nounits']
    result=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=30)
    rows=parse_csv(result.stdout);assert len(rows)==1 and int(rows[0][0])==candidate
    gpu=dict(index=int(rows[0][0]),uuid=rows[0][1].strip(),total_mib=int(rows[0][2]),used_mib=int(rows[0][3]),actual_free_mib=int(rows[0][4]),name=rows[0][5].strip())
    cmd=['nvidia-smi','-i',str(candidate),'--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits']
    result=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=30)
    processes=[];unknown=[]
    for row in parse_csv(result.stdout):
        if row[0].strip()!=gpu['uuid']:continue
        pid=int(row[1]);actual=proc_identity(pid);pair=records.get(pid)
        known=pair is not None and same_process(actual,pair[1])
        actual.update(memory_mib=int(row[2]),recorded_role=pair[0] if known else 'UNKNOWN',matched_recorded_pid_tick_sid_uid=known)
        if pair is not None:actual['recorded_command_sha256']=hashlib.sha256(pair[1]['command'].encode()).hexdigest()
        if known and actual['recorded_command_sha256']!=actual['command_sha256']:known=False;actual['matched_recorded_pid_tick_sid_uid']=False
        if not known:unknown.append(pid)
        processes.append(actual)
    return dict(candidate_gpu=gpu,compute_processes=processes,foreign_or_unbound_pids=unknown,
        original_shared_roles=allowed_roles,only_bound_original_run_processes=not unknown,
        sharing_original_training=candidate in [0,1,2,3],resource_authorization_to_allocate=False,
        process_ledger_path=str(path),process_ledger_fresh_sha256=hashlib.sha256(raw).hexdigest())
