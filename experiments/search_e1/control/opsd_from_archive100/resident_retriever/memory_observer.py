"""External CPU observer; direct resident-card memory.free, no GPU work."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time


def process_identity(pid):
    base=Path('/proc')/str(pid)
    fields=(base/'stat').read_text().rsplit(')',1)[1].split()
    uid=int(next(l.split()[1] for l in (base/'status').read_text().splitlines() if l.startswith('Uid:')))
    return dict(pid=pid,uid=uid,starttick=int(fields[19]))


def command_rows(fields,kind):
    text=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True)
    return [row for row in csv.reader(io.StringIO(text),skipinitialspace=True) if row]


def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-identity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples-output',type=Path,required=True)
    p.add_argument('--interval',type=float,default=1.)
    args=p.parse_args()
    if not math.isfinite(args.interval) or args.interval<.1:raise ValueError('Invalid observation interval')
    source=json.loads(args.source_identity.read_text())
    expected={k:source[k] for k in ['pid','uid','starttick']}
    if process_identity(expected['pid'])!=expected:raise RuntimeError('PID/UID/starttick mismatch')
    uuid_list=source['identity']['expected_gpu_uuids']
    if len(uuid_list)!=4:raise RuntimeError('Exactly four approved UUIDs required')
    first=None;samples=0;minima={u:float('inf') for u in uuid_list};peaks={u:0 for u in uuid_list};foreign_history={}
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    while True:
        try:actual=process_identity(expected['pid'])
        except OSError:break
        if actual!=expected:break
        gpu=command_rows('index,uuid,memory.total,memory.used,memory.free','gpu')
        apps=command_rows('pid,gpu_uuid,used_gpu_memory','compute-apps')
        now=time.time();first=now if first is None else first;samples+=1
        rows=[];foreign=[]
        for physical,uid in enumerate(uuid_list):
            found=[g for g in gpu if g[1]==uid]
            if len(found)!=1 or int(found[0][0])!=physical:raise RuntimeError('GPU identity changed')
            g=found[0];proc=[a for a in apps if int(a[0])==expected['pid'] and a[1]==uid]
            used=sum(int(a[2]) for a in proc)
            minima[uid]=min(minima[uid],int(g[4]));peaks[uid]=max(peaks[uid],used)
            foreign.extend([dict(pid=int(a[0]),gpu_uuid=a[1],used_gpu_memory_mib=int(a[2])) for a in apps if a[1]==uid and int(a[0])!=expected['pid']])
            rows.append(dict(index=physical,uuid=uid,total_mib=int(g[2]),used_mib=int(g[3]),actual_free_mib=int(g[4]),candidate_process_mib=used))
        sample=dict(time=now,candidate_pid=expected['pid'],source_identity_sha256=source['source_identity_sha256'],physical_gpus=[0,1,2,3],cards=rows,foreign_compute_apps=foreign)
        for app in foreign:
            key=(app['pid'],app['gpu_uuid'])
            previous=foreign_history.setdefault(key,dict(**app,first_seen_unix=now,last_seen_unix=now))
            previous['last_seen_unix']=now
        with args.samples_output.open('a') as f:f.write(json.dumps(sample)+'\n')
        summary=dict(candidate_pid=expected['pid'],candidate_uid=expected['uid'],candidate_starttick=expected['starttick'],source_identity_sha256=source['source_identity_sha256'],physical_gpus=[0,1,2,3],gpu_uuids=uuid_list,free_measurement='nvidia-smi memory.free',sample_start_unix=first,sample_end_unix=now,updated_at_unix=time.time(),samples=samples,min_free_mib=min(minima.values()),max_process_mib=max(peaks.values()),minimum_actual_free_by_uuid_mib=minima,max_process_by_uuid_mib=peaks,current=sample,foreign_compute_apps_seen=list(foreign_history.values()),observer=dict(pid=os.getpid(),source_sha256=script_sha),scope='max_process is maximum per-card allocation, not total across four cards')
        atomic_json(args.output,summary)
        time.sleep(args.interval)


if __name__=='__main__':main()
