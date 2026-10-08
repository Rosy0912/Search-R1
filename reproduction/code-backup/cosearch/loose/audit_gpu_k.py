import json,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch')
r=root/'runs/cosearch-local-gpu-five-20261003-k';job=2961
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
records={}
for line in raw.splitlines():
    uid,pid=[v.strip() for v in line.split(',')]
    proc=Path('/proc')/pid
    try:
        cgroup=(proc/'cgroup').read_text()
        if f'/job_{job}/' not in cgroup:continue
        title=(proc/'cmdline').read_text(errors='replace').replace('\x00',' ')
        ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:continue
    entry=records.setdefault(pid,dict(title=title,cgroup=cgroup,start_ticks=ticks,uuids=[]))
    entry['uuids'].append(uid)
actors={pid:entry for pid,entry in records.items() if 'ray::WorkerDict' in entry['title']}
main={pid:entry for pid,entry in records.items() if 'ray::main_task' in entry['title']}
passed=(len(actors)==8 and all(len(set(e['uuids']))==1 for e in actors.values()) and len({e['uuids'][0] for e in actors.values()})==8)
record=dict(job=job,run=r.name,captured_at=time.time(),records=records,
            actor_distinct_eight_physical_gpus=passed,ranker_main_gpu_context=main)
(r/'gpu-ownership-latest.json').write_text(json.dumps(record,indent=2)+'\n')
if passed and not (r/'actor-gpu-ownership-first.json').exists():
    (r/'actor-gpu-ownership-first.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(job=job,actor_pids=list(actors),eight_distinct=passed,
                     main={pid:e['uuids'] for pid,e in main.items()})))
