"""CPU-only scheduling gate: retained E has priority over curriculum GPUs."""
import json
from pathlib import Path
import subprocess
import sys
import time

OLD = Path('/home/jmzhang/Workspace/Search-State-Curriculum-20261013-v1')
SCIENCE = 'training-v4-E-2026101395'
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
FINALS = ['final-v4-'+SCIENCE+'-'+str(i) for i in range(4)]
STAGES = [SCIENCE, 'endpoint-v2-'+SCIENCE, 'canonical-v2-'+SCIENCE,
          *FINALS, 'final-audit-v4-'+SCIENCE]

def snapshot():
    receipts={name:json.loads(p.read_text()) for name in STAGES
              if (p:=OLD/'jobs'/name/'receipt.json').exists()}
    assert receipts[SCIENCE]['job_id']==138418
    ids=','.join(str(v['job_id']) for v in receipts.values())
    raw=subprocess.check_output(['sacct','-X','-n','-P','-j',ids,
                                 '--format=JobIDRaw,User,State,ExitCode'],text=True)
    records={x[0]:x for line in raw.splitlines() if (x:=line.split('|')) and x[0].isdigit()}
    states={}
    for name,r in receipts.items():
        x=records[str(r['job_id'])];assert x[1]=='jmzhang'
        states[name]=dict(job=r['job_id'],state=x[2].split()[0],exit=x[3])
    return states

def main():
    base=Path(sys.argv[1]);deadline=time.time()+46*3600
    while time.time()<deadline:
        states=snapshot()
        science=states[SCIENCE]['state']
        active=any(x['state'] not in TERMINAL for x in states.values())
        final_finished=all(states.get(n,{}).get('state') in TERMINAL for n in FINALS)
        # A scientific/endpoint failure cannot cause a later evaluation in the
        # existing E-only controller. Its CPU archive may finish independently.
        failure=any(x['state'] in TERMINAL-{'COMPLETED'} for x in states.values())
        cap_closed=time.time()>1792039690.177718
        release=(science in TERMINAL and not active and (final_finished or failure or cap_closed))
        record=dict(at=time.time(),states=states,release=release,
                    final_finished=final_finished,prior_failure=failure,prior_cap_closed=cap_closed,
                    meaning='Resource-ordering decision only; not E scientific qualification or EM')
        (base/'receipts/E-resource-gate.latest.json').write_text(json.dumps(record,indent=2))
        if release:
            (base/'receipts/E-resource-gate.released.json').write_text(json.dumps(record,indent=2))
            print(json.dumps(record),flush=True)
            return
        time.sleep(60)
    raise TimeoutError('E pipeline still owns scheduling priority; curriculum remains unscheduled')

if __name__=='__main__':main()
