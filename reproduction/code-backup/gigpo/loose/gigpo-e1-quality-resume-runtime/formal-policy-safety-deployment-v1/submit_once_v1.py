"""Template: once direct submission on h100-8; uncertain response keeps claim, never retry."""
import datetime
import json
import os
from pathlib import Path
import re
import socket
import subprocess
from deployment_contract_v1 import ROOT,HELPERS,CONTROLS,SCOPE,READY,INTENT,RECEIPT,require,sha,read_bound,write_once,validate_ready


def main(binding_record,review_record):
    require(socket.gethostname().split('.')[0]=='h100-8','Only authorized h100-8 may submit')
    require(Path(binding_record['path'])==READY,'Only exact new formal READY')
    require(not os.path.lexists(INTENT) and not os.path.lexists(RECEIPT),'Existing or uncertain formal claim: inspect, never repeat')
    binding=read_bound(binding_record);validate_ready(binding)
    review=read_bound(review_record)
    require(review['status']=='PASS' and review['scope']==SCOPE and review['binding_sha256']==binding_record['sha256']
        and review['submit_helper_sha256']==sha(Path(__file__)) and review['extra_AB_diagnostic_Adam_required']==0,
        'Missing exact independent READY/once review')
    # Primitives match reviewed calibrationv9. All full SHAs are checked again
    # by the frozen formal launcher before GPU, not inferred from metadata.
    for item in binding['files']+binding['evidence']:
        require(Path(item['path']).is_file() and not Path(item['path']).is_symlink()
            and Path(item['path']).stat().st_size==item['bytes'],'Input changed after READY')
    queue=subprocess.check_output(['squeue','-h','-w','h100-8','-o','%i %j %T'],text=True)
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
    require(not queue.strip() and not compute.strip(),'Authorized node occupied: no formal claim created')
    write_once(INTENT,dict(status='CLAIMED_NO_RESUBMISSION',scope=SCOPE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        binding=binding_record,review=review_record,retained_step25=25,new_outer275=275,queue=queue,gpu_compute=compute))
    command=['sbatch','--parsable',str(CONTROLS/'formal_v1.sbatch'),binding_record['path'],binding_record['sha256'],binding['repo'],binding['helpers']]
    try:
        completed=subprocess.run(command,text=True,capture_output=True,timeout=60)
        matched=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?',completed.stdout.strip())
        result=dict(command=command,stdout=completed.stdout,stderr=completed.stderr,returncode=completed.returncode,binding_sha256=binding_record['sha256'])
        if completed.returncode==0 and matched is not None:result.update(status='SUBMITTED_EXACTLY_ONCE',job_id=matched.group(1))
        else:result['status']='UNCERTAIN_OR_FAILED_NO_RETRY'
    except BaseException as error:
        write_once(RECEIPT,dict(status='UNCERTAIN_OR_FAILED_NO_RETRY',command=command,error_type=type(error).__name__,error=str(error),binding_sha256=binding_record['sha256']))
        raise
    write_once(RECEIPT,result)
    require(result['status']=='SUBMITTED_EXACTLY_ONCE','Ambiguous submission: preserve claim and inspect, never retry')
    return result


if __name__=='__main__':
    import sys
    require(len(sys.argv)==3,'Need READY record JSON and exact once review record JSON')
    print(json.dumps(main(*[json.loads(Path(x).read_text()) for x in sys.argv[1:]])))
