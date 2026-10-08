import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

root=Path(__file__).resolve().parent.parent
os.chdir(root)
def submit(args):
    result=subprocess.check_output(['sbatch','--parsable']+args,text=True).strip().split(';')[0]
    if not result.isdigit():
        raise RuntimeError('Unexpected Slurm receipt: '+result)
    return result
pilot=submit(['scripts/lumia_pilot.sbatch'])
full=submit(['--dependency=afterok:'+pilot,'--kill-on-invalid-dep=yes',
             '--export=ALL,PILOT_DIR='+str(root/'runs'/('pilot-'+pilot)), 'scripts/lumia_full.sbatch'])
receipt={'created_at_utc':datetime.now(timezone.utc).isoformat(),'pilot_id':int(pilot),'full_id':int(full),
         'pilot_dir':str(root/'runs'/('pilot-'+pilot)),'full_dir':str(root/'runs'/('paper-3b-'+full)),
         'full_launch_condition':'afterok:'+pilot, 'gpus':4,'partition':'L40S','qos':'normal'}
(root/'runs'/('submission-'+pilot+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
(root/'runs'/'latest_submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
