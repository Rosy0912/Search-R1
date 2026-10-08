"""Collect immutable evidence after real review. Never invent passed or reviewer records."""
import argparse,json,os
from pathlib import Path
from verify_formal_prerequisites import COMMIT,OWN,loads,own_file,digest,verify_receipt,verify_deployed
p=argparse.ArgumentParser();p.add_argument('--roles',required=True);p.add_argument('--output',required=True);p.add_argument('--pilot-job',required=True);p.add_argument('--reload-job',required=True);p.add_argument('--micro',required=True,type=int,choices=(2,4,8));a=p.parse_args()
roles=loads(Path(a.roles).read_text());output=Path(a.output)
assert output.is_absolute() and output.parent.resolve().is_relative_to(OWN.resolve()) and not output.exists()
assert set(roles)=={'pilot_raw_log','pilot_commit','pilot_slurm','pilot_exit','pilot_finished','reload_result','reload_commit','reload_slurm','reload_exit','reload_cleanup','independent_review'}
r={'source_commit':COMMIT,'pilot_job':a.pilot_job,'reload_job':a.reload_job,'picked_local_micro':a.micro,'picked_global_micro':8*a.micro,'evidence':{role:{'path':str(own_file(path,OWN)),'sha256':digest(own_file(path,OWN))} for role,path in roles.items()}}
# The manual independent record must already exist and bind these concrete hashes.
# Neither this collector nor the receipt itself manufactures approval.
verify_deployed()
pending=output.with_name(output.name+'.pending-unaccepted')
assert not pending.exists();pending.write_text(json.dumps(r,indent=2)+'\n')
verify_receipt(pending)
pending.rename(output)
print('CONCRETE_EVIDENCE_COLLECTED_AND_CONTENT_VALIDATED',str(output),digest(output),flush=True)
