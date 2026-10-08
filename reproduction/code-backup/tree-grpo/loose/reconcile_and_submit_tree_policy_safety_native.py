"""No blind retry: first inspect the own unique job name, logs and run paths."""
import json,re,shlex,subprocess,datetime
from pathlib import Path
from remote import route
journal=Path('outputs/tree-instruct-300/PPO_SAFETY_NATIVE_DISPATCH.json');record=json.loads(journal.read_text())
assert record['status']=='SUBMISSION_OUTCOME_UNKNOWN_DO_NOT_RESUBMIT'
query=subprocess.run(route('python3 /dev/stdin'),input=Path('work/discover_tree_policy_safety_native.py').read_text(),capture_output=True,text=True)
evidence=Path('outputs/tree-instruct-300/PPO_SAFETY_NATIVE_RECONCILE.json')
assert query.returncode==0,query.stderr
state=json.loads(query.stdout);evidence.write_text(json.dumps(state,indent=2)+'\n')
assert state['squeue_rc']==0 and not state['squeue'].strip() and not state['logs'] and not state['runs'],'Existing job/evidence; reconcile, do not submit'
backup=journal.with_name('PPO_SAFETY_NATIVE_DISPATCH_ATTEMPT1.json');assert not backup.exists();backup.write_bytes(journal.read_bytes())
record.update(status='RETRY_INTENT_AFTER_CONTROLLER_AND_LOG_ABSENCE',reconciled_at=datetime.datetime.now().astimezone().isoformat(),reconciliation_evidence=str(evidence));journal.write_text(json.dumps(record,indent=2)+'\n')
result=subprocess.run(route(shlex.join(record['request'])),text=True,capture_output=True)
record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
answer=result.stdout.strip()
if result.returncode==0 and re.fullmatch(r'\d+(;[^\n]+)?',answer):record.update(status='SUBMITTED_NOT_COMPLETED',job_id=int(answer.split(';')[0]))
else:record['status']='SUBMISSION_OUTCOME_UNKNOWN_DO_NOT_RESUBMIT'
journal.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));assert 'job_id' in record
