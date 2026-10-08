"""Submit only independently reviewed synthetic8 GPU checks, once."""
import argparse,datetime,hashlib,json,re,shlex,subprocess
from pathlib import Path
from remote import route
p=argparse.ArgumentParser();p.add_argument('--review-sha256',required=True);a=p.parse_args()
pack=Path('work/tree-policy-safety-native-v2');receipt=Path('work/tree-policy-safety-native-v2-review/PRE_DISPATCH_RECEIPT.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(receipt)==a.review_sha256
r=json.loads(receipt.read_text());assert r['status']=='READY_FOR_SYNTHETIC_NATIVE8_POLICY_SAFETY_V2_GATE_ONLY'
assert r['code_commit']=='0980ec343e90fdd39a91ecc91ba61e594281c063'
assert r['manifest_sha256']==sha(pack/'manifest.json')
assert r['wrapper_sha256']==sha(pack/'start.sh')
base='/dataset/shaoshitong/search-r1-rl/tree-grpo'
args=['sbatch','--parsable','-p','gpu','-w','h100-9','-N','1','--ntasks=1','--gres=gpu:8','--cpus-per-task=32','--mem=256G','--exclusive','--time=00:30:00','--job-name=tree-policy-safety-native-v2','--output='+base+'/logs/tree-policy-safety-native-%j.out','--wrap=bash '+base+'/candidates/tree-policy-safety-native-0980ec3-v2/start.sh']
out=Path('outputs/tree-instruct-300/PPO_SAFETY_NATIVE_V2_DISPATCH.json')
record=dict(status='SUBMIT_INTENT_NOT_JOB_SUCCESS',time=datetime.datetime.now().astimezone().isoformat(),scope='Synthetic8rank tests only; no3B training',request=args,receipt_sha256=a.review_sha256,code_commit=r['code_commit'],manifest_sha256=r['manifest_sha256'])
with out.open('x') as f:json.dump(record,f,indent=2)
result=subprocess.run(route(shlex.join(args)),text=True,capture_output=True)
record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
answer=result.stdout.strip()
if result.returncode==0 and re.fullmatch(r'\d+(;[^\n]+)?',answer):record.update(status='SUBMITTED_NOT_COMPLETED',job_id=int(answer.split(';')[0]))
else:record['status']='SUBMISSION_OUTCOME_UNKNOWN_DO_NOT_RESUBMIT'
out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
assert 'job_id' in record,record
