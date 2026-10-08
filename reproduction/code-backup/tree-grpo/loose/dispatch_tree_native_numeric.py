import hashlib,json,re,shlex,subprocess
from pathlib import Path
from remote import route
receipt=Path('work/tree-numeric-candidate-review/NATIVE_PRE_DISPATCH_RECEIPT.json')
assert hashlib.sha256(receipt.read_bytes()).hexdigest()=='e5a1d126ff116afadd9de8400938eb2c79eccb83f74e246942f0ac83aa63109e'
r=json.loads(receipt.read_text());assert r['status']=='READY_FOR_PRE_DISPATCH_NATIVE_NUMERIC_GATE_ONLY'
assert r['candidate_commit']=='296217a4b676a26a92fc729f5d9882b785dbfa96'
assert hashlib.sha256(Path('work/start_tree_native_numeric_296217a.sh').read_bytes()).hexdigest()==r['wrapper_sha256']
assert hashlib.sha256(Path('work/tree-native-numeric-296217a/manifest.json').read_bytes()).hexdigest()==r['manifest_sha256']
out=Path('outputs/evidence/tree-native-numeric-dispatch.json');assert not out.exists(),'Dispatch already recorded'
root='/dataset/shaoshitong/search-r1-rl/tree-grpo'
args=['sbatch','--parsable','-p','gpu','-w','h100-9','-N','1','--ntasks=1','--gres=gpu:8','--cpus-per-task=32','--mem=256G','--exclusive','--time=00:30:00','--job-name=tree-numeric-296217a','--output='+root+'/logs/tree-native-numeric-%j.out','--wrap=bash '+root+'/tools/start_tree_native_numeric_296217a.sh']
answer=subprocess.check_output(route(shlex.join(args),'h100-16'),text=True).strip();assert re.fullmatch(r'\d+(;[^\n]+)?',answer),answer
record=dict(job_id=int(answer.split(';')[0]),candidate_commit=r['candidate_commit'],receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),manifest_sha256=r['manifest_sha256'],wrapper_sha256=r['wrapper_sha256'],slurm_request=args,scope='Only native numerical/storage fixtures; not new training',scheduler_state_at_submit='UNKNOWN_SUBMITTED_NOT_SUCCESS')
out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
