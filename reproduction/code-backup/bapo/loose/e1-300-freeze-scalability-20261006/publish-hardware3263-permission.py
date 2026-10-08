"""Exclusive publication of independently signed allocated verify-only permission."""
import pathlib,json,hashlib,subprocess,sys
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006')
run=root/'runs/bapo-instruct300-hardware-3263'
source='b7119791b86951727cea36a1881e8a525c932494'
payload=json.load(sys.stdin)
assert root==root.resolve() and run==run.resolve() and not run.is_symlink()
assert (run/'WAITING_FOR_INDEPENDENT_RECOVERY_PERMISSION').is_file()
assert (run/'source-commit.txt').read_text().strip()==source==subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain'],text=True)
m=run/'recovery-manifest.json';manifest=json.loads(m.read_text());assert manifest['target_job_id']==3263 and manifest['target_run_root']==str(run)
files=payload['files'];assert set(files)=={'HARDWARE3263-VERIFY-ONLY-INDEPENDENT-REVIEW.md','recovery-permission.json'}
raw_permission=files['recovery-permission.json']['text'].encode();permit=json.loads(raw_permission)
assert permit['status']=='PASS_INDEPENDENT_BAPO_RESUME25_GPU_VERIFY_PERMISSION'
assert permit['target_job_id']==3263 and permit['target_source_commit']==source and permit['target_run_root']==str(run)
assert permit['manifest_sha256']==hashlib.sha256(m.read_bytes()).hexdigest()
assert permit['sampling_continuity']=='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT'
report=run/'HARDWARE3263-VERIFY-ONLY-INDEPENDENT-REVIEW.md';report_raw=files[report.name]['text'].encode()
assert permit['independent_review']=={'path':str(report),'sha256':hashlib.sha256(report_raw).hexdigest(),'bytes':len(report_raw)}
for name in [report.name,'recovery-permission.json']:
 item=files[name];raw=item['text'].encode();assert hashlib.sha256(raw).hexdigest()==item['sha256']
 p=run/name
 assert not p.exists() and not p.is_symlink()
 with p.open('xb') as f:f.write(raw)
print(json.dumps({'status':'INDEPENDENT_VERIFY_ONLY_PERMISSION_PUBLISHED_NOT_GPU_ACCEPTANCE','job':3263,'source':source,'permission_sha256':hashlib.sha256(raw_permission).hexdigest()}))
