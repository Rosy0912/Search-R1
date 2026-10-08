"""Publish a new independently signed receipt; acceptance requires actual gate after publication."""
import pathlib,json,hashlib,sys
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006');run=root/'runs/bapo-instruct300-hardware-3263';source='b7119791b86951727cea36a1881e8a525c932494'
p=json.load(sys.stdin);assert set(p['files'])=={'HARDWARE3263-FINAL-INDEPENDENT-REVIEW.md','hardware-acceptance.json'}
assert root==root.resolve() and run==run.resolve() and (run/'source-commit.txt').read_text().strip()==source
receipt=json.loads(p['files']['hardware-acceptance.json']['text'])
assert receipt['status']=='PASS_INDEPENDENT_E1_HARDWARE_REVIEW' and receipt['source_commit']==source and receipt['job_id']==3263 and receipt['root']==str(root) and receipt['run_root']==str(run)
assert receipt['real_production_restore_independent_review_passed'] is True and receipt['production_native_v3_roundtrip_independent_review_passed'] is True
assert json.loads((run/'own-process-cleanup.json').read_text())['remaining']==[]
report_name='HARDWARE3263-FINAL-INDEPENDENT-REVIEW.md';raw=p['files'][report_name]['text'].encode();b=receipt['independent_review']
assert b['path']==str(root/'logs'/report_name) and b['sha256']==hashlib.sha256(raw).hexdigest()
if 'bytes'in b:assert b['bytes']==len(raw)
for name in [report_name,'hardware-acceptance.json']:
 item=p['files'][name];raw=item['text'].encode();assert hashlib.sha256(raw).hexdigest()==item['sha256']
 path=root/'logs'/name;assert not path.exists() and not path.is_symlink()
 with path.open('xb')as f:f.write(raw)
print(json.dumps({'status':'NEW_INDEPENDENT_HARDWARE_RECEIPT_PUBLISHED_PENDING_ACTUAL_GATE','job_id':3263,'receipt_sha256':hashlib.sha256(raw).hexdigest()}))
