import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
def put(name,value):(HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
previous=WORK/'tree-fresh200-zero-v9-context-free-review'
assert sha(previous/'RECEIPT.json')=='6a369c25886e95dd73a4ed79f0b17e6a1b9bc3b3e1dba9d033237dfc680441f5'
prior=json.loads((previous/'RECEIPT.json').read_text())
frozen={}
for name,expected in prior['artifacts'].items():
    assert sha(previous/name)==expected,name
    frozen[name]=dict(ref(previous/name),matches_previous_seal=True)
frozen['RECEIPT.json']=dict(ref(previous/'RECEIPT.json'),matches_previous_seal=True)
assert frozen['CHILD_SIBLING_QUERY_COUNTEREXAMPLE.json']['sha256']=='51d6b55d322f13397091b8cc2337a5d2cefe136abfaa0795bbcbd2a2d48c3aa4'
v9_audit=json.loads((previous/'BYTE_AUDIT.json').read_text())
old_wrapper={n:dict(ref(Path(x['path'])),matches=sha(x['path'])==x['sha256']) for n,x in v9_audit['wrapper_files'].items()};assert all(x['matches'] for x in old_wrapper.values())
old_helper={n:dict(ref(Path(x['path'])),matches=sha(x['path'])==x['sha256']) for n,x in v9_audit['helper_files'].items()};assert all(x['matches'] for x in old_helper.values())
audit=json.loads((HERE/'BYTE_AUDIT.json').read_text());audit['prior_v9_rejection_files_unchanged']=frozen;audit['old_v9_wrapper_unchanged']=old_wrapper;audit['old_v7_helper_unchanged']=old_helper
audit['owner_v10_initial_failure_and_rerun']={name:ref(WORK/'tree-fresh200-mu-runtime-v10'/name) for name in ['V10_ALL_CPU_FIRST_ACTUAL.stderr','V10_ALL_CPU_FIRST_ACTUAL.stdout','V10_AFFECTED_FIXTURE_RERUN.stderr','V10_AFFECTED_FIXTURE_RERUN.stdout','V8_ACTUAL_TRACKED_RUNTIME_DRY_SELECTION.json','V10_ACTUAL_TRACKED_RUNTIME_DRY_SELECTION.json','SOURCE_CPU_RECEIPT_V10.json']}
put('BYTE_AUDIT.json',audit)
r=json.loads((HERE/'RECEIPT.json').read_text());r['reviewer_context']['continued_original_fork_none_review']=True;r['prior_v9_rejection_preserved']=True;r['prior_v9_review']=ref(previous/'RECEIPT.json');r['prior_v9_byte_audit']=ref(previous/'BYTE_AUDIT.json')
r['source']['acceptance_limited_to_exact_f7_commit']=True;r['source']['future_utf8_repair_source_accepted']=False
r['actual_native_capture_metadata_accepted']=False;r['actual_historical_tokenizer_as_fresh_accepted']=False;r['actual_slurm_executed']=False;r['actual_remote_executed']=False;r['actual_gpu_executed']=False
r['stages']['fresh_mu_calibration']['blockers'].append('Any UTF8 source repair requires new source commit/pins and independent review; f7 acceptance cannot be inherited')
for name in ['BYTE_AUDIT.json','REVIEW.md','HELPER_CONTEXT_COMPATIBILITY_CPU.json','run_independent_cpu.py','INDEPENDENT_CPU_ACTUAL.stdout','INDEPENDENT_CPU_ACTUAL.stderr','seal_review.py','finalize_review.py']:
    r['artifacts'][name]=sha(HERE/name)
put('RECEIPT.json',r)
print(json.dumps(dict(status=r['status'],receipt_sha256=sha(HERE/'RECEIPT.json'),audit_sha256=sha(HERE/'BYTE_AUDIT.json'),review_sha256=sha(HERE/'REVIEW.md'),helper_sha256=sha(HERE/'HELPER_V8_RECEIPT.json'))))
