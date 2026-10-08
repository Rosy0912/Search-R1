"""Read exact code/metadata only; write independent review artifacts only."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'work/tree-behavior-mu-utf8-v2'
COMMIT='02b1ea2688ff3fb032ee6699003a7e09d651f166'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def bound(relative,expected):
    path=ROOT/relative;raw=path.read_bytes();assert sha(raw)==expected,(relative,sha(raw))
    return json.loads(raw)
def git(*args):return subprocess.check_output(['git','-C',str(SOURCE),*args])
start=time.process_time()
manifest=bound('work/tree-fresh200-utf8-v2-owner/FROZEN_SOURCE_MANIFEST.json','8980c313dce065830f74e3fe8e9dc17e0cf5d4de5969346adfd3047ce71e5282')
owner=bound('work/tree-fresh200-utf8-v2-owner/UTF8_V2_SOURCE_INTERFACE_CPU_RECEIPT.json','5648c41dc787f95d750157fd876194182c6812665b6e4039db31a0ef86140ce2')
wrapper=bound('work/tree-fresh200-mu-runtime-v10/FILE_HASHES_V10.json','2f0f7df1b37ebbab9111cf3e0da976ff9be69e5fafcb9c5a4802ee09770ce917')
helper=bound('work/tree-fresh200-dispatch-v10/FILE_HASHES_V10.json','01b0883bc194a3aa0d4f4ce100bace352a0f4dc78c117e1bbcbe6d51a0f1bae0')
prior_wrapper=bound('work/tree-fresh200-zero-v10-context-free-review/WRAPPER_V10_RECEIPT.json','c0b3bfc5d3bee6cbed2f54025fc69240d1b9f35b7d7f2a2274a892f8f568b1bd')
prior_helper=bound('work/tree-fresh200-af67976-v10-v9-context-free-review/UTF8_REVIEWER_HELPER_V9_RECEIPT.json','e3d7b81cb94275a7cf3c4c249c7576ebd2bf390530894708dd900bd1159d0ea3')
assert git('rev-parse','HEAD').decode().strip()==COMMIT
assert not git('status','--porcelain=v1')
assert manifest['source_files']==owner['source_files'] and manifest['additional_runtime_files']==owner['additional_runtime_files']
source_files={}
for name,expected in {**manifest['source_files'],**manifest['additional_runtime_files']}.items():
    raw=(SOURCE/name).read_bytes();head=git('show',COMMIT+':'+name)
    assert raw==head and sha(raw)==expected
    source_files[name]=dict(path=str(SOURCE/name),sha256=sha(raw),bytes=len(raw),matches_git_head=True,matches_owner_manifest=True)
    previous=git('show','af67976ec3b544eddaaa68c35411a46b9db7af68:'+name)
    source_files[name]['unchanged_from_af67976']=raw==previous
wrapper_files={}
for name,expected in wrapper['wrapper_files'].items():
    raw=(ROOT/'work/tree-fresh200-mu-runtime-v10'/name).read_bytes()
    assert sha(raw)==expected==prior_wrapper['wrapper_files'][name]['sha256']
    wrapper_files[name]=dict(path=str(ROOT/'work/tree-fresh200-mu-runtime-v10'/name),sha256=sha(raw),bytes=len(raw),matches=True,matches_previous_review=True)
helper_files={}
for name,expected in helper['helper_files'].items():
    raw=(ROOT/'work/tree-fresh200-dispatch-v10'/name).read_bytes();assert sha(raw)==expected
    helper_files[name]=dict(path=str(ROOT/'work/tree-fresh200-dispatch-v10'/name),sha256=sha(raw),bytes=len(raw),matches=True)
    old=(ROOT/'work/tree-fresh200-dispatch-v9'/name).read_bytes()
    if name=='fresh200_dispatch_common.py':
        assert old.replace(b'af67976ec3b544eddaaa68c35411a46b9db7af68',COMMIT.encode())==raw
    else:assert old==raw
assert len(source_files)==19 and len(wrapper_files)==15 and len(helper_files)==3
assert {name for name,entry in source_files.items() if not entry['unchanged_from_af67976']}=={
    'search_r1/baselines/tree_raw_tokens.py','verl/workers/fsdp_workers.py'}
test_files={}
for name,expected in manifest['test_files'].items():
    raw=(SOURCE/name).read_bytes();assert sha(raw)==expected and raw==git('show',COMMIT+':'+name)
    test_files[name]=dict(sha256=sha(raw),bytes=len(raw),matches_git_head=True)
result=dict(git_head=COMMIT,git_tree_sha1=git('rev-parse','HEAD^{tree}').decode().strip(),initial_clean=True,
    final_clean=True,source_interface_count=15,additional_runtime_count=4,source_files=source_files,
    wrapper_files=wrapper_files,helper_files=helper_files,test_files=test_files,
    wrapper_inventory_sha256='2f0f7df1b37ebbab9111cf3e0da976ff9be69e5fafcb9c5a4802ee09770ce917',
    source_manifest_sha256='8980c313dce065830f74e3fe8e9dc17e0cf5d4de5969346adfd3047ce71e5282',
    source_owner_receipt_sha256='5648c41dc787f95d750157fd876194182c6812665b6e4039db31a0ef86140ce2',
    helper_inventory_sha256='01b0883bc194a3aa0d4f4ce100bace352a0f4dc78c117e1bbcbe6d51a0f1bae0',
    prior_wrapper_receipt_scope='Exact unchanged wrapper SOURCE/CPU only; prior f7 does not accept02b1 source',
    prior_helper_receipt_scope='Same prepare/submit bytes; common exact COMMIT-only change independently checked',
    actual_remote_executed=False,actual_slurm_executed=False,actual_gpu_executed=False,
    actual_assets_loaded=False,training_release=False,byte_audit_process_cpu_seconds=time.process_time()-start)
(OUT/'BYTE_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(source_files=len(source_files),wrapper_files=len(wrapper_files),helper_files=len(helper_files),
    source_clean=True,byte_audit_sha256=sha((OUT/'BYTE_AUDIT.json').read_bytes()),cpu_seconds=result['byte_audit_process_cpu_seconds'])))
