"""Only existing failed repeat2/3 move output to the admitted NFS; science stays exact."""
import copy,json,re
from pathlib import Path
from reproduction.trained_tree7b_v2 import contract as original
from reproduction.trained_tree7b_v2.contract import BASE,UID,ALLOCATION,RUNTIME,sha
NBASE=Path('/plm-dhw/zhangjunming/Workspace/Recursive-Tree-GRPO-7B-Ablations-20261020-nfs-recovery-v1')
ROOT=NBASE/'evaluations/tree179-commonv2-remaining-v1'
SCIENCE_SOURCE='1ace724201f75ddf839f942f0d8d4abce94f4b1c'
FAILURES={2: {'uid': '5b22766f-e86a-4ebf-8a78-a4da7b6d0f9d', 'job': 'zhangjunming-tg7-v2-r2-rec1', 'receipt_sha256': '3ff8a52a43683c3f2853b5eaa42b5d95f30033ed094b4eaff4a2cd7aa4adf1df', 'manifest_sha256': 'f5031124dd849969b5bc5fc61e5b0eb3434bed516c9703f114de49f158eb2728', 'archive_sha256': 'fb417ab29491eee6823eccc2f18921c8b1b6bb7d1207d9d5ce10d498370b9c98', 'file_count': 56}, 3: {'uid': 'b8918c34-ba8f-43c1-821d-1131ffe0f2e7', 'job': 'zhangjunming-tg7-v2-r3-rec1', 'receipt_sha256': '2e3794ba078892448b363777d393427e8b31b1ed7cc8c2aea7a5616266aee30e', 'manifest_sha256': '1fd9727411152c228a3688929f03ea3b7672b60b7ca222e1063e85bd13cc6d5c', 'archive_sha256': 'a43f01da7c197d4aef5d62649cc72472edb1e29966a3059d1c6ac02ccbd29024', 'file_count': 8}}
PLAN_SHA={2:'3065ff8c8b4aa554323a651e5bdad69f69b9008f1bb17089bb0792026dc70d76',3:'e5add9fd4a080764ebae6401753838307d7d219505e35608a0c9652ca13d3e64'}
def old_for(i):
 assert type(i) is int and i in (2,3)
 return FAILURES[i]['job'],FAILURES[i]['uid']
def job_name(i):
 old_for(i);return 'zhangjunming-tg7-v2-r'+str(i)+'-nfs1'
def validate_recovery(plan,reference):
 i=plan['repeat'];old_for(i)
 assert reference['repeat']==i and reference['repeat_count']==3 and reference['job']==original.job_name(i)
 assert reference['source_commit']==SCIENCE_SOURCE and reference['root']==str(original.ROOT/('r'+str(i)))
 value=copy.deepcopy(plan);r=value.pop('recovery')
 assert value['job']==job_name(i) and value['root']==str(ROOT/('r'+str(i)))
 value.update(job=reference['job'],root=reference['root']);assert value==reference,'Scientific plan differs'
 assert set(r)=={'superseded_uid','original_plan_sha256','failed_receipt_sha256','entry_source_commit','entry_source','entry_source_files','storage_admission_sha256'}
 assert r['superseded_uid']==FAILURES[i]['uid'] and r['original_plan_sha256']==PLAN_SHA[i]
 assert r['failed_receipt_sha256']==FAILURES[i]['receipt_sha256']
 assert re.fullmatch('[a-f0-9]{40}',r['entry_source_commit']) and r['entry_source_commit']!=SCIENCE_SOURCE
 assert r['entry_source']==str(NBASE/'sources'/r['entry_source_commit'])
 assert re.fullmatch('[a-f0-9]{64}',r['storage_admission_sha256'])
 files=r['entry_source_files'];assert files and all(re.fullmatch('[a-f0-9]{64}',h) for h in files.values())
 for n,h in reference['source_files'].items():assert files[n]==h,n
 for n in files:assert not Path(n).is_absolute() and '..' not in Path(n).parts
 return Path(plan['root'])
def verify(plan):
 i=plan['repeat'];path=original.ROOT/'plans'/('r'+str(i)+'.json');assert sha(path)==PLAN_SHA[i]
 reference=json.loads(path.read_text());root=validate_recovery(plan,reference);rows,_=original.verify(reference)
 f=ROOT/('failed-r'+str(i)+'-preservation.json');assert sha(f)==FAILURES[i]['receipt_sha256'];proof=json.loads(f.read_text())
 assert proof['uid']==FAILURES[i]['uid'] and proof['scheduler']['state']=='FAILED' and proof['no_scientific_acceptance']
 assert proof['archive_sha256']==FAILURES[i]['archive_sha256'] and proof['file_count']==FAILURES[i]['file_count']
 assert sha(ROOT/('failed-r'+str(i)+'-manifest.json'))==FAILURES[i]['manifest_sha256']
 r=plan['recovery'];storage=ROOT/'storage-admission.json';assert sha(storage)==r['storage_admission_sha256'];v=json.loads(storage.read_text())
 assert v['stable_capacity_and_fsync_atomic_rename_verified'] and v['minimum_available_bytes']>6000000000000
 for n,h in r['entry_source_files'].items():assert sha(Path(r['entry_source'])/n)==h,n
 return rows,root
def validate_gate(plan,submission,gate):
 actual=original.validate_gate(plan,submission,gate);i=plan['repeat'];assert plan['job']==job_name(i)
 assert submission['superseded_uid']==gate['superseded_uid']==FAILURES[i]['uid']
 assert submission['entry_source_commit']==gate['entry_source_commit']==plan['recovery']['entry_source_commit']
 return actual
