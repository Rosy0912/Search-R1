"""Independent exact candidate release validation CPU fixtures; never real key/sign/transport."""
import ast,hashlib,importlib.util,json,subprocess,sys,types
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_release',HERE/'publish_first5_release.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def actual_fixture():
 m=json.loads((HERE/'actual-snapshots/3261-20261005T201537333521Z/original-files/full_state/global_step_5/manifest.json').read_bytes());run=m['run_root']
 rec=lambda name,ch:{'path':run+'/review/CPU_FIXTURE_ONLY_'+name+'.json','sha256':ch*64}
 evidence={k:rec(k,'3') for k in p.ROLES}
 a=types.SimpleNamespace(job_id='3261',source=p.SOURCE,tools_sha256=p.TOOLS_SHA,driver_pid=1533312,driver_start='1829085316',manifest_sha256='8c6e17b2af0026b1267a2422c1437aa00461da46e0a1911caea3633bfef473d3',root_review_sha256='4'*64,independent_review_sha256='5'*64,unsigned_payload_remote_path=run+'/review/CPU_FIXTURE_ONLY_unsigned.json',unsigned_payload_sha256='6'*64,evidence_spec_remote_path=run+'/review/CPU_FIXTURE_ONLY_spec.json',evidence_spec_sha256='7'*64)
 def review(role):return {'version':1,'role':role,'status':'PASS_ACTUAL_FIRST5_EVIDENCE','source_commit':p.SOURCE,'job_id':'3261','driver_pid':1533312,'pause_nonce':m['pause_nonce'],'checkpoint_manifest_sha256':a.manifest_sha256,'evidence':evidence}
 payload={k:m[k] for k in p.IDENTITY};payload.update(status='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW',checkpoint_manifest_sha256=a.manifest_sha256,checkpoint_files=m['files'],independent_review=rec('independent','5'),evidence={**evidence,'root_review':rec('root','4'),'preparation_audit':rec('prep','8')},train_log_prefix={'live_path':run+'/train.log','snapshot_path':run+'/review/CPU_FIXTURE_ONLY_prefix.txt','bytes':5,'sha256':'9'*64})
 return a,m,payload,evidence,review('ROOT_AUDITOR'),review('INDEPENDENT_REVIEWER')

def test_actual_manifest_3261_contract_positive_join_only():
 a,m,payload,e,r,i=actual_fixture();run,records=p.check_inputs(a,m,payload,e,r,i)
 assert run==m['run_root'] and len(m['files'])>30
 assert all(k in payload['checkpoint_files'] for k in ('actor/training_state/rank_7.pt','driver/driver.pt','resolved_config'))

@pytest.mark.parametrize('status',['PASS_INDEPENDENT_CPU_OVERLAP_SOFTWARE_ONLY','PASS_INDEPENDENT_SOURCE_BOUND_SOFTWARE_ONLY','ACTUAL_PRE_GPU_INDEPENDENT_REVIEW_PASS_INITIALIZATION_ONLY'])
def test_existing_software_or_pre_gpu_pass_cannot_release_actual_first5(status):
 a,m,payload,e,r,i=actual_fixture();i['status']=status
 with pytest.raises(ValueError,match='external review'):p.check_inputs(a,m,payload,e,r,i)

def test_publication_is_after_fresh_guard_and_signature_check_not_raw_exit0():
 ast.parse(p.LIVE_BODY)
 branch=p.LIVE_BODY[p.LIVE_BODY.index("elif request['action']=='publish':"):]
 assert branch.index('before=validate_live()')<branch.index('gate.verify_rsa_signature(')<branch.index('exclusive(final,raw)')<branch.index('after=validate_live(require_paused=False)')
 assert 'private.pem' not in p.LIVE_BODY and "request_remote('check')" in (HERE/'publish_first5_release.py').read_text()
 assert (HERE/'publish_first5_release.py').read_text().index("request_remote('check')")<(HERE/'publish_first5_release.py').read_text().index('public=key_for_root_execution()')
