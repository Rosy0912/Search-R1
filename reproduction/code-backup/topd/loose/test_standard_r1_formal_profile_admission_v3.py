import pathlib,json
import pytest
from test_standard_r1_formal_profile_admission import fixture
from standard_r1_formal_profile_admission_v3 import profile_admission,job_id

def actual_fixture(tmp,job):
 c,g,put=fixture(tmp)
 # Re-publish every dependent SHA edge in actual contract order.
 review=json.loads(pathlib.Path(g['independent_profile_review']['path']).read_text());cfg=json.loads(pathlib.Path(review['profile_effective_config']['path']).read_text());cfg.update(job=str(job),cgroup='1:freezer:/slurm_h100-11/uid_0/job_%d/step_batch\n0::/system.slice/slurmd.service\n'%job)
 review['profile_effective_config']=put('evidence/effective-runtime-config.json',cfg)
 verifier=json.loads(pathlib.Path(review['profile_resource_verifier']['path']).read_text());verifier['job_id']=job
 for rank,ref in enumerate(verifier['rank_records']):
  raw=json.loads(pathlib.Path(ref['path']).read_text());raw.update(job=str(job),hostname='h100-11',cgroup=cfg['cgroup']);verifier['rank_records'][rank]=put('evidence/actor-resource-audit/actor-updates-rank%d-outer2.json'%rank,raw)
 review['profile_resource_verifier']=put('verifier.json',verifier)
 result=json.loads(pathlib.Path(c['profile_resource_proof']['path']).read_text());result.update(job=str(job),effective_config_sha256=review['profile_effective_config']['sha256']);c['profile_resource_proof']=put('evidence/train-result.json',result)
 semantic=json.loads(pathlib.Path(review['semantic_report']['path']).read_text());semantic.update(job_id=job,train_result=c['profile_resource_proof'],effective_config=review['profile_effective_config']);review['semantic_report']=put('semantic.json',semantic)
 terminal=put('terminal.json',{'job_id':job,'returncode':0,'stdout':'JobId=%d JobState=COMPLETED ExitCode=0:0'%job});review.update(profile_job_id=job,reviewed_profile_result_sha256=c['profile_resource_proof']['sha256'],prior_profile_terminal=terminal)
 g.update(prior_profile_job_id=job,prior_profile_terminal=terminal,independent_profile_review=put('review.json',review));return c,g,put
@pytest.mark.parametrize('job',[3170,9001])
def test_explicit_future_job_all_actual_contract_edges(tmp_path,job):
 c,g,_=actual_fixture(tmp_path,job);assert profile_admission(c,g,tmp_path)['job_id']==job
@pytest.mark.parametrize('bad',[True,'３１７０','+3170',' 3170','03170','3170.0'])
def test_only_positive_ascii_decimal_identity(bad):
 with pytest.raises(ValueError):job_id(bad)

def test_one_rank_different_job_rejected_even_same_counts(tmp_path):
 c,g,put=actual_fixture(tmp_path,9001);review=json.loads(pathlib.Path(g['independent_profile_review']['path']).read_text());v=json.loads(pathlib.Path(review['profile_resource_verifier']['path']).read_text());raw=json.loads(pathlib.Path(v['rank_records'][3]['path']).read_text());raw['job']='3170';v['rank_records'][3]=put('evidence/actor-resource-audit/actor-updates-rank3-outer2.json',raw);review['profile_resource_verifier']=put('verifier.json',v);g['independent_profile_review']=put('review.json',review)
 with pytest.raises(ValueError,match='another job'):profile_admission(c,g,tmp_path)

def test_pending_profile_never_accepted(tmp_path):
 c,g,put=actual_fixture(tmp_path,3170);r=json.loads(pathlib.Path(c['profile_resource_proof']['path']).read_text());r['status']='RUNNING';c['profile_resource_proof']=put('evidence/train-result.json',r);review=json.loads(pathlib.Path(g['independent_profile_review']['path']).read_text());review['reviewed_profile_result_sha256']=c['profile_resource_proof']['sha256'];g['independent_profile_review']=put('review.json',review)
 with pytest.raises(ValueError,match='not complete'):profile_admission(c,g,tmp_path)
