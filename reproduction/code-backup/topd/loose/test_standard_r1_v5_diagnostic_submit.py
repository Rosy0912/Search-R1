import pathlib,json,hashlib
import pytest
from standard_r1_v5_failed_profile_admission import validate_failed3163,TOOLS_MANIFEST
HERE=pathlib.Path(__file__).resolve().parent.parent

def setup(tmp):
 logs=tmp/'logs';logs.mkdir()
 def put(name,raw):
  p=logs/name;p.write_bytes(raw);return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()}
 terminal=put('standard3163-captured-controller-terminal.json',(HERE/'outputs/standard3163-captured-controller-terminal.json').read_bytes())
 cleanup=put('owned-cleanup.json',(HERE/'outputs/remote-evidence/standard-r1-profile3163/owned-cleanup.json').read_bytes())
 manifest={'path':str(tmp/'tools/search-r1-32b-grpo100-tools-v5/files.json'),'sha256':TOOLS_MANIFEST};source={'path':'source','sha256':'SOURCE_CONTRACT'}
 review=put('review.json',json.dumps({'status':'PASS','scope':'SEARCH_R1_32B_PROFILE2_SOURCE_AND_ALLOCATION_REVIEW','reviewed_static_recipe_sha256':'CPU_TOY','actual_native_passed':40,'actual_native_skipped':0,'actual_source_native_job_id':999}).encode())
 c={'mode':'profile2','expected_outer_steps':2,'tools':str(tmp/'tools/search-r1-32b-grpo100-tools-v5'),'tools_manifest':manifest,'source_manifest':source}
 g={'prior_failed_job_id':3163,'prior_terminal':terminal,'prior_run_cleanup':cleanup,'allowed_attempt':'profile2-v5','actual_native_passed':40,'actual_native_skipped':0,'actual_source_native_job_id':999,'independent_review':review,'reviewed_static_recipe_sha256':'CPU_TOY','tools_manifest':manifest,'source_manifest':source}
 return c,g

def test_actual_failed3163_bytes_accept_only_diagnostic_contract(tmp_path):
 c,g=setup(tmp_path);cl=validate_failed3163(c,g,'profile2-v5',tmp_path);assert cl['own_gpu_records_remaining']==[]
@pytest.mark.parametrize('mode,attempt',[('train100','profile2-v5'),('profile2','profile2-v4'),('train100','train100-v5')])
def test_no_formal_or_wrong_attempt(tmp_path,mode,attempt):
 c,g=setup(tmp_path);c['mode']=mode
 with pytest.raises(ValueError):validate_failed3163(c,g,attempt,tmp_path)
@pytest.mark.parametrize('field,value',[('actual_native_passed',36),('actual_native_skipped',1),('prior_failed_job_id',3159)])
def test_bad_native_or_job_rejected(tmp_path,field,value):
 c,g=setup(tmp_path);g[field]=value
 with pytest.raises(ValueError):validate_failed3163(c,g,'profile2-v5',tmp_path)
def test_actual_terminal_mutation_rejected(tmp_path):
 c,g=setup(tmp_path);p=pathlib.Path(g['prior_terminal']['path']);p.write_text(p.read_text().replace('JobState=FAILED','JobState=COMPLETED'))
 with pytest.raises(ValueError):validate_failed3163(c,g,'profile2-v5',tmp_path)
def test_old_helper_body_keeps_unknown_and_whole_cgroup_checks():
 s=(HERE/'work/submit_standard_r1_gpu_v5_diagnostic.py').read_text()
 assert "status='UNKNOWN_BEFORE_SUBMISSION'" in s and "job in (3159,3163)" in s
 assert "'/job_3163/' in cg" in s and 'fields[19]' in s and 'validate_failed3163(c,g' in s
