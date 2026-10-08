"""CPU admission boundaries only; no actual GPU proof fabricated or submission."""
import ast, copy, json, pathlib
import pytest
import standard_r1_v14_failed_profile_admission as admission
import submit_standard_r1_gpu_v14_profile as submit
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def inputs():
 c=dict(mode='profile2',expected_outer_steps=2,tools=str(ROOT/'tools/search-r1-32b-grpo100-tools-v14'),candidate=str(ROOT/'tools/search-r1-32b-grpo100-source-v4'),cuda_allocator=dict(environment_variable='PYTORCH_CUDA_ALLOC_CONF',value='backend:native,max_split_size_mb:128',backend='native'))
 c['tools_manifest']=dict(path=c['tools']+'/files.json',sha256=admission.TOOLS_MANIFEST)
 c['source_manifest']=dict(path=c['candidate']+'/r1-files.json',sha256=admission.SOURCE_MANIFEST)
 g=dict(allowed_attempt='profile2-v14',prior_failed_job_id=3204,actual_native_passed=87,actual_native_skipped=0,tools_manifest=c['tools_manifest'],source_manifest=c['source_manifest'])
 return c,g

def branch():
 return next(n for n in ast.parse(submit.REMOTE).body if isinstance(n,ast.If) and isinstance(n.test,ast.Constant) and n.test.value is True)
def invoke(c,g):
 exec(compile(ast.Module(body=[branch()],type_ignores=[]),'real-v14-unconditional-admission','exec'),dict(c=c,g=g,P={'attempt':'profile2-v14'},ROOT=ROOT,sys=__import__('sys'),validate_failed3204=admission.validate_failed3204))
def test_actual_branch_missing_independent_proof_rejects():
 c,g=inputs()
 with pytest.raises(KeyError,match='independent_review'):invoke(c,g)
@pytest.mark.parametrize('key',['prior_failed_job_id','allowed_attempt'])
def test_missing_failure_or_attempt_rejects(key):
 c,g=inputs();g.pop(key)
 with pytest.raises((KeyError,ValueError)):invoke(c,g)
@pytest.mark.parametrize('change',[lambda c,g:c.update(mode='formal100'),lambda c,g:c.update(expected_outer_steps=100),lambda c,g:g.update(prior_failed_job_id=3188),lambda c,g:g.update(actual_native_passed=76),lambda c,g:g.update(actual_native_skipped=1),lambda c,g:c['source_manifest'].update(sha256='0'*64)])
def test_actual_admission_wrong_route_source_or_native_rejects(change):
 c,g=inputs();change(c,g)
 with pytest.raises((KeyError,ValueError)):invoke(c,g)
def test_render_has_bound_helper_and_new_failure_without_execution(tmp_path):
 gate=tmp_path/'NOT_APPROVAL.json';gate.write_text('{}');code=submit.render({'mode':'profile2'},gate,'profile2-v14')
 compile(code,'render-only','exec')
 assert "job==3204" in code and "'/job_3204/'" in code
 assert 'validate_failed3204(c,g' in code and 'source_device_correctness' in code
 assert 'UNKNOWN_BEFORE_SUBMISSION' in code and 'expandable_segments:True' not in code
 with pytest.raises(ValueError):submit.render({'mode':'formal100'},gate,'formal100')
@pytest.mark.parametrize('status',['UNKNOWN_BEFORE_SUBMISSION','FAILED'])
def test_actual_existing_unknown_or_failed_record_no_resubmit(status):
 node=next(n for n in ast.walk(ast.parse(submit.REMOTE)) if isinstance(n,ast.If) and ast.unparse(n.test)=='record.exists()')
 class Record:
  def exists(self):return True
  def read_text(self):return json.dumps(dict(status=status,canonical='x',gate_sha256='x'))
 with pytest.raises(AssertionError):exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-record','exec'),dict(record=Record(),json=json,canonical='x',P={'gate_sha256':'x'}))
