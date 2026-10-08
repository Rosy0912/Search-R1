# Prepared test source only. Do not execute before actual immutable V8 gate,
# tools native result/log and real config are installed; missing files FAIL.
"""Actual3181/3172/new bound native bytes; path-only readonly transport mapping, no fake PASS raw."""
import ast,copy,hashlib,json,pathlib
import pytest
import standard_r1_v8_failed_profile_admission as admission
import submit_standard_r1_gpu_v8_diagnostic as submit
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')

def setup(monkeypatch):
 g=json.loads((BASE/'outputs/standard-r1-profile2-v8-bound-source-allocation-gate.json').read_text())
 native=BASE/'outputs/remote-evidence'/('standard-r1-native'+str(g['actual_source_native_job_id']))
 c=json.loads((native/'static-config.json').read_text())
 files={g['independent_review']['path']:BASE/'outputs/standard-r1-profile2-v8-bound-independent-review.json',
 g['prior_terminal']['path']:BASE/'outputs/standard3181-captured-controller-terminal.json',
 g['prior_failed_result']['path']:BASE/'outputs/remote-evidence/standard-r1-profile3181-small/train-result.json',
 g['prior_run_cleanup']['path']:BASE/'outputs/remote-evidence/standard-r1-profile3181-small/owned-cleanup.json',
 g['streamed_source_native17']['path']:BASE/'outputs/remote-evidence/streamed-adamw-native3172/result.json',
 g['native_tools_result']['path']:native/'result.json',
 g['native_tools_log']['path']:native/'pytest.log'}
 rb=pathlib.Path.read_bytes;rt=pathlib.Path.read_text
 monkeypatch.setattr(pathlib.Path,'read_bytes',lambda p:rb(files.get(str(p),p)))
 monkeypatch.setattr(pathlib.Path,'read_text',lambda p,*a,**kw:rt(files.get(str(p),p),*a,**kw))
 return c,g

def test_actual_failed3181_with_actual_new_native_results(monkeypatch):
 c,g=setup(monkeypatch)
 cl=admission.validate_failed3181(c,g,'profile2-v8',ROOT)
 assert cl['cleanup']['actual_cgroup_empty'] and cl['sampler_errors'] # failure stays failure

@pytest.mark.parametrize('key',['prior_terminal','prior_failed_result','prior_run_cleanup','native_tools_result','native_tools_log','streamed_source_native17','independent_review'])
def test_missing_actual_ref_rejected(monkeypatch,key):
 c,g=setup(monkeypatch);g.pop(key)
 with pytest.raises((KeyError,ValueError)):admission.validate_failed3181(c,g,'profile2-v8',ROOT)

@pytest.mark.parametrize('mutate',[lambda c,g:c.update(mode='formal100'),lambda c,g:c.update(expected_outer_steps=100),lambda c,g:g.update(actual_native_passed=40),lambda c,g:c['cuda_allocator'].update(value='expandable_segments:False'),lambda c,g:g.update(actual_native_skipped=1),lambda c,g:g.update(actual_source_native_job_id=3172),lambda c,g:g.update(source_streamed_adamw_native_job_id=3174),lambda c,g:c['source_manifest'].update(sha256='0'*64),lambda c,g:g['prior_terminal'].update(sha256='0'*64),lambda c,g:g['native_tools_log'].update(sha256='0'*64)])
def test_wrong_scope_source_native_or_raw_rejected(monkeypatch,mutate):
 c,g=setup(monkeypatch);mutate(c,g)
 with pytest.raises((KeyError,ValueError)):admission.validate_failed3181(c,g,'profile2-v8',ROOT)

def test_actual_rendered_branch_calls_strict_admission(monkeypatch):
 c,g=setup(monkeypatch);tree=ast.parse(submit.REMOTE)
 node=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Constant) and n.test.value is True)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-v8-branch','exec'),{'c':c,'g':g,'P':{'attempt':'profile2-v8'},'ROOT':ROOT,'validate_failed3181':admission.validate_failed3181})
 g.pop('prior_failed_result')
 with pytest.raises(KeyError):exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-v8-branch','exec'),{'c':c,'g':g,'P':{'attempt':'profile2-v8'},'ROOT':ROOT,'validate_failed3181':admission.validate_failed3181})

def test_render_only_compiles_and_refuses_other_attempt():
 gate=BASE/'outputs/standard-r1-profile2-v8-bound-source-allocation-gate.json';g=json.loads(gate.read_text());native=BASE/'outputs/remote-evidence'/('standard-r1-native'+str(g['actual_source_native_job_id']));c=json.loads((native/'static-config.json').read_text())
 code=submit.render(c,gate,'profile2-v8');compile(code,'render-only','exec')
 assert 'UNKNOWN_BEFORE_SUBMISSION' in code and 'profile2-v8' in code and "timeout=30" in code
 with pytest.raises(ValueError):submit.render(c,gate,'formal100')

def record_branch():
 tree=ast.parse(submit.REMOTE)
 return next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='record.exists()')

class Record:
 def __init__(self,data):self.data=data
 def exists(self):return True
 def read_text(self):return json.dumps(self.data)

@pytest.mark.parametrize('status',['UNKNOWN_BEFORE_SUBMISSION','FAILED','SUBMITTED'])
def test_actual_existing_record_never_resubmits(status):
 node=record_branch();data={'status':status,'canonical':'old' if status=='SUBMITTED' else 'good','gate_sha256':'good'}
 # No subprocess name in globals: any accidental submission would fail this test.
 with pytest.raises(AssertionError):exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-record-branch','exec'),{'record':Record(data),'json':json,'canonical':'good','P':{'gate_sha256':'good'}})

def test_known_exact_submitted_record_returns_without_sbatch():
 node=record_branch();d={'status':'SUBMITTED','canonical':'good','gate_sha256':'good','job_id':9999}
 env={'record':Record(d),'json':json,'canonical':'good','P':{'gate_sha256':'good'}}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-record-branch','exec'),env)
 assert env['d']==d
