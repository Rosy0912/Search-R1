# Prepared test source only. Do not execute before actual immutable V9 gate,
# tools native result/log and real config are installed; missing files FAIL.
"""Actual3188/3172/new bound native bytes; path-only readonly transport mapping, no fake PASS raw."""
import ast,copy,hashlib,json,pathlib
import pytest
import standard_r1_v9_failed_profile_admission as admission
import submit_standard_r1_gpu_v9_diagnostic as submit
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')

def setup(monkeypatch):
 monkeypatch.syspath_prepend(str(BASE/'work/search-r1-32b-grpo100-tools-v9'))
 g=json.loads((BASE/'outputs/standard-r1-profile2-v9-bound-source-allocation-gate.json').read_text())
 native=BASE/'outputs/remote-evidence'/('standard-r1-native'+str(g['actual_source_native_job_id']))
 c=json.loads((BASE/'outputs/standard-r1-profile2-v9-bound-static-config.json').read_text())
 files={g['independent_review']['path']:BASE/'outputs/standard-r1-profile2-v9-bound-independent-review.json',
 g['prior_terminal']['path']:BASE/'outputs/remote-evidence/standard-r1-profile3188-small/standard3188-captured-controller-terminal.json',
 g['prior_failed_result']['path']:BASE/'outputs/remote-evidence/standard-r1-profile3188-small/train-result.json',
 g['prior_run_cleanup']['path']:BASE/'outputs/remote-evidence/standard-r1-profile3188-small/owned-cleanup.json',
 g['streamed_source_native17']['path']:BASE/'outputs/remote-evidence/streamed-adamw-native3172/result.json',
 g['native_tools_result']['path']:native/'result.json',
 g['native_tools_log']['path']:native/'pytest.log'}
 rb=pathlib.Path.read_bytes;rt=pathlib.Path.read_text
 peer=c['telemetry_policy']['actual_independent_native_review']
 bysha={}
 for directory in (BASE/'outputs',BASE/'work/qwen25-32b-window-safety-candidate-v5'):
  for p in directory.rglob('*'):
   if p.is_file() and not p.is_symlink() and p.stat().st_size<=512*1024 and p.suffix in ('.json','.md','.log','.txt'):
    bysha[hashlib.sha256(rb(p)).hexdigest()]=p
 peer_local=bysha[peer['sha256']];files[peer['path']]=peer_local
 review=json.loads(rt(peer_local))
 for name in ('source_manifest','native_cpu_result','native_cpu_log','independent_source_review','inherited_runtime_observation_ref'):
  r=review[name];files[r['path']]=bysha[r['sha256']]
 inherited=json.loads(rt(files[review['inherited_runtime_observation_ref']['path']]))
 for r in inherited['raw_refs']:
  if r['sha256'] not in bysha:
   for candidate in (BASE/'outputs').rglob(pathlib.Path(r['path']).name):
    if candidate.is_file() and hashlib.sha256(rb(candidate)).hexdigest()==r['sha256']:bysha[r['sha256']]=candidate;break
  files[r['path']]=bysha[r['sha256']]
 monkeypatch.setattr(pathlib.Path,'read_bytes',lambda p:rb(files.get(str(p),p)))
 monkeypatch.setattr(pathlib.Path,'read_text',lambda p,*a,**kw:rt(files.get(str(p),p),*a,**kw))
 return c,g

def test_actual_failed3188_with_actual_new_native_results(monkeypatch):
 c,g=setup(monkeypatch)
 cl=admission.validate_failed3188(c,g,'profile2-v9',ROOT)
 assert cl['cleanup']['actual_cgroup_empty'] and cl['sampler_errors'] # historical failure stays failure

@pytest.mark.parametrize('key',['prior_terminal','prior_failed_result','prior_run_cleanup','native_tools_result','native_tools_log','streamed_source_native17','independent_review'])
def test_missing_actual_ref_rejected(monkeypatch,key):
 c,g=setup(monkeypatch);g.pop(key)
 with pytest.raises((KeyError,ValueError)):admission.validate_failed3188(c,g,'profile2-v9',ROOT)

@pytest.mark.parametrize('mutate',[lambda c,g:c.update(mode='formal100'),lambda c,g:c.update(expected_outer_steps=100),lambda c,g:g.update(actual_native_passed=40),lambda c,g:c['cuda_allocator'].update(value='expandable_segments:False'),lambda c,g:g.update(actual_native_skipped=1),lambda c,g:g.update(actual_source_native_job_id=3172),lambda c,g:g.update(source_streamed_adamw_native_job_id=3174),lambda c,g:c['source_manifest'].update(sha256='0'*64),lambda c,g:g['prior_terminal'].update(sha256='0'*64),lambda c,g:g['native_tools_log'].update(sha256='0'*64)])
def test_wrong_scope_source_native_or_raw_rejected(monkeypatch,mutate):
 c,g=setup(monkeypatch);mutate(c,g)
 with pytest.raises((KeyError,ValueError)):admission.validate_failed3188(c,g,'profile2-v9',ROOT)

def test_actual_rendered_branch_calls_strict_admission(monkeypatch):
 c,g=setup(monkeypatch);tree=ast.parse(submit.REMOTE)
 node=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Constant) and n.test.value is True)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-v9-branch','exec'),{'c':c,'g':g,'P':{'attempt':'profile2-v9'},'ROOT':ROOT,'sys':__import__('sys'),'validate_failed3188':admission.validate_failed3188})
 g.pop('prior_failed_result')
 with pytest.raises(KeyError):exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-v9-branch','exec'),{'c':c,'g':g,'P':{'attempt':'profile2-v9'},'ROOT':ROOT,'sys':__import__('sys'),'validate_failed3188':admission.validate_failed3188})

def test_render_only_compiles_and_refuses_other_attempt():
 gate=BASE/'outputs/standard-r1-profile2-v9-bound-source-allocation-gate.json';g=json.loads(gate.read_text());native=BASE/'outputs/remote-evidence'/('standard-r1-native'+str(g['actual_source_native_job_id']));c=json.loads((BASE/'outputs/standard-r1-profile2-v9-bound-static-config.json').read_text())
 code=submit.render(c,gate,'profile2-v9');compile(code,'render-only','exec')
 assert 'UNKNOWN_BEFORE_SUBMISSION' in code and 'profile2-v9' in code and "timeout=30" in code
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


def test_preparation_render_contains_exact_new_policy_and_prior_failure(tmp_path):
 gate=tmp_path/'CPU_RENDER_NOT_APPROVAL.json';gate.write_text('{"scope":"CPU_RENDER_ONLY_NOT_APPROVAL"}')
 code=submit.render({'mode':'profile2'},gate,'profile2-v9');compile(code,'cpu-render-not-executed','exec')
 assert admission.TOOLS_MANIFEST=='ad422ea17014180a88500446faf73175908b3c2fad9a1c8bd50e071ca51fa68d' and admission.NATIVE_TOOLS_COUNT==76
 assert 'backend:native,max_split_size_mb:128' in code and 'validate_policy_review' in code and 'UNKNOWN_BEFORE_SUBMISSION' in code
 assert 'validate_failed3188' in code and 'expandable_segments:True' not in code
 assert admission.FAILED_RESULT_SHA=='358b2286ed69056a2b6a2dceb7de5353846e75ec9cb2b589dbfcd4fbcb409fdb'

def test_preparation_unconditional_real_admission_rejects_missing_proof():
 # Execute the actual embedded branch against deliberately incomplete source evidence.
 c={'mode':'profile2','expected_outer_steps':2,'tools':str(ROOT/'tools/search-r1-32b-grpo100-tools-v9'),'tools_manifest':{'path':str(ROOT/'tools/search-r1-32b-grpo100-tools-v9/files.json'),'sha256':admission.TOOLS_MANIFEST},'candidate':str(ROOT/'tools/search-r1-32b-grpo100-source-v2'),'source_manifest':{'path':str(ROOT/'tools/search-r1-32b-grpo100-source-v2/r1-files.json'),'sha256':admission.SOURCE_MANIFEST},'cuda_allocator':{'environment_variable':'PYTORCH_CUDA_ALLOC_CONF','value':'backend:native,max_split_size_mb:128','backend':'native'}}
 g={'allowed_attempt':'profile2-v9','prior_failed_job_id':3188,'actual_native_passed':76,'actual_native_skipped':0,'tools_manifest':c['tools_manifest'],'source_manifest':c['source_manifest']}
 tree=ast.parse(submit.REMOTE);node=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Constant) and n.test.value is True)
 with pytest.raises(KeyError,match='independent_review'):
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-unconditional-v9','exec'),{'c':c,'g':g,'P':{'attempt':'profile2-v9'},'ROOT':ROOT,'sys':__import__('sys'),'validate_failed3188':admission.validate_failed3188})
