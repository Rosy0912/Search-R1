import ast,pathlib
import pytest
from standard_r1_v5_failed_profile_admission import validate_failed3163
from test_standard_r1_v5_diagnostic_submit import setup
HERE=pathlib.Path(__file__).resolve().parent

def execute_actual_admission_branch(c,g,attempt,root):
 tree=ast.parse((HERE/'submit_standard_r1_gpu_v5_diagnostic_v2.py').read_text());raw=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'import fcntl,hashlib,json' in n.value);rt=ast.parse(raw)
 node=next(n for n in rt.body if isinstance(n,ast.If) and "P['attempt'] == 'profile2-v5'" in ast.unparse(n.test))
 exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-rendered-admission-branch','exec'),{'c':c,'g':g,'P':{'attempt':attempt},'ROOT':root,'validate_failed3163':validate_failed3163})

@pytest.mark.parametrize('key',['prior_run_cleanup','prior_failed_job_id','prior_terminal'])
def test_actual_rendered_branch_missing3163_evidence_rejected(tmp_path,key):
 c,g=setup(tmp_path);g.pop(key)
 with pytest.raises((ValueError,KeyError)):execute_actual_admission_branch(c,g,'profile2-v5',tmp_path)

def test_actual_rendered_complete_named_diagnostic_passes(tmp_path):
 c,g=setup(tmp_path);execute_actual_admission_branch(c,g,'profile2-v5',tmp_path)

def test_old3159_other_attempt_does_not_enter3163_branch(tmp_path):
 execute_actual_admission_branch({}, {'prior_failed_job_id':3159},'profile2-v4',tmp_path)
