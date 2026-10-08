"""Submission contract fixtures only, not independent telemetry approval."""
import hashlib,importlib.util,json
from pathlib import Path
import pytest
P=Path(__file__).parent/'submit_window_eval64_v3_gpu.py'
spec=importlib.util.spec_from_file_location('submit64_v3_contract',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def receipt(tmp_path):
 p=tmp_path/'CONTRACT_ONLY_NOT_APPROVAL.json'
 d=json.loads((P.parent.parent/'outputs/qwen25-32b-window-dualburst3074-telemetry-native-review.json').read_text())
 p.write_text(json.dumps(d))
 return p,hashlib.sha256(p.read_bytes()).hexdigest()

def test_generated_python_parses_and_gates_precede_single_sbatch(tmp_path):
 p,h=receipt(tmp_path);remote=m.build_remote(p,h)
 code=remote.split("python3 - <<'INNER'\n",1)[1].rsplit('\nINNER',1)[0];compile(code,'contract','exec')
 assert code.count("['sbatch'")==1
 assert code.index('UNKNOWN_BEFORE_SUBMISSION')<code.index("['sbatch'")
 assert code.index("dd16dd1e")<code.index("['sbatch'")
 assert code.index("'actual_zero_own_remeasurements'")<code.index("['sbatch'")
 assert 'CONTRACT_ONLY_NOT_APPROVAL.json' in code and 'REVIEW_SHA' not in code
 assert 'native3074-review.json' not in code

@pytest.mark.parametrize('fault',['sha','job','status','runtime','guard','unsafe_name'])
def test_invalid_binding_fails_before_remote(tmp_path,fault):
 p,h=receipt(tmp_path)
 if fault=='sha':h='0'*64
 elif fault=='unsafe_name':new=tmp_path/'bad`name.json';p.rename(new);p=new
 else:
  d=json.loads(p.read_text());key={'job':'native_job_id','status':'status','runtime':'source_sha256','guard':'adapter_sha256'}[fault];d[key]='wrong';p.write_text(json.dumps(d));h=hashlib.sha256(p.read_bytes()).hexdigest()
 with pytest.raises(AssertionError):m.build_remote(p,h)
