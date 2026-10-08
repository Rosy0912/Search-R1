"""Real source-chain contracts, no model load or hardware claims."""
from pathlib import Path
import hashlib,json,ast,re,pytest
W=Path(__file__).parent
E=W/'qwen25-32b-trained-eval64-r1-v2';S=W/'topd-32b-native-score-gate-r1-v2';I=W/'topd-32b-pipeline-ipc-r1-v2';O=W/'topd-32b-shared-owner-r1-v2';P=W/'topd-shared32b-supervisor-r1-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_actual_compiled_chain_same_gate():
 h=sha(S/'score_gate.py')
 for p in (I/'topd32b_shared_pipeline/readiness.py',O/'owner.py',P/'dependencies.py'):
  text=p.read_text();assert h in text and 'topd-32b-native-score-gate-r1-v2' in text
 assert sha(S/'tool-files.json') in (P/'dependencies.py').read_text()
 assert sha(E/'files.json') in (S/'score_gate.py').read_text()
def test_real_owner_import_pins_all_ipc_bytes():
 pins=json.loads((O/'ipc-source-pins.json').read_text())
 assert pins=={p.name:sha(p) for p in (I/'topd32b_shared_pipeline').glob('*.py')}
 assert sha(O/'ipc-source-pins.json') in (O/'import_gate.py').read_text()
@pytest.mark.parametrize('name',['backend.py','protocol.py'])
def test_real_ipc_mathematics_bytes_unchanged(name):
 assert (I/'topd32b_shared_pipeline'/name).read_bytes()==(W/'topd-32b-pipeline-ipc-candidate-v2/topd32b_shared_pipeline'/name).read_bytes()
def test_same64_generation_functions_ast_unchanged():
 old=W/'qwen25-32b-trained-eval64-candidate-v2/eval64.py'
 def functions(p):
  s=p.read_text().replace('TRAINED_G1_SAME64','TRAINED_SEARCH_R1_SAME64').replace('SEARCH_G1_ACTUAL100_EXPORT','SEARCH_R1_ACTUAL100_EXPORT');t=ast.parse(s);return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
 a,b=functions(old),functions(E/'eval64.py')
 # run only changes teacher metadata; all other actual functions exact.
 assert set(a)==set(b)
 for name in a:
  assert a[name]==b[name],name
 assert 'second_quality_cutoff_applied=False' in (E/'eval64.py').read_text()
def test_source_manifests_real_bytes():
 for d in (E,I,O,P):
  m=json.loads((d/'files.json').read_text())
  for name,v in m['files'].items():assert sha(d/name)==v['sha256'] and (d/name).stat().st_size==v['size']
 for name,h in json.loads((S/'tool-files.json').read_text()).items():assert sha(S/name)==h
def test_original_method_and_new_kind_bound():
 for p in (S/'score_gate.py',I/'topd32b_shared_pipeline/readiness.py'):
  assert 'e23b1cdb73736723ff422a9c041f2678b5e8f04cb9a496bd219ad87827b5ef7d' in p.read_text()
 assert 'context_policy.policies(' in (O/'owner.py').read_text()
 assert 'SEARCH_R1_GRPO100_32B' in (P/'dependencies.py').read_text()
