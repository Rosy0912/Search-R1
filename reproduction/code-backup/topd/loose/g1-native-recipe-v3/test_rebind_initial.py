import copy,json,pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import rebind_initial as r
ACTUAL=pathlib.Path(__file__).with_name('actual3102-static-config.fixture.json')
def test_actual3102_recipe_only_source_and_external_gate_changed():
 old=json.loads(ACTUAL.read_text());out,changed=r.derive(old,'own/pipeline','newmanifest','newgate')
 assert set(changed)=={'pipeline_manifest_sha256','source_bindings','independent_gate'}
 for k in ('refit_args','base_model_inventory','base64_result_sha256','input_bindings','retriever_command','evidence_root'):assert out[k]==old[k]
 assert out['independent_gate']['sha256'] is None and r.canonical(old)==r.OLD_CANONICAL

def test_actual3102_data_or_recipe_mutation_refused():
 old=json.loads(ACTUAL.read_text());old['refit_args']['nq']=299
 with pytest.raises(ValueError,match='whole static recipe'):r.derive(old,'p','sha','gate')
