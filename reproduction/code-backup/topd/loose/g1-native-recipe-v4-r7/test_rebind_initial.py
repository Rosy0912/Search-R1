import copy,json,pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import rebind_initial as r
ACTUAL=pathlib.Path(__file__).with_name('actual3102-static-config.fixture.json')
def test_actual3102_recipe_only_source_and_external_gate_changed():
 old=json.loads(ACTUAL.read_text());out,changed=r.derive(old,'own/pipeline','newmanifest','newgate')
 assert set(changed)=={'producer','producer_manifest_sha256','pipeline_manifest_sha256','source_bindings','independent_gate','evidence_root','retriever_health_url','retriever_command','refit_args','probe_command'}
 for k in ('base_model_inventory','base64_result_sha256','input_bindings'):assert out[k]==old[k]
 assert out['retriever_health_url'].endswith('/docs') and out['probe_health_url']==old['probe_health_url']
 assert out['independent_gate']['sha256'] is None and r.canonical(old)==r.OLD_CANONICAL

def test_actual3102_data_or_recipe_mutation_refused():
 old=json.loads(ACTUAL.read_text());old['refit_args']['nq']=299
 with pytest.raises(ValueError,match='whole static recipe'):r.derive(old,'p','sha','gate')

def test_fresh_original_refit_args_and_protocol_only_delta():
 old=json.loads(ACTUAL.read_text());out,changed=r.derive(old,'own/pipeline','newmanifest','newgate')
 assert out['retriever_command'][:3]==old['retriever_command'][:3] and out['retriever_command'][4:]==old['retriever_command'][4:]
 assert out['retriever_command'][3]=='own/pipeline/retrieval_server.py'
 oldargs=dict(old['refit_args']);newargs=dict(out['refit_args'])
 assert newargs.pop('work_dir')!=oldargs.pop('work_dir') and newargs.pop('manifest_root')!=oldargs.pop('manifest_root')
 assert newargs==oldargs
 assert 'r4r7' in out['refit_args']['work_dir'] and 'r4r7' in out['refit_args']['manifest_root']

def test_probe_manifest_keeps_same_original_caller_fresh_new_path():
 old=json.loads(ACTUAL.read_text());out,_=r.derive(old,'own/pipeline','sha','gate')
 i=old['probe_command'].index('--manifest')+1
 assert out['probe_command'][:i]==old['probe_command'][:i] and out['probe_command'][i+1:]==old['probe_command'][i+1:]
 assert out['probe_command'][i]==str(pathlib.Path(out['refit_args']['manifest_root'])/('live_'+out['experiment'])/'manifest.json')


def test_new_scoped_producer_binding_is_complete_and_original_math_fields_unchanged():
 old=json.loads(ACTUAL.read_text());out,_=r.derive(old,'own/pipeline','sha','gate')
 assert out['producer'].endswith('/g1-staged-calibration-producer-v6')
 b=next(b for b in out['source_bindings'] if b['kind']=='producer')
 assert b['root']==out['producer'] and b['manifest']=={'path':out['producer']+'/files.json','sha256':out['producer_manifest_sha256']}
 assert out['owner_context']==old['owner_context'] and out['training_command']==old['training_command']
