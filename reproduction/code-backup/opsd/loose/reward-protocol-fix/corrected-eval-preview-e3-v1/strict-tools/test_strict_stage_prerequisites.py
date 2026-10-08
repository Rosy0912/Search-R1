"""CPU negative/schema tests only; never synthesize actual500/HF acceptance."""
import importlib.util,hashlib,json,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
import strict_stage_prerequisites as gate

def receipt(tmp_path):
 value=json.loads((HERE/'layered-receipt.UNFILLED.json').read_text());p=tmp_path/'receipt.json';p.write_text(json.dumps(value));return p,value

def save(p,value):p.write_text(json.dumps(value));return gate.digest(p)

def test_frozen_upstream_helpers_exact():gate.verify_upstream_sources()

@pytest.mark.parametrize('role',['teacher','student','HF500','',None])
def test_unknown_roles_reject(tmp_path,role):
 p,value=receipt(tmp_path)
 with pytest.raises(ValueError,match='role'):gate.read_receipt(p,gate.digest(p),role,str(tmp_path),tmp_path)

@pytest.mark.parametrize('role',['base','hf500'])
def test_wrong_role_path_reject(tmp_path,role):
 p,value=receipt(tmp_path)
 with pytest.raises(ValueError,match='Model path'):gate.read_receipt(p,gate.digest(p),role,str(tmp_path/'other'),tmp_path)

@pytest.mark.parametrize('field,value',[('training_commit','0'*40),('formal_job','2999'),('eval_commit',gate.COMMIT),('eval_commit','UNFILLED')])
def test_source_job_or_unfilled_corrected_source_reject(tmp_path,field,value):
 p,record=receipt(tmp_path);record[field]=value;sha=save(p,record);model=tmp_path/gate.MODEL_ROLES['base'].relative_to(gate.OWN)
 with pytest.raises(ValueError):gate.read_receipt(p,sha,'base',str(model),tmp_path)

def test_missing_receipt_file_or_hash_cotamper_reject(tmp_path):
 with pytest.raises(AssertionError):gate.bound_file(dict(path=str(tmp_path/'missing'),sha256='0'*64),tmp_path)
 p,value=receipt(tmp_path);sha=gate.digest(p);p.write_text('{}')
 with pytest.raises(ValueError,match='external SHA'):gate.bound_file(dict(path=str(p),sha256=sha),tmp_path)

@pytest.mark.parametrize('record',[{}, {'path':'UNFILLED','sha256':'UNFILLED'}, {'path':'/tmp/no','sha256':'0'*64,'passed':True}])
def test_not_opaque_passing_flags(record,tmp_path):
 with pytest.raises((ValueError,AssertionError)):gate.bound_file(record,tmp_path)

def test_evidence_outside_own_scope_reject(tmp_path):
 other=tmp_path/'other';other.mkdir();p=tmp_path/'file';p.write_text('real bytes')
 with pytest.raises(AssertionError):gate.bound_file(dict(path=str(p),sha256=gate.digest(p)),other)

def test_model_roles_require_complete_pins(tmp_path):
 model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
 with pytest.raises(ValueError):gate.verify_model_files(dict(directory=str(model),files='UNFILLED'))
 with pytest.raises(ValueError):gate.verify_model_files(dict(directory=str(model),files={'config.json':gate.digest(model/'config.json')}))

@pytest.mark.parametrize('job',['UNFILLED','2994','2996','2999','3007','abc',None])
def test_old_or_unfilled_hf_jobs_rejected(job,tmp_path):
 with pytest.raises(ValueError,match='reload job'):gate.verify_hf500(dict(hf500_reload_job=job),tmp_path)

def test_training_proof_missing_rejects(tmp_path):
 with pytest.raises((ValueError,AssertionError)):gate.verify_training(dict(formal_completion_receipt='UNFILLED'),tmp_path)

def test_runtime_supplement_cannot_be_a_passing_flag(tmp_path):
 with pytest.raises(ValueError):gate.verify_runtime_supplement(dict(runtime_supplement={'passed':True}),tmp_path)

@pytest.mark.parametrize('role',['base','hf500'])
def test_default_entry_exit78_no_future_proof(role):
 result=subprocess.run([sys.executable,str(HERE/'strict_stage_prerequisites.py'),'--receipt','UNFILLED','--receipt-sha256','0'*64,'--model-role',role,'--model-directory',str(gate.MODEL_ROLES[role])],capture_output=True,text=True)
 assert result.returncode==78 and not result.stdout and 'BLOCKED PREVIEW' in result.stderr

def test_upstream_helper_tamper_rejects(monkeypatch):
 original=gate.digest
 monkeypatch.setattr(gate,'digest',lambda p:'0'*64 if Path(p).name=='periodic_validation.py' else original(p))
 with pytest.raises(ValueError,match='helper bytes'):gate.verify_upstream_sources()

def test_shards_without435_index_rejects(tmp_path):
 (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'fake':'one.safetensors'}}))
 with pytest.raises(ValueError,match='435'):gate.shard_keys(tmp_path)

def test_dependency_closure_all_helpers_byte_pinned():gate.verify_dependency_bundle()

def test_helper_dependency_mutation_rejected_even_top_entry_unchanged(monkeypatch):
 original=gate.digest
 monkeypatch.setattr(gate,'digest',lambda p:'0'*64 if Path(p).name=='activation_verification.py' else original(p))
 with pytest.raises(ValueError,match='dependency bytes'):gate.verify_dependency_bundle()

def test_dependency_manifest_cotampering_rejected(monkeypatch):
 original=gate.digest
 monkeypatch.setattr(gate,'digest',lambda p:'0'*64 if Path(p).name=='verifier-dependencies.json' else original(p))
 with pytest.raises(ValueError,match='dependency manifest'):gate.verify_dependency_bundle()
