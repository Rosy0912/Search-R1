"""Meaningful CPU tensors/contracts; never actual32B/model-forward evidence."""
import pathlib,json,importlib.util,pytest,torch
from safetensors.torch import save_file
P=pathlib.Path(__file__).with_name('review.py');s=importlib.util.spec_from_file_location('bounded_hf_review',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def weights(tmp_path,change=False,dtype=torch.float32,nonfinite=False):
 dirs=[tmp_path/'base',tmp_path/'changed']
 for d in dirs:
  d.mkdir();(d/'config.json').write_text(json.dumps(dict(model_type='qwen2',vocab_size=3,hidden_size=2,tie_word_embeddings=False)))
 x=torch.tensor([[1.0001,2.],[3.,4.],[5.,6.]])
 y=x.to(dtype)
 if change:y=y.clone();y[1,1]+=.5
 if nonfinite:y=y.clone();y[2,0]=float('nan')
 save_file({'model.embed_tokens.weight':x,'lm_head.weight':x.clone()},str(dirs[0]/'model.safetensors'))
 save_file({'model.embed_tokens.weight':y,'lm_head.weight':y.clone()},str(dirs[1]/'model.safetensors'))
 return dirs
def test_true_semantic_delta_reports_all_names_and_elements(tmp_path):
 r=m.load('original_semantic_gate').compare_weights(*weights(tmp_path,True));assert r['changed_tensor_names']==['lm_head.weight','model.embed_tokens.weight'];assert r['compared_elements_including_tied_aliases']==12;assert r['max_abs_difference']==.5
@pytest.mark.parametrize('case',['identical','dtypeonly','nonfinite'])
def test_packaging_cast_and_nonfinite_are_not_training(tmp_path,case):
 d=weights(tmp_path,dtype=torch.bfloat16 if case=='dtypeonly' else torch.float32,nonfinite=case=='nonfinite')
 with pytest.raises(ValueError):m.load('original_semantic_gate').compare_weights(*d)
def test_original_helper_source_tamper_rejected(tmp_path,monkeypatch):
 monkeypatch.setitem(m.PINS,'original_semantic_gate.py','0'*64)
 with pytest.raises(ValueError,match='helper source'):m.load('original_semantic_gate')
def inventory():
 names={'config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors.index.json'}|{'part%d.safetensors'%i for i in range(17)}
 return dict(tensor_count=771,files={n:{'size':1,'sha256':'f'*64} for n in names},shards=sorted(n for n in names if n.endswith('.safetensors')))
@pytest.mark.parametrize('case',['missing','extra','schema','drift'])
def test_strict_complete_hf_set_rejects_bad_receipts(case):
 import copy
 e=inventory();a=copy.deepcopy(e)
 if case=='missing':a['files'].pop(a['shards'][0])
 if case=='extra':a['files']['unindexed.safetensors']={'size':1,'sha256':'a'*64};e['files']=a['files']
 if case=='schema':a['tensor_count']=770
 if case=='drift':a['files']['config.json']['sha256']='0'*64
 with pytest.raises(ValueError):m.strict_inventory(a,e,True)
def test_profile2_must_complete_and_cleanup_before_any_scan(tmp_path,monkeypatch):
 records={'r':{'status':'FAILED','cleanup_passed':False,'error':'real failure'},'c':{}}
 monkeypatch.setattr(m,'ref',lambda r:(tmp_path,records[r['path']]))
 binding=dict(scope='REQUEST_BOUNDED_CPU_HF_SEMANTIC_SCAN_NOT_APPROVAL',mode='profile2',train_result={'path':'r'},effective_config={'path':'c'})
 with pytest.raises(ValueError,match='runtime/cleanup'):m.binding_records(binding)
def test_exclusive_result_does_not_replace_original(tmp_path):
 p=tmp_path/'receipt.json';m.publish(p,{'actual':'first'})
 with pytest.raises(ValueError):m.publish(p,{'actual':'second'})
 assert json.loads(p.read_text())=={'actual':'first'}

@pytest.mark.parametrize('name',['config.json','generation_config.json'])
@pytest.mark.parametrize('case',['wrong_eos','wrong_pad','bool_eos','empty_eos','wrong_bos'])
def test_actual_special_ids_reject_semantic_stop_drift(name,case):
 from types import SimpleNamespace
 tok=SimpleNamespace(eos_token_id=151643,pad_token_id=151643,bos_token_id=None)
 cfg=dict(eos_token_id=151643,pad_token_id=None,bos_token_id=151643)
 if case=='wrong_eos':cfg['eos_token_id']=151645
 if case=='wrong_pad':cfg['pad_token_id']=0
 if case=='bool_eos':cfg['eos_token_id']=True
 if case=='empty_eos':cfg['eos_token_id']=[]
 if case=='wrong_bos':cfg['bos_token_id']=42
 with pytest.raises(ValueError):m.check_special_ids(cfg,tok,{'bos_token_id':151643},name)
def test_official_bos_override_and_legal_eos_list_preserved():
 from types import SimpleNamespace
 tok=SimpleNamespace(eos_token_id=151643,pad_token_id=151643,bos_token_id=None)
 m.check_special_ids(dict(eos_token_id=[151643],pad_token_id=151643,bos_token_id=151643),tok,{'bos_token_id':151643},'generation_config.json')

@pytest.mark.parametrize('value',[True,False,0,-1,'','-1','1.0',' 1','１','0',None])
def test_job_identity_rejects_non_decimal_or_nonpositive(value):
 with pytest.raises(ValueError):m.job_id(value)
def test_real_producer_environment_decimal_job_identity():
 import os
 actual=dict(job=os.environ.get('SLURM_JOB_ID','3163'))
 assert m.job_id(actual['job'])==m.job_id(3163)
 assert m.job_id('3164')!=m.job_id(actual['job'])
