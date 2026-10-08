import ast,json,pathlib,sys
import pytest,torch
from safetensors.torch import save_file
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import trained_assets as a
import eval64

def checkpoint(root,name,value):
 p=root/name;p.mkdir();save_file({'weight':value},str(p/'one.safetensors'));(p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':'one.safetensors'}}));return p

def test_semantic_dtype_repack_is_not_training(tmp_path):
 x=torch.arange(100,dtype=torch.float32).reshape(10,10)
 b=checkpoint(tmp_path,'base',x.to(torch.bfloat16));m=checkpoint(tmp_path,'model',x)
 with pytest.raises(RuntimeError,match='No actual numerical'):a.semantic_changes(b,m,64)

def test_true_change_is_measured_across_chunk_boundaries(tmp_path):
 x=torch.zeros(33,4);y=x.clone();y[-1,-1]=.25
 result=a.semantic_changes(checkpoint(tmp_path,'base',x),checkpoint(tmp_path,'model',y),32)
 assert result['changed_elements']==1 and result['compared_elements']==132

def test_nonfinite_change_cannot_count_as_learning(tmp_path):
 x=torch.zeros(10,4);y=x.clone();y[5,2]=float('nan')
 with pytest.raises(RuntimeError,match='Nonfinite'):a.semantic_changes(checkpoint(tmp_path,'base',x),checkpoint(tmp_path,'model',y),32)

def test_real100_profile2_and_missing_steps_and_export_identity(tmp_path):
 m=tmp_path/'global_step_100';valid={'actual_completed_outer_steps':list(range(1,101)),'export_step':100,'export_path':str(m)}
 a.check_outer(valid,m)
 for wrong in (dict(valid,export_step=2),dict(valid,actual_completed_outer_steps=list(range(1,100))),dict(valid,export_path=str(tmp_path/'global_step_2'))):
  with pytest.raises(RuntimeError,match='actual100'):a.check_outer(wrong,m)

def test_no_actual_training_binding_rejected_without_loading_model(tmp_path,monkeypatch):
 p=tmp_path/'binding';p.write_text(json.dumps({'status':'INVENTORY_VERIFIED','teacher_kind':'OFFICIAL_QWEN2_5_32B_BASE'}))
 monkeypatch.setattr(a,'inventory',lambda *args:pytest.fail('Base/absent training must not load'))
 with pytest.raises(RuntimeError,match='actual G1'):a.gate(p,tmp_path,set())

def test_trained_quality_is_reported_without_second_cutoff():
 rows=[dict(uid=str(i),sample_position=i,em=0.,f1=0.) for i in range(64)]
 result=eval64.quality_gate(rows)
 assert result['correct']==0 and result['denominator']==64 and result['second_quality_cutoff_applied'] is False
 assert 'candidate_quality_accepted' not in result and 'required_correct' not in result
 rows[-1]['uid']='0'
 with pytest.raises(RuntimeError,match='unique'):eval64.quality_gate(rows)

def test_original_generation_and_mask_and_qa_functions_byte_unchanged():
 here=pathlib.Path(__file__).parent;sealed=here.parent/'qwen25-32b-terminal-barrier-candidate-v2/eval64.py'
 # Source bodies are identical, not a mock implementation of generation.
 before=sealed.read_text();after=(here/'eval64.py').read_text()
 for name in ('StandaloneAdapter','recording_manager_class','load_questions','child_execution_gate'):
  def body(source):
   tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
   return ast.get_source_segment(source,node)
  assert body(before)==body(after)
 assert before[before.index('    readiness ='):before.index('    write_json(output / \'result.json\'')]==after[after.index('    readiness ='):after.index('    write_json(output / \'result.json\'')]


def actual_tokenizer_fixture(tmp_path):
 from tokenizers import Tokenizer
 from tokenizers.models import WordLevel
 from transformers import PreTrainedTokenizerFast
 model=tmp_path/'special-model';base=tmp_path/'special-base';model.mkdir();base.mkdir()
 t=Tokenizer(WordLevel({'word':0,'<eos>':1,'<pad>':2,'<unk>':3},unk_token='<unk>'))
 tok=PreTrainedTokenizerFast(tokenizer_object=t,eos_token='<eos>',pad_token='<pad>',unk_token='<unk>');tok.save_pretrained(model)
 # The native Base config may carry BOS==EOS with actual tokenizer BOS None.
 (base/'config.json').write_text(json.dumps({'bos_token_id':1,'eos_token_id':1,'pad_token_id':None}))
 for name in ('config.json','generation_config.json'):(model/name).write_text(json.dumps({'bos_token_id':1,'eos_token_id':1,'pad_token_id':2}))
 return model,base

def test_actual_tokenizer_legal_base_bos_and_list_eos(tmp_path):
 model,base=actual_tokenizer_fixture(tmp_path)
 assert a.check_special_ids(model,base)=={'bos':None,'eos':1,'pad':2}
 (model/'generation_config.json').write_text(json.dumps({'eos_token_id':[1],'pad_token_id':2}))
 a.check_special_ids(model,base)

@pytest.mark.parametrize('name,key,value',[('config.json','eos_token_id',3),('generation_config.json','eos_token_id',[1,3]),('config.json','pad_token_id',1)])
def test_actual_tokenizer_wrong_stop_ids_rejected(tmp_path,name,key,value):
 model,base=actual_tokenizer_fixture(tmp_path);p=model/name;c=json.loads(p.read_text());c[key]=value;p.write_text(json.dumps(c))
 with pytest.raises(RuntimeError,match='tokenizer/config'):a.check_special_ids(model,base)
