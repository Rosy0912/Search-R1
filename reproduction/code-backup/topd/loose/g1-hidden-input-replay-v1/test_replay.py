import pathlib,importlib.util,copy,pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
s=importlib.util.spec_from_file_location('hidden_replay',pathlib.Path(__file__).with_name('replay.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
SOURCE=pathlib.Path(__file__).resolve().parents[1]/'g1-staged-calibration-producer-v5/original/ts_probe_dump.py'
def setup():
 tokenizer=Tokenizer(WordLevel({'[PAD]':0,'[UNK]':1,'a':2,'b':3},unk_token='[UNK]'));tokenizer.pre_tokenizer=Whitespace()
 tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='[PAD]',unk_token='[UNK]',padding_side='left')
 expr=m.expression(SOURCE);texts={k:['a','a b','a '*4100] for k in ('commit_ctx','ans_ctx','prompts')};data={}
 for k,values in texts.items():
  data[k]=[]
  # The last row was a distinct batch: replay must preserve those boundaries.
  for chunk in (values[:2],values[2:]):
   enc=tok(chunk,return_tensors='pt',add_special_tokens=False,padding=True,truncation=True,max_length=4096);data[k].append({n:enc[n].tolist() for n in ('input_ids','attention_mask')})
 return tok,expr,texts,data
def test_real_CPU_tokenizer_original_AST_preserves_leftpad_and_truncation():
 tok,expr,texts,data=setup();assert data['prompts'][0]['input_ids'][0]==[0,2];assert len(data['prompts'][1]['input_ids'][0])==4096
 result=m.batches(tok,expr,texts,data);assert all(v['rows']==3 and v['batches']==2 for v in result.values())
@pytest.mark.parametrize('field',['input_ids','attention_mask'])
def test_one_real_recorded_token_or_mask_tamper_rejected(field):
 tok,expr,texts,data=setup();data['commit_ctx'][0][field][0][0]=1
 with pytest.raises(ValueError,match='tokenizer mismatch'):m.batches(tok,expr,texts,data)
def test_missing_original_last_batch_cannot_count_as_full():
 tok,expr,texts,data=setup();data['prompts'].pop()
 with pytest.raises(ValueError,match='Incomplete'):m.batches(tok,expr,texts,data)
def test_rebatching_identical_text_changes_padding_and_must_fail():
 tok,expr,texts,data=setup();data['prompts']=[{'input_ids':[r for b in data['prompts'] for r in b['input_ids']],'attention_mask':[r for b in data['prompts'] for r in b['attention_mask']]}]
 with pytest.raises(ValueError,match='batch shape'):m.batches(tok,expr,texts,data)
def test_original_source_drift_refused_before_tokenizer_expression(tmp_path):
 p=tmp_path/'dump.py';p.write_bytes(SOURCE.read_bytes()+b'\n')
 with pytest.raises(ValueError,match='source drift'):m.expression(p)
