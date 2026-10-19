"""Exercise the real five-call worker loop on CPU with a deterministic fake LLM."""
import json
import importlib.metadata as metadata
import os
from pathlib import Path
from types import SimpleNamespace,ModuleType
import tempfile
import unittest
from unittest.mock import patch
from . import driver

class Tokenizer:
 eos_token_id=151645;pad_token_id=151643
 def apply_chat_template(self,messages,**kw):return self.encode(messages[0]['content'])
 def encode(self,text,**kw):return [ord(c) for c in text]
 def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)

class Parameter:
 requires_grad=True
 def requires_grad_(self,x):self.requires_grad=x
 def numel(self):return 1

class DriverTests(unittest.TestCase):
 def exercise(self,texts):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'output').mkdir();data=root/'data.jsonl'
   data.write_text(json.dumps(dict(final_index=0,question_id='nq:test:0',source='nq',gold_answers=['Paris'],prompt='Q'))+'\n')
   calls=[];queries=[];parameter=Parameter();tok=Tokenizer()
   class LLM:
    def __init__(self,**kwargs):
     assert os.environ['CUDA_VISIBLE_DEVICES']=='0' and os.environ['CUDA_DEVICE_ORDER']=='PCI_BUS_ID'
     self.llm_engine=SimpleNamespace(model_executor=SimpleNamespace(driver_worker=SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(parameters=lambda:[parameter])))))
    def generate(self,prompt_token_ids,sampling_params,use_tqdm):
     assert sampling_params.temperature==0 and sampling_params.max_tokens==500
     index=len(calls);calls.append(prompt_token_ids)
     return [SimpleNamespace(prompt_token_ids=ids,outputs=[SimpleNamespace(token_ids=tok.encode(texts[index]),finish_reason='stop')]) for ids in prompt_token_ids]
   torch=ModuleType('torch');torch.cuda=SimpleNamespace(device_count=lambda:1);torch.set_grad_enabled=lambda x:None;torch.is_grad_enabled=lambda:False
   transformers=ModuleType('transformers');transformers.AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:tok)
   vllm=ModuleType('vllm');vllm.LLM=LLM;vllm.SamplingParams=lambda **kw:SimpleNamespace(**kw)
   def search(url,q):
    queries.extend(q);return [[{'document':{'contents':'Title\nParis'}}] for _ in q],'hash'
   plan=dict(root=str(root),model='fixture',model_identity='fixture',model_files={},data=str(data),source_commit='fixture',retriever_url='http://fixture')
   with patch.dict('sys.modules',torch=torch,transformers=transformers,vllm=vllm),patch.object(metadata,'version',return_value='0.6.3'),patch.object(driver,'search',side_effect=search),patch.object(driver,'cuda_runtime_uuid',return_value='GPU-fixture'),patch.dict(os.environ,dict(os.environ)):
    driver.worker(plan,0,'GPU-fixture',24000)
   row=json.loads((root/'output/rank-0.predictions.jsonl').read_text())
   proof=json.loads((root/'output/rank-0.complete.json').read_text())
   self.assertEqual(proof['raw_emitted_tokens'],sum(len(x) for x in texts[:len(calls)]))
   self.assertEqual(proof['search_calls'],len(queries))
   self.assertFalse(parameter.requires_grad)
   return row,calls,queries
 def test_four_searches_then_final_answer(self):
  row,calls,queries=self.exercise(['<search>q</search>']*4+['<answer>Paris</answer>'])
  self.assertEqual(len(calls),5);self.assertEqual(len(queries),4)
  self.assertEqual(row['em'],1);self.assertEqual(row['search_calls'],4)
  self.assertTrue(row['events'][-1]['final_only']);self.assertFalse(row['events'][-1]['actual_retrieval'])
 def test_no_http_for_fifth_search(self):
  row,calls,queries=self.exercise(['<search>q</search>']*5)
  self.assertEqual(len(calls),5);self.assertEqual(len(queries),4)
  self.assertIsNone(row['prediction']);self.assertEqual(row['em'],0)
 def test_answer_em_not_gated_by_format(self):
  row,calls,queries=self.exercise(['unclosed</information><answer>Paris</answer>'])
  self.assertEqual(len(calls),1);self.assertEqual(queries,[])
  self.assertEqual(row['em'],1);self.assertEqual(row['format_valid'],0)
 def test_invalid_action_has_final_opportunity(self):
  row,calls,queries=self.exercise(['invalid']*4+['<answer>Paris</answer>'])
  self.assertEqual(len(calls),5);self.assertEqual(row['em'],1);self.assertEqual(queries,[])

if __name__=='__main__':unittest.main()
