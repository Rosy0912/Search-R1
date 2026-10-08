"""Target-only actual tokenizer/dataset checks; absence is failure when explicitly required."""
import hashlib,json,os,re,unittest
from pathlib import Path

class ActualPromptDataset(unittest.TestCase):
 def test_calibration_context_and_question_identity_match_actual_grpo_dataset(self):
  base=Path(os.environ.get('SEARCH_G1_BASE','/data/search-g1'))
  if not (base/'data/calibration-300.parquet').exists():
   if os.environ.get('SEARCH_G1_REQUIRE_REAL_DATASET')=='1':self.fail('Required real target data missing')
   self.skipTest('Actual target dataset required for this integration check')
  import pandas as pd
  from transformers import AutoTokenizer
  from verl.utils.dataset.rl_dataset import RLHFDataset
  from ts_probe_dump import parse_row
  tok=AutoTokenizer.from_pretrained(base/'assets/qwen2.5-3b-instruct',local_files_only=True,padding_side='left')
  dataset=RLHFDataset(str(base/'data/calibration-300.parquet'),tok,max_prompt_length=4096,cache_dir=str(base/'cache/dataset'))
  frame=pd.read_parquet(base/'data/calibration-300.parquet');audit=json.loads((base/'data/audit.json').read_text())
  ids=[]
  for i,(_,row) in enumerate(frame.iterrows()):
   prompt,golds,raw=parse_row(row,tok);expected=tok.encode(prompt,add_special_tokens=False)
   item=dataset[i];actual=item['input_ids'][item['attention_mask'].bool()].tolist()
   self.assertEqual(expected,actual);self.assertLessEqual(len(expected),2048)
   starts=list(re.finditer(r'Question:\s*',raw));question=raw[starts[-1].end():]
   ids.append(hashlib.sha256(re.sub(r'\s+',' ',question).strip().casefold().encode()).hexdigest())
  self.assertEqual(len(frame),300);self.assertEqual(set(ids),set(audit['calibration_question_ids']))
  self.assertFalse(set(ids)&set(audit['validation_question_ids']))
if __name__=='__main__':unittest.main()
