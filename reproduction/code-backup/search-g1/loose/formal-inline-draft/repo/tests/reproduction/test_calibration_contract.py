import ast,json,pathlib,re,string,unittest,types
ROOT=pathlib.Path(__file__).resolve().parents[2]
def functions(path,names,ns):
 tree=ast.parse(path.read_text());selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
 return ns
class CalibrationContract(unittest.TestCase):
 def test_calibration_uses_same_instruct_chat_and_keeps_raw_question(self):
  ns=functions(ROOT/'ts_probe_dump.py',{'parse_row'},{'np':types.SimpleNamespace(ndarray=type('Array',(),{})),'json':json})
  class Tokenizer:
   chat_template='qwen'
   def apply_chat_template(self,chat,add_generation_prompt,tokenize):
    self.args=(chat,add_generation_prompt,tokenize)
    return '<user>'+chat[0]['content']+'</user><assistant>'
  tok=Tokenizer();chat=[{'role':'user','content':'Question: Where?'}]
  prompt,golds,raw=ns['parse_row']({'prompt':chat,'reward_model':{'ground_truth':{'target':['Paris']}}},tok)
  self.assertEqual(tok.args,(chat,True,False));self.assertEqual(prompt,'<user>Question: Where?</user><assistant>');self.assertEqual(raw,'Question: Where?');self.assertEqual(golds,['Paris'])
 def test_calibration_and_reward_have_identical_unicode_and_punctuation_em(self):
  ns=functions(ROOT/'verl/utils/reward_score/qa_em.py',{'normalize_answer','em_check'},{'re':re,'string':string})
  functions(ROOT/'ts_probe_dump.py',{'normalize','em_score'},ns)
  for answer,gold in [('U.S.','US'),('北京','北京'),('The Côte','Côte'),('Rome','Paris')]:
   self.assertEqual(ns['em_score'](answer,[gold]),ns['em_check'](answer,[gold]))
 def test_acceptance_gate_precedes_next_rollout_counter(self):
  source=(ROOT/'verl/trainer/ppo/ray_trainer.py').read_text()
  self.assertLess(source.index('accept_five_updates(checkpoint, five_update_records)'),source.index('                self.global_steps += 1'))
if __name__=='__main__':unittest.main()
