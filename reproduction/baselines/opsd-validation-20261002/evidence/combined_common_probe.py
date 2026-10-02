import ast,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/reproduction'))
from verl import DataProto
from search_r1.baselines.evaluation import batch_answer_metrics,aggregate_answer_metrics
from search_r1.baselines.preflight import check_selected_gpus
from test_reward_and_validation import CharTokenizer
from test_final_checkpoint import FakeTrainer,real_iteration_tail

def data(segments):
 prompt='Teacher-only answer: Paris. <answer>Paris</answer> original question'
 response=''.join(t for t,m in segments); mask=[1]*len(prompt)
 for t,m in segments:mask += [int(m)]*len(t)
 return DataProto.from_dict(dict(prompts=torch.tensor([[ord(c) for c in prompt]],dtype=torch.long),responses=torch.tensor([[ord(c) for c in response]],dtype=torch.long),attention_mask=torch.ones((1,len(prompt)+len(response)),dtype=torch.long),info_mask=torch.tensor([mask],dtype=torch.long)),non_tensors={'data_source':np.array(['nq'],dtype=object),'reward_model':np.array([{'ground_truth':{'target':['Lutetia','Paris']}}],dtype=object)})
cases=[('prompt_only',[('<search>France</search>',True)],0.),('observation_answer',[('<search>France</search>',True),('<information><answer>Paris</answer></information>',False),('<search>more</search>',True)],0.),('stitched_tag',[('<ans',True),('external',False),('wer>Paris</answer>',True)],0.),('stitched_content',[('<answer>Pa',True),('external',False),('ris</answer>',True)],0.),('legit_answer',[('<search>France</search>',True),('<information>Paris</information>',False),('<answer>The PARIS!</answer>',True)],1.)]
rows=[]
for name,segments,expected in cases:
 row=batch_answer_metrics(data(segments),CharTokenizer())[0]
 assert row['em']==expected and row['f1']==expected,(name,row)
 assert row['generated_tokens']==sum(len(t) for t,m in segments if m)
 rows.append({'case':name,'metrics':row})
inv={'0':dict(uuid='GPU-test',memory_used=15.,memory_total=100.,utilization=15.)}
assert len(check_selected_gpus(['0'],inv,{'GPU-test'}))==1
for key,val in [('utilization',15.01),('memory_used',15.01),('utilization',float('nan')),('memory_total',0.)]:
 bad={'0':dict(inv['0'],**{key:val})}
 try: check_selected_gpus(['0'],bad,{'GPU-test'})
 except RuntimeError:pass
 else:raise AssertionError('invalid/busy GPU accepted')
smoke=FakeTrainer(5,6);smoke.config.trainer.smoke_only=True;real_iteration_tail()(smoke)
assert smoke.events==[('update',5,1),('save',5,1)]
formal=FakeTrainer(500,501);formal.config.trainer.smoke_only=False;real_iteration_tail()(formal)
assert formal.events[-2:]==[('final_data',501,1),('validate',501,1)]
source=ast.parse((ROOT/'search_r1/baselines/main.py').read_text())
main=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
guard=next(n for n in main.body if isinstance(n,ast.If) and 'config.trainer.smoke_only' in ast.unparse(n.test))
from types import SimpleNamespace
module=ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[]));code=compile(module,'actual_smoke_guard','exec')
exec(code,{'config':SimpleNamespace(trainer=SimpleNamespace(smoke_only=True,total_training_steps=6))})
try:exec(code,{'config':SimpleNamespace(trainer=SimpleNamespace(smoke_only=True,total_training_steps=7))})
except ValueError:pass
else:raise AssertionError('oversized smoke accepted')
print(json.dumps({'adversarial_evaluator_cases':rows,'low_process_boundary_and_invalid_occupancy':True,'actual_main_smoke_guard_6_allowed_7_rejected':True,'smoke_update5_terminal_events':smoke.events,'formal_update500_terminal_events':formal.events,'scope':'CPU evaluator/guards/trainer-tail only; no GPU or production3B validation'},indent=2))
