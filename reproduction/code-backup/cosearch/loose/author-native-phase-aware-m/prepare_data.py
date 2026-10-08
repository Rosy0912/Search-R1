"""Use same retained questions/targets; author's native prompt, no target in prompt."""
import ast,hashlib,json,sys
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch');native=root/'author-native'
sys.path.insert(0,str(root/'repo'))
from search_r1.baselines.holdout import prepare_common_holdout
c=OmegaConf.load(root/'runs/cosearch-local-gpu-five-20261003-n/resolved.yaml')
split=prepare_common_holdout(c)
tree=ast.parse((native/'repo/verl/verl/tools/utils/prompts.py').read_text())
prompt=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SEARCH_R1_CoT_PROMPT' for t in n.targets))
(native/'data').mkdir(exist_ok=True)
report={'split_content_identity':split['content_identity'],'prompt_constant':'SEARCH_R1_CoT_PROMPT','gold_in_inference_prompt':False,'outputs':{}}
for label,source in [('train',str(c.data.train_files)),('validation-64',str(c.data.val_files))]:
 frame=pd.read_parquet(source);identities=[]
 for i in frame.index:
  row=frame.loc[i];messages=list(row['prompt']);content=messages[-1]['content'];assert 'Question: ' in content;question=content.rsplit('Question: ',1)[1].strip();assert question and not question.startswith('[{')
  answers=list(row['reward_model']['ground_truth']['target']);assert answers
  identities.append(json.dumps([question,answers],ensure_ascii=False))
  frame.at[i,'prompt']=[{'role':'user','content':prompt.format(question=question)}]
  extra=dict(row['extra_info']);extra['question']=question;frame.at[i,'extra_info']=extra
 out=native/'data'/f'{label}.parquet';assert not out.exists(), 'Do not overwrite identity-bound inputs'
 frame.to_parquet(out,index=False)
 report['outputs'][label]={'source':source,'rows':len(frame),'question_answers_sha256':hashlib.sha256('\n'.join(identities).encode()).hexdigest(),'parquet_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
assert report['outputs']['validation-64']['rows']==64
(native/'data/identity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
