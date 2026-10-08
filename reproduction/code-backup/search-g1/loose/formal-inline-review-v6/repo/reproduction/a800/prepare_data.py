"""Question-disjoint genuine calibration and fixed R1 validation64 on A800."""
import hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path('/data/search-g1');source=Path('/data/search-r1/data/paper-grpo')

def prompt(row):
    value=row['prompt']
    if isinstance(value,str):value=json.loads(value)
    return value[0]['content']

def question(row):
    text=prompt(row)
    matches=list(re.finditer(r'Question:\s*',text))
    if not matches:raise ValueError('Unknown question prompt schema')
    return re.sub(r'\s+',' ',text[matches[-1].end():]).strip().casefold()

def qid(row):return hashlib.sha256(question(row).encode()).hexdigest()

train=pd.read_parquet(source/'train.parquet');validation=pd.read_parquet(source/'validation-64.parquet')
assert len(validation)==64
train_ids=[qid(r) for _,r in train.iterrows()];val_ids=[qid(r) for _,r in validation.iterrows()]
assert len(set(val_ids))==64
# Reserve150 unique questions from each training source, seed-fixed across lengths.
rng=np.random.RandomState(42);selected=[];seen=set(val_ids)
for name in ('nq','hotpotqa'):
 idx=[i for i in range(len(train)) if train.iloc[i]['data_source']==name]
 rng.shuffle(idx);chosen=[]
 for i in idx:
  if train_ids[i] not in seen:
   selected.append(i);chosen.append(i);seen.add(train_ids[i])
   if len(chosen)==150:break
 assert len(chosen)==150, 'Insufficient genuinely unique calibration questions'
calibration=train.iloc[selected].copy();cal_ids=set(train_ids[i] for i in selected)
# Remove all duplicate questions, including duplicates in a different source.
keep=[i for i,x in enumerate(train_ids) if x not in cal_ids and x not in set(val_ids)]
training=train.iloc[keep].copy();training_ids={train_ids[i] for i in keep}
assert not training_ids & cal_ids and not training_ids & set(val_ids) and not cal_ids & set(val_ids)
root=BASE/'data';root.mkdir(exist_ok=True)
training.to_parquet(root/'train.parquet',index=False);calibration.to_parquet(root/'calibration-300.parquet',index=False);validation.to_parquet(root/'validation-64.parquet',index=False)
audit={'train_count':len(training),'source_train_count':len(train),'calibration_count':300,'validation_count':64,'removed_train_rows':len(train)-len(training),'question_disjoint':True,'calibration_source_counts':calibration['data_source'].value_counts().to_dict(),'validation_source_counts':validation['data_source'].value_counts().to_dict(),'calibration_question_ids':sorted(cal_ids),'validation_question_ids':val_ids,'selection_seed':42,'g1_train_data_adaptation':'Shared NQ+HotpotQA R1 training source with genuine question-disjoint calibration held out, rather than per-domain source recipe.'}
(root/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if 'question_ids' not in k},indent=2))
