"""Fixed common stage512 at baseline,step5 and periodic50; no duplicate validation512; final full51713."""
import os,pathlib,time
from pipeline import atomic

def evaluate(trainer,step):
 from torch.utils.data import DataLoader
 from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
 path='/data/user/jxu957/experiments/topd-20261008/runtime/assets/common-instruct-0b002b/validation.parquet'
 dataset=RLHFDataset(parquet_files=path,tokenizer=trainer.tokenizer,prompt_key=trainer.config.data.prompt_key,max_prompt_length=trainer.config.data.max_prompt_length,filter_prompts=True,return_raw_chat=trainer.config.data.get('return_raw_chat',False),truncation='error')
 if len(dataset)!=512:raise RuntimeError('Actual shared stage512 differs')
 old_dataset,old_loader=trainer.val_dataset,trainer.val_dataloader
 try:
  trainer.val_dataset=dataset;trainer.val_dataloader=DataLoader(dataset,batch_size=32,shuffle=False,drop_last=False,collate_fn=collate_fn);metrics=trainer._validate()
 finally:trainer.val_dataset,trainer.val_dataloader=old_dataset,old_loader
 record={'scope':'ACTUAL_SHARED_STAGE512_NOT_FINAL_TEST','step':step,'rows':512,'validation_batch_size':32,'metrics':metrics,'epoch':time.time()}
 root=pathlib.Path(os.environ['TOPD_RUN_DIRECTORY']);atomic(root/('stage512-step%d.json'%step),record)
 print('TOPD_STAGE512_JSON='+__import__('json').dumps(record,sort_keys=True),flush=True)
 return metrics

def install():
 from search_r1.baselines.controller import Controller
 original=Controller.stage_validation
 def stage(self,step=None):
  if self.method=='topd' and step is not None and (step==5 or step%50==0):return {'topd/stage512/'+k:v for k,v in evaluate(self.trainer,step).items()}
  return original(self,step)
 Controller.stage_validation=stage
