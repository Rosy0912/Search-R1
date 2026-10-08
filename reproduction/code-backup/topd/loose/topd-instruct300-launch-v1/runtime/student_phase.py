"""Fixed common stage64 at baseline and step5; periodic validation stays512."""
import os,pathlib,time
from pipeline import atomic

def evaluate(trainer,step):
 from torch.utils.data import DataLoader
 from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
 path='/dataset/shaoshitong/search-r1-rl/topd/assets/common-instruct-0b002b/stage.parquet'
 dataset=RLHFDataset(parquet_files=path,tokenizer=trainer.tokenizer,prompt_key=trainer.config.data.prompt_key,max_prompt_length=trainer.config.data.max_prompt_length,filter_prompts=True,return_raw_chat=trainer.config.data.get('return_raw_chat',False),truncation='error')
 if len(dataset)!=64:raise RuntimeError('Actual shared stage64 differs')
 old_dataset,old_loader=trainer.val_dataset,trainer.val_dataloader
 try:
  trainer.val_dataset=dataset;trainer.val_dataloader=DataLoader(dataset,batch_size=64,shuffle=False,drop_last=False,collate_fn=collate_fn);metrics=trainer._validate()
 finally:trainer.val_dataset,trainer.val_dataloader=old_dataset,old_loader
 record={'scope':'ACTUAL_SHARED_STAGE64_NOT_FINAL_TEST','step':step,'rows':64,'metrics':metrics,'epoch':time.time()}
 root=pathlib.Path(os.environ['TOPD_RUN_DIRECTORY']);atomic(root/('stage64-step%d.json'%step),record)
 print('TOPD_STAGE64_JSON='+__import__('json').dumps(record,sort_keys=True),flush=True)
 return metrics

def install():
 from search_r1.baselines.controller import Controller
 original=Controller.stage_validation
 def stage(self,step=None):
  if self.method=='topd' and step==5:return {'topd/stage64/'+k:v for k,v in evaluate(self.trainer,5).items()}
  return original(self,step)
 Controller.stage_validation=stage
