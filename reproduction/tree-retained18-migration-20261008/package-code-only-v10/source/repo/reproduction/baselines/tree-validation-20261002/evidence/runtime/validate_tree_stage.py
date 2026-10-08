"""Independent CPU-only stage evidence reviewer; never emits stage acceptance."""
import argparse, hashlib, json, math, os, pathlib, re, subprocess, sys
os.environ['CUDA_VISIBLE_DEVICES']=''
NUMBER=re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
ANSI=re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
REQUIRED={'training/actual_outer_updates','actor/optimizer_steps','actor/grad_norm','actor/pg_loss','actor/entropy_loss','actor/pg_clipfrac','actor/ppo_kl'}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def parse_console(path,expected):
 records=[]
 for raw in path.read_text(errors='strict').splitlines():
  line=ANSI.sub('',raw);hit=re.search(r'(?<![\w/])step:(\d+) - ',line)
  if not hit:continue
  fields={}
  for field in line[hit.end():].split(' - '):
   key,sep,value=field.partition(':')
   if not sep or key in fields or not NUMBER.fullmatch(value.strip()):raise ValueError('Invalid/duplicate/nonfinite console metric: '+field)
   n=float(value);assert math.isfinite(n),'Nonfinite metric';fields[key]=n
  if 'training/actual_outer_updates' not in fields:continue
  assert REQUIRED<=fields.keys(),'Missing actor metrics'
  step=int(hit.group(1));count=fields['training/actual_outer_updates']
  assert count.is_integer() and count==step,'Actual updates and outer step differ'
  assert fields['actor/optimizer_steps'].is_integer() and fields['actor/optimizer_steps']>0,'No actual optimizer update'
  assert fields['actor/grad_norm']>0,'No positive finite gradient evidence'
  records.append({'step':step,'metrics':fields})
 assert [r['step'] for r in records]==list(range(1,expected+1)),'Missing/extra/duplicate updates or wrong stage length'
 return records
def inspect(run,stage):
 import yaml
 expected={'two':2,'five':5,'formal':500}[stage]
 assert run.is_dir(),'No real run directory'
 cost=json.loads((run/'cost.json').read_text());assert cost['stage']==stage and cost['exit_code']==0 and cost['allocated_gpus']==8
 assert cost['end_epoch']>=cost['start_epoch'] and int(cost['job_id'])>0
 records=parse_console(run/'stage.log',expected)
 cfg=yaml.safe_load((run/'resolved.yaml').read_text());trainer=cfg['trainer']
 assert trainer['total_training_steps']==expected+1,'Wrong configured budget'
 assert trainer['smoke_only']==(stage!='formal'),'Wrong stage mode'
 assert cfg['baselines']['method']=='tree_grpo','Wrong algorithm'
 checkpoint_root=pathlib.Path(trainer['default_local_dir']).resolve();assert checkpoint_root.is_relative_to(run.resolve()),'Checkpoint outside own run'
 checkpoint=checkpoint_root/'actor'/('global_step_'+str(expected));statepath=checkpoint_root/'algorithm'/('global_step_'+str(expected))/'state.json'
 assert checkpoint.is_dir() and statepath.is_file(),'No matching final HF and algorithm checkpoint'
 state=json.loads(statepath.read_text());assert state['method']=='tree_grpo' and state['global_step']==expected and state['resume_supported'] is False
 for key in ('base_commit','algorithm_source_commit'):assert state[key]==cfg['baselines'][key],'Algorithm source identity differs'
 assert any(checkpoint.glob('*.safetensors')) or any(checkpoint.glob('*.bin')),'No saved weights'
 return records,cost,cfg,state,statepath,checkpoint

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);p.add_argument('--stage',choices=['two','five','formal'],required=True);p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--checkpoint-checker',type=pathlib.Path,default=pathlib.Path(__file__).with_name('check_real_checkpoint.py'));a=p.parse_args()
 assert not a.output.exists(),'Refuse evidence overwrite'
 records,cost,cfg,state,statepath,checkpoint=inspect(a.run,a.stage)
 assert a.base.resolve()==pathlib.Path(cfg['actor_rollout_ref']['model']['path']).resolve(),'Immutable base does not match resolved actor model'
 child=a.output.with_name(a.output.stem+'-hf-reload.json');assert not child.exists()
 subprocess.run([sys.executable,str(a.checkpoint_checker),'--base',str(a.base),'--checkpoint',str(checkpoint),'--output',str(child)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 hf=json.loads(child.read_text());assert hf['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED' and hf['all_parameters_finite'] and hf['cpu_forward_logits_finite'] and hf['changed_parameters']>0
 assert pathlib.Path(hf['checkpoint']).resolve()==checkpoint
 report={'status':'REAL_STAGE_EVIDENCE_READY_FOR_INDEPENDENT_REVIEW','not_stage_acceptance':True,'stage':a.stage,'actual_outer_updates':len(records),'run':str(a.run.resolve()),'records':records,'cost':cost,'algorithm_state':state,'checkpoint':str(checkpoint),'hf_reload_report':str(child.resolve()),'source_sha256':{str(x.resolve()):digest(x) for x in [a.run/'stage.log',a.run/'cost.json',a.run/'resolved.yaml',statepath,child,pathlib.Path(__file__),a.checkpoint_checker]},'exact_optimizer_rng_dataloader_resume':False,'effect_validation_complete':False}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps({'status':report['status'],'stage':a.stage,'actual_outer_updates':len(records),'output':str(a.output)}))
if __name__=='__main__':main()
