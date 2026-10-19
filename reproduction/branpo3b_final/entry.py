"""Only final trained-model identity adaptation; original common runner unchanged."""
from pathlib import Path
import argparse,json,os,subprocess
from .contract import *

def inspect(plan):
 from reproduction.seven_dataset.contracts import validate_model_files
 require(sha(TEMPLATE)==TEMPLATE_SHA,'Reference template changed');template=json.loads(Path(TEMPLATE).read_text())
 evidence={}
 for name,row in plan['evidence'].items():
  require(Path(row['path'])==Path(ROOT)/'inputs'/name and sha(row['path'])==row['sha256'],'Evidence binding changed')
  evidence[name]=json.loads(Path(row['path']).read_text())
 require(set(evidence)=={'cold.json','native.json','hf.json','export-intent.json'},'Incomplete final lineage')
 native,hf=evidence['native.json'],evidence['hf.json']
 validate_lineage(evidence['cold.json'],native,hf,evidence['export-intent.json']);out=validate_plan(plan,template,native,hf)
 src=Path(BASE)/'sources'/plan['evaluation_code_commit']
 for name,h in SCIENCE.items():require(sha(src/name)==h,'Scientific source changed: '+name)
 validate_model_files(plan)
 # Validate the original200 reference protocol in full, never load its weights
 # for inference. The trained final weights are substituted only after this.
 from reproduction.seven_dataset.entry import inspect_inputs
 cfg,expected=inspect_inputs(template)
 from reproduction.seven_dataset.model_view import validate_view
 validate_view(plan,cfg.baselines.tree.token_preservation.tokenizer_sha256)
 from omegaconf import OmegaConf
 OmegaConf.set_struct(cfg,False)
 cfg.actor_rollout_ref.model.path=plan['inference_model'];cfg.actor_rollout_ref.ref.model_path=plan['inference_model']
 cfg.trainer.default_local_dir=str(out/'output/unused-checkpoints')
 from search_r1.baselines.controller import validate_config
 validate_config(cfg)
 require(cfg.max_turns==4 and cfg.force_final_answer is False and cfg.retriever.topk==3 and len(expected)==51713,'Common protocol changed')
 return cfg,expected,out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
 plan=json.loads(Path(args.plan).read_text());cfg,expected,root=inspect(plan)
 if args.validate_only:
  print(json.dumps(dict(status='BRANPO3B_FINAL150_COMMON51713_INPUTS_VERIFIED_NO_INFERENCE',questions=len(expected),repeat_index=plan['repeat_index'])));return
 require(os.environ['SLURM_JOB_NAME']==plan['job'] and os.environ['SLURM_JOB_ID'].isdigit(),'Wrong allocation')
 row=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'],text=True)
 require('JobState=RUNNING' in row and 'UserId=jmzhang(1066)' in row and 'gres/gpu=8' in row and 'mem=600G' in row,'Wrong actual resources')
 inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
 require(len(inv)==8 and len({v.split(',')[1].strip() for v in inv})==8 and all(float(v.split(',')[-1])>=24000 for v in inv),'Eight compatible GPUs required')
 require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'No GPU preemption')
 with (root/'entry-once.json').open('x') as f:json.dump(dict(job_id=os.environ['SLURM_JOB_ID'],job=plan['job'],gpus=inv,plan_sha256=sha(args.plan)),f,indent=2)
 from reproduction.seven_dataset.entry import run_frozen
 run_frozen(plan,args.plan,cfg,expected,root)

if __name__=='__main__':main()
