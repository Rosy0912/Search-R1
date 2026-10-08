"""Blocked corrected final native evaluation supervisor; no training dispatch."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'strict-tools'))
from training_evidence import digest,loads,OWN
from hf_activation_capture import capture_lease
from runtime_capture import process_identity
from post_eval_check import EVAL_COMMIT

def model_record(stage):
 from strict_stage_prerequisites import verify_model_files
 record={'directory':stage['model_directory'],'files':stage['model_files']}
 verify_model_files(record);return record

def validate_config(config,stage):
 from omegaconf import OmegaConf
 from search_r1.baselines import corrected_evaluation as ce
 if config.trainer.val_only is not True or config.trainer.val_before_train is not True or config.data.val_batch_size!=64 or config.data.val_data_num is not None:raise ValueError('Exact full-final eval-only batch64 required')
 if config.baselines.method!='opsd' or config.baselines.holdout_eval_only or config.baselines.corrected_evaluation.enabled is not True or not config.data.return_raw_chat:raise ValueError('Native corrected scoring-view configuration required')
 if config.baselines.local_runtime_only is not True or config.baselines.local_runtime_host!='h100-10':raise ValueError('Same-node runtime only required')
 if config.retriever.url!='http://127.0.0.1:18084/retrieve' or config.trainer.n_gpus_per_node!=8 or config.trainer.nnodes!=1:raise ValueError('Native same-node eight GPU search required')
 if str(Path(config.actor_rollout_ref.model.path).resolve())!=stage['model_directory']:raise ValueError('Actual evaluated model role differs')
 s=config.baselines.corrected_evaluation
 if s.input_manifest_sha256!=stage['input_manifest_sha256'] or str(Path(s.reviewed_stage_manifest).resolve())!=stage['reviewed_stage_path']:raise ValueError('Config actual stage/input differs')
 inputs=ce.load_inputs(s.input_manifest_path,s.input_manifest_sha256)
 files=OmegaConf.to_container(config.data.final_val_files) if not isinstance(config.data.final_val_files,str) else [config.data.final_val_files]
 if [str(Path(p).resolve()) for p in files]!=[item['path'] for item in inputs['inputs']]:raise ValueError('Final actual source file set differs')
 ce.verify_budget(config,stage);ce.verify_runtime_sources(stage)
 return inputs

def run(config_path,stage_path,stage_sha,output):
 from omegaconf import OmegaConf
 from search_r1.baselines import corrected_evaluation as ce,preflight
 job=os.environ.get('SLURM_JOB_ID','');repo=Path(ce.__file__).resolve().parents[2]
 if not re.fullmatch('[0-9]+',job) or repo==OWN/'repo':raise ValueError('Actual corrected own allocation and separate source checkout required')
 if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=EVAL_COMMIT:raise ValueError('Actual corrected eval source differs')
 stage_path=Path(stage_path).resolve();config_path=Path(config_path).resolve();out=Path(output).resolve()
 if digest(stage_path)!=stage_sha or os.environ['OPSD_CORRECTED_EVAL_STAGE_SHA256']!=stage_sha:raise ValueError('Actual external stage SHA differs')
 cfg=OmegaConf.load(config_path);stage=ce.load_stage(cfg.baselines.corrected_evaluation);validate_config(cfg,stage)
 if out.exists() or not out.is_relative_to(OWN):raise ValueError('New own immutable eval evidence directory required')
 out.mkdir(parents=True)
 preflight.validate_inherited_gpu_leases(8)
 inventory,processes=preflight.gpu_snapshot();selected=preflight.check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','),inventory,processes)
 if len(selected)!=8:raise ValueError('Actual selected8 GPU preflight differs')
 leases=loads(os.environ['SEARCH_R1_GPU_LEASES']);fds=tuple([9]+[int(r['fd']) for r in leases])
 capture={'job_id':job,'stage_sha256':stage_sha,'process':process_identity(os.getpid()),'selected_gpu_preflight':selected,'global_lease':capture_lease(9,OWN/'gpu-job.lock'),'gpu_leases':[capture_lease(int(r['fd']),Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(r['uuid']+'.lock')) for r in leases]}
 (out/'launch-capture.json').write_text(json.dumps(capture,sort_keys=True))
 (out/'model-before.json').write_text(json.dumps(model_record(stage),sort_keys=True))
 # The reviewed job script has already launched/verified the actual local GPU retrieval
 # and resource sampler. Main's actual8-worker RPC and guards run before generation.
 command=[sys.executable,'-u','-m','search_r1.baselines.main','--config-path',str(config_path.parent),'--config-name',config_path.stem]
 with (out/'eval.log').open('xb') as log:
  result=subprocess.run(command,cwd=repo,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,pass_fds=fds)
 (out/'exit.txt').write_text(str(result.returncode)+'\n')
 if result.returncode:raise RuntimeError('Corrected native eval child exited nonzero')
 (out/'model-after.json').write_text(json.dumps(model_record(stage),sort_keys=True))
 (out/'finished.txt').write_text('CORRECTED_FINAL_EVAL_EXITED_51713\n')

if __name__=='__main__':
 print('BLOCKED PREVIEW: corrected actual500/HF receipt/stage/job fields UNFILLED',file=sys.stderr)
 raise SystemExit(78)
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--stage',required=True);parser.add_argument('--stage-sha256',required=True);parser.add_argument('--output',required=True)
 args=parser.parse_args();run(args.config,args.stage,args.stage_sha256,args.output)
