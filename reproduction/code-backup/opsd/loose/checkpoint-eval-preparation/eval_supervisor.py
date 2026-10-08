"""Fresh HF5 validation64 only. No optimizer/outer or final-test execution."""
import hashlib,json,os,re,subprocess,time,math
from pathlib import Path
from omegaconf import OmegaConf
from search_r1.baselines.preflight import validate_inherited_gpu_leases
from search_r1.baselines.holdout import prepare_common_holdout
from search_r1.baselines.controller import validate_config
root=Path(os.environ['SEARCH_R1_BASE']);out=Path(os.environ['OPSD_EVIDENCE'])
checkpoint=root/'runs/opsd-acceptance-2960/output/checkpoints/opsd-gpu-acceptance-5/actor/global_step_5'
base_teacher=root/'assets-root/assets/qwen2.5-3b'
assert os.environ['OPSD_TEACHER']==str(base_teacher) and checkpoint!=base_teacher
assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF','')==''
lease_fds=tuple(int(r['fd']) for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES']))
for attempt in range(30):
 try:validate_inherited_gpu_leases(8);break
 except RuntimeError:
  if attempt==29:raise
  time.sleep(2)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
reload_record=json.loads((root/'runs/opsd-hf-reload-2962/checkpoint-reload.json').read_text())
assert reload_record['slurm_job_id']=='2962' and reload_record['checkpoint']==str(checkpoint)
weights={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in checkpoint.glob('*.safetensors')}
assert weights and weights==reload_record['weights'], 'HF5 changed since successful fresh2962 reload'
(out/'hf5-weights.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),weights=weights,source_commit=os.environ['OPSD_EXPECTED_COMMIT'],job_id=os.environ['SLURM_JOB_ID']),indent=2))
args=[f'actor_rollout_ref.model.path={checkpoint}',f'baselines.distillation.teacher_path={base_teacher}',
'baselines.distillation.teacher_device=cuda','baselines.distillation.loss_checkpointing=true',
'baselines.distillation.max_student_logits_gib=16.0','baselines.local_runtime_only=true',
'baselines.local_runtime_host=h100-10','retriever.url=http://127.0.0.1:18084/retrieve',
'baselines.seed=42','+trainer.val_only=true','trainer.total_training_steps=6','trainer.smoke_only=true',
'data.val_batch_size=64','trainer.val_before_train=true','trainer.test_freq=5','trainer.save_freq=5',
'actor_rollout_ref.actor.ppo_micro_batch_size=16','trainer.experiment_name=opsd-hf5-validation64-only',
'actor_rollout_ref.rollout.log_prob_micro_batch_size=64','actor_rollout_ref.ref.log_prob_micro_batch_size=64']
entry=['bash','reproduction/baselines/run.sh','opsd']
resolved=subprocess.check_output(entry+args+['--cfg','job','--resolve'],text=True,pass_fds=lease_fds)
(out/'resolved.yaml').write_text(resolved);cfg=OmegaConf.create(resolved)
assert cfg.trainer.val_only is True and cfg.trainer.val_before_train is True
assert cfg.trainer.total_training_steps==6 and cfg.trainer.smoke_only is True
assert cfg.trainer.n_gpus_per_node==8 and cfg.data.val_batch_size==64
assert cfg.data.val_data_num is None and cfg.data.train_data_num is None
assert cfg.actor_rollout_ref.model.path==str(checkpoint)
assert cfg.baselines.distillation.teacher_path==str(base_teacher)
assert cfg.baselines.distillation.teacher_device=='cuda'
assert cfg.baselines.local_runtime_only is True and cfg.baselines.local_runtime_host=='h100-10'
assert cfg.retriever.url=='http://127.0.0.1:18084/retrieve'
assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size==16
assert cfg.actor_rollout_ref.rollout.n_agent==5 and cfg.actor_rollout_ref.rollout.temperature==1.0
assert cfg.max_turns==4
assert cfg.baselines.seed==42 and cfg.baselines.holdout.seed==42
manifest=prepare_common_holdout(cfg);validate_config(cfg)
assert manifest['content_identity']['sha256']=='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'
import pandas as pd
counts={k:int(v) for k,v in pd.read_parquet(cfg.data.val_files,columns=['data_source'])['data_source'].value_counts().items()}
assert sum(counts.values())==64
(out/'expected-validation-counts.json').write_text(json.dumps(counts,indent=2))
text=OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)))
(out/'reviewed-entry-config.yaml').write_text(text)
os.environ.update(SEARCH_R1_BASELINE_AUDIT_DIR=str(out/'audit'),SEARCH_R1_BASELINE_RUN_ID=os.environ['SLURM_JOB_ID'],SEARCH_R1_BASELINE_COMMIT=os.environ['OPSD_EXPECTED_COMMIT'],SEARCH_R1_BASELINE_CONFIG_SHA256=hashlib.sha256(text.encode()).hexdigest(),SEARCH_R1_BASELINE_DATA_IDENTITY=manifest['content_identity']['sha256'])
with (out.parent/'evaluation.log').open('w') as f:
 result=subprocess.run(entry+args,stdout=f,stderr=subprocess.STDOUT,pass_fds=lease_fds)
assert result.returncode==0, ('Evaluation child failed',result.returncode)
log=re.sub(r'\x1b\[[0-9;]*m','',(out.parent/'evaluation.log').read_text())
assert 'SEARCH_R1_FINAL_METRICS_JSON=' not in log, 'Full final evaluation must not run'
assert 'actor/optimizer_steps:' not in log, 'No actor outer is permitted'
marker='BASELINE_EVALUATION_METRICS_JSON='
records=[json.loads(line.split(marker,1)[1]) for line in log.splitlines() if marker in line]
assert len(records)==1
metrics=records[0];assert metrics['val/qa_count/all']==64
for source,count in counts.items():
 assert metrics[f'val/qa_count/{source}']==count and metrics[f'val/evaluated_count/{source}']==count
for source in set(counts)|{'all'}:
 for key in ('f1','em','answer_format','format_rate'):
  v=metrics[f'val/{key}/{source}'];assert math.isfinite(v) and 0<=v<=1
expected={'val/em/all':.046875,'val/f1/all':.0546875,'val/format_rate/all':0.,'val/answer_format/all':.28125}
comparison={key:{'pilot_after5':value,'fresh_HF5':metrics[key],'delta':metrics[key]-value} for key,value in expected.items()}
(out/'fresh-hf5-validation64.json').write_text(json.dumps(dict(execution_validated=True,scope='fresh_HF5_validation64_only_no_outer_no_final_test',actual_outer_updates=0,final_test_questions=0,job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['OPSD_EXPECTED_COMMIT'],checkpoint=str(checkpoint),teacher_initialbase=str(base_teacher),metrics=metrics,comparison=comparison,accuracy_pass_claim=False,exact_reproduction_claim=False,weights=weights),indent=2,allow_nan=False))
print('FRESH_HF5_VALIDATION64_ONLY_COMPLETE',json.dumps(comparison),flush=True)
