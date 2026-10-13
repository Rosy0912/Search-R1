"""Passive attempt-cost publication; reads real Controller.generate metrics only.

This does not count optimizer writes or outer credit. The persisted actor/driver
attempt decisions supply those independently. Failed generation counts remain
unknown; raw failure ledgers/logs are retained rather than guessed as zero.
"""
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

KEYS=('generated_tokens','prefill_tokens','retrieval_calls','trajectories','padded_rows',
      'context_mismatches','raw_slots','raw_emitted_tokens','raw_retained_tokens',
      'raw_discarded_tail_tokens','raw_inherited_leaf_tokens',
      'semantic_prefix_scan_cpu_seconds','semantic_prefix_scan_wall_seconds')


def wrap_controller(Controller,root):
    if getattr(Controller,'_fresh_training_cost_observer_installed',False):
        raise ValueError('Training cost observer already installed')
    original=Controller.generate
    @functools.wraps(original)
    def generate(self,manager,gen_batch,qa_batch,metrics):
        trainer=self.trainer;attempt=trainer.training_attempts;candidate=trainer.global_steps
        if type(attempt) is not int or attempt<1 or candidate!=trainer.actual_outer_updates+1:
            raise ValueError('Real consumed attempt/candidate required before training generation')
        start=time.time();clock=time.perf_counter();cpu=time.process_time();error=None;costs=None
        try:
            result=original(self,manager,gen_batch,qa_batch,metrics)
            costs={key:metrics['rollout/'+key] for key in KEYS if 'rollout/'+key in metrics}
            if not {'generated_tokens','prefill_tokens','retrieval_calls','trajectories','padded_rows','raw_emitted_tokens'}<=set(costs):
                raise ValueError('Actual native-token rollout cost fields missing')
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in costs.values()):
                raise ValueError('Actual rollout cost fields invalid')
            return result
        except BaseException as exc:
            error=repr(exc);raise
        finally:
            root_path=Path(root);root_path.mkdir(parents=True,exist_ok=True)
            value=dict(status='TRAINING_ROLLOUT_COST_COMPLETE' if error is None else 'TRAINING_ROLLOUT_COST_FAILED',
                attempt_id=attempt,candidate_outer=candidate,completed_outer_before=trainer.actual_outer_updates,
                question_rows=len(qa_batch.batch),costs=costs,error=error,start_epoch=start,end_epoch=time.time(),
                wall_seconds=time.perf_counter()-clock,trainer_process_cpu_seconds=time.process_time()-cpu,
                actual_optimizer_writes=None,outer_credit_not_inferred=True,
                allocation_kind='user_authorized_shared_systemd' if os.environ.get('TREE_SHARED_TRAINING_MODE')=='1' else 'exclusive_slurm',
                physical_gpus=8,slurm_reserved_gpus=0 if os.environ.get('TREE_SHARED_TRAINING_MODE')=='1' else 8,
                resource_protocol=os.environ.get('TREE_SHARED_RESOURCE_PROTOCOL','exclusive_slurm_dual15_v1'),
                job_id=os.environ.get('SLURM_JOB_ID'),run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
                code_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
                data_identity=os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY'),model_identity=os.environ.get('SEARCH_R1_MODEL_IDENTITY'),
                observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Actual original Controller.generate metric ledger only; question rows before five-leaf expansion; native emitted scalar counts include DP padding; prefill is submitted token events, not unique KV/FLOPs. Costs include inherited ledger conventions; cached savings and worker CPU are not inferred. Credit/write/skip decisions join separately by attempt_id.')
            with (root_path/f'attempt-{attempt:06d}.json').open('x') as stream:
                json.dump(value,stream,indent=2,allow_nan=False)
    Controller.generate=generate;Controller._fresh_training_cost_observer_installed=True


def install():
    from search_r1.baselines import controller
    path=Path(os.environ['TREE_INSTRUCT_TOOLS']);manifest=json.loads((path/'manifest.json').read_text())
    actual=Path(inspect.getsourcefile(controller.Controller))
    expected=manifest['files']['source/repo/search_r1/baselines/controller.py']
    if hashlib.sha256(actual.read_bytes()).hexdigest()!=expected:
        raise ValueError('Training cost wrapper actual source differs from frozen package')
    wrap_controller(controller.Controller,Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'training-rollout-cost')
