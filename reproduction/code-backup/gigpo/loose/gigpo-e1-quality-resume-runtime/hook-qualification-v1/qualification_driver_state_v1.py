"""Actual diagnostic driver/RNG/sampler/iterator capture and reload.

This supplies the concrete driver portion missing from calibration v8's
worker-only roundtrip. It never restores actor weights, initializes CUDA or
claims HF semantic acceptance. Diagnostic A/B must separately restore all
eight workers from the immutable complete25 and prove actual sampling equality.
"""
import copy
import os
from pathlib import Path
from omegaconf import OmegaConf
from qualification_contract_v1 import require
from calibration_state_v1 import state_digest
from search_r1.baselines.recovery import cpu_snapshot,rng_state,restore_rng


def capture_driver_state(trainer,iterator):
    require(trainer.train_dataloader.num_workers==0 and trainer.recovery_sampler is not None,
            'Exact driver replay requires the actual unprefetched recoverable sampler')
    controller=trainer.baseline_controller
    return dict(schema=1,pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
                    if Path('/proc/self/stat').exists() else 'CPU_LOCAL_NO_PROC_NOT_GPU_IDENTITY',
        rng=cpu_snapshot(rng_state()),sampler=cpu_snapshot(trainer.recovery_sampler.state_dict()),
        iterator=dict(base_seed=iterator._base_seed,num_yielded=iterator._num_yielded),
        config=OmegaConf.to_container(trainer.config,resolve=True),
        data_identity=copy.deepcopy(trainer.instruct_data_identity),
        logical_step=trainer.global_steps,actual_new_updates=trainer.actual_outer_updates,
        kl_controller=cpu_snapshot(vars(trainer.kl_ctrl)),
        algorithm=dict(method=controller.method,stage=cpu_snapshot(controller.stage.state_dict()),
            parameters=OmegaConf.to_container(controller.config.gigpo,resolve=True),
            loss_reduction=controller.config.loss_reduction,
            base_commit=controller.config.base_commit,source_commit=controller.config.algorithm_source_commit),
        quality_health=copy.deepcopy(getattr(trainer,'quality_health',None)),
        validation_trace_serial=getattr(trainer,'validation_trace_serial',None))


def reload_driver_state(trainer,saved):
    """Restore real saved driver state and rebuild the same zero-worker cursor.

    The constructor's base-seed RNG draw is deliberately undone by restoring
    the saved driver RNG after reconstruction. Iterator private bookkeeping is
    restored explicitly; no data are fetched or loss counters inferred here.
    The real next DataProto batch must subsequently be compared by the adapter.
    """
    require(saved.get('schema')==1 and saved['pid']==os.getpid(),
            'Diagnostic driver cannot be replaced at same-process reload')
    require(trainer.train_dataloader.num_workers==0 and trainer.recovery_sampler is not None,
            'Driver replay cannot hide worker prefetch or an unrecoverable sampler')
    controller=trainer.baseline_controller
    immutable=dict(config=OmegaConf.to_container(trainer.config,resolve=True),
        data_identity=trainer.instruct_data_identity,method=controller.method,
        parameters=OmegaConf.to_container(controller.config.gigpo,resolve=True),
        loss_reduction=controller.config.loss_reduction,base_commit=controller.config.base_commit,
        source_commit=controller.config.algorithm_source_commit)
    original=dict(config=saved['config'],data_identity=saved['data_identity'],
        **{key:saved['algorithm'][key] for key in
            ('method','parameters','loss_reduction','base_commit','source_commit')})
    require(state_digest(immutable)==state_digest(original),'Driver algorithm/data/config changed; cannot replay')
    require(type(saved['iterator']['base_seed']) is int and type(saved['iterator']['num_yielded']) is int
            and saved['iterator']['num_yielded']>=0,'Invalid actual iterator progress')
    trainer.recovery_sampler.load_state_dict(saved['sampler'])
    vars(trainer.kl_ctrl).clear();vars(trainer.kl_ctrl).update(copy.deepcopy(saved['kl_controller']))
    controller.stage.load_state_dict(copy.deepcopy(saved['algorithm']['stage']))
    trainer.global_steps=saved['logical_step'];trainer.actual_outer_updates=saved['actual_new_updates']
    trainer.quality_health=copy.deepcopy(saved['quality_health'])
    trainer.validation_trace_serial=saved['validation_trace_serial']
    iterator=iter(trainer.train_dataloader)
    iterator._base_seed=saved['iterator']['base_seed']
    iterator._num_yielded=saved['iterator']['num_yielded']
    restore_rng(saved['rng'])
    now=capture_driver_state(trainer,iterator)
    require(state_digest(now)==state_digest(saved),'Actual saved driver/RNG/cursor roundtrip differs')
    return iterator
