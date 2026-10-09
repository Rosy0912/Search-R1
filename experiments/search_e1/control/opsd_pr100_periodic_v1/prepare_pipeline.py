"""CPU-only templates; final publication/storage are supplied before READY."""
import argparse
from pathlib import Path
import uuid
try:
    from .utils import *
    from . import prepare
except ImportError:
    from utils import *
    import prepare

def make(training_ref,base,helper_root,output):
    train,tf=checked_json(training_ref);base=Path(base).resolve();helper_root=Path(helper_root).resolve()
    require(train['schema']=='opsd-pr100-config-v1' and train['variant']=='PR' and train['scientific']['outer_iterations']==100,'Wrong PR100 training')
    require(base!=Path(train['paths']['base']).resolve(),'Pipeline output cannot overwrite training')
    evaluations=[]
    for n in prepare.BOUNDARIES:
        cfg=prepare.template(tf,base/f'eval-{n:03d}',helper_root,n)
        ref=atomic_json(base/'receipts'/f'eval-{n:03d}-template.json',cfg,exclusive=True)
        evaluations.append(dict(completed_outers=n,template=ref))
    value=dict(schema='opsd-pr100-periodic-pipeline-v1',state='NOT_READY',run_id=uuid.uuid4().hex,study_id=train['study_id'],
        training_config=tf,source_commit=None,helper_source_manifest=None,publication=None,
        helper_root=str(helper_root),output_root=str(base/'pipeline'),pipeline_lock=str(base/'pipeline.lock'),
        evaluations=evaluations,baseline_summary=None,baseline_version='bf16_zero',startup_previous_driver_terminal=None,
        resources=dict(exclusive_gpu_sequence=True,parent_holds_gpu_locks=False),
        mirror_root=str(Path(train['paths']['base'])/'preserved'/'boundary-checkpoint-copies'),
        instructions=dict(outer_iterations=100,evaluate_every=10,primary_slots=5120,resume_same_training_run=True,
            evaluate_raw_boundary=True,no_best_fallback=True,no_additional_fp16_evaluation=True,no_new_baseline_generation=True,
            outputs_remote_only=True,no_checkpoint_transfer_start_condition=True))
    ref=atomic_json(output,value,exclusive=True)
    return dict(accepted=True,ready=False,gpu_started=False,pipeline_template=ref,evaluation_templates=evaluations)

def main():
    p=argparse.ArgumentParser()
    for name in ('training-config','training-config-sha256','base','helper-root','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(make(dict(path=a.training_config,sha256=a.training_config_sha256),a.base,a.helper_root,a.output)))
if __name__=='__main__':main()
