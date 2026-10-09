"""CPU preparation for retrying only evaluation10 after the actual train10 close."""
import argparse
import copy
from pathlib import Path
import uuid
try:
    from .utils import *
    from . import prepare
except ImportError:
    from utils import *
    import prepare

def make(previous_ref,base,helper_root,output):
    old,op=checked_json(previous_ref);train,tf=checked_json(old['training_config'])
    require(old['schema']=='opsd-pr100-periodic-pipeline-v1' and old['state']=='READY'
        and train['schema']=='opsd-pr100-config-v1' and train['variant']=='PR','Wrong original PR100 pipeline')
    base=Path(base).resolve();helper_root=Path(helper_root).resolve()
    require(helper_root.name=='opsd_pr100_periodic_v1','Runtime final namespace must preserve original frozen training admission')
    require(base!=Path(old['output_root']).resolve(),'Recovery cannot overwrite the old pipeline')
    evaluations=[]
    for n in prepare.BOUNDARIES:
        cfg=prepare.template(tf,base/f'eval-{n:03d}',helper_root,n)
        ref=atomic_json(base/'receipts'/f'eval-{n:03d}-template.json',cfg,exclusive=True)
        evaluations.append(dict(completed_outers=n,template=ref))
    value=copy.deepcopy(old)
    value.update(state='NOT_READY',run_id=uuid.uuid4().hex,source_commit=None,helper_source_manifest=None,publication=None,
        helper_root=str(helper_root),output_root=str(base/'pipeline'),evaluations=evaluations,
        # The same lock serializes old/new exact pipeline owners. The old owner
        # must actually terminate; backup never grants that resource transition.
        pipeline_lock=old['pipeline_lock'],mirror_root=str(base/'boundary-checkpoint-copies'))
    value['recovery']=dict(schema='opsd-pr100-evaluation10-recovery-v1',previous_pipeline_config=op,
        previous_supervisor_terminal=None,completed_training_terminal=None,completed_boundary=None,
        failed_evaluation_actual_exit=None,previous_cpu_registry=None,observer_binding=None,
        old_mirror_returncode='UNKNOWN',skip_train10=True,reuse_live_observer=True,
        preserve_original_failed_evaluation=True,unchanged_training_config=tf,no_checkpoint_reset=True)
    ref=atomic_json(output,value,exclusive=True)
    return dict(accepted=True,ready=False,gpu_started=False,pipeline_template=ref,evaluation_templates=evaluations,
        training_config=tf,first_action='Existing accepted train10 -> fresh evaluation512; no sampling/training0 replay')

def main():
    p=argparse.ArgumentParser()
    for name in ('previous-pipeline-config','previous-pipeline-config-sha256','base','helper-root','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(make(dict(path=a.previous_pipeline_config,sha256=a.previous_pipeline_config_sha256),a.base,a.helper_root,a.output)))
if __name__=='__main__':main()
