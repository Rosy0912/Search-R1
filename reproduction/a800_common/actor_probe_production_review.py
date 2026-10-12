"""Actual prepared deployment and independent review, before any fresh GPU claim."""
import importlib.metadata
import os
import sys
from pathlib import Path

from reproduction.common8.actor_probe_rank_tail import _sealed_bytes
from reproduction.common8.policy_calibration_plan import _strict_json,document_sha256
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_contract import actor_probes
from search_g1.actor_probe_recovery import identity

PREPARED='logs/actor-probe-production-prepared.json'
REVIEW='logs/actor-probe-production-deployment-review.json'
MODE='remote_prepared_actor_probe_v1'


def read(base,name):
    raw,file=_sealed_bytes(Path(base),name,limit=32*1024**2)
    return _strict_json(raw),file


def unlaunched(base,run,*,pipeline_entry=False):
    for name in ('logs','artifacts','checkpoints'):
        path=base/name;assert path==path.resolve() and path.is_dir() and path.stat().st_uid==os.getuid()
    assert actor_probes(run) and run.get('actor_probe_recovery') is None
    assert run['completed_updates']==0 and run['initial_probe_version'] is None
    for name in ('pipeline-launch-claim.json','training-launch-claim.json','process-ledger.json',
                 'native-allocation-admission.json','pipeline-status.json','actual-updates.jsonl'):
        path=base/'logs'/name;assert not path.exists() and not path.is_symlink(), 'Deployment already attempted; observe its exact handle'
    assert not Path(run['train_log']).exists() and not Path(run['train_log']).is_symlink()
    assert not list((base/'artifacts').iterdir()) and not list((base/'checkpoints').iterdir())
    assert not list((base/'logs').glob('gpu-samples-*.jsonl'))
    if pipeline_entry:
        claim,_=read(base,'logs/prepared-launch-claim.json')
        assert claim['run_id']==run['run_id'] and claim['execution_id']==run['execution_id']
        assert claim['runtime_source_commit']==run['source_commit'] and claim['no_automatic_retry'] is True
        # Popen's launch receipt may still be fsyncing in the SSH supervisor.
        # The exclusive prepared claim exists before Popen and is mandatory.
        path=base/'logs/deployment-launch-receipt.json'
        if path.exists():
            receipt,_=read(base,'logs/deployment-launch-receipt.json')
            from reproduction.a800.owned_processes import identity as process_identity
            actual=process_identity(os.getpid())
            assert receipt['pid']==os.getpid() and receipt['sid']==receipt['pid']
            assert receipt['start_ticks']==actual['start_ticks'] and receipt['sid']==actual['sid']
            assert actual['uid']==os.getuid()
            assert all(receipt[key]==run[key] for key in ('run_id','execution_id','source_commit'))
    else:
        for name in ('prepared-launch-claim.json','deployment-launch-receipt.json','deployment-pipeline.log'):
            path=base/'logs'/name;assert not path.exists() and not path.is_symlink()


def environment_versions():
    return {name:importlib.metadata.version(name) for name in
        ('torch','vllm','hydra-core','transformers','ray','scikit-learn','pyarrow','numpy')}


def actual_preparation(base,run,*,pipeline_entry=False):
    """CPU/file/config reads only; does not allocate CUDA or start components."""
    from reproduction.common8.actor_probe_head_admission import verify_inputs_and_source
    from reproduction.common8.configuration import resolved
    from reproduction.common8.formal_inline_safety import verify_method
    from reproduction.common8.formal_contract import verify_formal_config
    from reproduction.a800_common.predecessor_admission import admit
    from reproduction.common8.prepare_run import verify_model
    from omegaconf import OmegaConf
    base=Path(base);unlaunched(base,run,pipeline_entry=pipeline_entry)
    assert Path(__file__).resolve().parents[2] == base/'repo'
    intent,intent_file=read(base,'preparation/deployment-intent.json')
    assert intent['preparation_mode']==MODE and intent['reward_schedule']=='g1_through100_then_binary'
    assert intent['source_commit']==run['source_commit'] and intent['source_sha256']==run['source_sha256']
    assert intent['base']==str(base) and intent['predecessor_completion']==run['predecessor_completion']
    verify_inputs_and_source(base,run)
    for row in intent['inputs']:
        path=base/row['target'];assert path==path.resolve() and path.is_file()
        assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
    model=verify_model(Path(run['initial_model']),base/'preparation/official-model-manifest-original.json')
    assert model==run['initial_model_sha256'] and run['initial_model']==run['reference_model']
    method,method_file=read(base,'preparation/formal-inline-method.json');verify_method(method,run)
    bound,config,argv=resolved(base);assert bound==run
    document=OmegaConf.to_container(config,resolve=True);formal=verify_formal_config(document,run,base)
    predecessor=admit(base,run)
    return dict(role='actual_remote_prepared_actor_probe_deployment',identity=identity(run),
        deployment_id=intent['deployment_id'],source_map_sha256=document_sha256(run['source_sha256']),
        intent_file=intent_file,method_file=method_file,input_sha256=run['input_sha256'],
        official_model_sha256=model,resolved_config=document,actual_argv_overrides=argv,formal_contract=formal,
        predecessor={k:predecessor[k] for k in ('predecessor','actual_complete_outers','actual_optimizer_writes','scope')},
        environment_versions=environment_versions(),python_executable=run['python_executable'],
        output_paths={name:run[name] for name in ('checkpoint_dir','train_log','ray_tmp_dir')},
        training_physical_indices=run['training_physical_indices'],auxiliary_physical_indices=run['auxiliary_physical_indices'],
        excluded_physical_indices=run['excluded_physical_indices'],same_process_after_five=True,
        exact_remote_inputs_verified=True,source_verified=True,initial_actor='official fresh',
        inherited_training_state=False,services_started=False,trainer_started=False,GPU_claimed=False,
        initial_probe_fitting_exercised=False,actor_RPC_exercised=False,native_algorithm_accepted=False,
        production_launch_admission=False)


def verify_documents(review,prepared,run,prepared_sha):
    assert review['role']=='independent_actual_actor_probe_production_deployment_review' and review['accepted'] is True
    assert review['identity']==prepared['identity']==identity(run)
    assert review['prepared_receipt_sha256']==prepared_sha
    assert review['source_map_sha256']==prepared['source_map_sha256']==document_sha256(run['source_sha256'])
    assert review['independent_initialization_without_parent_context'] is True
    assert review['actual_prepared_deployment_reviewed'] is True and review['actual_source_reviewed'] is True
    assert isinstance(review['reviewer_task'],str) and review['reviewer_task'].startswith('/root/')
    assert isinstance(review['review_summary'],str) and review['review_summary'].strip()
    assert review['native_algorithm_accepted'] is False
    return review


def verify(base,run,*,pipeline_entry=False):
    base=Path(base)
    prepared,file=read(base,PREPARED);review,_=read(base,REVIEW)
    verify_documents(review,prepared,run,file['sha256'])
    actual=actual_preparation(base,run,pipeline_entry=pipeline_entry)
    assert prepared['preparation']==actual, 'Actual deployment changed after the independent review'
    disk,disk_file=read(base,'logs/checkpoint-disk-admission.json')
    assert disk_file['sha256']==prepared['disk_admission_sha256']
    assert disk['required_free_bytes']==prepared['required_free_bytes']
    assert disk['run_id']==run['run_id'] and disk['accepted'] is True
    import shutil
    assert shutil.disk_usage(base).free>=disk['required_free_bytes'], 'Checkpoint space changed after review'
    return dict(independent_actual_deployment_review_verified=True,prepared_receipt_sha256=file['sha256'])


def verify_before_popen(base,run):
    """Shared prepared launcher gate, including direct callers of its code."""
    assert actor_probes(run)
    assert Path(sys.prefix).resolve()==Path('/data/search-g1/env').resolve(), \
        'Actor-probe launcher must use the exact reviewed remote runtime'
    return verify(base,run)
