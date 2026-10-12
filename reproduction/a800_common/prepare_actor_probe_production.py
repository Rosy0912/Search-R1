"""Remote-only official fresh deployment; stop at a reviewable CPU preparation.

Canonical inputs copy within A800 from the exact accepted predecessor; source
copies from the separately verified published checkout. No QA/model/state or
environment bytes transit the Mac, and no GPU service/trainer is launched here.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from reproduction.a800_common.deploy import DATA_NAMES,OFFICIAL_SHA,SHARED_SHA
from reproduction.a800_common.actor_probe_production_review import MODE,PREPARED,actual_preparation,read
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.source_files import runtime_files
from reproduction.common8.verify_data import digest
from reproduction.common8.calibration_runtime import write_once
from search_g1.actor_probe_contract import experiment_fields
from search_g1.actor_probe_recovery import identity

TARGETS={*('data/'+name for name in DATA_NAMES),
    'preparation/official-model-manifest-original.json','preparation/formal-inline-method-template.json'}


def validate_intent(intent):
    from reproduction.a800_common.predecessor_admission import validate_binding
    assert intent['role']=='fresh_a800_common_deployment_intent' and intent['host']=='baai-a800'
    assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
    assert intent['base']=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id']
    assert re.fullmatch('[0-9a-f]{40}',intent['source_commit']) and intent['source_sha256']
    assert re.fullmatch('[0-9a-f]{64}',intent['source_staging_receipt_sha256'])
    for name,sha in intent['source_sha256'].items():
        path=Path(name)
        assert not path.is_absolute() and '..' not in path.parts and re.fullmatch('[0-9a-f]{64}',sha)
    assert intent['preparation_mode']==MODE and intent['reward_schedule']=='g1_through100_then_binary'
    assert intent['training_world_size']==4 and intent['training_physical_indices']==[0,1,2,3]
    assert intent['auxiliary_physical_indices']==[5,6] and intent['same_process_after_five'] is True
    assert intent['target_updates']==200 and 'initial_probes_source_base' not in intent
    parent=validate_binding(intent['predecessor_completion'])
    stage=Path(intent['source_staging_base'])
    assert re.fullmatch('/data/shaoshitong/Search-G1-actor-probes-source-[0-9a-f]{32}',str(stage))
    assert len(intent['inputs'])==len(TARGETS) and {row['target'] for row in intent['inputs']}==TARGETS
    for row in intent['inputs']:
        assert row['remote_source']==str(Path(parent['base'])/row['target']) and 'source' not in row
        assert type(row['bytes']) is int and row['bytes']>0 and re.fullmatch('[0-9a-f]{64}',row['sha256'])
    by_target={row['target']:row for row in intent['inputs']}
    assert by_target['data/manifest.json']['sha256']==SHARED_SHA
    assert by_target['preparation/official-model-manifest-original.json']['sha256']==OFFICIAL_SHA
    return intent


def environment(base,ray=None):
    env=dict(os.environ)
    for key in tuple(env):
        if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH',
            'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    env.update(SEARCH_G1_BASE=str(base),SEARCH_G1_PROFILE='common8',SEARCH_G1_TRAINING_WORLD_SIZE='4',
        CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONPATH=str(base/'repo'),PATH='/data/search-g1/env/bin:'+env.get('PATH','/usr/bin:/bin'),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',
        C_INCLUDE_PATH='/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include',
        NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',NO_PROXY='127.0.0.1,localhost',RAY_DEDUP_LOGS='0')
    for name,value in {'TMPDIR':'tmp','HF_HOME':'cache/hf','XDG_CACHE_HOME':'cache',
        'TRITON_CACHE_DIR':'cache/triton','TORCHINDUCTOR_CACHE_DIR':'cache/inductor'}.items():env[name]=str(base/value)
    env['RAY_TMPDIR']=str(ray or base/'ray')
    return env


def copy_owned_input(source,target,expected):
    assert source==source.resolve() and source.is_file() and source.stat().st_uid==os.getuid()
    assert source.stat().st_size==expected['bytes'] and digest(source)==expected['sha256']
    assert target.parent==target.parent.resolve() and not target.exists() and not target.is_symlink()
    # Copy remote-to-remote; never hardlink writable run products to the parent.
    with source.open('rb') as src,target.open('xb') as dst:
        shutil.copyfileobj(src,dst,1024**2);dst.flush();os.fsync(dst.fileno())
    assert target.stat().st_size==expected['bytes'] and digest(target)==expected['sha256']


def verify_stage(intent):
    stage=Path(intent['source_staging_base']);repo=stage/'repo'
    assert stage==stage.resolve() and repo==repo.resolve()
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==intent['source_commit']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    actual={p.relative_to(repo).as_posix():digest(p) for p in runtime_files(repo)}
    assert actual==intent['source_sha256']
    receipt,file=read(stage,'logs/source-staging-complete.json')
    assert file['sha256']==intent['source_staging_receipt_sha256']
    assert receipt['role']=='a800_published_source_staging_only' and receipt['base']==str(stage)
    assert str(stage)=='/data/shaoshitong/Search-G1-actor-probes-source-'+receipt['staging_id']
    assert receipt['source_commit']==intent['source_commit'] and receipt['source_sha256']==actual
    assert receipt['exact_published_source_verified'] is True and receipt['trainer_started'] is False
    assert receipt['services_started'] is False and receipt['production_admission'] is False
    return repo


def layout(intent):
    """Validate actual predecessor completion before making the new namespace."""
    from reproduction.a800_common.predecessor_admission import admit
    intent=validate_intent(intent);stage_repo=verify_stage(intent)
    assert Path(__file__).resolve().parents[2]==stage_repo
    parent,_=read(Path(intent['predecessor_completion']['base']),'logs/run-manifest.json')
    # Admission reads actual accepted200/final/canonical receipts and process
    # births; a syntactically valid binding never substitutes for completion.
    prospective=dict(parent,run_id=intent['deployment_id'],execution_id=intent['deployment_id'],
        reward_schedule=intent['reward_schedule'],**experiment_fields(intent['reward_schedule']),
        predecessor_completion=intent['predecessor_completion'])
    prospective.pop('initial_probe_adoption_source_base',None)
    base=Path(intent['base']);admit(base,prospective)
    assert base.parent==base.parent.resolve() and base.parent.is_dir()
    base.mkdir(mode=0o700,exist_ok=False)
    for name in ('logs','preparation','data','tmp','ray','cache/triton','cache/inductor','cache/hf','artifacts','checkpoints'):
        (base/name).mkdir(parents=True,exist_ok=True)
    write_once(base/'preparation/deployment-intent.json',intent)
    for row in intent['inputs']:copy_owned_input(Path(row['remote_source']),base/row['target'],row)
    write_once(base/'logs/deployment-input-receipt.json',dict(deployment_id=intent['deployment_id'],
        inputs=[{key:row[key] for key in ('target','bytes','sha256')} for row in intent['inputs']],
        exact_original_remote_bytes=True,local_QA_transfer=False,training_data_partitioned=False))
    subprocess.run(['git','clone','--local','--no-hardlinks','--no-checkout',str(stage_repo),str(base/'repo')],check=True,capture_output=True)
    subprocess.run(['git','-C',str(base/'repo'),'checkout','--detach',intent['source_commit']],check=True,capture_output=True)
    actual={p.relative_to(base/'repo').as_posix():digest(p) for p in runtime_files(base/'repo')}
    assert actual==intent['source_sha256']
    write_once(base/'logs/deployment-source-receipt.json',dict(deployment_id=intent['deployment_id'],
        source_commit=intent['source_commit'],source_sha256=actual,source_transport='verified_published_remote_source_copy',
        source_is_exact_published_commit=True,uncommitted_source_copied=False))
    env=environment(base)
    with (base/'preparation/prepare-target.log').open('xb') as out:
        result=subprocess.run(['/data/search-g1/env/bin/python','-m','reproduction.a800_common.prepare_actor_probe_production',
            '--base',str(base)],cwd=base/'repo',env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,timeout=600)
    assert result.returncode==0, 'Remote CPU preparation failed; preserve this namespace and inspect the same handle'
    receipt,file=read(base,PREPARED)
    return dict(role='actor_probe_remote_prepared_for_independent_review',base=str(base),
        source_commit=intent['source_commit'],identity=receipt['identity'],prepared_receipt_sha256=file['sha256'],
        services_started=False,trainer_started=False,GPU_claimed=False,production_launch_admission=False)


def prepare(base):
    from reproduction.a800_common.prepare_run import prepare as prepare_run
    from reproduction.common8.formal_inline_safety import verify_method
    from reproduction.common8.disk_admission import admit as admit_disk
    base=Path(base).resolve();intent,_=read(base,'preparation/deployment-intent.json');validate_intent(intent)
    assert Path(__file__).resolve().parents[2]==base/'repo'
    assert digest(Path(__file__))==intent['source_sha256']['reproduction/a800_common/prepare_actor_probe_production.py']
    method,_=read(base,'preparation/formal-inline-method-template.json')
    method.update(source_commit=intent['source_commit'],source_map_sha256=document_sha256(intent['source_sha256']),created_unix=time.time())
    write_once(base/'preparation/formal-inline-method.json',method)
    run=prepare_run(base,Path('/data/search-g1/assets/qwen2.5-3b-instruct'),
        base/'preparation/official-model-manifest-original.json',intent['source_commit'],
        reward_schedule=intent['reward_schedule'],predecessor_completion=intent['predecessor_completion'])
    assert run['source_sha256']==intent['source_sha256'];verify_method(method,run)
    os.environ['RAY_TMPDIR']=run['ray_tmp_dir']
    disk=admit_disk(base);actual=actual_preparation(base,run)
    proof=dict(role='actor_probe_deployment_prepared_for_independent_review',identity=identity(run),
        source_map_sha256=actual['source_map_sha256'],preparation=actual,
        disk_admission_sha256=digest(base/'logs/checkpoint-disk-admission.json'),
        required_free_bytes=disk['required_free_bytes'],actual_free_bytes=disk['actual_free_bytes'],
        prepared=True,production_launch_admission=False,checked_unix=time.time())
    write_once(base/PREPARED,proof)
    write_once(base/'logs/deployment-status.json',dict(deployment_id=intent['deployment_id'],base=str(base),
        source_commit=intent['source_commit'],stage='PREPARED_FOR_INDEPENDENT_DEPLOYMENT_REVIEW',
        observed_unix=time.time(),run_id=run['run_id'],execution_id=run['execution_id'],trainer_started=False))
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);args=parser.parse_args()
    proof=prepare(args.base)
    print(json.dumps({key:proof[key] for key in ('role','identity','prepared','checked_unix')}))
