"""Explicit Lumia-to-PT transport of byte-identical config and full native state.

Only storage aliases and host admission change. Restoration still uses all
native model/Adam/scheduler/RNG/cursor equality checks. No fresh reset is allowed.
"""
import copy
import json
import os
from pathlib import Path
import socket
import time
from release.recursive_bran_a.ablation150_v1.contracts import sha
from .prepare import BASE,MODEL

OLD_BASE=Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1')
OLD_MODEL=Path('/home/jmzhang/Workspace/Search-E1-20261001/models/Qwen2.5-3B-Instruct')
STAGE=BASE/'migrations/budget5-parent2-v1'
PARENT_JOB='138706'
PARENT_SOURCE='bb271db4b5c19555ade5c1afc0b2f287c46e42e2'
RUN_ID='02ccdba4db9f4e68aacbb7edfa3042ab'
CONFIG_SHA='be8bfd13c2ea9e077d055be2b230cefcac365b2ddcb9d7a25a60da442f05b372'
ENV='RECURSIVE_ABLATION_PJ_BINDING'


def validate_inputs(binding,record,manifest):
    identity=manifest['identity']
    if (record['group']!='budget-5' or record['run_id']!=RUN_ID
            or record['source_commit']!=PARENT_SOURCE or record['config_sha256']!=CONFIG_SHA
            or binding['parent_job']!=PARENT_JOB or binding['completed_outer']!=2
            or manifest['status']!='COMPLETE_SAME_WORLD_STATE_SAVED'
            or identity['job_id']!=PARENT_JOB or identity['code_commit']!=PARENT_SOURCE
            or identity['config_sha256']!=CONFIG_SHA or identity['data_identity']!=record['data_identity']
            or identity['completed_step']!=2 or identity['actual_update_limit']!=150):
        raise ValueError('Transport must preserve the exact authorized budget5 parent boundary')
    workers=manifest['worker_states']
    if len(workers)!=8 or {w['identity']['rank'] for w in workers}!=set(range(8)):
        raise ValueError('Missing complete eight-rank state')
    for w in workers:
        if (w['identity']['world_size']!=8 or w['actual_total_optimizer_calls']!=40
                or w['scheduler_last_epoch']!=2 or w['full_worker_state_finite'] is not True
                or any(w['identity'][k]!=v for k,v in identity.items())):
            raise ValueError('Worker native optimizer/scheduler/identity changed')
    if manifest['driver']['next_step']!=3:raise ValueError('Driver cursor boundary changed')
    expected={Path(r['path']).name:r for r in workers+[manifest['driver']]}
    if len(expected)!=9 or len(binding['members'])!=9:raise ValueError('Nine exact native members required')
    for m in binding['members']:
        old=expected[m['name']]
        if m['original']!=old['path'] or m['sha256']!=old['sha256']:
            raise ValueError('Native member binding differs')
    return True


def load_inputs(stage=STAGE,verify_members=False):
    stage=Path(stage);b=json.loads((stage/'inputs.json').read_text())
    for name,h in b['metadata_sha256'].items():
        if sha(stage/name)!=h:raise ValueError('Parent metadata bytes changed: '+name)
    if sha(stage/'resolved.yaml')!=CONFIG_SHA or sha(stage/'manifest.json')!=b['manifest_sha256']:
        raise ValueError('Frozen config/manifest bytes changed')
    r=json.loads((stage/'configuration.json').read_text());m=json.loads((stage/'manifest.json').read_text())
    validate_inputs(b,r,m)
    for member in b['members']:
        path=stage/member['name']
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=member['bytes']:
            raise ValueError('Missing/incomplete transferred native file')
        if verify_members and sha(path)!=member['sha256']:raise ValueError('Transferred native SHA differs')
    return b,r,m


def aliases():
    if os.getuid()!=0 or os.environ.get('SEARCH_R1_EXECUTION_BACKEND')!='pj_pt' or os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Storage aliases require a real allocated PT container')
    for old,new in [(OLD_BASE,BASE),(OLD_MODEL,Path(MODEL))]:
        old.parent.mkdir(parents=True,exist_ok=True)
        if old.exists() or old.is_symlink():
            if not old.is_symlink() or old.resolve()!=new.resolve():raise ValueError('Conflicting existing storage path')
        else:
            try:old.symlink_to(new,target_is_directory=True)
            except FileExistsError:
                if not old.is_symlink() or old.resolve()!=new.resolve():raise ValueError('Racing conflicting storage alias')


def validate_gate(gate,binding,probe):
    if (gate['parent_job']!=PARENT_JOB or gate['parent_state'] not in ('CANCELLED','COMPLETED','FAILED','TIMEOUT')
            or gate['completed_outer']!=2 or gate['latest_durable_outer']!=2
            or gate['manifest_sha256']!=binding['manifest_sha256']
            or gate['parent_entry_sha256']!=binding['parent_entry_sha256']
            or gate['target_job']!=probe['job'] or gate['probe_sha256']!=sha(STAGE/('result-'+probe['job']+'.json'))
            or probe['status']!='INDEPENDENT_EIGHT_GPU_COLD_STATE_EQUALITY_VERIFIED'
            or probe['training_updates_executed']!=0 or probe['completed_outer']!=2
            or probe['parent_manifest_sha256']!=binding['manifest_sha256']
            or len(probe['ranks'])!=8 or {r['rank'] for r in probe['ranks']}!=set(range(8))
            or not all(r['actual_state_equal'] and r['observed_adam_steps']==[40] for r in probe['ranks'])):
        raise ValueError('Need terminal exact parent and actual eight-GPU cold-state equality before continuation')


def prepare_resume():
    b,parent,m=load_inputs();job=os.environ['SEARCH_R1_JOB_ID'];aliases()
    probe=json.loads((STAGE/('result-'+job+'.json')).read_text())
    gate=json.loads((STAGE/'cutover.json').read_text());validate_gate(gate,b,probe)
    if probe['probe_source']!=os.environ['RECURSIVE_SOURCE_COMMIT']:raise ValueError('Probe and training source differ')
    if gate['target_job']!=job:raise ValueError('Cutover is not for this allocation')
    base=BASE/'runs/budget-5';base.mkdir(parents=True,exist_ok=False)
    for name in ('receipts','output','logs','tmp','cache','checkpoints'):(base/name).mkdir()
    config=base/'receipts/resolved.yaml';config.write_bytes((STAGE/'resolved.yaml').read_bytes())
    boundary=base/'checkpoints/native-rolling/global_step_2';boundary.mkdir(parents=True)
    for member in b['members']:os.link(STAGE/member['name'],boundary/member['name'])
    (boundary/'manifest.json').write_bytes((STAGE/'manifest.json').read_bytes())
    retriever=BASE/'receipts/pj-retriever-ready.json';retrieval=json.loads(retriever.read_text())
    if not retrieval['validation']['top3_identical']:raise ValueError('Unvalidated destination retrieval')
    record=dict(parent,base=str(base),output=str(base/'output'),config=str(config),source_commit=os.environ['RECURSIVE_SOURCE_COMMIT'],
        model=str(OLD_MODEL),data=str(OLD_BASE/'common-data'),execution_backend='pj_pt',allocation=job,
        scheduler_node=os.environ['RECURSIVE_PJ_SCHEDULER_NODE'],host=socket.gethostname().split('.')[0],
        configuration_host=parent['host'],fresh_authorized=False,parent_checkpoint=str(boundary/'manifest.json'),
        resume_manifest=str(boundary/'manifest.json'),resume_sha256=b['manifest_sha256'],parent_job=PARENT_JOB,
        parent_source_commit=PARENT_SOURCE,parent_entry_sha256=b['parent_entry_sha256'],
        input_binding=str(STAGE/'inputs.json'),input_binding_sha256=sha(STAGE/'inputs.json'),
        retriever_receipt=str(retriever),retriever_receipt_sha256=sha(retriever),retriever_url=retrieval['url'],
        cutover_sha256=sha(STAGE/'cutover.json'),probe_sha256=sha(STAGE/('result-'+job+'.json')),
        scope='Same frozen config bytes and full native continuation; actual new source/backend recorded; cross-hardware bitwise future rollouts not claimed.')
    receipt=base/'receipts/configuration.json';receipt.write_text(json.dumps(record,indent=2)+'\n');return receipt


def install():
    path=os.environ.get(ENV)
    if not path:return
    record=json.loads(Path(path).read_text());b,parent,_=load_inputs()
    if (record['run_id']!=RUN_ID or record['config_sha256']!=CONFIG_SHA or sha(record['config'])!=CONFIG_SHA
            or record['parent_source_commit']!=PARENT_SOURCE or record['parent_entry_sha256']!=b['parent_entry_sha256']
            or record['source_commit']!=os.environ['SEARCH_R1_BASELINE_COMMIT']
            or record['allocation']!=os.environ['SEARCH_R1_JOB_ID']
            or record['host']!=socket.gethostname().split('.')[0]):raise ValueError('PT continuation binding changed')
    from search_r1.baselines import local_runtime
    if getattr(local_runtime.check_local_runtime,'_pj_transport',False):return
    original=local_runtime.check_local_runtime
    def check(config,ray_nodes=None):
        if config.baselines.local_runtime_host!=parent['host']:raise ValueError('Frozen configuration host changed')
        projected=copy.deepcopy(config);projected.baselines.local_runtime_host=record['host']
        return original(projected,ray_nodes)
    check._pj_transport=True;local_runtime.check_local_runtime=check


def worker_setup():
    install()
    from release.recursive_bran_a.ablation150_v1.bootstrap import install as native_install
    native_install()
