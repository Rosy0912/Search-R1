"""Checked PJ -> Lumia path projection; native full-state loaders stay unchanged."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from .pj_return_contracts import BASE, MODEL, PROFILES, project_config, validate_parent, validate_projection, mapped_manifest
from release.recursive_bran_a.ablation150_v1.contracts import read_receipt, sha

ENV = 'RECURSIVE_ABLATION_RETURN_BINDING'


def dump_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def load(path, verify_members=False):
    from omegaconf import OmegaConf
    path = Path(path).resolve()
    b = json.loads(path.read_text()); group = b['group']; p = PROFILES[group]
    stage = BASE / 'migrations' / p['stage']
    if path != stage / 'return-inputs.json' or path.is_symlink():
        raise ValueError('Unregistered migration binding')
    for name, digest in b['metadata_sha256'].items():
        if Path(name).name != name or sha(stage / name) != digest:
            raise ValueError('Migration metadata bytes changed')
    if sha(b['evidence_path'])!=b['evidence_sha256']:
        raise ValueError('Migration evidence changed')
    parent = json.loads((stage/'parent-configuration.json').read_text())
    entry = json.loads((stage/'parent-current-entry.json').read_text())
    original = json.loads((stage/'parent-manifest.json').read_text())
    validate_parent(parent, entry, original, group)
    if sha(stage/'parent-resolved.yaml') != p['config']:
        raise ValueError('Parent physical config bytes changed')
    record = read_receipt(stage/'return-configuration.json')
    if (record['source_commit'] != b['source_commit'] or record['run_id'] != p['run_id']
            or record['group'] != group or record['fresh_authorized'] is not False
            or record['parent_config_sha256'] != p['config']
            or record['input_binding_sha256'] != sha(path)
            or record['base'] != str(BASE/'runs'/group) or record['host'] != p['host']):
        raise ValueError('Continuation identity changed')
    validate_projection(OmegaConf.to_container(OmegaConf.load(stage/'parent-resolved.yaml'), resolve=True),
                        OmegaConf.to_container(OmegaConf.load(record['config']), resolve=True), group)
    for x in b['members']:
        f = stage/x['name']
        if (f.parent != stage or f.is_symlink() or not f.is_file() or f.stat().st_size != x['bytes']
                or (verify_members and sha(f) != x['sha256'])):
            raise ValueError('Incomplete or modified full-state member: '+x['name'])
    m = mapped_manifest(original, stage, b['members'])
    return b, record, m


def validate_probe(probe, binding, manifest, job):
    totals = manifest['identity']['guard_continue_accounting']['total_optimizer_calls_per_rank']
    if (probe['status'] != 'INDEPENDENT_EIGHT_GPU_COLD_STATE_EQUALITY_VERIFIED'
            or probe['job'] != job or probe['probe_source'] != binding['source_commit']
            or probe['parent_manifest_sha256'] != binding['metadata_sha256']['parent-manifest.json']
            or probe['completed_outer'] != PROFILES[binding['group']]['step']
            or probe['training_updates_executed'] != 0
            or len(probe['ranks']) != 8 or {x['rank'] for x in probe['ranks']} != set(range(8))
            or len({x['allocated_visible_device'] for x in probe['ranks']}) != 8):
        raise ValueError('Incomplete independent cold restoration')
    for row in probe['ranks']:
        expected = next(x for x in manifest['worker_states'] if x['identity']['rank'] == row['rank'])
        if (not row['actual_state_equal'] or row['observed_adam_steps'] != [totals[str(row['rank'])]]
                or row['scheduler_step'] != PROFILES[binding['group']]['step']
                or row['live_logical_sha256'] != expected['logical_state_sha256']
                or row['parent_sha256'] != expected['sha256']):
            raise ValueError('Cold worker equality or actual Adam count differs')
    if probe['driver_logical_sha256'] != manifest['driver']['logical_state_sha256']:
        raise ValueError('Driver cursor/RNG equality differs')


def validate_allocation(binding, record):
    from .ablation_entry import validate_allocation
    validate_allocation(record, os.environ, os.getuid(), socket.gethostname().split('.')[0])
    job = os.environ['SLURM_JOB_ID']
    submitted = json.loads((Path(record['input_binding']).parent/'return-submission.json').read_text())
    if (submitted['job_id'] != job or submitted['binding_sha256'] != sha(record['input_binding'])
            or os.environ['SLURM_JOB_NODELIST'] != PROFILES[binding['group']]['node']):
        raise ValueError('Only the unique registered eight-GPU allocation can continue')
    return job


def install():
    path = os.environ.get(ENV)
    if not path:
        return
    b, r, _ = load(path)
    submitted=json.loads((Path(path).parent/'return-submission.json').read_text())
    if (submitted['job_id']!=os.environ.get('SLURM_JOB_ID')
            or submitted['binding_sha256']!=sha(path)
            or r['host']!=socket.gethostname().split('.')[0] or os.getuid()!=1066):
        raise ValueError('Return worker is outside its registered allocation')
    if (r['source_commit'] != os.environ['SEARCH_R1_BASELINE_COMMIT']
            or r['config_sha256'] != os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256']
            or os.environ['RECURSIVE_ABLATION_RESUME_MANIFEST'] != str(Path(path).parent/'parent-manifest.json')
            or os.environ['RECURSIVE_ABLATION_RESUME_SHA256'] != b['metadata_sha256']['parent-manifest.json']):
        raise ValueError('Runtime continuation identity differs from frozen projection')
    from release.recursive_bran_a.ablation150_v1 import native_resume
    # Only file location admission is projected. Actual tensor identities, hashes,
    # full Adam/scheduler/RNG/cursor and live restored logical digests stay checked
    # by the original native load_worker/migrate_trainer implementations.
    native_resume.manifest = lambda: load(path)[2]


def worker_setup():
    install()
    from release.recursive_bran_a.ablation150_v1.bootstrap import install as native_install
    native_install()


def prepare(group, source, evidence_path):
    """One-time CPU preparation, after all nine destination files verify."""
    from omegaconf import OmegaConf
    from .prepare_ablation import DATA_SHA
    p = PROFILES[group]; stage = BASE/'migrations'/p['stage']; path = stage/'return-inputs.json'
    if path.exists() or (stage/'return-prepare-intent.json').exists():
        raise ValueError('Preparation already attempted; inspect, never replay')
    evidence_path = Path(evidence_path)
    evidence = json.loads(evidence_path.read_text())
    if (evidence['status'] != 'VERIFIED_MODEL_DATA_RETRIEVAL_AND_TERMINAL_PARENTS'
            or evidence['parents'][group]['job'] != p['job']
            or evidence['parents'][group]['observations'] != ['FAILED','FAILED']):
        raise ValueError('Require explicit terminal parent and destination identities')
    parent = json.loads((stage/'parent-configuration.json').read_text())
    entry = json.loads((stage/'parent-current-entry.json').read_text())
    m = json.loads((stage/'parent-manifest.json').read_text())
    validate_parent(parent, entry, m, group)
    if sha(stage/'parent-resolved.yaml') != p['config'] or len(source) != 40:
        raise ValueError('Parent config or frozen source missing')
    members=[]
    for row in m['worker_states']+[m['driver']]:
        name=Path(row['path']).name; f=stage/name
        if f.is_symlink() or not f.is_file() or sha(f) != row['sha256']:
            raise ValueError('Destination actual SHA not verified: '+name)
        members.append(dict(name=name,original=row['path'],bytes=f.stat().st_size,sha256=row['sha256']))
    dump_new(stage/'return-prepare-intent.json',dict(at=time.time(),group=group,source=source))
    base=BASE/'runs'/group
    for folder in ('receipts','output','logs','tmp','cache','checkpoints'):
        (base/folder).mkdir(parents=True,exist_ok=True)
    previous=base/'receipts/current-entry.json'; previous_sha=None
    if previous.exists():
        previous_sha=sha(previous); old=json.loads(previous.read_text()); job=str(old['allocation'])
        if not job.isdigit():raise ValueError('Unexpected existing Lumia predecessor')
        from .ablation_entry import check_previous_scheduler
        state=check_previous_scheduler(old,'not-a-job',lambda j:subprocess.check_output(
            ['sacct','-X','-n','-P','-j',j,'--format=JobIDRaw,State'],text=True))
        dump_new(stage/'previous-lumia-entry-proof.json',dict(entry=old,sha256=previous_sha,state=state))
    cfg=project_config(OmegaConf.to_container(OmegaConf.load(stage/'parent-resolved.yaml'),resolve=True),group)
    validate_projection(OmegaConf.to_container(OmegaConf.load(stage/'parent-resolved.yaml'),resolve=True),cfg,group)
    target_config=stage/'return-resolved.yaml'
    with target_config.open('x') as f:f.write(OmegaConf.to_yaml(OmegaConf.create(cfg)))
    retriever=BASE/'receipts/gpu-retriever-ready.json'
    if sha(retriever)!=evidence['retriever_receipt_sha256'] or sha(BASE/'common-data/manifest.json')!=DATA_SHA:
        raise ValueError('Destination retrieval/data identity changed')
    for x in evidence['model_files']:
        f=Path(MODEL)/x['file']
        if f.stat().st_size!=x['bytes'] or sha(f)!=x['sha256']:raise ValueError('Destination model/tokenizer changed')
    b=dict(version=1,group=group,source_commit=source,parent_job=p['job'],completed_outer=p['step'],
           members=members,previous_lumia_entry_sha256=previous_sha,
           evidence_path=str(evidence_path),evidence_sha256=sha(evidence_path),
           metadata_sha256={n:sha(stage/n) for n in ['parent-manifest.json','parent-resolved.yaml','parent-configuration.json','parent-current-entry.json','return-resolved.yaml']})
    dump_new(path,b)
    r=copy.deepcopy(parent)
    for key in ('checkpoint_storage_alias','input_binding','input_binding_sha256','cutover_sha256','probe_sha256','allocation','scheduler_node','configuration_host'):
        r.pop(key,None)
    r.update(base=str(base),output=str(base/'output'),config=str(target_config),config_sha256=sha(target_config),
        source_commit=source,model=MODEL,data=str(BASE/'common-data'),execution_backend='slurm_lumia',host=p['host'],
        retriever_receipt=str(retriever),retriever_receipt_sha256=sha(retriever),retriever_url=json.loads(retriever.read_text())['url'],
        fresh_authorized=False,training_submitted=False,parent_checkpoint=str(stage/'parent-manifest.json'),
        resume_manifest=str(stage/'parent-manifest.json'),resume_sha256=sha(stage/'parent-manifest.json'),
        parent_job=p['job'],parent_source_commit=m['identity']['code_commit'],parent_config_sha256=p['config'],
        parent_entry_sha256=sha(stage/'parent-current-entry.json'),input_binding=str(path),input_binding_sha256=sha(path),
        scope='Full native continuation with explicit physical path/host projection. Scientific configuration unchanged. Cross-hardware future bitwise equality not claimed.')
    dump_new(stage/'return-configuration.json',r)
    load(path,verify_members=False)
    # Preserve predecessor receipts before installing the continuation pointer.
    for name in ('configuration.json','resolved.yaml'):
        old=base/'receipts'/name
        if old.exists():
            with (stage/('previous-lumia-'+name)).open('xb') as f:f.write(old.read_bytes())
    from release.recursive_bran_a.ablation150_v1.progress import replace_json
    replace_json(base/'receipts/configuration.json',r)
    return path
