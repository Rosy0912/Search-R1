"""Failclosed production-owner candidate: one32B CPU copy/eight Unix endpoints.

No submission/approval generation; requires new reviewed32B proofs before load.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')
PIPELINE_PINS={'pipeline.py':'8d7a847daa6ffc84d62c24c0cdb6250c285c8433883fc37fdfe72b6f0254edde',
 'capacity_reference.py':'2d5da3fdd94b637c1625677b302df4df21395ef138f9de2ce484035eab7298ea',
 'native_gate.py':'c945b54294a4e7d222aa6a8c964f2ac8aa5349c0a3d3199532fe0dbba18e95b0'}
METHOD='e23b1cdb73736723ff422a9c041f2678b5e8f04cb9a496bd219ad87827b5ef7d'


def require(condition,message):
    if not condition:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def fresh_files(directory,files):
    require(isinstance(files,dict) and files,'Fresh complete file inventory required')
    directory=Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),'Fresh model/source directory missing')
    for name,digest in files.items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,'Nonlocal inventory member')
        path=directory/relative
        require(path.is_file() and all(not p.is_symlink() for p in [path]+[a for a in path.parents if directory in a.parents]),'Missing/symlink inventory member')
        expected = digest['sha256'] if isinstance(digest, dict) else digest
        if isinstance(digest, dict):
            require(path.stat().st_size == digest['size'], 'Fresh file size mismatch: '+name)
        require(sha(path)==expected,'Fresh file SHA mismatch: '+name)


def owner_assets(manifest,repo,pipeline_directory):
    from topd32b_shared_pipeline.readiness import validate_training_readiness,artifact
    require(manifest['kind']=='NATIVE_32B_TOPD_SHARED_PIPELINE','Hardware/CPU fixture cannot start production owner')
    proof=validate_training_readiness(manifest)
    source=artifact(manifest['source_manifest'])
    repo=Path(repo).resolve()
    require(repo==ROOT/'repo-shared32b','Only new own shared32B checkout permitted')
    require(source['commit']==manifest['commit']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'Actual source commit drift')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),'Dirty production source forbidden')
    require({'search_r1/baselines/controller.py','verl/trainer/ppo/ray_trainer.py','search_r1/baselines/distillation.py'}<=set(source['files']),'Incomplete integrated source binding')
    fresh_files(repo,source['files'])
    provenance=artifact(proof['teacher_provenance'])
    inventory=artifact(provenance['model_inventory'])
    require(inventory['status']=='FULL_HF_FILES_VERIFIED','Canonical complete HF inventory required')
    model=Path(inventory['model_directory']).resolve()
    require(ROOT in model.parents,'Remote/non-own teacher forbidden')
    require(provenance['model_inventory']['sha256']==proof['hf_inventory_sha256'], 'Canonical HF identity mismatch')
    fresh_files(model,inventory['files'])
    cfg=json.loads((model/'config.json').read_text())
    require(cfg['model_type']=='qwen2' and cfg['vocab_size']==152064 and cfg['num_hidden_layers']==64,'Actual full32B Qwen2 config required')
    directory=Path(pipeline_directory).resolve()
    require(directory==ROOT/'tools/topd-32b-pipeline-candidate','Own reviewed pipeline directory only')
    fresh_files(directory,PIPELINE_PINS)
    return proof,model,source


def pre_load_runtime_gate(manifest):
    import socket
    from topd32b_shared_pipeline.protocol import process_identity, in_parent_scope
    from search_r1.baselines.topd_local_gpu import lease_ownership
    require(socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11', 'Own native node required')
    require(os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_GPUS_ON_NODE','0'))==8
            and len(os.environ.get('SLURM_JOB_GPUS','').split(','))==8,'Actual allocated8 GPUs required before CPU load')
    require(int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=384*1024,'Actual384GiB owner/training budget required')
    require(in_parent_scope(process_identity(os.getpid())['cgroups'],manifest['parent_cgroup']), 'Owner outside actual Slurm cgroup')
    require(os.environ.get('TOPD32B_OWNER_PARENT')==str(os.getppid())
            and in_parent_scope(process_identity(os.getppid())['cgroups'],manifest['parent_cgroup']),
            'Actual direct owned supervisor required before CPU load')
    leases=json.loads(os.environ.get('SEARCH_R1_GPU_LEASES','[]'))
    require(os.environ.get('SEARCH_R1_GPU_PREFLIGHT')=='1' and len(leases)==8
            and [r['uuid'] for r in leases]==manifest['leased_uuids']
            and len(set(manifest['leased_uuids']))==8,'Actual inherited8 ordered UUID leases required')
    for record in leases:lease_ownership(record)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==','.join(manifest['leased_uuids']),'Visible UUID order differs')


def require_source_identity(backend,manifest,source):
    require(backend.identity(manifest)['code_commit']==manifest['commit']==source['commit'],'Owner run/source identity')

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-template',required=True);p.add_argument('--directory',required=True)
    p.add_argument('--repo',default=str(ROOT/'repo-shared32b'));p.add_argument('--pipeline',default=str(ROOT/'tools/topd-32b-pipeline-candidate'))
    p.add_argument('--check-only',action='store_true');args=p.parse_args()
    require(os.environ.get('AGENT_ROOT')==str(ROOT) and os.environ.get('SLURM_JOB_ID'),'Own allocated Slurm required')
    require(all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')),'Strict offline required')
    template=Path(args.manifest_template).resolve();require(ROOT/'logs' in template.parents and not template.is_symlink(),'Own manifest template required')
    manifest=json.loads(template.read_text())
    from import_gate import verified_ipc
    backend=verified_ipc()
    from topd32b_shared_pipeline.protocol import process_identity,write_private_json,private_directory
    pre_load_runtime_gate(manifest)
    proof,model,source=owner_assets(manifest,args.repo,args.pipeline)
    if args.check_only:
        print(json.dumps(dict(scope='DEPENDENCIES_ONLY_NOT_OWNER_READY',status='DEPENDENCIES_VERIFIED',teacher_path=str(model))))
        return
    import torch
    require(not torch.cuda.is_initialized(),'Owner must load CPU before any stage')
    directory=private_directory(args.directory)
    require(ROOT/'logs' in directory.parents or str(directory).startswith('/dev/shm/topd32b-'),'Own short socket directory required')
    actual=process_identity(os.getpid());actual['uid']=os.getuid()
    manifest['owner']=actual
    manifest['endpoints']=[dict(rank=rank,pid=actual['pid'],starttime=actual['starttime'],socket=str(directory/('rank'+str(rank)))) for rank in range(8)]
    require_source_identity(backend,manifest,source)
    backend.validate_manifest(manifest)
    # Verify original function from the explicitly bound new checkout.
    from search_r1.baselines.distillation import FrozenTeacher
    require(Path(inspect.getfile(inspect.unwrap(FrozenTeacher.log_probs))).resolve().is_relative_to(Path(args.repo).resolve()),'Foreign original FrozenTeacher import')
    require(hashlib.sha256(inspect.getsource(FrozenTeacher.log_probs).encode()).hexdigest()==METHOD,'Original teacher math drift')
    from transformers import AutoTokenizer
    student=AutoTokenizer.from_pretrained(str(ROOT/'assets-root/assets/qwen2.5-3b'),local_files_only=True)
    if student.pad_token_id is None:student.pad_token_id=student.eos_token_id
    started=time.time();teacher=FrozenTeacher(str(model),student,'cpu')
    sys.path.insert(0,str(Path(args.pipeline).resolve()))
    import pipeline,capacity_reference
    require(sha(inspect.getfile(pipeline))==PIPELINE_PINS['pipeline.py'],'Foreign pipeline import')
    leases=json.loads(os.environ.get('SEARCH_R1_GPU_LEASES','[]'))
    runtime=pipeline.PipelineRuntime(leases,manifest['parent_cgroup'],os.environ['SLURM_JOB_ID'],ROOT)
    scorer=pipeline.PipelineScorer(teacher,runtime,capacity_reference.device_map_32b(),set(student.get_vocab().values()))
    require(scorer.initial['parameter_sha256']==proof['fp32_parameter_sha256'],'Actual loaded selected32B parameter SHA mismatch')
    require(not torch.cuda.is_initialized(),'CPU-ready must not stage GPU')
    from topd32b_shared_pipeline.readiness import artifact
    import context_policy
    provenance=artifact(proof['teacher_provenance'])
    gate_ref=proof['teacher_gate_source'];gate_path=Path(gate_ref['path']).resolve()
    require(gate_path==ROOT/'tools/topd-32b-native-score-gate-r1-v2/score_gate.py'
            and gate_ref['sha256']=='fbc1507e5a1f1e6f16ba588763e58670f67436276191b8c19fbb4a765387097f'
            and sha(gate_path)==gate_ref['sha256'],'Canonical V7 gate source drift before context phase')
    gate_spec=importlib.util.spec_from_file_location('owner_context_canonical_gate',gate_path)
    canonical_gate=importlib.util.module_from_spec(gate_spec);gate_spec.loader.exec_module(canonical_gate)
    policies=context_policy.policies(gate_path.with_name('caller_contracts.py'),canonical_gate)
    policies['same_cpu_identity'](scorer.initial,provenance)
    authorization_identity=dict(pid=actual['pid'],starttime_ticks=actual['starttime'])
    write_private_json(directory,'owner-cpu-ready.json',dict(status='CPU_READY_NOT_GPU_SERVICE_PERMISSION',owner=actual,
        initial_cpu=scorer.initial,teacher_receipt_sha256=manifest['teacher_receipt_sha256'],context_policy_sha256=context_policy.CALLER_SHA))
    context_policy.wait_permission(directory/'allow-context.json',policies,'CONTEXT_ONLY',authorization_identity,
        manifest['teacher_receipt_sha256'],os.environ['SLURM_JOB_ID'])
    # Fresh original absolute dual15 check before the owner's first empty CUDA context.
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(8)
    context_ready=context_policy.empty_contexts(runtime,torch,manifest['leased_uuids'],authorization_identity,policies)
    write_private_json(directory,'owner-context-ready.json',context_ready)
    context_policy.wait_permission(directory/'allow-service.json',policies,'SCORING_ONLY',authorization_identity,
        manifest['teacher_receipt_sha256'],os.environ['SLURM_JOB_ID'])
    coordinator=backend.Coordinator(scorer,manifest,directory);server=backend.RankUnixServer(coordinator)
    try:
        server.start()
        write_private_json(directory,'manifest.json',manifest)
        write_private_json(directory,'owner-service-ready.json',dict(status='EMPTY8_CUDA_CONTEXTS_VERIFIED_SERVICE_READY_NOT_TRAINING_ACCEPTANCE',owner=actual,
            gpu_owner_process_count=1,rank_endpoint_count=8,initial_cpu=scorer.initial,
            load_cpu_wall_seconds=time.time()-started,context_ready=context_ready,source_commit=source['commit'],teacher_receipt_sha256=manifest['teacher_receipt_sha256']))
        server.serve()
        write_private_json(directory,'owner-done.json',dict(status='ALL_OUTER_PHASES_RETURNED_UNREVIEWED',owner=actual,completed_outer_updates=coordinator.step-1))
    except BaseException as exc:
        coordinator.fail(repr(exc));raise
    finally:server.close()


if __name__=='__main__':main()
