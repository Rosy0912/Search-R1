"""Stage source and frozen metadata only; never create a release or submit GPU."""
import argparse,hashlib,json,shutil,subprocess,sys,tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parent
OWNED=HERE/'tree-probability-probe-owned'
COMMIT='49a6b35ad11d5ce6a027a18a80035577822ca8dc'
ADAPTER='29822f98f8749cdbede4f62c91a0745d241a71c4bd16ce7e52e2f58126c3e273'
NAME='probability-probe-49a6b35-29822f9-r3'
REMOTE=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates')/NAME
STAGE=HERE/NAME

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--failure-receipt',required=True,type=Path)
    parser.add_argument('--failure-receipt-sha256',required=True)
    args=parser.parse_args()
    assert sha(args.failure_receipt)==args.failure_receipt_sha256
    failure_dir=WORKSPACE/'outputs/tree-instruct-300/gpu3280'
    failure=read(failure_dir/'result.json')
    assert failure['job_id']=='3280' and failure['status']=='FAILED_OR_UNKNOWN_ZERO_UPDATE_GPU_PROBE'
    assert failure['error']=='Actual8rank diagnostic failed' and failure['cleanup_residual']=={}
    assert failure['original_background_pid_start_ticks_preserved'] is True
    assert failure['actual_optimizer_writes'] is None and failure['planned_optimizer_writes']==0
    header_proof=read(WORKSPACE/'outputs/tree-instruct-300/PPO_PROBE_GPU3280_FAILURE_METADATA_RAW.json')
    assert header_proof['host']=='h100-9'
    compiler=header_proof['actual_same_node_own_cpu_only_compiler_fixture']
    assert compiler['rc']==0 and compiler['no_output_file_or_gpu_or_training'] is True
    repo=HERE/'tree-probability-probe-runner-v3'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==COMMIT
    assert subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)==''
    assert sha(OWNED/'resource_adapter_v2.py')==ADAPTER
    assert not STAGE.exists(),'A package stage must not be replaced'
    STAGE.mkdir();(STAGE/'repo').mkdir()
    archive=HERE/(NAME+'-source.tar')
    subprocess.run(['git','archive','--format=tar','--output',str(archive),COMMIT],cwd=repo,check=True)
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            assert member.isdir() or member.isfile(),'Source archive aliases are refused'
        stream.extractall(STAGE/'repo')  # Only regular files/directories with checked relative member paths.
    archive.unlink()
    for path in sorted((OWNED/'supervisor-candidate-r3').iterdir()):
        if path.is_file():shutil.copy2(path,STAGE/path.name)
    shutil.copy2(OWNED/'resource_adapter_v2.py',STAGE/'resource_adapter_v2.py')
    (STAGE/'audits').mkdir()
    audits={
      'runner-v3-source-receipt.json':OWNED/'v2/RUNNER_V3_SOURCE_CPU_RECEIPT.json',
      'adapter-v2-source-receipt.json':OWNED/'v2/RESOURCE_ADAPTER_V2_SOURCE_CPU_RECEIPT.json',
      'selection-actual3277-receipt.json':HERE/'tree-probe-selection-review/v2/ACTUAL_CPU3277_RECEIPT.json',
      'prior-c542-refusal.json':OWNED/'pre-dispatch/RECEIPT.json',
      'prior-r2-release.json':OWNED/'pre-dispatch-r2/RELEASE.json',
      'actual3280-failure-receipt.json':args.failure_receipt,
      'actual3280-original-result.json':failure_dir/'result.json',
      'actual3280-original-launcher.out':failure_dir/'tree-probability-probe-3280.out',
      'actual3280-original-probe.log':failure_dir/'probe.log',
      'actual3280-gcc-and-header-cpu-proof.json':WORKSPACE/'outputs/tree-instruct-300/PPO_PROBE_GPU3280_FAILURE_METADATA_RAW.json',
      'adapter-v2-actual-linux-toy.json':WORKSPACE/'outputs/tree-instruct-300/PPO_PROBE_ADAPTER_V2_FDINFO_CPU_ENV_RAW.json',
    }
    for name,path in audits.items():shutil.copy2(path,STAGE/'audits'/name)
    selection=read(WORKSPACE/'outputs/tree-instruct-300/cpu3277/selection.json')
    selection_sha=sha(WORKSPACE/'outputs/tree-instruct-300/cpu3277/selection.json')
    assert selection_sha=='bca59a4508a0627a04b3f1f64e632b69488208650fa375aac52ab7d16029f995'
    config=(OWNED/'ORIGINAL3268_RESOLVED.yaml').read_bytes()
    assert hashlib.sha256(config).hexdigest()==selection['identity']['config_sha256']
    parent=read(HERE/'tree-safety110-runtime/HEALTHY110_PARENT.json')
    index=read(OWNED/'HF110_INDEX.json')
    installed=read(WORKSPACE/'outputs/tree-instruct-300/PPO_PROBE_INSTALLED_SOURCE_METADATA_RAW.json')
    sources=[dict(path=str(REMOTE/'repo'/p.relative_to(STAGE/'repo')),sha256=sha(p))
        for p in sorted((STAGE/'repo').rglob('*')) if p.is_file()]
    workers=[v for v in selection['identity']['source_files'] if v['path'].endswith('/repo/verl/workers/fsdp_workers.py')]
    assert len(workers)==1
    root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
    bundle=dict(status='GPU_PROBE_BUNDLE_INDEPENDENTLY_RELEASED',output='@OWN_RUN_OUTPUT@',
        identity=selection['identity'],selection=selection,inventory=selection['inventory'],
        selection_file=dict(path=str(root/'instruct300/probe-selection-v2-result/selection.json'),sha256=selection_sha),
        config_path=str(root/'runs/tree-safety300-3268/resolved.yaml'),config_text=config.decode('utf-8'),
        original_worker_path=workers[0]['path'],original_worker_sha256=workers[0]['sha256'],
        hf_path=str(root/'runs/tree-recovery300-3234/output/checkpoints/tree-recovery300/actor/global_step_110'),
        hf_identity=parent['hf_identity'],hf_index=index,parent_state_manifest=dict(path=parent['path']),
        probe_mode='eval_like_original_compute_log_prob',runner_commit=COMMIT,
        native_logprob_sha256=sha(STAGE/'repo/verl/utils/torch_functional.py'),probe_source_files=sources,
        reviewed_resource_adapter=dict(path=str(REMOTE/'resource_adapter_v2.py'),sha256=ADAPTER),
        installed_sources=installed['sources'],installed_versions=installed['versions'])
    (STAGE/'bundle-template.json').write_text(json.dumps(bundle,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    external=read(WORKSPACE/'outputs/tree-instruct-300/PPO_PROBE_EXTERNAL_METADATA_RAW.json')
    spec={k:bundle[k] for k in ('runner_commit','reviewed_resource_adapter','probe_source_files','installed_sources','installed_versions')}
    external_files={e['path']:e['sha256'] for e in external['records']}
    for name in ('python-dev/usr/include/python3.10/Python.h','python-dev/usr/include/python3.10/pyconfig.h',
                 'python-dev/usr/include/x86_64-linux-gnu/python3.10/pyconfig.h'):
        record=header_proof['headers'][name]
        assert record['exists'] and not record['is_symlink'] and record['resolved']==str(root/name)
        external_files[str(root/name)]=record['sha256']
    external_files[str(root/'runs/tree-probability-probe-3280/result.json')]=sha(failure_dir/'result.json')
    spec.update(external_files=external_files,torchrun_port=26411)
    sys.path.insert(0,str(STAGE))
    from freeze_package import build_manifest
    from probe_package import atomic_json,DISPATCH_INTENT
    assert DISPATCH_INTENT==str(root/'instruct300/PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json')
    manifest=build_manifest(STAGE,spec)
    saved=atomic_json(STAGE/'package-manifest.json',manifest)
    record=dict(status='SOURCE_PACKAGE_FROZEN_WITHOUT_GPU_RELEASE',local_package=str(STAGE),remote_package=str(REMOTE),
        manifest_sha256=saved['sha256'],bundle_payload_sha256=manifest['bundle_payload_sha256'],
        files=len(manifest['files']),runner_commit=COMMIT,adapter_sha256=ADAPTER,selection_sha256=selection_sha,
        actual_gpu_submitted=False,actual_model_forwards=0,actual_optimizer_writes=0,prior_failed_probe=3280,prior_failure_receipt_sha256=args.failure_receipt_sha256,fixed_dispatch_intent=DISPATCH_INTENT)
    (OWNED/'PACKAGE_BUILD_R3.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,sort_keys=True))

if __name__=='__main__':main()
