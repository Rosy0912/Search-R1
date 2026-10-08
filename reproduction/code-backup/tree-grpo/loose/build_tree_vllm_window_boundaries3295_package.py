"""Freeze a new diagnostic's code and metadata; no release or job submission."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True

WORK = Path(__file__).resolve().parent
WS = WORK.parent
ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner-commit', required=True)
    parser.add_argument('--inputs-plan', required=True, type=Path)
    parser.add_argument('--inputs-plan-sha', required=True)
    args = parser.parse_args()
    repo = WORK/'tree-vllm-boundary-window-runner'
    launcher = WORK/'tree-vllm-window-boundaries3295/launcher'
    assert len(args.runner_commit) == 40
    assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip() == args.runner_commit
    assert subprocess.check_output(['git','status','--porcelain'], cwd=repo, text=True) == ''
    # Verify and consume the same bytes, including the copy in the package.
    plan_bytes = args.inputs_plan.read_bytes()
    assert hashlib.sha256(plan_bytes).hexdigest() == args.inputs_plan_sha
    plan = json.loads(plan_bytes)
    adapter_sha = sha(launcher/'resource_adapter_window.py')
    name = 'single-tp1-window-'+args.runner_commit[:8]+'-'+adapter_sha[:8]
    stage = WORK/name
    remote = ROOT/'instruct300/candidates'/name
    assert not stage.exists(), 'Do not replace an immutable candidate'
    prior = WORK/'probability-probe-49a6b35-29822f9-r3'
    prior_manifest = read(prior/'package-manifest.json')
    terminal = read(WS/'outputs/tree-instruct-300/gpu3282/result.json')
    assert terminal['job_id'] == '3282' and terminal['actual_optimizer_writes'] == 0
    assert terminal['cleanup_residual'] == {} and terminal['original_background_pid_start_ticks_preserved'] is True
    accepted = read(WS/'outputs/tree-instruct-300/PPO_PROBE_GPU3282_INDEPENDENT_MEASUREMENT_RECEIPT.json')
    assert accepted['job_id'] == '3282' and accepted['actual_measurement_execution_verified'] is True
    assert accepted['training_accepted'] is False and accepted['probability_gate_passed'] is False
    stage.mkdir()
    archive = WORK/(name+'-source.tar')
    subprocess.run(['git','archive','--format=tar','--output',str(archive),args.runner_commit],cwd=repo,check=True)
    (stage/'repo').mkdir()
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            assert member.isdir() or member.isfile()
            assert Path(member.name).suffix not in {'.pt','.safetensors','.parquet','.bin','.so'}
        stream.extractall(stage/'repo')
    archive.unlink()
    for filename in ['start.sh','start.py','supervisor.py','probe_package.py','dispatch_once.py',
                     'freeze_package.py','resource_adapter_window.py']:
        shutil.copy2(launcher/filename,stage/filename)
    audits = stage/'audits'
    audits.mkdir()
    for filename in ['PPO_PROBE_GPU3282_INDEPENDENT_MEASUREMENT_RECEIPT.json',
                     'PPO_VLLM_WINDOW_CORE_INDEPENDENT_RECEIPT.json',
                     'PPO_INSTALLED_KERNEL_BINARY_METADATA_RAW.json',
                     'PPO_WINDOW_GPU3295_INDEPENDENT_FULL_RAW_RECEIPT.json',
                     'PPO_WINDOW_GPU3295_CPU_EXTREMES.json']:
        shutil.copy2(WS/'outputs/tree-instruct-300'/filename,audits/filename)
    (audits/'metadata-only-input-plan.json').write_bytes(plan_bytes)
    sources = [dict(path=str(remote/'repo'/p.relative_to(stage/'repo')),sha256=sha(p))
               for p in sorted((stage/'repo').rglob('*')) if p.is_file()]
    bundle = copy.deepcopy(read(prior/'bundle-template.json'))
    bundle.update(runner_commit=args.runner_commit,probe_source_files=sources,
                  reviewed_resource_adapter=dict(path=str(remote/'resource_adapter_window.py'),sha256=adapter_sha),
                  runner_module='single_window_runner',window_inputs=plan)
    # The new launcher owns the exact count/profile contract; no guessed schema.
    sys.path.insert(0,str(stage))
    import probe_package
    bundle.update(window_profile=probe_package.WINDOW_PROFILE,
                  diagnostic_profile=probe_package.WINDOW_PROFILE,
                  window_contract=probe_package.WINDOW_CONTRACT,
                  window_inputs_sha256=probe_package.digest(plan))
    probe_package.validate_window_inputs(plan)
    installed = {}
    for folder in ['tree-vllm-installed-source-20261006','tree-vllm-installed-source-extra-20261006',
                   'tree-vllm-installed-engine-source-20261006','tree-vllm-installed-sampling-source-20261006',
                   'tree-vllm-installed-hook-base-source-20261006']:
        for relative, entry in read(WORK/folder/'SOURCE_MAP.json').items():
            assert relative not in installed
            installed[relative] = {key:entry[key] for key in ('path','sha256')}
    wrappers = read(WORK/'tree-vllm-installed-wrapper-source-20261006/SOURCE_MAP.json')
    installed['utils.py'] = {key:wrappers['vllm/utils.py'][key] for key in ('path','sha256')}
    bundle['vllm_installed_sources'] = installed
    boundary_sources = read(WORK/'tree-native-boundary-installed-source-20261006/SOURCE_MAP.json')
    hf_body = read(WORK/'tree-native-hf-body-source-20261006/SOURCE_MAP.json')
    bundle['boundary_method_sources'] = {
        'torch_module': {key:boundary_sources['torch/nn/modules/module.py'][key] for key in ('path','sha256')},
        'actor_norm': {key:hf_body['transformers/models/qwen2/modeling_qwen2.py'][key] for key in ('path','sha256')},
        'actor_embedding': {key:boundary_sources['torch/nn/modules/sparse.py'][key] for key in ('path','sha256')},
        'vllm_custom_op': {key:boundary_sources['vllm/model_executor/custom_op.py'][key] for key in ('path','sha256')},
        'vllm_norm': installed['model_executor/layers/layernorm.py'],
        'vllm_embedding': installed['model_executor/layers/vocab_parallel_embedding.py'],
    }
    bundle['torch_wrapper_sources'] = {'torch/utils/_contextlib.py':
        {key:wrappers['torch/utils/_contextlib.py'][key] for key in ('path','sha256')}}
    kernels = read(WS/'outputs/tree-instruct-300/PPO_INSTALLED_KERNEL_BINARY_METADATA_RAW.json')
    bundle['installed_kernel_binaries'] = kernels['records']
    assert bundle['installed_kernel_binaries'] == probe_package.KERNEL_BINARY_PINS
    facade = 'verl/third_party/vllm/vllm_v_0_6_3/llm.py'
    facade_entry = dict(path=str(remote/'repo'/facade),sha256=sha(stage/'repo'/facade))
    bundle['native_method_sources'] = dict(engine_add_request=installed['engine/llm_engine.py'],
                                         model_runner=installed['worker/model_runner.py'],facade=facade_entry)
    core = stage/'repo/vllm_window_weights.py'
    assert sha(core) == '46ef590201bfd34014ca5efdafd44a20907a242b5dc520c38ebfb8c5e206a350'
    bundle['weights_core'] = dict(path=str(remote/'repo/vllm_window_weights.py'),sha256=sha(core))
    (stage/'bundle-template.json').write_text(json.dumps(bundle,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    external = copy.deepcopy(prior_manifest['external_files'])
    external.pop(str(ROOT/'runs/tree-probability-probe-3280/result.json'),None)
    external[str(ROOT/'runs/tree-probability-probe-3282/result.json')] = sha(WS/'outputs/tree-instruct-300/gpu3282/result.json')
    external.update({entry['path']:entry['sha256'] for entry in bundle['boundary_method_sources'].values()})
    spec = {key:bundle[key] for key in ['runner_commit','reviewed_resource_adapter','probe_source_files',
                                      'installed_sources','installed_versions']}
    spec.update(external_files=external,torchrun_port=26431,
                diagnostic_profile=bundle['diagnostic_profile'],window_contract=bundle['window_contract'],
                window_inputs_sha256=bundle['window_inputs_sha256'],runner_module=bundle['runner_module'],
                vllm_installed_sources=bundle['vllm_installed_sources'],
                installed_kernel_binaries=bundle['installed_kernel_binaries'])
    import freeze_package
    manifest = freeze_package.build_manifest(stage,spec)
    saved = probe_package.atomic_json(stage/'package-manifest.json',manifest)
    result = dict(status='SINGLE_WINDOW_SOURCE_PACKAGE_FROZEN_WITHOUT_DISPATCH_RELEASE',
                  local_package=str(stage),remote_package=str(remote),manifest_sha256=saved['sha256'],
                  bundle_payload_sha256=manifest['bundle_payload_sha256'],files=len(manifest['files']),
                  runner_commit=args.runner_commit,adapter_sha256=adapter_sha,
                  inputs_plan_file_sha256=args.inputs_plan_sha,window_inputs_sha256=bundle['window_inputs_sha256'],
                  fixed_dispatch_intent=probe_package.DISPATCH_INTENT,actual_gpu_submitted=False,
                  actual_model_forwards=0,planned_optimizer_writes=0,actual_optimizer_writes=None)
    destination = WORK/'tree-vllm-window-boundaries3295/PACKAGE_BUILD.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
