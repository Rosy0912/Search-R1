#!/usr/bin/env python3
"""Read-only c5bf native-CPU import/config preparation; no evaluation launch."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

SOURCE = 'c5bf4c8db33562a259fa227a940dc69543901b98'
PARENT = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
COMMON_SHA = 'e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581'
CONFIG_NAME = 'ppo_opsd_final200_eval'
NATIVE_PYTHON = '/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python'
INVENTORY_KIND = 'ACTUAL_FIXED_EVAL_SOURCE_INVENTORY'
SCOPE = 'NATIVE_CPU_C5BF_IMPORT_AND_PARTIAL_CONFIG_ONLY_NOT_EVALUATION'
ADDED_REQUIRED = ('search_r1.baselines.final200_evaluation',
    'search_r1.baselines.corrected_evaluation', 'search_r1.baselines.corrected_runtime_capture',
    'search_r1.baselines.evaluation', 'search_r1.baselines.sharedholdout',
    'search_r1.llm_agent.generation', 'search_r1.qa_protocol',
    'verl.utils.reward_score.qa_em', 'verl.trainer.main_ppo', 'verl.utils.dataset.rl_dataset')
FUTURE_INTERPOLATIONS = {
    'actor_rollout_ref.model.path': '${oc.env:OPSD_FINAL200_ACTUAL_MODEL_DIRECTORY}',
    'baselines.final200_evaluation.reviewed_stage_manifest': '${oc.env:OPSD_FINAL200_ACTUAL_REVIEWED_STAGE}',
    'baselines.final200_evaluation.input_manifest_path': '${oc.env:OPSD_FINAL200_INPUT_MANIFEST}',
    'baselines.final200_evaluation.input_manifest_sha256': '${oc.env:OPSD_FINAL200_INPUT_MANIFEST_SHA256}',
    'baselines.final200_evaluation.output_dir': '${oc.env:OPSD_FINAL200_ACTUAL_FRESH_OUTPUT_DIRECTORY}'}
RETRIEVER_DECLARATION = '${oc.env:OPSD_E1_RETRIEVER_URL,http://127.0.0.1:8000/retrieve}'


def pinned_module(path, expected_sha, name):
    """Bootstrap exact helper bytes before executing any imported helper."""
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('Helper must have an absolute physical path')
    fields = ('st_dev', 'st_ino', 'st_size', 'st_uid', 'st_mode', 'st_mtime_ns', 'st_ctime_ns')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_size > 32 * 1024**2:
        raise ValueError('Helper ownership/type/size differs')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        chunks = []
        while True:
            value = os.read(fd, 1024**2)
            if not value: break
            chunks.append(value)
        ended = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if any(getattr(before, k) != getattr(s, k) for s in (opened, ended, after) for k in fields):
        raise ValueError('Helper changed during read')
    raw = b''.join(chunks)
    if type(expected_sha) is not str or not re.fullmatch('[0-9a-f]{64}', expected_sha) or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Helper exact bytes differ before execution')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition('.')[0]
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def common():
    return pinned_module(Path(__file__).resolve().parent / 'common.py', COMMON_SHA, 'eval_cpu_exact_common')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo)] + list(args))


def tree(repo):
    result = {}
    for item in git(repo, 'ls-tree', '-rz', '--full-tree', 'HEAD').split(b'\0'):
        if not item: continue
        metadata, raw_name = item.split(b'\t', 1)
        mode, kind, blob = metadata.decode('ascii').split()
        name = raw_name.decode('utf-8')
        if kind != 'blob' or mode not in ('100644', '100755') or name in result:
            raise ValueError('Unsupported source tree member')
        result[name] = (mode, blob)
    return result


def source_inventory(c, path, sha):
    value, inventory_record = c.read_json(path, sha)
    c.require(type(value) is dict and set(value) == {'version', 'kind', 'source_commit', 'parent_commit', 'member_count', 'remote_repository', 'files'}, 'Source inventory schema differs')
    for key, target in dict(version=1, kind=INVENTORY_KIND, source_commit=SOURCE, parent_commit=PARENT, member_count=857).items():
        c.typed(value[key], target, 'Source inventory header differs: ' + key)
    c.require(type(value['remote_repository']) is str, 'Actual published repository is required, not a future null path')
    repo = Path(value['remote_repository'])
    c.require(repo.is_absolute() and repo.resolve(strict=True) == repo and repo.is_dir() and repo.stat().st_uid == os.getuid(), 'Actual source repository differs')
    c.typed(git(repo, 'rev-parse', 'HEAD').decode().strip(), SOURCE, 'Actual evaluation HEAD differs')
    c.typed(git(repo, 'rev-parse', 'HEAD^').decode().strip(), PARENT, 'Actual evaluation parent differs')
    c.require(not git(repo, 'status', '--porcelain', '--untracked-files=all').strip(), 'Actual evaluation source is not clean')
    entries = tree(repo)
    c.require(type(value['files']) is dict and len(entries) == 857 and set(value['files']) == set(entries), 'Full c5bf source tree inventory differs')
    records = {}
    inodes = set()
    for name, (mode, blob) in entries.items():
        record = value['files'][name]
        c.require(type(record) is dict and set(record) == {'git_mode', 'git_blob', 'sha256', 'bytes'}, 'Source member schema differs')
        c.typed(record['git_mode'], mode, 'Source mode differs')
        c.typed(record['git_blob'], blob, 'Source blob differs')
        c.require(type(record['bytes']) is int and record['bytes'] >= 0, 'Source member byte count type differs')
        raw, actual = c.read_file(repo / name, record['sha256'], repo, keep_bytes=True, limit=32*1024**2)
        c.typed(actual['bytes'], record['bytes'], 'Source member byte count differs')
        c.typed(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest(), blob, 'Source bytes differ from HEAD blob')
        inode = (actual['device'], actual['inode'])
        c.require(inode not in inodes, 'Source members alias one inode')
        inodes.add(inode)
        records[name] = actual
    for folder, dirs, files in os.walk(repo, followlinks=False):
        dirs[:] = [n for n in dirs if n != '.git']
        c.require(not any(n.endswith(('.pyc', '.pyo')) for n in files), 'Existing project bytecode cache is forbidden')
    c.verify_record(inventory_record)
    return repo, value, inventory_record, records


def selected(value, dotted):
    for key in dotted.split('.'):
        if type(value) is not dict or key not in value:
            raise ValueError('Missing partial configuration field: ' + dotted)
        value = value[key]
    return value


def structural_config(c, raw):
    """Inspect authored scalar meanings without fabricating future paths."""
    expected = {
        'trainer.val_only': True, 'trainer.val_before_train': True, 'trainer.total_training_steps': 0,
        'trainer.smoke_only': False, 'trainer.n_gpus_per_node': 8, 'trainer.nnodes': 1,
        'baselines.method': 'opsd', 'baselines.holdout_eval_only': False,
        'baselines.opsd_e1.enabled': False, 'baselines.opsd_e1.gate_required': False,
        'baselines.opsd_e1.profile': None, 'baselines.corrected_evaluation.enabled': False,
        'baselines.final200_evaluation.enabled': True,
        'baselines.local_runtime_only': True, 'baselines.local_runtime_host': 'h100-10',
        'data.val_batch_size': 64, 'data.val_data_num': None, 'data.return_raw_chat': True,
        'data.max_start_length': 2048, 'data.max_prompt_length': 4096,
        'data.max_response_length': 500, 'data.max_obs_length': 500,
        'max_turns': 4, 'force_final_answer': False, 'do_search': True,
        'retriever.topk': 3, 'retriever.url': RETRIEVER_DECLARATION,
        'actor_rollout_ref.rollout.n_agent': 1, 'actor_rollout_ref.rollout.n': 1,
        'actor_rollout_ref.rollout.temperature': 1, 'actor_rollout_ref.rollout.top_p': 1,
        'actor_rollout_ref.rollout.top_k': -1, 'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.actor.use_kl_loss': False, 'actor_rollout_ref.actor.kl_loss_coef': 0,
        'actor_rollout_ref.actor.entropy_coeff': 0, 'algorithm.kl_ctrl.kl_coef': 0,
        'baselines.holdout.manifest_sha256': '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
        **FUTURE_INTERPOLATIONS}
    for key, target in expected.items():
        c.typed(selected(raw, key), target, 'Authored eval configuration differs: ' + key)
    return expected


def compose_partial(c, repo, evaluation):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    # No future stage, model or output values may influence this preparation.
    forbidden = [n for n in os.environ if n.startswith('OPSD_FINAL200_') or n == 'OPSD_E1_RETRIEVER_URL']
    c.require(not forbidden, 'Future eval/runtime URL environment must be absent for partial preparation')
    with initialize_config_dir(version_base=None, config_dir=str(repo / 'verl/trainer/config')):
        config = compose(config_name=CONFIG_NAME)
    raw = OmegaConf.to_container(config, resolve=False, throw_on_missing=False)
    checked = structural_config(c, raw)
    # This reads only existing scalar fields and the YAML-authored loopback
    # default. It neither resolves model/stage/output nor contacts a service.
    budget = evaluation.validate_eval_config(config)
    c.typed(budget['retriever.url'], 'http://127.0.0.1:8000/retrieve', 'Authored default loopback differs')
    return raw, dict(scope='PARTIAL_CONFIG_WITH_AUTHORED_DEFAULT_URL_NOT_RUNTIME_BUDGET',
        checked_fields=checked, authored_default_budget=budget,
        future_values={key: None for key in FUTURE_INTERPOLATIONS},
        future_interpolations=FUTURE_INTERPOLATIONS,
        runtime_resolved_config=False, retrieval_service_verified=False)


def native_runtime(c, job, step):
    c.typed(sys.platform, 'linux', 'Native CPU inspection requires Linux /proc identity')
    execution = c.execution_identity(job, step)
    c.typed(execution['uid'], 0, 'Native source CPU inspection must use actual own UID0')
    c.typed(sys.executable, NATIVE_PYTHON, 'Actual reviewed native interpreter invocation path differs')
    c.require(sys.version_info[:2] == (3, 10) and sys.dont_write_bytecode and sys.flags.optimize == 0, 'Native Python3.10 -B nonoptimized interpreter required')
    import torch
    c.require(re.fullmatch(r'2\.4(?:\.[0-9]+)?(?:\+[^\s]+)?', torch.__version__) is not None, 'Native Torch2.4 runtime required')
    c.typed(torch.cuda.is_initialized(), False, 'CPU preparation initialized CUDA')
    _, interpreter = c.read_file(Path(sys.executable).resolve(strict=True))
    _, torch_source = c.read_file(Path(torch.__file__).resolve(strict=True))
    execution['actual_process_argv'] = Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').decode('utf-8').split('\0')
    return execution, dict(torch_version=torch.__version__, cuda_is_initialized=False,
        python=sys.executable, python_version=sys.version, interpreter_binary=interpreter,
        torch_module_source=torch_source)


def write_once(c, path, value):
    path = Path(path)
    raw = c.canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return c.read_file(path, hashlib.sha256(raw).hexdigest())[1]


def inspect(args):
    c = common()
    _, self_record = c.read_file(Path(__file__).resolve(), args.self_sha256, keep_bytes=True)
    execution, runtime = native_runtime(c, args.execution_job_id, args.execution_step_id)
    repo, inventory, inventory_record, members = source_inventory(c, args.source_inventory, args.source_inventory_sha256)
    bootstrap_path = repo / 'reproduction/baselines/e1_import_bootstrap.py'
    bootstrap = pinned_module(bootstrap_path, members['reproduction/baselines/e1_import_bootstrap.py']['sha256'], 'reproduction.baselines.e1_import_bootstrap')
    bootstrap.repo_first(repo)
    required = tuple(dict.fromkeys(tuple(bootstrap.REQUIRED) + ADDED_REQUIRED))
    closure = bootstrap.collect(repo, required=required)
    actual, function_count = bootstrap.verify_imports(repo, closure, required=required)
    c.require(closure == actual and all(n in actual['modules'] for n in ADDED_REQUIRED), 'Evaluation import closure incomplete')
    for name, record in actual['modules'].items():
        if name in bootstrap.LIBRARY_MODULES: continue
        relative = str(Path(record['path']).relative_to(repo))
        c.typed(record['sha256'], members[relative]['sha256'], 'Actual import not bound to full source inventory')
    partial, meaning = compose_partial(c, repo, sys.modules['search_r1.baselines.final200_evaluation'])
    after_config, after_function_count = bootstrap.verify_imports(repo, closure, required=required)
    c.require(after_config == closure and after_function_count == function_count, 'Config check changed the actual executable project closure')
    import torch
    c.typed(torch.cuda.is_initialized(), False, 'Actual source/config imports initialized CUDA')
    for record in members.values(): c.verify_record(record)
    c.verify_record(runtime['interpreter_binary']); c.verify_record(runtime['torch_module_source'])
    c.verify_record(inventory_record); c.verify_record(self_record)
    c.typed(git(repo, 'rev-parse', 'HEAD').decode().strip(), SOURCE, 'Source HEAD changed after import')
    c.require(not git(repo, 'status', '--porcelain', '--untracked-files=all').strip(), 'Source changed after import')
    output = Path(args.output)
    c.require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent and output.parent.stat().st_uid == os.getuid(), 'Actual output parent must already exist and be owned')
    closure_record = write_once(c, output.parent / (output.name + '.closure.json'), closure)
    partial_record = write_once(c, output.parent / (output.name + '.partial-config.json'), partial)
    source_records = write_once(c, output.parent / (output.name + '.source-physical-records.json'), members)
    report = dict(version=1, kind='ACTUAL_C5BF_NATIVE_CPU_IMPORT_PARTIAL_CONFIG_PREPARATION',
        scope=SCOPE, passed=True, eval_source_commit=SOURCE, stored_training_source_commit=PARENT,
        stored_training_job_id='3304', execution=execution, runtime=runtime,
        argv=list(sys.argv), tool_source=self_record, source_inventory=inventory_record,
        source_tree_members=len(members), required_modules=list(required), module_count=len(closure['modules']),
        verified_live_function_code_count=function_count, import_closure=closure_record,
        partial_config=partial_record, source_physical_records=source_records, configuration=meaning,
        actual_eval_job_id=None, actual_model_directory=None, actual_stage_manifest=None,
        actual_input_manifest=None, actual_output_directory=None,
        evaluation_eligibility='BLOCKED_ACTUAL200_AND_REVIEWED_STAGE_REQUIRED',
        model_loaded=False, gpu_execution_verified=False, evaluation_started=False)
    result = write_once(c, output, report)
    print(c.canonical(dict(report=result, scope=SCOPE)).decode(), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-inventory', 'source-inventory-sha256', 'self-sha256', 'execution-job-id', 'execution-step-id', 'output'):
        parser.add_argument('--' + name, required=True)
    inspect(parser.parse_args())


if __name__ == '__main__': main()
