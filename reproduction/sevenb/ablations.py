"""New human-authorized 7B ablations; existing science and old runs stay frozen."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from .contract import REVISION, REPOSITORY, models, make_config
from reproduction.recursive_lumia.prepare_ablation import derive as derive_3b, condition_for
from release.recursive_bran_a.ablation150_v1.contracts import sha, projected_original, validate_config_dict
from release.recursive_bran_a.ablation150_v1 import PROTOCOL, EVALUATION_STEPS

ROOT = Path('/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-7B-Ablations-20261019-v1')
COMMON = Path('/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-20261015-v1')
MODEL = Path('/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-7B-20261019-v1/models/base') / REVISION
DATA = COMMON / 'common-data'
PYTHON = '/plm-shared/zhangjunming/Workspace/Search-Self-Evolution-20261006-PJ-runtime-v1/native-env/bin/python'
GROUPS = {'C1-root-only': 'c1', 'C2-random-final': 'c2', 'budget-5': 'b5', 'budget-20': 'b20', 'generation-2': 'd2'}
ALLOCATION = dict(gpus=8, cpus=64, memory_gib=900, machine_type='n2ls.ie.i60b', gpu_model='A100', gpu_memory_gib=80)
COMPILER = dict(path='/plm-shared/zhangjunming/Workspace/Search-Self-Evolution-20261006-PJ-runtime-v1/native-env/lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas',
                sha256='94dda54a1673940d56d3072c64ef7aacb125a1e64394586e10f77a7a5ff0ff87')
VERSIONS = {'torch': '2.4.0', 'vllm': '0.6.3', 'transformers': '4.46.3', 'ray': '2.51.2', 'flash-attn': '2.6.3', 'triton': '3.0.0'}
HOST = '__ALLOCATED_PJ_HOST__'


def run_root(group):
    if group not in GROUPS:
        raise ValueError('Only five currently retained3B conditions are authorized at7B')
    return ROOT / 'runs' / group


def job_name(group):
    run_root(group)
    name = 'zhangjunming-7b-abl-' + GROUPS[group] + '-v1'
    assert len(name) <= 32
    return name


def derive(group, host=HOST):
    from omegaconf import OmegaConf
    base = run_root(group)
    template = OmegaConf.to_container(make_config(base, MODEL, 'gpu-l40s-2'), resolve=True)
    # Reuse precisely the existing five ablation transformations and hooks.
    cfg = derive_3b(template, group, base, 'gpu-l40s-2')
    cfg['baselines']['local_runtime_host'] = host
    for key, name in [('train_files', 'train.parquet'), ('val_files', 'validation.parquet'), ('final_val_files', 'final.parquet')]:
        cfg['data'][key] = str(DATA / name)
    cfg['baselines']['holdout'].update(manifest_path=str(DATA / 'manifest.json'), data_root=str(DATA))
    cfg['baselines']['bapo']['stage_validation_files'] = str(DATA / 'stage.parquet')
    cfg['trainer']['experiment_name'] = 'sevenb-ablation-' + group
    return cfg


def validate_plan(plan):
    from omegaconf import OmegaConf
    group = plan['group']; root = run_root(group)
    assert plan['protocol'] == 'sevenb_recursive_ablation150_pj_v1'
    assert plan['job'] == job_name(group) and plan['root'] == str(root)
    assert plan['output'] == str(root / 'output') and plan['allocation'] == ALLOCATION
    assert plan['model'] == str(MODEL) and plan['model_revision'] == REVISION
    assert plan['model_files'] == {v['file']: v['sha256'] for v in models()['base']['files']}
    assert plan['data_root'] == str(DATA) and plan['condition'] == condition_for(group)
    assert plan['actual_update_limit'] == 150 and plan['scheduler_scale'] == 200 and plan['warmup_rounds'] == 57
    assert plan['fresh_from_common_initialization'] and plan['parent_checkpoint'] is None
    assert plan['no_auto_retry'] and plan['no_hostname_affinity'] and plan['compiler'] == COMPILER
    assert plan['required_versions'] == VERSIONS
    assert len(plan['source_commit']) == 40 and len(plan['run_id']) == 32
    assert OmegaConf.to_container(OmegaConf.create(plan['config_text']), resolve=True) == derive(group)
    validate_config_dict(derive(group))
    return root


def verify(plan):
    root = validate_plan(plan)
    for name, expected in plan['wrapper_files'].items():
        assert sha(Path(plan['source']) / name) == expected, name
    from reproduction.seven_dataset.contracts import DATA_SHA, validate_model_files
    validate_model_files(dict(model=str(MODEL), model_files=plan['model_files']))
    c = json.loads((MODEL / 'config.json').read_text())
    assert (c['model_type'], c['hidden_size'], c['num_hidden_layers'], c['num_attention_heads'], c['num_key_value_heads']) == ('qwen2', 3584, 28, 28, 4)
    assert sha(DATA / 'manifest.json') == DATA_SHA == plan['data_identity']
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for split, count in [('train', 169028), ('validation', 512), ('stage', 64), ('final', 51713)]:
        row = manifest['splits'][split]
        assert row['rows'] == count and sha(DATA / row['path']) == row['sha256']
    assert sha(plan['reference_plan']) == plan['reference_plan_sha256']
    assert sha(plan['retriever_receipt']) == plan['retriever_receipt_sha256']
    assert json.loads(Path(plan['retriever_receipt']).read_text())['execution_backend'] == 'pj_pt'
    import importlib.metadata as metadata
    for name, expected in VERSIONS.items():
        assert metadata.version(name).split('+')[0] == expected, name
    assert sha(COMPILER['path']) == COMPILER['sha256']
    assert 'release 12.4,' in subprocess.check_output([COMPILER['path'], '--version'], text=True)
    return root


def validate_gate(plan, sub, gate):
    assert gate['plan_sha256'] == sub['plan_sha256']
    a = gate['actual']
    assert a['name'] == sub['job'] == plan['job'] and a['uid'] == sub['uid']
    assert a['state'] == 'RUNNING' and a['resources'] == ALLOCATION
    return a


def record_for(plan, host):
    from omegaconf import OmegaConf
    root = run_root(plan['group']); path = root / 'receipts' / 'resolved.yaml'
    cfg = derive(plan['group'], host)
    text = OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True)
    with path.open('x') as f:
        f.write(text)
    record = dict(protocol=PROTOCOL, group=plan['group'], run_id=plan['run_id'], config=str(path), config_sha256=sha(path),
                  source_commit=plan['source_commit'], model=str(MODEL), data=str(DATA), data_identity=plan['data_identity'],
                  base=str(root), output=str(root / 'output'), fresh_authorized=True, parent_checkpoint=None,
                  execution_backend='pj_pt', allocation=plan['job'], scheduler_node=host, host=host,
                  condition=plan['condition'], actual_update_limit=150, evaluation_steps=list(EVALUATION_STEPS),
                  scheduler_scale=200, warmup_rounds=57, model_size='7B', model_revision=REVISION,
                  retriever_receipt=plan['retriever_receipt'], retriever_receipt_sha256=plan['retriever_receipt_sha256'])
    receipt = root / 'receipts' / 'configuration.json'
    with receipt.open('x') as f:
        json.dump(record, f, indent=2)
    return record, receipt


def train(planfile):
    plan = json.loads(Path(planfile).read_text()); root = verify(plan)
    host = socket.gethostname().split('.')[0]
    record, receipt = record_for(plan, host)
    output = root / 'output'; output.mkdir(exist_ok=False)
    from reproduction.recursive_lumia.ablation_entry import runtime_environment
    env = runtime_environment(record, output, receipt)
    env.pop('RECURSIVE_A_LUMIA_FRESH_PROFILE')
    env.update(SEARCH_R1_EXECUTION_BACKEND='pj_pt', SEARCH_R1_JOB_ID=plan['job'],
               SEARCH_R1_MODEL_IDENTITY=REPOSITORY + '@' + REVISION, SEARCH_R1_PJ_NODE=host,
               RECURSIVE_A_PJ_FRESH_PROFILE='1')
    os.environ.update(env)
    from release.recursive_bran_a.ablation150_v1.bootstrap import install, worker_setup
    install()
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(record['config'])
    from search_r1.baselines.controller import validate_config
    validate_config(cfg)
    from search_r1.baselines.preflight import validate_gpu_lease
    validate_gpu_lease(cfg)
    tick = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    entry = dict(time=time.time(), pid=os.getpid(), uid=os.getuid(), tick=tick, host=host,
                 allocation=plan['job'], run_id=plan['run_id'], group=plan['group'], output=str(output),
                 config_sha256=record['config_sha256'], source_commit=plan['source_commit'],
                 execution_backend='pj_pt', configuration_receipt=str(receipt), model_size='7B')
    with (root / 'receipts' / 'current-entry.json').open('x') as f:
        json.dump(entry, f, indent=2)
    from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
    prepare_click()
    import ray
    inherited = {k: v for k, v in os.environ.items() if k.startswith(('SEARCH_R1_', 'RECURSIVE_A_', 'RECURSIVE_ABLATION_', 'RECURSIVE_LUMIA_', 'TREE_'))}
    inherited.update(PYTHONPATH=plan['source'], TRITON_PTXAS_PATH=COMPILER['path'], TOKENIZERS_PARALLELISM='false', NCCL_DEBUG='WARN')
    error = None
    try:
        ctx = ray.init(address='local', num_gpus=8, num_cpus=64, include_dashboard=False,
                       object_store_memory=8 * 1024**3, _node_ip_address='127.0.0.1', _temp_dir='/tmp/' + plan['job'],
                       runtime_env={'env_vars': inherited, 'worker_process_setup_hook': worker_setup})
        os.environ['RAY_ADDRESS'] = ctx.address_info['gcs_address']
        from search_r1.baselines.main import main
        main.__wrapped__(cfg)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        ray.shutdown()
        with (output / 'entry-exit.json').open('x') as f:
            json.dump(dict(at=time.time(), rc=0 if error is None else 1, error=error), f)


def launch(planfile):
    plan = json.loads(Path(planfile).read_text()); root = verify(plan)
    assert os.environ.get('SEARCH_R1_EXECUTION_BACKEND') == 'pj_pt'
    assert os.environ.get('SEARCH_R1_JOB_ID') == plan['job'] and not os.environ.get('SLURM_JOB_ID')
    intent = json.loads((root / 'submission-intent.json').read_text())
    assert intent['plan_sha256'] == sha(planfile) and intent['job'] == plan['job']
    deadline = time.monotonic() + 180
    while not ((root / 'submission.json').exists() and (root / 'scheduler-release-once.json').exists()):
        if time.monotonic() >= deadline:
            raise TimeoutError('No exact UID scheduler release; no training started')
        time.sleep(.5)
    sub = json.loads((root / 'submission.json').read_text()); gate = json.loads((root / 'scheduler-release-once.json').read_text())
    actual = validate_gate(plan, sub, gate)
    inv = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(inv) == 8 and len({r.split(',')[1].strip() for r in inv}) == 8
    assert all('A100' in r and float(r.split(',')[3]) >= 80000 for r in inv)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    assert not (root / 'checkpoints').exists() and not (root / 'output').exists()
    with (root / 'entry-once.json').open('x') as f:
        json.dump(dict(at=time.time(), actual=actual, gpus=inv, plan_sha256=sha(planfile)), f, indent=2)
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(r.split(',')[0].strip() for r in inv), TRITON_PTXAS_PATH=COMPILER['path'])
    for name in ['ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES']:
        os.environ.pop(name, None)
    subprocess.run([sys.executable, '-m', 'reproduction.sevenb.ablation_smoke', str(root / 'compiler-smoke.json')], check=True, timeout=120)
    from http.server import ThreadingHTTPServer
    from reproduction.recursive_lumia.proxy import Handler
    ready = json.loads(Path(plan['retriever_receipt']).read_text())
    server = ThreadingHTTPServer(('127.0.0.1', 29146), Handler); server.remote = ready['url']
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        child = subprocess.Popen([sys.executable, '-m', 'search_r1.baselines.preflight', '--', sys.executable,
                                  '-m', 'reproduction.sevenb.ablations', 'train', str(planfile)])
        rc = child.wait()
        with (root / 'actual-parent-wait.json').open('x') as f:
            json.dump(dict(at=time.time(), pid=child.pid, rc=rc, job=plan['job']), f)
        raise SystemExit(rc if rc >= 0 else 128 - rc)
    finally:
        server.shutdown(); server.server_close()


if __name__ == '__main__':
    (train if sys.argv[1] == 'train' else launch)(sys.argv[2])
