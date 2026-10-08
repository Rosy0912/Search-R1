"""Only new numeric hardware evidence can authorize fresh training."""
import json
import math
import os
import subprocess
from pathlib import Path
from verify_shared_inputs import ROOT, SOURCE, digest, owned, verify as verify_shared

def verify(root=ROOT, command=subprocess.check_output, shared_check=verify_shared):
    receipt = json.loads(owned(root / 'logs/hardware-acceptance.json', root).read_text())
    if receipt.get('status') != 'PASS_INDEPENDENT_REVIEW' or receipt.get('source_commit') != SOURCE:
        raise ValueError('Independent new-source hardware review missing')
    source = command(['git', '-C', str(root / 'repo'), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = command(['git', '-C', str(root / 'repo'), 'status', '--porcelain'], text=True)
    if source != SOURCE or dirty:
        raise ValueError('Current new fixed source changed')
    shared_check(root)
    fields = {'environment_sha256': 'logs/environment-acceptance.json',
              'assets_manifest_sha256': 'assets-root/assets.json',
              'derived_assets_record_sha256': 'logs/derived-assets.sha256',
              'common_split_manifest_sha256': 'logs/common-split-manifest.json',
              'shared_inputs_sha256': 'logs/shared-inputs.json',
              'isolation_setup_sha256': 'logs/isolation-setup.json'}
    for key, name in fields.items():
        if receipt.get(key) != digest(owned(root / name, root)):
            raise ValueError('Current identity hash mismatch: ' + key)
    evidence = receipt.get('evidence')
    review = receipt.get('independent_review')
    if not isinstance(evidence, list) or not isinstance(review, dict):
        raise ValueError('Missing independent direct bindings')
    seen = set()
    for item in evidence + [review]:
        path = owned(Path(item['path']), root)
        if digest(path) != item['sha256']:
            raise ValueError('Evidence changed: ' + str(path))
        seen.add(str(path))
    for name in fields.values():
        if str(root / name) not in seen:
            raise ValueError('Identity evidence not directly bound: ' + name)
    job = receipt.get('slurm_job_id', '')
    if not isinstance(job, str) or not job.isascii() or not job.isdigit():
        raise ValueError('Invalid hardware scheduler identity')
    run = root / 'runs' / ('bapo-hardware-' + job)
    if receipt.get('run_root') != str(run) or run.is_symlink() or run.resolve() != run.absolute():
        raise ValueError('Actual new hardware run identity mismatch')
    names = ('source-commit.txt', 'slurm-allocation.txt', 'hardware.log',
             'HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY', 'retrieval-comparison.json',
             'retrieval-cpu.json', 'retrieval-gpu.json', 'gpu-embeddings.npz',
             'parity-cpu.log', 'parity-gpu.log', 'parity-cpu-time.log', 'parity-gpu-time.log',
             'cpu-process-monitor.jsonl', 'gpu-monitor.csv', 'process-monitor.csv',
             'resource-watch.jsonl', 'gpus-before.csv', 'gpus-exit.csv',
             'processes-before.csv', 'own-process-cleanup.json')
    for name in names:
        if str(run / name) not in seen:
            raise ValueError('Actual hardware evidence not bound: ' + name)
    for rank in range(8):
        path=run/'numeric-fixture'/('rank-'+str(rank)+'.json')
        if str(path) not in seen:raise ValueError('Real numeric rank not directly bound')
        numeric=json.loads(path.read_text())
        if numeric.get('status')!='PASS' or numeric.get('source_commit')!=SOURCE or numeric.get('rank')!=rank or numeric.get('world')!=8:raise ValueError('Real numeric rank identity')
        records=numeric.get('records',[])
        if len(records)!=22:raise ValueError('Real numeric rank cases incomplete')
        expected={(case,orig,offload) for case in ['normal_native_and_fullvector_oracle','finite_gradient_fp32_square_overflow','one_rank_nonsharded_nan_collective_rejection','empty_one_rank_sharded_gradients'] for orig in [False,True] for offload in [False,True]}
        expected|={(case,orig,None) for case in ['pure_NO_SHARD_all_empty_gradients','pure_NO_SHARD_one_rank_inf_rejection'] for orig in [False,True]}
        expected|={('gpu_low_var_kl_saturated_forward_backward_fp64_oracle',None,None),('gpu_low_var_kl_nonfinite_inputs_refused',None,None)}
        actual={(r.get('case'),r.get('orig'),r.get('offload')) for r in records}
        if actual!=expected:raise ValueError('Missing or duplicate numeric rank cases')
        for item in records:
            if 'norm' in item and not math.isfinite(item['norm']):raise ValueError('Nonfinite numeric norm evidence')
            if item['case']=='gpu_low_var_kl_saturated_forward_backward_fp64_oracle':
                if item.get('points')!=408:raise ValueError('Incomplete KL oracle inputs')
                if any(not isinstance(item.get(k),(int,float)) or not math.isfinite(item[k]) or not 0<=item[k]<=2e-5 for k in ['max_value_error','max_grad_error']):raise ValueError('Bad actual GPU KL oracle error')
            if item['case']=='gpu_low_var_kl_nonfinite_inputs_refused' and item.get('count')!=3:raise ValueError('Missing nonfinite GPU KL rejection')
        if Path(numeric.get('gradient_norm_file','')).resolve()!=root/'repo/search_r1/baselines/gradient_norm.py' or Path(numeric.get('core_algos_file','')).resolve()!=root/'repo/verl/trainer/ppo/core_algos.py':raise ValueError('Numeric function imported wrong source')
    if (run / 'source-commit.txt').read_text().strip() != SOURCE:
        raise ValueError('Hardware measurement source mismatch')
    cleanup = json.loads((run / 'own-process-cleanup.json').read_text())
    if cleanup.get('own_marker') != str(run) or cleanup.get('remaining') != []:
        raise ValueError('New hardware own cleanup incomplete')
    if receipt.get('launch_stage_sha256') != digest(owned(root / 'tools/stage.sh', root)):
        raise ValueError('Actual stage identity mismatch')
    manifest = root / 'tools/hardware-launch-hashes-numeric-v1.json'
    if str(manifest) not in seen:
        raise ValueError('Hardware actual tool manifest not bound')
    tools = json.loads(owned(manifest, root).read_text())
    required_tools = {'stage.sh', 'stage-gpu-inner.sh', 'hardware.sh', 'monitor.sh',
                      'resource_watch.py', 'owned_processes.py', 'verify_hardware_receipt.py',
                      'verify_shared_inputs.py', 'flash_preflight.py', 'bapo-retrieval-parity.py',
                      'cpu-gate.py', 'prepare-stage64-proof.sh','numerics-gpu-fixture.py'}
    if not isinstance(tools, dict) or not required_tools.issubset(tools):
        raise ValueError('Actual launch tool manifest incomplete')
    for name, sha in tools.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Tool manifest path escapes')
        path = owned(root / 'tools' / rel, root)
        if str(path) not in seen or digest(path) != sha:
            raise ValueError('Actual hardware tool not bound: ' + name)
    raw = command(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    actual = [uid.strip() for line in raw.splitlines() for idx, uid in [line.split(',')] if idx.strip() in visible or uid.strip() in visible]
    if len(actual) != 8 or len(set(actual)) != 8 or set(actual) != set(receipt.get('gpu_uuids', [])):
        raise ValueError('Actual eight hardware UUIDs differ')
    identity = json.loads((root / 'logs/common-split-manifest.json').read_text())['content_identity']['sha256']
    if receipt.get('common_split_content_identity') != identity:
        raise ValueError('Common split content identity changed')
    print('HARDWARE_RECEIPT_NEW_NUMERIC_SOURCE_ENV_ASSETS_SPLIT_UUID_EVIDENCE_OK')

if __name__ == '__main__':
    verify()
