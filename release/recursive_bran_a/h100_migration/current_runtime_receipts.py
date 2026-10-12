"""Read deployment/config/gate/cache receipts only; never recompute or mutate training."""
import hashlib
import json
import pathlib
import time

base = pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
receipt = json.loads((base/'receipts/configuration.json').read_text())
config = pathlib.Path(receipt['config'])
config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
assert config_sha == receipt['config_sha256'] == '653193a4eb06bd28667f78ac14a66a2d1c97200ca8199b8521dcc2d3de35b319'
assert receipt['run_id'] == '1c806d15752d44a7ac99687f0737d7e3'
source = base/'source-71d4e69'
files = ['release/recursive_bran_a/h100_migration/train.py',
         'search_r1/baselines/tree_guard_continue.py',
         'search_r1/baselines/candidate_qa_reward.py',
         'search_r1/baselines/tree_behavior_mu.py',
         'verl/trainer/ppo/ray_trainer.py', 'verl/workers/fsdp_workers.py',
         'verl/workers/sharding_manager/fsdp_vllm.py',
         'verl/workers/rollout/vllm_rollout/vllm_rollout.py']
file_sha = {f: hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files}
gate_path = base/'retriever/receipts/full-scan-http-gate.json'
gate = json.loads(gate_path.read_text())
assert gate['accepted'] and gate['candidate_pid'] == 3713706
proc = pathlib.Path('/proc')/str(gate['candidate_pid'])
stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
assert proc.stat().st_uid == 0 and int(stat[19]) == gate['candidate_starttick']
cache = []
for line in (base/'logs/training.log').read_text(errors='replace').splitlines():
    if 'RECURSIVE_WEIGHT_CACHE_PHASE=' in line:
        try:
            cache.append(json.JSONDecoder().raw_decode(line.split('RECURSIVE_WEIGHT_CACHE_PHASE=',1)[1])[0])
        except ValueError:
            pass
print(json.dumps(dict(time=time.time(), run_id=receipt['run_id'], config_sha256=config_sha,
    declared_science_source=receipt['source_commit'], deployed_runtime='71d4e69', file_sha256=file_sha,
    gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
    gate={k: gate.get(k) for k in ['accepted','candidate_pid','candidate_starttick',
        'source_identity_sha256','tests','actual_directfree_mib','actual_gpu_apps','limitations']},
    cache_phase_records=cache,
    scope='Fresh published source/config/finite gate receipts and recorded cache counters; no repeated GPU gate or tensor audit')))
