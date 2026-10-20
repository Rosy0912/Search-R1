"""Read-only C20 lineage gate for the future fixed-three commonv2 evaluations.

This module has no scheduler mutation, inference, deployment or download entry.
It consumes the independent C20 proof and its locally SHA-backed small files.
Full actual old-runtime HF reload and full51713 inference are separate gates.
"""
from pathlib import Path
import hashlib
import json
import math
import re

JOB = '139134'
NAME = 'zhangjunming-7b-original-mem-v1'
SOURCE = '92942a32161071b85cc70906050143ed58244eb1'
PLAN = 'b71e43e5f979bed427dc26bf0953524cf151b37bc8f9f36a8ed4cade5c44cdf5'
ROOT = '/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-20261018-v1/runs/original-hostmem-v1'
NATIVE = ROOT + '/checkpoints/training_state/global_step_200'
MODEL = ROOT + '/checkpoints/actor/global_step_200'
STATUS = 'INDEPENDENT_FINAL200_TERMINAL_PHYSICAL_SHA_ADAM_RNG_SCHEDULER_DRIVER_AND_HF_IDENTITY_VERIFIED'
REQUIRED_SMALL = {'manifest.json', 'plan.json', 'submission.json', 'submission-intent.json',
                  'entry-once.json', 'resolved.yaml', 'driver.pt', 'training-returned.json',
                  'training-exit.code', 'train-start.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_value(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


def positive_integer(value):
    return type(value) is int and value > 0


def scheduler(row):
    require(len(row) >= 6 and row[:3] == [JOB, 'jmzhang', NAME], 'Wrong training allocation')
    require(row[3] == 'COMPLETED' and row[4] == '0:0', 'Training is failed or uncertain')
    resources = dict(part.split('=', 1) for part in row[5].split(',') if '=' in part)
    require(all(resources.get(k) == v for k, v in {'gres/gpu': '8', 'cpu': '64', 'mem': '900G'}.items()),
            'Training resources differ')
    # Final200 requires the exact successful terminal allocation.
    return row[3]


def validate(proof, manifest, plan, submission):
    require(proof['status'] == STATUS and proof['job_id'] == JOB and proof['source'] == SOURCE,
            'Wrong native acceptance lineage')
    require(proof['plan_sha256'] == PLAN and proof['completed_native_step'] == 200,
            'Wrong native boundary')
    for key in ('local_small_evidence_sha_verified', 'external_cpu_only', 'process_rng_unchanged',
                'no_cuda_initialization', 'no_training_mutation', 'not_scientific_evaluation_acceptance', 'training_terminal0_and_return_verified',
                'no_new_cold_gpu_resume_attempt', 'pre_evaluation_boundary_only'):
        require(proof[key] is True, 'Missing C20 gate: ' + key)
    training_state = scheduler(proof['scheduler'])
    returned = proof['training_returned']
    require(set(returned) == {'at', 'job'} and returned['job'] == JOB and
            type(returned['at']) in (int, float) and math.isfinite(returned['at']),
            'Wrong training return evidence; no invented update-count field')
    require(submission['job_id'] == JOB and submission['job'] == NAME and submission['root'] == ROOT
            and submission['plan_sha256'] == PLAN and submission['source'] == SOURCE,
            'Wrong original training submission')
    require(submission['allocation']['gpus'] == 8 and submission['allocation']['cpus'] == 64
            and submission['allocation']['memory_gib'] == 900, 'Wrong submitted resources')
    require(plan['evaluation_code_commit'] == SOURCE and
            plan['model_revision'] == 'a09a35458c702b33eeacc393d103063234e8bc28', 'Wrong source or initialization')
    require(manifest['status'] == 'COMPLETE_SAME_WORLD_STATE_SAVED' and manifest['hf_checkpoint'] == MODEL,
            'Wrong saved checkpoint')
    ident = manifest['identity']
    require(ident['job_id'] == JOB and ident['code_commit'] == SOURCE and
            ident['completed_step'] == 200 and ident['actual_update_limit'] == 200,
            'Final200 boundary must match the200 training limit')
    require(ident['data_identity'] == plan['data_manifest_sha256'], 'Training data binding differs')
    require(hash_value(proof['manifest_sha256']) and hash_value(ident['config_sha256']),
            'Invalid physical identity')
    require(manifest['driver']['next_step'] == 201 and
            manifest['driver']['path'] == NATIVE + '/driver.pt', 'Driver boundary differs')
    require(manifest['driver']['exact_reload_verified'] is False,
            'Pre-evaluation200 is not the post-evaluation reload boundary')
    require(proof['files']['driver.pt']['sha256'] == manifest['driver']['sha256'] and
            proof['files']['resolved.yaml']['sha256'] == ident['config_sha256'], 'Driver/config SHA differs')
    accounting = ident['guard_continue_accounting']
    require(proof['driver_actual_optimizer_accounting'] == accounting, 'Driver accounting differs')
    totals = accounting['total_optimizer_calls_per_rank']
    require(set(totals) == {str(i) for i in range(8)} and
            all(positive_integer(v) for v in totals.values()), 'Missing actual optimizer counters')
    workers = proof['workers']; saved_workers = manifest['worker_states']
    require(len(workers) == len(saved_workers) == 8 and
            {v['rank'] for v in workers} == set(range(8)) and
            {v['identity']['rank'] for v in saved_workers} == set(range(8)), 'All8 ranks required')
    physical = proof['physical_rank_files']
    require(set(physical) == {'rank-%02d.pt' % i for i in range(8)}, 'Physical rank coverage differs')
    saved_by_rank = {v['identity']['rank']: v for v in saved_workers}
    for worker in workers:
        rank = worker['rank']; record = saved_by_rank[rank]; item = physical['rank-%02d.pt' % rank]
        require(all(record['identity'].get(k) == v for k, v in ident.items()) and
                record['identity']['world_size'] == 8, 'Saved rank identity differs')
        require(record['full_worker_state_finite'] is True and
                record['exact_same_world_reload_verified'] is False, 'Wrong pre-evaluation worker boundary')
        require(item['path'] == record['path'] == NATIVE + '/rank-%02d.pt' % rank and
                positive_integer(item['bytes']) and item['bytes'] == record['bytes'] and
                hash_value(item['sha256']) and item['sha256'] == record['sha256'], 'Physical rank SHA differs')
        expected = totals[str(rank)]
        require(worker['actual_adam_min'] == worker['actual_adam_max'] ==
                record['actual_total_optimizer_calls'] == expected, 'Measured Adam accounting differs')
        require(worker['optimizer_states'] == record['optimizer_state_entries'] and
                positive_integer(worker['optimizer_states']), 'Incomplete saved Adam state')
        require(worker['scheduler_last_epoch'] == record['scheduler_last_epoch'] == 200,
                'Scheduler200 absent')
        lrs = worker['learning_rates']
        require(lrs == record['optimizer_lrs'] and len(lrs) > 0 and
                all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in lrs),
                'Learning rate is not finite and positive')
        require(worker['all_complete_state_finite'] is True and
                worker['stored_cpu_numpy_python_and_cuda_rng_verified'] is True and
                hash_value(worker['logical_state_sha256']) and
                worker['logical_state_sha256'] == record['logical_state_sha256'], 'Saved full state/RNG differs')
    sampler = proof['sampler']
    require(sampler['rows'] == 169028 and sampler['epoch'] == 0 and
            positive_integer(sampler['batch_size']) and sampler['cursor'] == 200 * sampler['batch_size'],
            'Saved data cursor differs')
    hf = proof['hf_identity']
    require(hf == manifest['hf_identity'] and isinstance(hf['files'], dict), 'HF identity differs')
    require(hf['sha256'] == hashlib.sha256(json.dumps(hf['files'], sort_keys=True).encode()).hexdigest(),
            'HF file manifest digest differs')
    require({'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
             'model.safetensors.index.json'} <= set(hf['files']), 'Incomplete HF metadata')
    require(any(n.endswith('.safetensors') for n in hf['files']) and
            all(Path(n).name == n and hash_value(v) for n, v in hf['files'].items()), 'Unsafe or incomplete HF files')
    return dict(training_job_id=JOB, training_source=SOURCE, actual_training_iterations=200,
                training_limit_remains=200, boundary='pre-evaluation', model=MODEL,
                training_state=training_state, repeat_count=3,
                actual_optimizer_calls_per_rank=dict(totals), no_cold_GPU_resume_claim=True,
                actual_old_runtime_full_HF_reload_and_full51713_still_required=True)


def read_local_bundle(receipt, evidence_root):
    """Verify only original C20 small evidence; never read model/Adam payloads."""
    evidence_root = Path(evidence_root).resolve(); proof = json.loads(Path(receipt).read_text())
    require(proof.get('local_small_evidence_sha_verified') is True, 'Original local SHA gate missing')
    require(REQUIRED_SMALL <= set(proof['files']), 'Missing original small evidence')
    raw = {}
    for name, row in proof['files'].items():
        relative = Path(row['local_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe local evidence path')
        target = evidence_root / relative
        require(target.resolve().is_relative_to(evidence_root) and not target.is_symlink(),
                'Local evidence escaped or is a symlink')
        require(target.is_file() and 0 <= target.stat().st_size == row['bytes'] < 16 * 1024**2,
                'Original small evidence size differs')
        data = target.read_bytes(); require(hashlib.sha256(data).hexdigest() == row['sha256'],
                                            'Original small evidence SHA differs: ' + name)
        raw[name] = data
    require(proof['files']['plan.json']['sha256'] == PLAN and
            proof['files']['manifest.json']['sha256'] == proof['manifest_sha256'],
            'Plan/manifest physical binding differs')
    manifest, plan, submission, intent, entry = [json.loads(raw[n]) for n in
        ('manifest.json', 'plan.json', 'submission.json', 'submission-intent.json', 'entry-once.json')]
    require(intent['plan_sha256'] == entry['plan_sha256'] == PLAN and entry['job_id'] == JOB and
            len(entry['gpus']) == len({v.split(',')[0] for v in entry['gpus']}) == 8,
            'Original entry/intent binding differs')
    returned, started = [json.loads(raw[n]) for n in ('training-returned.json', 'train-start.json')]
    require(returned == proof['training_returned'] and set(returned) == {'at', 'job'} and
            returned['job'] == started['job'] == JOB and raw['training-exit.code'].strip() == b'0',
            'Final training return/exit differs')
    require(all(type(v) in (int, float) and math.isfinite(v) for v in (returned['at'], started['at'])) and
            returned['at'] >= started['at'] and started['config_sha256'] == manifest['identity']['config_sha256'],
            'Final training time/config differs')
    return validate(proof, manifest, plan, submission)
