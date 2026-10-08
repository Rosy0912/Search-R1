"""Draft approval adapter; only accepts separate actual independent30 evidence.

It does not run a reviewer, promote calibration26, restart workers or fabricate
source/thresholds. Caller must arrange actual live-state collection and independent
first5+HF/complete-state review before this adapter can produce approval bytes.
"""
import json
import re
from pathlib import Path
from formal_protection_contract_v1 import require, read_bound, sha, DATA, MODEL, MANIFEST


def approval_payload(binding, ready_receipt, actual_independent_review_receipt):
    require(binding['status'] == 'REVIEWED_FOR_PRODUCTION' and binding['code_commit'], 'No production READY')
    ready = read_bound(ready_receipt)
    review = read_bound(actual_independent_review_receipt)
    manifest = ready['manifest']
    identity = manifest['identity']
    source = binding['code_commit']
    path = Path(manifest['path'])
    require(str(identity['job_id']).isdecimal() and re.fullmatch(r'[0-9a-f]{40}', source)
            and re.fullmatch(r'[0-9a-f]{64}', manifest['sha256'])
            and path.is_absolute() and path.is_file() and not path.is_symlink()
            and sha(path) == manifest['sha256'] and json.loads(path.read_text()) == identity,
            'Actual first5 saved manifest/job/source identity missing or changed')
    expected = dict(job_id=str(identity['job_id']), source=MODEL, logical_step=30,
                    actual_new_updates=5, manifest_sha256=manifest['sha256'], restore_manifest_sha256=MANIFEST)
    require(identity['code_commit'] == source and identity['data_identity'] == DATA
            and identity['source'] == MODEL and identity['logical_step'] == 30
            and identity['actual_new_updates'] == 5 and ready['driver_cursor'] == 15360,
            'Actual first5 ready state belongs to another experiment')
    require(review['status'] == 'PASS' and all(review.get(k) == v for k, v in expected.items()),
            'No independently accepted actual first5')
    require(review['source_commit'] == source and review['config_sha256'] == identity['config_sha256']
            and review['production_artifact_sha256'] == binding['production_artifact']['sha256']
            and review['ready_sha256'] == ready_receipt['sha256'], 'First5 does not bind current production/source/config/pause')
    require(review['actual_native_acceptance_sha256'] == binding['native_acceptance']['review_sha256'],
            'First5 uses different actual source native acceptance')
    checks = review['first5_actual_checks']
    for key in ('all8_nonzero_parameter_change','finite_loss_and_gradient','same_driver_and8worker_pids_start_ticks',
                'raw_retrieval_and_fixed_reference','microbatch_global_sequence_mean','raw_trajectories_and_gigpo_state',
                'stage64_health','baseline512_health','full_hf_semantic_reload','allrank_model_adam_scheduler_rng_actual_reload',
                'driver_rng_cursor_permutation_gigpo_actual_roundtrip','approved_pause_unchanged'):
        require(checks[key] is True, 'Missing actual first5 acceptance: ' + key)
    require(checks['adam_per_rank'] == 50 and checks['gradient_micro_per_rank'] == 800
            and checks['saved_and_actual_restored_adam_total'] == 550
            and checks['scheduler'] == 30 and checks['cursor'] == 15360, 'First5 actual optimizer/state budget differs')
    require(len(ready['actual_state_roundtrip']) == 8
            and {r['rank'] for r in ready['actual_state_roundtrip']} == set(range(8))
            and all(r['logical_step'] == 30 for r in ready['actual_state_roundtrip']), 'Missing actual8rank step30 load')
    paused = ready['checkpoint_pause_identity']
    saved = [r['pause_identity'] for r in identity['files'] if 'pause_identity' in r]
    require(paused == saved and len({(r['pid'], r['start_ticks']) for r in paused}) == 8,
            'Actual fullworker tensors/Adam/RNG/PID pause differs')
    # These extra fields make the new review explain stronger control evidence;
    # the original source's wait gate also rechecks manifest, worker identity and
    # driverRNG immediately before31. No writer may restart/reset to simulate it.
    return dict(expected, source_commit=source, config_sha256=identity['config_sha256'],
                production_artifact_sha256=binding['production_artifact']['sha256'],
                review_path=actual_independent_review_receipt['path'],
                review_sha256=actual_independent_review_receipt['sha256'], ready_sha256=ready_receipt['sha256'],
                same_process_next_fetch=31, actual_new5_counted_in275=True)
