"""Read back the actual driver/data boundary around frozen teacher warmup.

Keep this caller-side audit separate from the immutable teacher sampling code:
adding a diagnostic must not silently change a reusable rollout namespace.
"""
import copy
import json
import os
from pathlib import Path
import time


def capture(runtime, stream):
    from search_g1.training_state import driver_rng, cpu_tree
    return cpu_tree(copy.deepcopy(dict(
        driver_rng=driver_rng(), stream=stream.state(),
        selection_rng=runtime.selection_rng.bit_generator.state)))


def verify(before, after):
    from search_g1.full_state_restore import equal_tree
    equal_tree(before, after)
    return dict(driver_rng_equal=True, actual_candidate_stream_equal=True,
                selector_rng_equal=True)


def run(trainer, manager, runtime, stream):
    import torch
    from search_g1.training_state import atomic_save
    from search_g1.qprobe_warmup import run as teacher_warmup, write_once
    from search_g1.qprobe_features import sha_file
    from search_g1.full_state_restore import equal_tree

    base = Path(os.environ['SEARCH_G1_BASE']).resolve()
    manifest_path = base/'logs/run-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if runtime.completed_outer != 0 or runtime.head.update_count or trainer.global_steps != 1:
        raise ValueError('WARMUP_DRIVER_AUDIT_REQUIRES_FRESH_PRE_OUTER1_STATE')
    directory = runtime.output_dir/'teacher-warmup-driver-boundary'
    before = atomic_save(directory/'before.pt', capture(runtime, stream))
    result = teacher_warmup(trainer, manager, runtime, stream)
    after = atomic_save(directory/'after.pt', capture(runtime, stream))
    # Compare fresh disk readbacks, not a receipt's assertion or saved cursor alone.
    for proof in (before, after):
        path = Path(proof['path'])
        if path.stat().st_size != proof['bytes'] or sha_file(path) != proof['sha256']:
            raise ValueError('WARMUP_DRIVER_BOUNDARY_FULL_SHA_CHANGED')
    measured = verify(torch.load(before['path'], map_location='cpu', weights_only=False),
                      torch.load(after['path'], map_location='cpu', weights_only=False))
    if not result or result['fitted_questions'] != 5120 or runtime.head.completed_outer or runtime.head.update_count:
        raise ValueError('WARMUP_DRIVER_AUDIT_REQUIRES_ALL_5120_FITTED_BEFORE_FORMAL_WORK')
    head = result['head_checkpoint']
    if Path(head['path']).stat().st_size != head['bytes'] or sha_file(head['path']) != head['sha256']:
        raise ValueError('WARMUP_HEAD_FULL_SHA_CHANGED')
    saved_head = torch.load(head['path'], map_location='cpu', weights_only=True)
    equal_tree(runtime.head.state_dict(), saved_head)
    proof = dict(status='FROZEN_TEACHER_ACTUAL_DRIVER_DATA_HEAD_READBACK_PASS',
        run_id=manifest['run_id'], source_commit=manifest['source_commit'],
        run_manifest_sha256=sha_file(manifest_path), before=before, after=after,
        **measured, fitted_questions=5120, reused_questions=result['reused_questions'],
        completed_training_outers=0, head_state_equal=True,
        head_scheduler=None, head_scheduler_policy='constant learning rate Adam; no scheduler object',
        head_checkpoint=head, checked_unix=time.time())
    write_once(directory/'complete.json', proof)
    return result
