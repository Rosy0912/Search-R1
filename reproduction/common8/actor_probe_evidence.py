"""Check real per-rank actor readout receipts against completed outer records."""

import json
from pathlib import Path

from search_g1.actor_probe_contract import actor_probes


def verify_actor_readouts(base, run, records, *, retained_prefix=None):
    assert actor_probes(run)
    ranks = run['training_world_size']
    files=[]
    if retained_prefix is not None:
        from reproduction.common8.actor_probe_retained_prefix import verify_prefix
        from reproduction.common8.actor_probe_rank_tail import _sealed_log
        verify_prefix(base,run,retained_prefix,records[-1]['step'])
        assert records==retained_prefix['selected_updates']
    proof = []
    for rank in range(ranks):
        path = Path(base)/'logs'/f'actor-probe-features-rank-{rank}.jsonl'
        if retained_prefix is None:
            receipts = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            receipts,metadata=_sealed_log(Path(base).resolve(),path.relative_to(base).as_posix())
            files.append(metadata)
        for record in records:
            matching = [r for r in receipts if r['binding']['attempt_id'] == record['attempt_id']]
            assert len(matching) == 1, 'Missing or duplicate real actor RPC receipt'
            receipt = matching[0]; binding = receipt['binding']
            assert receipt['rank'] == rank and receipt['detached'] is True
            assert binding['run_id'] == run['run_id'] and binding['execution_id'] == run['execution_id']
            assert binding['logical_update'] == record['step'] and binding['completed_outer'] == record['step'] - 1
            assert binding['optimizer_writes'] == record['optimizer_writes_before']
            assert binding['attempt_index'] == record['attempt_index'] and binding['head_version'] == record['probe_version']
            assert record['probe_actor_binding']['row_order_verified'] is True
            assert record['probe_actor_binding']['feature_rows'] == 5120
            per_rank = 5120 // ranks
            assert receipt['row_ids'] == list(range(rank * per_rank, (rank + 1) * per_rank))
            assert receipt['feature_rows'] == per_rank and len(receipt['feature_sha256']) == 64
            if record['step'] == 1:
                assert receipt['isolation_before'] and receipt['isolation_before'] == receipt['isolation_after']
            proof.append(dict(rank=rank, step=record['step'], feature_sha256=receipt['feature_sha256']))
    return dict(accepted=True, actual_rank_count=ranks, actual_completed_outers=len(records), receipts=proof,
                retained_prefix_files=files if retained_prefix is not None else None)
