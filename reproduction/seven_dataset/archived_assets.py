"""Resolve the two recorded budget5 migration aliases without editing receipts.

The archived config, run identity, model bytes and data hashes remain binding.
This only resolves the same physical assets used by the existing PJ trainer.
No filesystem aliases are created and no pruned native state is inspected.
"""
from pathlib import Path
from .contracts import DATA_SHA, require

PJ_BASE = '/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-20261015-v1'
OLD_DATA = '/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1/common-data'
OLD_MODEL = '/home/jmzhang/Workspace/Search-E1-20261001/models/Qwen2.5-3B-Instruct'
PJ_MODEL = '/plm-shared/zhangjunming/Workspace/Search-Self-Evolution-20261006-PJ-runtime-v1/model'
BUDGET_RUN = '02ccdba4db9f4e68aacbb7edfa3042ab'


def resolve_assets(base, group, backend, receipt):
    actual = {k: str(receipt[k]) for k in ('data', 'model')}
    binding = {}
    if backend == 'pj_pt' and group == 'budget-5':
        require(str(base) == PJ_BASE and receipt['run_id'] == BUDGET_RUN
                and receipt['data_identity'] == DATA_SHA
                and receipt['group'] == group, 'Unregistered budget5 storage migration')
        for kind, old, new in [('data', OLD_DATA, PJ_BASE+'/common-data'),
                               ('model', OLD_MODEL, PJ_MODEL)]:
            require(actual[kind] in (old, new), 'Unregistered migrated '+kind+' path')
            if actual[kind] == old:
                binding[kind] = {'archived': old, 'physical': new}
                actual[kind] = new
    return {k: Path(v) for k, v in actual.items()}, binding
