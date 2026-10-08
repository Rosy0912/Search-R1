"""Only normalize the exact source52 audit-row int64 representation.

The source-generated integer IDs and row order must already be correct. This
function does not sort, repair, remove duplicates, change tokens/other metadata,
consume RNG or alter any GiGPO state. Baseline A can call this same metadata
adapter alone without installing the probability observer or admission arm.
"""
import numpy as np


def normalize_exact_audit_rows(data):
    if len(data)!=2560:
        raise ValueError('Audit metadata normalization requires the actual full512x5 batch')
    row=data.non_tensor_batch.get('audit_trajectory_row')
    if not isinstance(row,np.ndarray) or row.ndim!=1 or len(row)!=len(data):
        raise ValueError('Missing/malformed source audit metadata; cannot normalize')
    if row.dtype!=np.dtype(np.int64):
        raise ValueError('Expected the exact source52 int64 audit metadata representation')
    if row.tolist()!=list(range(len(data))):
        raise ValueError('Source audit integer values/order differ; cannot repair provenance')
    replacement=row.astype(object,copy=True)
    if not all(type(value) is int for value in replacement) or replacement.tolist()!=row.tolist():
        raise ValueError('Audit object conversion changed actual integer identity')
    data.non_tensor_batch['audit_trajectory_row']=replacement
    data.check_consistency()
    return dict(scope='EXACT_AUDIT_METADATA_REPRESENTATION_ONLY',rows=len(data),
                before_dtype='int64',after_dtype='object',values_and_order_preserved=True)
