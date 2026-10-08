"""Original row .get->None semantics across shards for four trajectory-only fields."""
import numpy as np
from verl import DataProto
KEYS = ('tool_call_details', 'messages', 'initial_query', 'answers')


def concat_main(outputs, record, preserve):
    assert outputs and all(isinstance(v, DataProto) for v in outputs)
    mixed = {}
    for key in KEYS:
        present = [key in v.non_tensor_batch for v in outputs]
        for value, exists in zip(outputs, present):
            if exists:
                field = value.non_tensor_batch[key]
                assert isinstance(field, np.ndarray) and field.dtype == np.dtype(object)
                assert field.shape == (len(value),), ('Malformed present optional trajectory field', key)
        if any(present) and not all(present):
            mixed[key] = present
    if not mixed:
        return DataProto.concat(outputs)
    # Missing-key and an explicit None entry remain distinguishable in evidence.
    evidence = [dict(shard=i, rows=len(value), fields={key:dict(
                    field_present=present[i], original_field=value.non_tensor_batch[key]
                    if present[i] else None) for key,present in mixed.items()})
                for i,value in enumerate(outputs)]
    preserve('optional-trajectory-metadata-original-shards', evidence)
    record('optional_trajectory_metadata_missing_shards', dict(
        fields={key:dict(missing_shards=[i for i,exists in enumerate(present) if not exists],
            missing_rows=sum(len(v) for v,exists in zip(outputs,present) if not exists),
            present_explicit_none_rows=sum(sum(x is None for x in v.non_tensor_batch[key])
                                          for v,exists in zip(outputs,present) if exists))
                for key,present in mixed.items()},
        semantics='original_per_row_extra_fields_get_None', training_rows_changed=False))
    normalized=[]
    for i,value in enumerate(outputs):
        missing=[key for key,present in mixed.items() if not present[i]]
        if not missing:
            normalized.append(value)
        else:
            fields=dict(value.non_tensor_batch)
            for key in missing:fields[key]=np.full(len(value),None,dtype=object)
            normalized.append(type(value)(batch=value.batch,non_tensor_batch=fields,meta_info=value.meta_info))
    return DataProto.concat(normalized)
