"""Metadata-only plan and in-job CPU preparation; tensor assets remain on h100-9."""
import hashlib
import json
from pathlib import Path

import torch

from search_r1.baselines.probability_probe_capture import file_sha, tensor_digest, require
from search_r1.baselines.probability_probe_contracts import ASSET_ROOT
from search_r1.baselines.probability_probe_io import load_tensor_verified
from window_contracts import (ORIGINAL_SELECTION_SHA, ORIGINAL_READY_SHA, ORIGINAL_CONFIG_SHA,
    PLANNED, anchors, bound_json, validate_inputs, json_payload_sha)

POSITION_REPORT_SHA = '326fe29839606f1d52dfc7cb68e8858f92e3f12f1178965343d88dbdde4bfff3'


def original_tokenizer_ids(hf_config):
    ids = {key:hf_config.get(key) for key in ('pad_token_id','eos_token_id','bos_token_id')}
    require(type(ids['pad_token_id']) is int and type(ids['eos_token_id']) is int and
            (ids['bos_token_id'] is None or type(ids['bos_token_id']) is int),
            'Actual bound HF pad/eos must be integers; BOS may be explicitly None')
    # HF serializes only nondefault fields: absent BOS has the None default.
    # PAD/EOS still require explicit integers, and the actual tokenizer is
    # compared exactly against these IDs before any model forward.
    return ids


def position_report_identity(tensor):
    """Byte convention of the already accepted position CPU report."""
    require(tensor.device.type == 'cpu', 'Original identity comparison requires CPU tensor')
    digest = hashlib.sha256(json.dumps(dict(dtype=str(tensor.dtype),shape=list(tensor.shape)),sort_keys=True).encode())
    for block in tensor.detach().contiguous().reshape(-1).view(torch.uint8).split(8*1024**2):
        digest.update(block.numpy().tobytes())
    return dict(dtype=str(tensor.dtype),shape=list(tensor.shape),shape_and_bytes_sha256=digest.hexdigest())


def freeze_metadata_plan(selection, position_report, *, selection_file, original_ready, hf_config_file, root=ASSET_ROOT):
    """Pure local metadata: no PT/HF tensor read and no actual-prepare claim."""
    require(selection_file['sha256'] == ORIGINAL_SELECTION_SHA and original_ready['sha256'] == ORIGINAL_READY_SHA,
            'Original frozen identities changed')
    require(selection['identity']['config_sha256'] == ORIGINAL_CONFIG_SHA and
            selection['cpu_results']['position']['sha256'] == POSITION_REPORT_SHA,
            'Bound original position/config metadata changed')
    inspections = [x for x in position_report['inspections'] if x.get('kind') == 'raw_same_turn' and x.get('generation_call') == 5]
    require(len(inspections) == 8 and {x['rank'] for x in inspections} == set(range(8)), 'CPU source shape report incomplete')
    metadata = {Path(entry['path']).name:entry for entry in selection['raw_metadata']}
    ranks = []
    for rank in range(8):
        asset = selection['inventory']['rank-%02d-call-00005.pt'%rank]
        record = metadata['rank-%02d-call-00005.json'%rank]
        entry = next(x for x in inspections if x['rank'] == rank)['inspection']
        identity = entry['full_input_tensor_identity']; rows,width = identity['input_ids']['shape']
        require(asset['rank'] == rank and asset['generation_call'] == 5 and asset['row_count'] == rows == 320,
                'Original fixed320 call5 inventory/report differs')
        require(entry['label'] == asset['path'], 'Source shape report names another asset')
        require(asset['original_micro4_groups'] == [list(range(i,i+4)) for i in range(0,320,4)],
                'Original80 complete adjacent4 groups changed')
        ranks.append(dict(rank=rank,local_rows=320,prompt_width=width-500,source_response_width=500,
            original_physical_rows=list(range(320)),expected_actor_micro4_calls=80,
            source_snapshot=dict(path=asset['path'],sha256=asset['sha256']),source_record=record,
            source_full_input_identity=identity,actual_tensor_rows_verified=False))
    result = dict(schema_version=1,status=PLANNED,frozen=True,asset_root=str(root),generation_call=5,
        original_selection_sha256=ORIGINAL_SELECTION_SHA,original_ready_sha256=ORIGINAL_READY_SHA,
        original_config_sha256=ORIGINAL_CONFIG_SHA,original_ready=original_ready,
        source_position_report=selection['cpu_results']['position'],hf_config_file=hf_config_file,
        original_identity=selection['identity'],ranks=ranks,anchors=anchors(),global_requests=2560,
        maximum_new_emitted_tokens=1280000,actual_model_forwards=0,actual_optimizer_writes=0,
        actual_tensor_rows_verified=False,CPU_job_executed_by_this_plan=False)
    return validate_inputs(result,root)


def materialize_rank_inputs(bundle, rank):
    """Run within the future owned job, CPU Gloo phase BEFORE CUDA selection."""
    require(not torch.cuda.is_initialized(), 'Rank prompt materialization must precede CUDA initialization')
    inputs = validate_inputs(bundle['window_inputs']); local = inputs['ranks'][rank]
    require(local['rank'] == rank, 'Rank plan order changed')
    record = bound_json(local['source_record']['path'],local['source_record']['sha256'])
    ready = bound_json(inputs['original_ready']['path'],inputs['original_ready']['sha256'])
    workers = sorted(ready['workers'],key=lambda worker:worker['rank'])
    worker = workers[rank]
    require(worker['rank'] == rank and worker['world_size'] == 8 and worker['optimizer_committed'] == 0,
            'Actual original zero-write parent worker changed')
    entries = [e for e in worker['diagnostic_entries'] if e['path'] == local['source_record']['path']]
    require(len(entries) == 1 and entries[0]['sha256'] == local['source_record']['sha256'] and
            entries[0]['snapshot_path'] == local['source_snapshot']['path'] and
            entries[0]['snapshot_sha256'] == local['source_snapshot']['sha256'], 'Ready/record/snapshot provenance differs')
    for key in ('code_commit','config_sha256','data_identity','model_identity','job_id','run_id'):
        require(record[key] == worker[key], 'Original parent/raw identity mismatch: '+key)
    require(record['local_rows'] == record['saved_rows'] == 320 and record['scheduler_step'] == 110 and
            record['snapshot_path'] == local['source_snapshot']['path'] and
            record['snapshot_sha256'] == local['source_snapshot']['sha256'], 'Complete call5 record changed')
    hf_config = bound_json(inputs['hf_config_file']['path'],inputs['hf_config_file']['sha256'])
    ids = original_tokenizer_ids(hf_config)
    payload = load_tensor_verified(local['source_snapshot']['path'],local['source_snapshot']['sha256'])
    actual = {key:position_report_identity(payload[key]) for key in ('input_ids','attention_mask','position_ids')}
    require(actual == local['source_full_input_identity'], 'Actual CPU fullinput bytes differ from accepted position report')
    report = bound_json(inputs['source_position_report']['path'],inputs['source_position_report']['sha256'])
    matches = [x for x in report['inspections'] if x.get('kind') == 'raw_same_turn' and x.get('generation_call') == 5 and x['rank'] == rank]
    require(len(matches) == 1 and matches[0]['inspection']['full_input_tensor_identity'] == actual,
            'Frozen source position report does not bind these actual tensors')
    prepared, meta = prepare_payload(payload,record,rank,ids['pad_token_id'])
    require(meta['prompt_width'] == local['prompt_width'] and meta['local_rows'] == 320,
            'New actual CPU materialization changed original320 complete prompts')
    prepared['tokenizer_metadata'] = dict(ids=ids,source_file=inputs['hf_config_file'])
    prepared['cpu_preparation'] = dict(status='ACTUAL_CALL5_CPU_INPUTS_VERIFIED_BEFORE_CUDA',
        rank=rank,local_rows=320,original_physical_rows=list(range(320)),
        original_generation_call=5,expected_actor_micro4_calls=80,
        actual_tensor_rows_verified=True,cuda_initialized=False,
        source_snapshot=local['source_snapshot'],source_record=local['source_record'],
        source_full_input_identity=actual,prepared_prompt_identity=meta['tensor_identity'],
        window_inputs_sha256=json_payload_sha(inputs),model_forwards=0,optimizer_writes=0)
    require(not torch.cuda.is_initialized(), 'Actual rank CPU load unexpectedly initialized CUDA')
    return prepared


def prepare_payload(payload, record, rank, pad_token_id):
    """Derive the full prompt slice, preserving source row/order/padding bytes."""
    require(type(pad_token_id) is int and type(rank) is int and 0 <= rank < 8, 'Explicit pad/rank required')
    require(record['rank'] == rank and record['world_size'] == 8 and record['generation_call'] == 5 and
            record['saved_rows'] == record['local_rows'] and record['optimizer_committed'] == 0,
            'Original full call5 diagnostic metadata required')
    required = ('input_ids', 'attention_mask', 'position_ids', 'responses')
    require(all(isinstance(payload.get(k), torch.Tensor) and payload[k].device.type == 'cpu' and
                payload[k].dtype == torch.int64 for k in required), 'Complete original CPU int64 buffers required')
    ids, mask, pos, responses = (payload[k] for k in required)
    require(ids.ndim == responses.ndim == 2 and mask.shape == pos.shape == ids.shape and
            ids.shape[0] == responses.shape[0] == record['local_rows'], 'Original buffer row/shape changed')
    width = responses.shape[1]
    require(0 < width <= 500 and width < ids.shape[1] and torch.equal(ids[:, -width:], responses),
            'Raw response must be the complete input suffix')
    prompt_width = ids.shape[1]-width
    require(0 < prompt_width <= 4096 and bool(((mask == 0) | (mask == 1)).all()), 'Invalid original prompt/mask')
    batch = {k: payload[k][:, :prompt_width].clone() for k in ('input_ids', 'attention_mask', 'position_ids')}
    lengths = []
    for row in range(len(ids)):
        valid = batch['attention_mask'][row].bool()
        length = int(valid.sum()); lengths.append(length)
        require(length > 0 and valid.tolist() == [False]*(prompt_width-length)+[True]*length,
                'Original rollout expects exactly left-padded prompts')
        require(batch['position_ids'][row][valid].tolist() == list(range(length)),
                'Original prompt position history is not zero-based contiguous')
        nonpad = torch.nonzero(batch['input_ids'][row] != pad_token_id, as_tuple=False)
        require(len(nonpad) and int(nonpad[0]) == prompt_width-length,
                'Original ID-based padding strip differs from attention-valid prompt')
    require(payload.get('distribution') == record['distribution'], 'Saved original distribution changed')
    metadata = dict(rank=rank, local_rows=len(ids), prompt_width=prompt_width,
        original_physical_rows=list(range(len(ids))), prompt_lengths=lengths,
        attention_tokens=sum(lengths), expected_actor_micro4_calls=(len(ids)+3)//4,
        tensor_identity={key:tensor_digest(value) for key,value in batch.items()},
        source_response_width=width, source_scope='All original call5 physical prompts; no selection/reordering/retokenization')
    return dict(batch=batch, metadata=metadata), metadata
