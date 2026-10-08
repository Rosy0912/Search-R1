"""Read-only, CPU-only reconstruction of failed 3289 action provenance.

This is diagnostic evidence, never a probability acceptance or training gate.
Raw tensors stay on H100.  No model, forward, restore, optimizer, or Ray API.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
RUN = ROOT / 'runs/gigpo-e1-policy-safety-calibration25-job3289-1791251062110516012'
SOURCE = '52befe968c856162d1bc79cd0171bb38309f4f9e'
BINDING = '7cc3098a35d6f56e809e293e2f26642cc365eac8c49c1ddaedb592e989c79ce2'
INVENTORY = '0cc1015260365ad2ec0b644fe28b04105e16490932eb182d993c4b394fb26079'
GENERATION_SHA = '8aab854727dd08c152f937ea8b434f6716daea5721ffb6ce4cf15af8dc8f28b0'
MODEL = ROOT / 'assets/qwen2.5-3b-instruct-aa8e725'
SCOPE = 'READONLY_CPU_ACTION_DIAGNOSTIC_NOT_PROBABILITY_ACCEPTANCE'
MAX_EXAMPLES = 20
MAX_OUTPUT_BYTES = 4 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return dict(index=index, left=a, right=b)
    if len(left) != len(right):
        index = min(len(left), len(right))
        return dict(index=index, left=left[index] if index < len(left) else None,
                    right=right[index] if index < len(right) else None)
    return None


def trim_terminal_pad(tokens, pad):
    end = len(tokens)
    while end and tokens[end - 1] == pad:
        end -= 1
    return tokens[:end]


def bounded_text(value, limit=4000):
    encoded = value.encode('utf-8')
    return dict(text=value[:limit], truncated=len(value) > limit, characters=len(value),
                utf8_bytes=len(encoded), full_utf8_sha256=hashlib.sha256(encoded).hexdigest())


def checked_file(record):
    path = Path(record['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Missing/nonregular/symlink original')
    require(path.resolve() == path, 'Noncanonical original path')
    stat = path.stat()
    require(stat.st_size == record['bytes'] and sha256(path) == record['sha256'], 'Original bytes/SHA differ: ' + str(path))
    # Inode/mtime are observations in the actual failure inventory, not substitutes for SHA.
    for key, value in [('inode', stat.st_ino), ('mtime_ns', stat.st_mtime_ns)]:
        if key in record:
            require(value == record[key], 'Original ' + key + ' differs: ' + str(path))
    return dict(path=str(path), bytes=stat.st_size, sha256=record['sha256'],
                inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)


def extract_frozen_postprocess(path, torch):
    """Compile the two exact source methods; skip unrelated trainer/Tracking imports."""
    text = Path(path).read_text()
    tree = ast.parse(text)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LLMGenerationManager')
    wanted = [node for node in cls.body if isinstance(node, ast.FunctionDef)
              and node.name in ('_batch_tokenize', '_postprocess_responses')]
    require(len(wanted) == 2, 'Frozen postprocess methods missing')
    module = ast.Module(body=[ast.ClassDef(name='FrozenPostprocess', bases=[], keywords=[], body=wanted,
                                         decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(torch=torch, List=list)
    exec(compile(module, str(path), 'exec'), namespace)
    instance = namespace['FrozenPostprocess']()
    instance.config = SimpleNamespace(no_think_rl=False)
    method_digest = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    return instance, method_digest


def byte_decoder():
    # Qwen2 BPE ByteLevel vocabulary alphabet. Unknown alphabet => no byte conclusion.
    values = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    chars = list(values)
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return {chr(char): value for value, char in zip(values, chars)}


def text_boundary_evidence(tokenizer, sampled, decoded, processed):
    """Distinguish evidence of incomplete UTF8 from a closing cut within a token.

    A byte-level explanation is emitted only if the known Qwen alphabet maps every
    nonspecial token and its decoded text equals the actual tokenizer output.
    """
    mapping = byte_decoder()
    special = set(tokenizer.all_special_ids)
    pieces = tokenizer.convert_ids_to_tokens(sampled)
    blobs = []
    ends = []
    total = 0
    for token_id, piece in zip(sampled, pieces):
        if token_id in special:
            blob = b''
        elif not isinstance(piece, str) or any(char not in mapping for char in piece):
            return dict(byte_evidence_scope='UNAVAILABLE_UNKNOWN_TOKEN_ALPHABET', closing_inside_sampled_token=None)
        else:
            blob = bytes(mapping[char] for char in piece)
        blobs.append(blob)
        total += len(blob)
        ends.append(total)
    raw = b''.join(blobs)
    try:
        strict = raw.decode('utf-8', errors='strict')
        strict_valid, error = True, None
    except UnicodeDecodeError as exc:
        strict, strict_valid = None, False
        error = dict(start=exc.start, end=exc.end, reason=exc.reason,
                     reaches_end=exc.end == len(raw), byte_tail_hex=raw[-16:].hex())
    result = dict(byte_evidence_scope='DERIVED_QWEN_BYTELEVEL_VOCABULARY_NOT_ENGINE_BYTE_TENSOR',
        strict_utf8_valid=strict_valid, strict_utf8_error=error,
        byte_stream_sha256=hashlib.sha256(raw).hexdigest(), byte_stream_length=len(raw),
        replacement_characters=decoded.count('\ufffd'),
        byte_decode_matches_actual_text=strict == decoded if strict_valid else False,
        closing_inside_sampled_token=None)
    positions = {tag: decoded.find(tag) for tag in ('</search>', '</answer>')}
    result['closing_character_positions'] = positions
    selected = '</search>' if positions['</search>'] >= 0 else '</answer>' if positions['</answer>'] >= 0 else None
    result['selected_closing'] = selected
    if strict_valid and strict == decoded and selected is not None:
        end_character = positions[selected] + len(selected)
        cut = len(decoded[:end_character].encode('utf-8'))
        result.update(closing_cut_byte_offset=cut, closing_at_sampled_token_boundary=cut in ends,
                      closing_inside_sampled_token=cut not in ends,
                      processed_equals_exact_text_cut=processed == decoded[:end_character])
        start = 0
        for index, end in enumerate(ends):
            if start < cut < end:
                result['closing_cut_sampled_token_index'] = index
                result['closing_cut_sampled_token_id'] = sampled[index]
                break
            start = end
    return result


def mapping_check(raw, row, local_index, pad):
    prompts = raw['tensors']['prompts'][local_index]
    actual_prefix = prompts[prompts.ne(pad)].tolist()
    engine_prefix = raw['actual_engine_prompt_token_ids'][local_index]
    return dict(worker_prompt_matches_observer=actual_prefix == row['generation_prefix'],
                engine_prompt_matches_observer=engine_prefix == row['generation_prefix'],
                worker_engine_prefix_match=actual_prefix == engine_prefix)


def retained_replay(tokens, row, initial_prompt_length, cap=4096):
    previous_response = len(row['scoring_prefix']) - initial_prompt_length
    require(previous_response >= 0, 'Invalid stored scoring-prefix length')
    return tokens[:max(0, cap - previous_response)]


def selected_scoring_action(scored, row_index, step_id):
    tensor = scored['tensors']
    selected = tensor['step_ids'][row_index].eq(step_id) & tensor['loss_mask'][row_index].bool()
    return tensor['responses'][row_index][selected].tolist()


def row_diagnostic(tokenizer, sampled, returned, reencoded, text, row, initial_prompt_length,
                   mapping, scored_tokens, padded_sample_width, pad, eos):
    raw_decoded = tokenizer.decode(returned, skip_special_tokens=True)
    retained_decoded = tokenizer.decode(row['retained_action_tokens'], skip_special_tokens=True)
    full_reencoded = tokenizer.encode(raw_decoded, add_special_tokens=False)
    retained = row['retained_action_tokens']
    expected = retained_replay(reencoded, row, initial_prompt_length)
    raw_effective = trim_terminal_pad(sampled, pad)  # Middle PAD is never silently deleted.
    flags = dict(length_exceeds_v9_padded_sample=len(retained) > padded_sample_width,
        sampled_prefix_matches_retained=sampled[:len(retained)] == retained and len(sampled) >= len(retained),
        retained_matches_exact_source_postprocess_and_cap=expected == retained,
        retained_matches_effective_scoring_action=scored_tokens == retained,
        returned_response_starts_with_engine_sample=returned[:len(sampled)] == sampled,
        generation_matches_scoring_prefix=row['generation_prefix'] == row['scoring_prefix'], **mapping)
    return dict(source=row['source'], step_id=row['step_id'], flags=flags,
        sampled_raw_width=len(sampled), v9_global_padded_sample_width=padded_sample_width,
        first_pad_index=next((i for i, value in enumerate(sampled) if value == pad), None),
        raw_sampled_nonpad_count=sum(value != pad for value in sampled),
        raw_sampled_trim_terminal_pad_only=raw_effective,
        raw_middle_pad_positions=[i for i, value in enumerate(raw_effective) if value == pad],
        eos_positions=[i for i, value in enumerate(sampled) if value == eos],
        pad_positions=[i for i, value in enumerate(sampled) if value == pad],
        returned_response_width=len(returned), retained_length=len(retained),
        postprocessed_length=len(reencoded), expected_retained_length=len(expected),
        decode_reencode_full_length=len(full_reencoded),
        full_decode_reencode_equals_sampled_without_terminal_pad=full_reencoded == raw_effective,
        full_decode_reencode_equals_sampled_after_explicit_special_removal=full_reencoded ==
            [value for value in raw_effective if value not in set(tokenizer.all_special_ids)],
        raw_special_ids_removed_by_decode=[dict(index=i, token_id=value)
            for i, value in enumerate(raw_effective) if value in set(tokenizer.all_special_ids)],
        first_difference_sampled_retained=first_difference(sampled, retained),
        first_difference_replayed_retained=first_difference(expected, retained),
        raw_decoded=bounded_text(raw_decoded), processed_text=bounded_text(text),
        retained_decoded=bounded_text(retained_decoded), retained_text_equals_processed=retained_decoded == text,
        generation_prefix_length=len(row['generation_prefix']), scoring_prefix_length=len(row['scoring_prefix']),
        generation_prefix_token_sha256=token_digest(row['generation_prefix']),
        scoring_prefix_token_sha256=token_digest(row['scoring_prefix']),
        raw_sampled_tokens=sampled, retained_action_tokens=retained,
        replayed_postprocessed_tokens=reencoded,
        scoring_action_token_sha256=token_digest(scored_tokens),
        byte_boundary=text_boundary_evidence(tokenizer, sampled, raw_decoded, text))


def analyze(spec, torch, tokenizer, processor):
    originals = {row['path']: row for row in spec['raw_material_inventory']}
    turns = json.loads((RUN / 'training-prefix-turns.json').read_text())
    require(len(turns) == 4 and {row['source'] for row in turns[0]} == set(range(2560)), 'Wrong actual turn/source coverage')
    initial_lengths = {row['source']: len(row['scoring_prefix']) for row in turns[0]}
    score = torch.load(RUN / 'actual-rollout-2560.pt', map_location='cpu', weights_only=False, mmap=True)
    ids = [int(value) for value in score['non_tensors']['audit_trajectory_row']]
    require(len(ids) == 2560 and set(ids) == set(range(2560)), 'Scored audit-row permutation is incomplete')
    by_scoring_row = {source: index for index, source in enumerate(ids)}
    require(score['metadata']['baseline_outer_step'] == 26 and score['scope'] == 'GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION', 'Wrong scored batch scope')
    counters = dict(actions=0, empty_retained_actions=0, length_overflow=0, sampled_prefix_mismatch=0,
        exact_postprocess_replay_mismatch=0, scoring_action_mismatch=0, worker_prompt_mismatch=0,
        engine_prompt_mismatch=0, returned_raw_sample_mismatch=0, generation_scoring_prefix_mismatch=0,
        strict_utf8_invalid=0, incomplete_utf8_tail=0, closing_inside_sampled_token=0,
        full_decode_reencode_changed=0)
    lengths = {key: {} for key in ('sampled_width', 'retained_length', 'postprocessed_length')}
    examples = []
    coverage = []
    seen = set()
    pad, eos = spec['pad_token_id'], spec['eos_token_id']
    for serial, rows in enumerate(turns):
        tensors = []
        for rank in range(8):
            path = RUN / f'probabilities/train/call-{serial}-rank-{rank}.json'
            record = json.loads(path.read_text())
            require(record['serial'] == serial and record['rank'] == rank and record['phase'] == 'train'
                    and record['source_commit'] == SOURCE, 'Wrong actual per-turn JSON identity')
            raw_path = RUN / f'probabilities/train/call-{serial}-rank-{rank}.pt'
            require(record['raw']['path'] == str(raw_path)
                    and record['raw']['sha256'] == originals[str(raw_path)]['sha256']
                    and record['raw']['bytes'] == originals[str(raw_path)]['bytes'], 'JSON/PT receipt mismatch')
            raw = torch.load(raw_path, map_location='cpu', weights_only=False, mmap=True)
            require(raw['phase'] == 'train' and raw['serial'] == serial and raw['rank'] == rank, 'Wrong raw PT identity')
            require(len(raw['sampled_ids']) == len(raw['tensors']['responses']) == record['rows'], 'Raw returned row mismatch')
            tensors.append(raw)
        widths = [raw['sampled_ids'].shape[1] for raw in tensors]
        counts = [raw['sampled_ids'].shape[0] for raw in tensors]
        maximum = max(widths)
        require(len(set(counts)) == 1, 'v9 uniform-rank indexing would differ; preserve mapping error')
        total = sum(counts)
        require(total >= len(rows) and total - len(rows) < 8, 'Unexpected real GPU padding coverage')
        all_returned = torch.cat([raw['tensors']['responses'] for raw in tensors])
        require(all_returned.device.type == 'cpu', 'CUDA tensor encountered')
        processed, text = processor._postprocess_responses(all_returned[:len(rows)])
        coverage.append(dict(serial=serial, active_rows=len(rows), rank_rows=counts,
                             engine_sample_widths=widths, v9_global_padded_sample_width=maximum))
        for index, row in enumerate(rows):
            require(isinstance(row['source'], int) and isinstance(row['step_id'], int), 'Noninteger actual source/step')
            key = (row['source'], row['step_id'])
            require(key not in seen and row['source'] in initial_lengths, 'Duplicate/unknown source step')
            seen.add(key)
            rank, local = divmod(index, counts[0])
            raw = tensors[rank]
            sample = raw['sampled_ids'][local].tolist()
            # Match v9 cat padding exactly without allocating another tensor.
            sample_v9 = sample + [pad] * (maximum - len(sample))
            returned = all_returned[index].tolist()
            recoded = processed[index][processed[index].ne(pad)].tolist()
            mapping = mapping_check(raw, row, local, pad)
            scoring = selected_scoring_action(score, by_scoring_row[row['source']], row['step_id'])
            diag = row_diagnostic(tokenizer, sample_v9, returned, recoded, text[index], row,
                                  initial_lengths[row['source']], mapping, scoring, maximum, pad, eos)
            diag.update(serial=serial, rank=rank, local_row=local,
                        actual_engine_row_width=len(sample), raw_pt=originals[str(RUN / f'probabilities/train/call-{serial}-rank-{rank}.pt')])
            flags = diag['flags']
            counters['actions'] += 1
            counters['empty_retained_actions'] += int(not row['retained_action_tokens'])
            for name, flag in [('length_overflow', 'length_exceeds_v9_padded_sample'),
                ('sampled_prefix_mismatch', 'sampled_prefix_matches_retained'),
                ('exact_postprocess_replay_mismatch', 'retained_matches_exact_source_postprocess_and_cap'),
                ('scoring_action_mismatch', 'retained_matches_effective_scoring_action'),
                ('worker_prompt_mismatch', 'worker_prompt_matches_observer'),
                ('engine_prompt_mismatch', 'engine_prompt_matches_observer'),
                ('returned_raw_sample_mismatch', 'returned_response_starts_with_engine_sample'),
                ('generation_scoring_prefix_mismatch', 'generation_matches_scoring_prefix')]:
                counters[name] += int(flags[flag] if name == 'length_overflow' else not flags[flag])
            byte = diag['byte_boundary']
            counters['strict_utf8_invalid'] += int(byte.get('strict_utf8_valid') is False)
            error = byte.get('strict_utf8_error') or {}
            counters['incomplete_utf8_tail'] += int(error.get('reaches_end') is True
                                                   and error.get('reason') == 'unexpected end of data')
            counters['closing_inside_sampled_token'] += int(byte.get('closing_inside_sampled_token') is True)
            counters['full_decode_reencode_changed'] += int(not diag['full_decode_reencode_equals_sampled_after_explicit_special_removal'])
            for name, length in [('sampled_width', maximum), ('retained_length', len(row['retained_action_tokens'])),
                                 ('postprocessed_length', len(recoded))]:
                lengths[name][str(length)] = lengths[name].get(str(length), 0) + 1
            bad = flags['length_exceeds_v9_padded_sample'] or not all(value for key, value in flags.items()
                                                                    if key != 'length_exceeds_v9_padded_sample')
            if bad:
                # Preserve actual length-failure cases even when earlier benign
                # parser recodings already filled the bounded example list.
                diag['_priority'] = (0 if flags['length_exceeds_v9_padded_sample'] else
                    1 if not flags['retained_matches_exact_source_postprocess_and_cap'] else
                    2 if not all(mapping.values()) else 3)
                examples.append(diag)
                examples.sort(key=lambda item: (item['_priority'], item['serial'], item['rank'], item['local_row']))
                del examples[MAX_EXAMPLES:]
    require({source for source, step in seen} == set(range(2560)), 'Incomplete actual source coverage')
    for example in examples:
        example.pop('_priority')
    if counters['worker_prompt_mismatch'] or counters['engine_prompt_mismatch'] or counters['returned_raw_sample_mismatch']:
        finding = 'ROW_OR_INPUT_IDENTITY_ERROR_REQUIRES_INVESTIGATION'
    elif counters['exact_postprocess_replay_mismatch'] or counters['scoring_action_mismatch']:
        finding = 'REPLAY_OR_SCORING_RECORD_DIFF_REQUIRES_INVESTIGATION'
    elif counters['sampled_prefix_mismatch']:
        finding = 'EXACT_FROZEN_POSTPROCESS_PRODUCES_DIFFERENT_SAMPLED_TOKEN_PATH'
    else:
        finding = 'NO_ACTION_TOKEN_PATH_MISMATCH_OBSERVED_NO_PROBABILITY_ACCEPTANCE'
    return dict(status='DIAGNOSTIC_COMPLETE_NOT_ACCEPTANCE', counters=counters, length_histograms=lengths,
        structural_finding=finding,
        turn_coverage=coverage, unique_source_steps=len(seen), covered_sources=2560, bounded_bad_examples=examples,
        bad_examples_limit=MAX_EXAMPLES, example_selection='LENGTH_FAILURE_THEN_REPLAY_OR_MAPPING_FAILURE_THEN_TOKEN_MISMATCH_ORIGINAL_ORDER',
        findings_are_not_probability_acceptance=True, probabilities_or_thresholds_selected=False)


def validate_spec(spec, failure):
    require(spec['scope'] == SCOPE and spec['source_commit'] == SOURCE and spec['binding_sha256'] == BINDING,
            'Wrong diagnostic scope/source/binding')
    require(spec['actual_run'] == str(RUN) and spec['job_id'] == 3289 and spec['node'] == 'h100-8', 'Wrong actual run')
    require(spec['pad_token_id'] == 151643 and spec['eos_token_id'] == 151645, 'Official PAD/EOS identity changed')
    require(failure['run'] == str(RUN) and failure['job_id'] == 3289 and failure['node'] == 'h100-8', 'Wrong failure original')
    original = [{k: row[k] for k in ('path', 'bytes', 'sha256', 'inode', 'mtime_ns')} for row in failure['raw_material_inventory']]
    require(spec['raw_material_inventory'] == original, 'Raw inventory not exact actual failure original')
    expected = {str(RUN / name) for name in ('training-prefix-turns.json', 'actual-rollout-2560.pt')}
    expected.update(str(RUN / f'probabilities/train/call-{serial}-rank-{rank}.{suffix}')
                    for serial in range(4) for rank in range(8) for suffix in ('json', 'pt'))
    require(len(original) == 66 and len({row['path'] for row in original}) == 66
            and {row['path'] for row in original} == expected, 'Actual 4x8 raw inventory differs')
    require(len(spec['official_model_metadata']) == 11, 'Missing official 11 metadata identities')
    require(all(Path(row['path']).parent == MODEL and not row['path'].endswith('.safetensors')
                for row in spec['official_model_metadata']), 'Metadata load would include weights/other model')
    require(len({row['path'] for row in spec['official_model_metadata']}) == 11, 'Duplicate official metadata')
    control_paths = {str(RUN / name) for name in ('run-result.json', 'resolved-config.yaml')}
    actual_control = [{k: row[k] for k in ('path', 'bytes', 'sha256', 'inode', 'mtime_ns')}
                      for row in failure['originals'] if row['path'] in control_paths]
    require(spec['control_originals'] == actual_control and len(actual_control) == 2,
            'Diagnostic control originals do not match actual failure evidence')


def execute(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU diagnostic requires CUDA_VISIBLE_DEVICES empty')
    require(socket.gethostname().split('.')[0] == 'h100-8', 'Raw tensor diagnostic must stay on h100-8')
    input_path = Path(args.identity)
    inventory_path = Path(args.inventory)
    output = Path(args.output)
    require(input_path.parent == ROOT / 'logs' and inventory_path.parent == ROOT / 'logs', 'Diagnostic inputs must be original ROOT/logs')
    require(output.parent == ROOT / 'logs' and output.name == 'CALIBRATION3289-ACTION-IDENTITY-CPU-DIAGNOSTIC-v1.json'
            and not output.exists(), 'Diagnostic output must be new unique bounded ROOT/logs file')
    require(sha256(input_path) == args.identity_sha256 and sha256(inventory_path) == INVENTORY, 'Wrong frozen input/original failure SHA')
    spec = json.loads(input_path.read_text())
    failure = json.loads(inventory_path.read_text())
    validate_spec(spec, failure)
    helper = Path(__file__).resolve()
    require(helper == ROOT / 'tools/diagnostic-action-3289-v1/diagnose_actions_v1.py'
            and sha256(helper) == spec['diagnostic_helper_sha256'], 'Diagnostic helper path/bytes changed')
    binding_path = ROOT / 'logs/policy-safety-calibration-52befe9-reviewed-binding-v9.json'
    require(sha256(binding_path) == BINDING, 'Actual calibration launch binding changed')
    actual_binding = json.loads(binding_path.read_text())
    require(actual_binding['code_commit'] == SOURCE and actual_binding['pad_token_id'] == spec['pad_token_id'], 'Actual launch binding identity differs')
    closure = actual_binding['files']
    by_path = {row['path']: row for row in closure}
    require(len(by_path) == len(closure), 'Duplicate actual binding files')
    for row in spec['official_model_metadata'] + spec['source_files']:
        require({key: row[key] for key in ('path', 'bytes', 'sha256')} == by_path[row['path']], 'Identity does not derive from actual launch binding')
    inputs = spec['raw_material_inventory'] + spec['official_model_metadata'] + spec['source_files'] + spec['control_originals']
    inputs += [dict(path=str(binding_path), bytes=binding_path.stat().st_size, sha256=BINDING),
               dict(path=str(input_path), bytes=input_path.stat().st_size, sha256=args.identity_sha256),
               dict(path=str(inventory_path), bytes=inventory_path.stat().st_size, sha256=INVENTORY),
               dict(path=str(helper), bytes=helper.stat().st_size, sha256=spec['diagnostic_helper_sha256'])]
    require(len({row['path'] for row in inputs}) == len(inputs), 'Duplicate diagnostic input')
    before = [checked_file(row) for row in inputs]
    actual_result = json.loads((RUN / 'run-result.json').read_text())
    require(actual_result['status'] == 'FAILED' and actual_result['job_id'] == '3289'
            and actual_result['source_commit'] == SOURCE and actual_result['binding_sha256'] == BINDING,
            'Wrong actual failed run control identity')
    import yaml
    config = yaml.safe_load((RUN / 'resolved-config.yaml').read_text())
    require(config['algorithm']['no_think_rl'] is False and config['max_turns'] == 4
            and config['force_final_answer'] is False, 'Actual text processing configuration differs')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    require(not torch.cuda.is_initialized() and torch.__version__.startswith('2.4.'), 'Wrong CPU Torch/runtime')
    torch.set_num_threads(4)
    source = ROOT / 'repo-policy-safety-52befe9/search_r1/llm_agent/generation.py'
    require(sha256(source) == GENERATION_SHA, 'Frozen generation source differs')
    ast_digest = tokenizer = None
    try:
        processor, ast_digest = extract_frozen_postprocess(source, torch)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
        require(tokenizer.pad_token_id == spec['pad_token_id'] and tokenizer.eos_token_id == spec['eos_token_id'], 'Actually loaded tokenizer differs')
        processor.tokenizer = tokenizer
        result = analyze(spec, torch, tokenizer, processor)
    except Exception as exc:
        result = dict(status='DIAGNOSTIC_FAILED_NOT_ACCEPTANCE', error_class=type(exc).__name__,
                      error=str(exc)[:2000], findings_are_not_probability_acceptance=True,
                      probabilities_or_thresholds_selected=False)
    after = [checked_file(row) for row in inputs]
    require(before == after and not torch.cuda.is_initialized(), 'Read-only diagnostic inputs/CUDA state changed')
    result.update(scope=SCOPE, job_id=3289, source_commit=SOURCE, binding_sha256=BINDING, actual_run=str(RUN),
        torch_version=torch.__version__, cuda_initialized_before=False, cuda_initialized_after=False,
        model_loaded=False, forward_calls=0, restore_calls=0, optimizer_writes=0, retained_updates_contributed=0,
        exact_frozen_postprocess_ast_sha256=ast_digest,
        loaded_tokenizer_class=type(tokenizer).__name__ if tokenizer is not None else None,
        official_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
        originals_before=before, originals_after=after, all_originals_unchanged=True)
    raw = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    require(len(raw) <= MAX_OUTPUT_BYTES, 'Diagnostic aggregate exceeds bounded output')
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(path=str(output), bytes=len(raw), sha256=sha256(output), scope=SCOPE)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', required=True)
    parser.add_argument('--identity-sha256', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--output', required=True)
    execute(parser.parse_args())
