"""Explicit, read-only CPU checkpoint audit; no model/optimizer/CUDA entry.

All eight saved optimizer states and both adapters are actually read. Earlier
token logs receive metadata/inventory checks; only the last committed mini's
token records are hashed and reconstructed. Source100 weights are NOT rehashed.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
STAT_KEYS = ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')
SCOPE = 'actual_training_forward_before_optimizer'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def stat_from(s):
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def path_under(path, root):
    p, root = Path(path), Path(root).resolve()
    require(p.is_absolute() and not p.is_symlink() and p.resolve().is_relative_to(root), 'Artifact path escaped bound root')
    return p.resolve()


class Reader:
    """Fixed-handle reads, plus end-of-audit metadata continuity, without writes."""
    def __init__(self):
        self.seen = {}

    def remember(self, proof):
        old = self.seen.get(proof['path'])
        require(old is None or all(old[k] == proof[k] for k in STAT_KEYS), 'Artifact changed between reads')
        self.seen[proof['path']] = proof
        return proof

    def metadata(self, ref):
        require(isinstance(ref, dict) and isinstance(ref.get('path'), str), 'Expected explicit file reference')
        path = Path(ref['path'])
        require(path.is_absolute() and not path.is_symlink() and path.is_file(), 'Missing/aliased artifact')
        proof = dict(path=str(path.resolve()), **stat_from(path.stat()))
        for key in STAT_KEYS:
            if key in ref:
                require(ref[key] == proof[key], 'Artifact metadata differs: ' + key)
        return self.remember(proof)

    def read(self, ref):
        before = self.metadata(ref)
        require(isinstance(ref.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Explicit SHA256 required')
        with Path(before['path']).open('rb') as h:
            require(stat_from(os.fstat(h.fileno())) == {k: before[k] for k in STAT_KEYS}, 'Opened inode changed')
            raw = h.read()
            require(stat_from(os.fstat(h.fileno())) == {k: before[k] for k in STAT_KEYS}, 'Artifact changed while reading')
        require(stat_from(Path(before['path']).stat()) == {k: before[k] for k in STAT_KEYS}, 'Artifact replaced while reading')
        digest = hashlib.sha256(raw).hexdigest()
        require(digest == ref['sha256'], 'Artifact SHA mismatch')
        return raw, self.remember(dict(before, sha256=digest))

    def json(self, ref):
        raw, proof = self.read(ref)
        return json.loads(raw), proof

    def finish(self):
        for path, proof in self.seen.items():
            require(stat_from(Path(path).stat()) == {k: proof[k] for k in STAT_KEYS}, 'Artifact changed before audit completion')
        return len(self.seen)


def fixed_handle_state(ref, torch, reader):
    """Hash and CPU-load the exact same opened inode; never create an optimizer."""
    before = reader.metadata(ref)
    with Path(before['path']).open('rb') as handle:
        require(stat_from(os.fstat(handle.fileno())) == {k: before[k] for k in STAT_KEYS}, 'State opened inode differs')
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(4 << 20), b''):
            h.update(block)
        require(h.hexdigest() == ref['sha256'], 'State SHA mismatch')
        handle.seek(0)
        state = torch.load(handle, map_location='cpu', weights_only=False)
        require(stat_from(os.fstat(handle.fileno())) == {k: before[k] for k in STAT_KEYS}, 'State mutated during CPU load')
    require(stat_from(Path(before['path']).stat()) == {k: before[k] for k in STAT_KEYS}, 'State replaced during CPU load')
    return state, reader.remember(dict(before, sha256=h.hexdigest()))


def recursive_equal(a, b, torch):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), 'Cross-rank optimizer tensor differs')
    elif isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), 'Cross-rank optimizer keys differ')
        for key in a:
            recursive_equal(a[key], b[key], torch)
    elif isinstance(a, (tuple, list)):
        require(type(a) is type(b) and len(a) == len(b), 'Cross-rank optimizer sequence differs')
        for left, right in zip(a, b):
            recursive_equal(left, right, torch)
    else:
        require(type(a) is type(b) and a == b, 'Cross-rank optimizer scalar/group differs')


def inspect_adam(optimizer, updates, parameter_count, science, torch):
    groups, entries = optimizer['param_groups'], optimizer['state']
    parameters = [p for group in groups for p in group['params']]
    require(len(parameters) == parameter_count == len(set(parameters)), 'Adam parameter inventory differs from adapter')
    require(set(entries) == set(parameters), 'Missing/extra actual Adam moments')
    opt = science['optimizer']
    for group in groups:
        require(group['lr'] == science['learning_rate'] and list(group['betas']) == opt['betas'] and group['eps'] == opt['eps'] and group['weight_decay'] == opt['weight_decay'], 'Actual Adam groups differ from scientific config')
    for entry in entries.values():
        require(set(entry) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Incomplete actual Adam state')
        require(all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and torch.isfinite(t).all().item() for t in entry.values()), 'Nonfinite/nonCPU actual Adam tensor')
        require(entry['step'].numel() == 1 and float(entry['step']) == updates, 'Actual Adam step differs')
        left, right = entry['exp_avg'], entry['exp_avg_sq']
        require(left.is_floating_point() and right.is_floating_point() and left.dtype == right.dtype and left.shape == right.shape, 'Adam moment shape/dtype differs')
        require((right >= 0).all().item(), 'Negative actual Adam squared moment')
    return dict(actual_state_parameters=len(entries), actual_adam_steps=[updates], finite_optimizer_tensors=3 * len(entries))


def bind_dependencies(config, reader):
    root = Path(config['paths']['helper_root']).resolve()
    mf, proof = reader.json(config['helper_source_manifest'])
    require(Path(proof['path']).parent == root, 'Helper manifest outside configured helper root')
    files = mf['files']; proofs = {}
    for name, expected in files.items():
        path = path_under(root / name, root)
        _, proofs[name] = reader.read(dict(expected, path=str(path)))
    modules = {}
    # trainer imports flat objective; reload only from explicitly bound files.
    for name in ('objective', 'checkpoint', 'trainer'):
        path = root / (name + '.py')
        require(path.name in proofs, 'Imported science file absent from immutable manifest')
        cached = sys.modules.get(name)
        require(cached is None or Path(getattr(cached, '__file__', '')).resolve() == path, 'Scientific module name shadowed')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # Compile the verified source bytes, never a timestamp-valid cached pyc.
        raw, _ = reader.read(dict(path=str(path), sha256=proofs[path.name]['sha256']))
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        require(Path(module.__file__).resolve() == path, 'Loaded scientific module escaped helper root')
        reader.read(dict(path=str(path), sha256=proofs[path.name]['sha256']))
        modules[name] = module
    modules['trainer'].validate_science(config)
    return modules, dict(manifest=proof, files=proofs)


def bind_source_and_mask(config, reader):
    source, source_proof = reader.json(config['source_model_manifest'])
    model_root = Path(config['paths']['model']).resolve()
    files = source.get('files', source.get('source_files'))
    require(isinstance(files, list) and len(files) == 12, 'Expected twelve source model identities')
    identities = []
    for ref in files:
        path_under(ref['path'], model_root)
        require(re.fullmatch('[0-9a-f]{64}', ref['sha256']) and integer(ref['bytes']), 'Malformed expected source file identity')
        identities.append(dict(expected=ref, current_metadata=reader.metadata(ref)))
    require(len({p['current_metadata']['path'] for p in identities}) == 12 and len({(p['current_metadata']['device'], p['current_metadata']['inode']) for p in identities}) == 12, 'Source model paths/inodes aliased')
    actual = {str(p.resolve()) for p in model_root.rglob('*') if p.is_file()}
    require(actual == {p['current_metadata']['path'] for p in identities}, 'Source model directory inventory changed')
    runtime, runtime_proof = reader.json(config['runtime_manifest'])
    entries = runtime.get('files', runtime)
    name = 'search_e1/core.py'; expected = entries[name]
    if isinstance(expected, str):
        expected = dict(sha256=expected)  # Native frozen305 manifest is bare path->SHA.
    raw, core_proof = reader.read(dict(expected, path=str(path_under(Path(config['paths']['runtime']) / name, config['paths']['runtime']))))
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('information_spans', 'response_encoding')]
    require({n.name for n in nodes} == {'information_spans', 'response_encoding'}, 'Frozen mask functions absent')
    namespace = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(core_proof['path']), 'exec'), namespace)
    return namespace['response_encoding'], dict(source_manifest=source_proof, expected_source_files=identities, runtime_manifest=runtime_proof, frozen_mask_source=core_proof, scope='Source100 manifest fresh SHA and twelve expected identities/current metadata; source weight bytes NOT rehashed. Only SHA-bound frozen char-offset mask functions executed; no model/tokenizer load.')


def checkpoint_identity(cp, ref, config, expected_updates=None, expected_outer=None):
    require(cp['schema'] == 'opsd-online-checkpoint-v1' and cp['accepted'] is True, 'Checkpoint incomplete')
    require(cp['run_id'] == config['run_id'] and cp['config_sha256'] == config['_config_sha256'] and cp['runtime_identity'] == config['runtime_manifest']['sha256'] and cp['source_model_identity'] == config['source_model_manifest']['sha256'], 'Checkpoint identity mismatch')
    updates, outer = cp['optimizer_updates'], cp['cursor']['outer_index']
    require(integer(updates) and integer(outer), 'Invalid checkpoint counters')
    require(expected_updates is None or updates == expected_updates, 'Explicit target update differs')
    require(expected_outer is None or outer == expected_outer, 'Explicit target outer differs')
    root = Path(config['paths']['output_root']).resolve() / 'checkpoints'
    directory = root / ('update-%08d-outer-%05d' % (updates, outer))
    require(Path(ref['path']).resolve() == directory / 'manifest.json', 'Checkpoint not exact committed canonical path')
    path_under(ref['path'], root)
    require(cp['complete_rank_count'] == 8 and cp['equal_optimizer_steps'] is True, 'Incomplete eight-rank declaration')
    require(len(cp['all_rank_adapter_sha256']) == 8 and len(set(cp['all_rank_adapter_sha256'])) == 1, 'Adapter rank digests differ')
    require(len(cp['ranks']) == 8 and {r['rank'] for r in cp['ranks']} == set(range(8)), 'Rank inventory differs')
    for row in cp['ranks']:
        require(Path(row['file']['path']).resolve() == directory / ('rank-%d-state.pt' % row['rank']), 'Rank state path escaped checkpoint')
        path_under(row['file']['path'], directory)
    require(Path(cp['adapter']['path']).resolve() == directory / 'adapter', 'Adapter directory escaped checkpoint')
    for row in cp['adapter']['files']:
        path_under(row['path'], directory / 'adapter')
    return directory


def load_adapter(cp, reader, trainer, torch):
    # Optional dependency required only for real artifact reads, not CPU fixtures.
    from safetensors.torch import load
    refs = [r for r in cp['adapter']['files'] if Path(r['path']).name == 'adapter_model.safetensors']
    require(len(refs) == 1, 'Expected one actual adapter tensor file')
    for ref in cp['adapter']['files']:
        reader.read(ref)
    raw, proof = reader.read(refs[0]); tensors = load(raw)
    require(tensors and all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and t.is_floating_point() and torch.isfinite(t).all().item() for t in tensors.values()), 'Nonfinite/nonCPU adapter')
    digest = trainer.tensor_digest(tensors)
    require(all(d == digest for d in cp['all_rank_adapter_sha256']), 'Exported adapter differs from saved rank digests')
    return tensors, proof, digest


def adapter_delta(current, prior, torch):
    require(set(current) == set(prior), 'Adapter parameter names differ')
    groups = {}; changed = 0; maximum = 0.0
    for name, value in current.items():
        before = prior[name]
        require(value.dtype == before.dtype and value.shape == before.shape, 'Adapter dtype/shape differs from baseline')
        require(value.numel() > 0, 'Empty adapter tensor')
        count = int(not torch.equal(value, before)); delta = float((value.float() - before.float()).abs().max())
        changed += count; maximum = max(maximum, delta)
        role = 'LoRA_A' if 'lora_A' in name else 'LoRA_B' if 'lora_B' in name else 'other'
        group = groups.setdefault(role, dict(tensors=0, changed=0, max_absolute_delta=0.0))
        group['tensors'] += 1; group['changed'] += count; group['max_absolute_delta'] = max(group['max_absolute_delta'], delta)
    require(changed > 0 and maximum > 0, 'No actual adapter change against explicit baseline')
    return dict(changed=changed, max_absolute_delta=maximum, per_role=groups)


def validate_updates(cp, outer, order, reader, config, previous):
    cursor = cp['cursor']['next_pair_cursor']; K = cp['cursor']['usable_pair_count']
    before = outer['cumulative_optimizer_updates_before']; refs = cp['outer_update_receipts']
    require(integer(cursor, 1) and integer(K, 1) and cursor <= K <= 512, 'UNSUPPORTED_K0_OR_NO_COMMITTED_UPDATE: use separate no-update audit')
    require(cursor == K or cursor % 32 == 0, 'Partial cursor split a native mini')
    require(integer(before) and cp['optimizer_updates'] == before + math.ceil(cursor / 32) and len(refs) == math.ceil(cursor / 32), 'Native update/cursor count differs')
    prior_exposure = previous['cumulative_pair_exposure'] if previous else 0
    require(cp['cumulative_pair_exposure'] == prior_exposure + cursor, 'Cumulative real pair exposure differs')
    summaries = []; trace_refs = []; seen_pairs = []; last = None; offset = 0
    previous_digest = previous['all_rank_adapter_sha256'][0] if previous else None
    for index, ref in enumerate(refs, 1):
        path_under(ref['path'], Path(outer['outer_dir']) / 'training')
        value, proof = reader.json(ref); actual = min(32, cursor - offset); batch = order[offset:offset + actual]
        require(value['schema'] == 'opsd-online-real-update-v1' and value['accepted'] is True and value['run_id'] == config['run_id'] and value['outer_index'] == cp['cursor']['outer_index'] and value['optimizer_updates'] == before + index, 'Native update identity differs')
        require(value['global_actual_pairs'] == actual and value['cumulative_pair_exposure'] == prior_exposure + offset + actual, 'Native actual tail denominator/exposure differs')
        require(value['loss_reduction'] == 'trajectory_mean' and value['micro_batch'] == 1 and value['learning_rate'] == config['scientific']['learning_rate'] and value['loss_scope'] == SCOPE and value['reference_student_prefix_shared'] is True and value['frozen_teacher'] is True, 'Native objective scope/config differs')
        if previous_digest is not None:
            require(value['pre_update_adapter_sha256'] == previous_digest, 'Native adapter digest chain broken')
        previous_digest = value['post_update_adapter_sha256']
        ranks = value['all_rank_actual_token_metrics']
        require(len(ranks) == 8 and {r['rank'] for r in ranks} == set(range(8)), 'Missing/duplicate native rank metrics')
        pairs = []
        for rank in ranks:
            require(finite(rank['grad_norm']), 'Nonfinite native gradnorm')
            require([p['pair_index'] for p in rank['pairs']] == batch[rank['rank']::8], 'Actual rank partition differs from native shuffled order')
            for pair in rank['pairs']:
                require(integer(pair['valid_tokens'], 1) and integer(pair['clipped_vocabulary_count']) and finite(pair['raw_loss']) and finite(pair['post_loss']), 'Native pair count/loss nonfinite or invalid')
                path_under(pair['token_log']['path'], Path(outer['outer_dir']) / 'training')
                reader.metadata(pair['token_log']); trace_refs.append(pair['token_log']); seen_pairs.append(pair['pair_index']); pairs.append(pair)
        require(len(pairs) == actual and sum(p['valid_tokens'] for p in pairs) == value['global_valid_tokens'], 'Native global pair/token counts differ')
        require(len({r['grad_norm'] for r in ranks}) == 1, 'Native global gradnorm differs across ranks')
        summaries.append(dict(receipt=proof, optimizer_updates=value['optimizer_updates'], global_actual_pairs=actual, global_valid_tokens=value['global_valid_tokens'], trajectory_mean_pre_clip_kl=sum(p['raw_loss'] for p in pairs) / actual, trajectory_mean_post_clip_loss=sum(p['post_loss'] for p in pairs) / actual, clipped_vocabulary_count=sum(p['clipped_vocabulary_count'] for p in pairs), learning_rate=value['learning_rate'], grad_norm=ranks[0]['grad_norm']))
        last = value; offset += actual
    require(len(seen_pairs) == len(set(seen_pairs)) == cursor and len(trace_refs) == len({r['path'] for r in trace_refs}) == cursor, 'Committed pair/token trace inventory duplicated')
    require(previous_digest == cp['all_rank_adapter_sha256'][0], 'Last native post-update adapter differs from checkpoint')
    return summaries, last, trace_refs


def validate_tail(header, records, pair, rank, update, config, encoding_rows, pair_rows, native_mask):
    require(header['schema'] == 'opsd-online-training-token-header-v1' and header['run_id'] == config['run_id'] and header['outer_index'] == update['outer_index'] and header['optimizer_update'] == update['optimizer_updates'] and header['rank'] == rank and header['pair_index'] == pair['pair_index'], 'Actual tail header identity differs')
    require(header['encoded_sha256'] == encoding_rows[pair['pair_index']] and header['student_pre_update_adapter_sha256'] == update['pre_update_adapter_sha256'] and header['frozen_teacher_identity'] == config['source_model_manifest']['sha256'] and header['sampling_model_identity'] == config['_outer_sampling_identity'], 'Tail source/sampling/encoding provenance differs')
    require(header['component_clip'] == config['scientific']['component_clip'] and header['topk_cap'] == config['logging']['topk_max_tokens_per_outer'] and header['loss_scope'] == SCOPE and header['full_vocab_persisted'] is False, 'Tail loss/logging scope differs')
    identity = header['identity']; original = pair_rows[identity['pair_row_sha256']]
    require(identity['question_id'] == original['question_id'] and identity['reference_reward'] == original['reference_reward'] == 1 and identity['student_reward'] == original['student_reward'] and identity['student_response_sha256'] == hashlib.sha256(original['student'].encode()).hexdigest() and identity['reference_response_sha256'] == hashlib.sha256(original['reference'].encode()).hexdigest(), 'Tail original private pair identity differs')
    ids, mask, offsets = (header[k] for k in ('token_ids', 'policy_mask', 'token_offsets'))
    require(isinstance(ids, list) and ids and len(ids) == len(mask) == len(offsets) and all(integer(i) for i in ids) and all(type(m) in (int, bool) and m in (0, 1) for m in mask), 'Invalid actual token/mask shape or values')
    require(all(isinstance(o, (tuple, list)) and len(o) == 2 and all(integer(v) for v in o) and 0 <= o[0] <= o[1] <= len(original['student']) for o in offsets), 'Invalid stored native char offsets')
    def stored_tokenizer(text, **kwargs):
        require(text == original['student'] and kwargs == dict(add_special_tokens=False, return_offsets_mapping=True), 'Unexpected frozen mask tokenizer call')
        return dict(input_ids=ids, offset_mapping=offsets)
    _, expected_mask = native_mask(stored_tokenizer, original['student'], original['student_policy_spans'])
    require(mask == expected_mask, 'Actual policy mask differs from frozen spans/information exclusion')
    positions = [i for i, m in enumerate(mask) if m]
    require(len(records) == pair['valid_tokens'] == len(positions) and [r['token_index'] for r in records] == positions, 'Tail masked positions missing/duplicate/out of order')
    for record in records:
        pos = record['token_index']
        require(record['policy_mask'] is True and record['token_id'] == ids[pos] and record['token_offsets'] == offsets[pos], 'Tail actual token/offset/mask differs')
        require(record['loss_scope'] == SCOPE and record['teacher_forced_fixed_student_prefix'] is True and record['actual_generation_scores'] is False, 'Tail is not native pre-update training forward')
        for field in ('pre_clip_full_vocabulary_kl', 'post_clip_full_vocabulary_loss', 'clipped_teacher_probability_mass', 'clipped_vocabulary_fraction', 'trajectory_contribution', 'global_batch_contribution'):
            require(finite(record[field]), 'Nonfinite actual tail metric')
        count, vocab = record['clipped_vocabulary_count'], record['vocabulary_size']
        require(integer(vocab, 1) and integer(count) and count <= vocab and -1e-7 <= record['clipped_teacher_probability_mass'] <= 1 + 1e-7, 'Invalid actual clipping statistics')
        require(math.isclose(record['clipped_vocabulary_fraction'], count / vocab, rel_tol=1e-12, abs_tol=1e-12), 'Clip fraction denominator differs')
        if config['scientific']['component_clip'] is None:
            require(count == 0 and math.isclose(record['pre_clip_full_vocabulary_kl'], record['post_clip_full_vocabulary_loss'], rel_tol=1e-9, abs_tol=1e-9), 'Disabled clip still changed loss')
        contribution = record['post_clip_full_vocabulary_loss'] / len(records)
        require(math.isclose(record['trajectory_contribution'], contribution, rel_tol=1e-12, abs_tol=1e-12) and math.isclose(record['global_batch_contribution'], contribution / update['global_actual_pairs'], rel_tol=1e-12, abs_tol=1e-12), 'Trajectory/global actual tail normalization differs')
    for field, summary in (('pre_clip_full_vocabulary_kl', 'raw_loss'), ('post_clip_full_vocabulary_loss', 'post_loss')):
        require(math.isclose(sum(r[field] for r in records) / len(records), pair[summary], rel_tol=1e-9, abs_tol=1e-9), 'Actual tail loss does not reconstruct native summary')
    require(sum(r['clipped_vocabulary_count'] for r in records) == pair['clipped_vocabulary_count'], 'Actual tail clipping count differs')
    return dict(rank=rank, pair_index=pair['pair_index'], valid_tokens=len(records), raw_loss=pair['raw_loss'], post_loss=pair['post_loss'], clipped_vocabulary_count=pair['clipped_vocabulary_count'], actual_policy_mask_reconstructed_from_frozen_char_offsets=True, tokenizer_or_logits_recomputed=False, native_tail_denominator=update['global_actual_pairs'])


def audit(args):
    started = time.time(); reader = Reader()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CUDA_VISIBLE_DEVICES empty required')
    config, config_proof = reader.json(dict(path=args.config, sha256=args.config_sha256)); config['_config_sha256'] = config_proof['sha256']
    require(os.geteuid() == config['owner']['uid'], 'Effective UID differs from configured owner')
    modules, source_helpers = bind_dependencies(config, reader)
    mask, source_identity = bind_source_and_mask(config, reader)
    trainer = modules['trainer']; checkpoint = modules['checkpoint']
    import torch
    require(not torch.cuda.is_initialized(), 'CPU audit unexpectedly initialized CUDA'); torch.set_num_threads(1)
    target_ref = dict(path=args.checkpoint, sha256=args.checkpoint_sha256)
    cp, cp_proof = reader.json(target_ref); directory = checkpoint_identity(cp, target_ref, config, args.expected_updates, args.expected_outer)
    require(cp['cursor']['next_pair_cursor'] > 0, 'UNSUPPORTED_K0_OR_NO_COMMITTED_UPDATE: use separate no-update audit')
    _, native_proofs = checkpoint.check_checkpoint(target_ref, config, config['run_id'], args.expected_updates, args.expected_outer)
    base_ref = dict(path=args.baseline_checkpoint, sha256=args.baseline_sha256)
    prior_cp, prior_cp_proof = reader.json(base_ref); checkpoint_identity(prior_cp, base_ref, config)
    require(cp['optimizer_updates'] > prior_cp['optimizer_updates'], 'Target must be after explicit baseline')
    current, current_proof, current_digest = load_adapter(cp, reader, trainer, torch)
    prior, prior_proof, prior_digest = load_adapter(prior_cp, reader, trainer, torch)
    delta = adapter_delta(current, prior, torch); parameter_count = len(current)
    ranks = []; first = None
    for row in sorted(native_proofs['ranks'], key=lambda r: r['rank']):
        state, proof = fixed_handle_state(row['file'], torch, reader)
        trainer.verify_rank_state(state, cp, row['rank'], config)
        actual = inspect_adam(state['optimizer'], cp['optimizer_updates'], parameter_count, config['scientific'], torch)
        if first is None:
            first = state['optimizer']
        else:
            recursive_equal(first, state['optimizer'], torch)
        ranks.append(dict(rank=row['rank'], file=proof, **actual, optimizer_digest=trainer.optimizer_tensor_digest(state['optimizer']), adapter_digest=state['adapter_tensor_sha256'], full_actual_moments_and_groups_equal_rank0=True, rng_cuda_state_count=len(state['rng']['cuda_all'])))
        del state
    outer, outer_proof = reader.json(cp['training_input']['outer_manifest'])
    require(outer['run_id'] == config['run_id'] and outer['config_sha256'] == config['_config_sha256'] and outer['outer_index'] == args.expected_outer and Path(outer['outer_dir']).resolve() == Path(config['paths']['output_root']).resolve() / 'outers' / ('outer-%05d' % args.expected_outer), 'Bound training outer identity/path differs')
    previous = None
    if outer.get('previous_complete_checkpoint'):
        previous, _ = reader.json(outer['previous_complete_checkpoint']); checkpoint_identity(previous, outer['previous_complete_checkpoint'], config)
    trainer.verify_sampling_identity(outer, config, previous)
    require(cp['sampling_model_identity'] == outer['sampling_model_identity'], 'Checkpoint fixed sampling identity differs')
    config['_outer_sampling_identity'] = outer['sampling_model_identity']
    order_value, order_proof = reader.json(cp['training_input']['order']); encoding, encoding_proof = reader.json(cp['training_input']['encoding'])
    K = cp['cursor']['usable_pair_count']; order = order_value['pair_indices']
    require(encoding['K'] == K and len(encoding['rows']) == K and all(isinstance(s, str) and re.fullmatch('[0-9a-f]{64}', s) for s in encoding['rows']) and encoding['sha256'] == canonical_sha(encoding['rows']) and order_value['encoded_sha256'] == encoding['sha256'] == cp['cursor']['encoded_sha256'] and order_proof['sha256'] == cp['cursor']['order_sha256'] and order_value['outer_index'] == args.expected_outer and sorted(order) == list(range(K)) and all(integer(i) for i in order), 'Native encoding/order/cursor identities differ')
    require(cp['training_input']['pairs'] == outer['pairs'], 'Training pair reference differs from immutable outer')
    raw, pair_proof = reader.read(cp['training_input']['pairs']); private_pairs = [json.loads(line) for line in raw.splitlines()]
    require(len(private_pairs) == encoding['P'] <= 512, 'Private mined P count differs')
    pair_rows = {canonical_sha(row): row for row in private_pairs}; require(len(pair_rows) == len(private_pairs), 'Duplicate private mined row')
    summaries, last, trace_refs = validate_updates(cp, outer, order, reader, config, previous)
    readbacks = []
    for rank in last['all_rank_actual_token_metrics']:
        for pair in rank['pairs']:
            raw, proof = reader.read(pair['token_log']); require(raw.endswith(b'\n'), 'Tail token log lacks complete final newline')
            lines = raw.splitlines(); header = json.loads(lines[0]); records = [json.loads(line) for line in lines[1:]]
            readbacks.append(dict(file=proof, **validate_tail(header, records, pair, rank['rank'], last, config, encoding['rows'], pair_rows, mask)))
    require(len(readbacks) == last['global_actual_pairs'], 'Actual tail trace inventory incomplete')
    # Fresh rehash tiny immutable imported sources; all other read artifacts keep
    # exact inode/mtime/ctime/size continuity through completion.
    for proof in source_helpers['files'].values():
        reader.read(proof)
    reader.read(source_identity['frozen_mask_source'])
    checked_artifacts = reader.finish(); require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    return dict(schema='opsd-online-explicit-checkpoint-independent-cpu-audit-v1', accepted=True, started_unix=started, ended_unix=time.time(), run_id=config['run_id'], config=config_proof, source_helpers=source_helpers, source_identity=source_identity, checkpoint=cp_proof, optimizer_updates=cp['optimizer_updates'], cursor=cp['cursor'], cumulative_pair_exposure=cp['cumulative_pair_exposure'], protection_reason=cp['protection_reason'], ranks=ranks, adapter=dict(current_file=current_proof, baseline_file=prior_proof, baseline_manifest=prior_cp_proof, current_tensor_digest=current_digest, baseline_tensor_digest=prior_digest, actual_tensor_count=parameter_count, all_finite=True, **delta), training_inputs=dict(outer=outer_proof, order=order_proof, encoding=encoding_proof, pairs=pair_proof), native_update_summaries=summaries, actual_native_training_token_trace_count=len(trace_refs), latest_native_token_readbacks=readbacks, end_metadata_continuity_artifact_count=checked_artifacts, token_scope='All committed current-outer summary receipts validated; earlier token logs metadata/path/inventory only. All latest mini token records fresh SHA and native scalar/mask/offset/actual global denominator reconstruction. Stored tokenizer IDs/prefix IDs and logits NOT reevaluated; no historical full-vocabulary arrays or earlier-record claim.', baseline_scope='Explicit baseline manifest identity and actual finite adapter only; baseline Adam states not reread.', rng_scope='Native private Python/Numpy state-container validation and saved Torch CPU/CUDA uint8 shapes; no global/production RNG restored or sampled.', no_model_loaded=True, no_optimizer_created_or_mutated=True, no_node_output_written=True, no_http_gpu_signals_git=True, cuda_initialized_before=False, cuda_initialized_after=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'config-sha256', 'checkpoint', 'checkpoint-sha256', 'baseline-checkpoint', 'baseline-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--expected-updates', type=int, required=True); parser.add_argument('--expected-outer', type=int, required=True)
    args = parser.parse_args()
    try:
        result = audit(args)
    except Exception as error:
        # Avoid exception repr: third-party parsers can include private raw rows.
        print(json.dumps(dict(schema='opsd-online-explicit-checkpoint-independent-cpu-audit-v1', accepted=False, error_type=type(error).__name__, scope='Rejected audit; no training/runtime mutation. Inspect privately for missing dependencies or unsupported zero-update outer.'), sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
