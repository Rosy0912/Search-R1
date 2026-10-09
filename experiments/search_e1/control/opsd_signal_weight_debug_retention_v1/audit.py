"""Fresh retained native student generation/HTTP and teacher diagnostic audit."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
try:
    from .common import (VARIANTS, ORIGINAL_SHA, STUDY_ROOT, require, checked_file,
                         read, rows, same, bind_helper, canonical)
except ImportError:
    from common import (VARIANTS, ORIGINAL_SHA, STUDY_ROOT, require, checked_file,
                        read, rows, same, bind_helper, canonical)

SCHEMA = 'opsd-signal-debug-retention-v1'
REQUEST = 'opsd-signal-debug-retention-request-v1'
RUNTIME_SHA = '3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc'
STUDY_SHA = '86db5c8e1a86f6b5fd27f880ca062d193b4623e29869e8fa2aa3dba0e8e46c8e'
CONFIG_SHAS = {
    'G_POS': 'e7c0659bc6544d3440b7f67b1ba2f976c56c79d8371e23b4dbe614cbb894f661',
    'G_NEG': '2c37c851193dbc9a2d2512a72dd49310b9b8bcc14ac903159af4a491ef773a0d',
    'H_POS': '934f3e1ab8f70d7a1faa14b9806e9ddaecdbaea6afc97668ca3389a5c7237612',
    'H_NEG': 'ddb7d18dfcd5826f6aede00bcb507cebe553d8af5a28b9ef3d299c137b92c8bc',
}


class Inventory:
    """Never print private contents; track and finally rehash all referenced files."""
    def __init__(self):
        self.files = {}

    def file(self, ref):
        proof = checked_file(ref)
        previous = self.files.get(proof['path'])
        require(previous is None or previous == proof, 'Conflicting referenced file identity')
        self.files[proof['path']] = proof
        return proof

    def json(self, ref):
        value, proof = read(ref); self.file(proof)
        return value, proof

    def rows(self, ref):
        value, proof = rows(ref); self.file(proof)
        return value, proof

    def finish(self):
        proofs = [self.file(self.files[p]) for p in sorted(self.files)]
        return dict(files=len(proofs), bytes=sum(p['bytes'] for p in proofs),
                    sha256=canonical(proofs), fresh_end_sha_recheck=True)


def plain_ref(proof):
    return {k: proof[k] for k in ('path', 'sha256')}


def ints(value):
    return isinstance(value, list) and all(type(x) is int and x >= 0 for x in value)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def generation_trace(records, rollouts, rank, model_sha):
    """Native vLLM rows are per active turn, not one row per trajectory."""
    counts = Counter(len(row['policy_spans']) for row in rollouts)
    require(len(rollouts) == 320, 'Logical worker must retain all320 rollouts')
    for row in rollouts:
        end = 0
        require(isinstance(row['response'], str), 'Native rollout response missing')
        for start, stop in row['policy_spans']:
            require(type(start) is int and type(stop) is int and end <= start <= stop <= len(row['response']),
                    'Native authored policy span bounds invalid')
            end = stop
    expected_calls = max(counts, default=0)
    require(expected_calls <= 5, 'Native five-turn budget changed')
    grouped = {}
    for event in records:
        call = event['call']; index = event['index']
        require(type(call) is int and type(index) is int and call >= 0 and index >= 0,
                'Invalid native generation call/index')
        require(event['rank'] == rank and event['sampling_model_identity'] == model_sha,
                'Generation trace rank/raw model differs')
        require(finite_number(event['started']) and finite_number(event['ended'])
                and event['ended'] >= event['started'], 'Incomplete native generation timing')
        require(ints(event['prompt_token_ids']) and ints(event['generated_token_ids'])
                and isinstance(event['native_output_text'], str)
                and isinstance(event['input_text_sha256'], str) and len(event['input_text_sha256']) == 64,
                'Actual raw prompt/generated IDs or native output text missing')
        require('finish_reason' in event and 'stop_reason' in event,
                'Actual native termination evidence missing')
        require(index not in grouped.setdefault(call, {}), 'Duplicated native generation slot')
        grouped[call][index] = event
    require(set(grouped) == set(range(expected_calls)), 'Native generation call sequence incomplete')
    for call, events in grouped.items():
        active = [row for row in rollouts if len(row['policy_spans']) > call]
        expected = len(active)
        require(set(events) == set(range(expected)), 'Active-turn native raw generation slots incomplete')
        for index, row in enumerate(active):
            start, end = row['policy_spans'][call]
            require(events[index]['native_output_text'] == row['response'][start:end],
                    'Raw native generation text differs from the authored rollout span')
    require(len(records) == sum(len(row['policy_spans']) for row in rollouts),
            'Native generated-turn trace does not cover every authored rollout span')
    return dict(rows=len(records), generated_token_ids=sum(len(e['generated_token_ids']) for e in records),
                calls=expected_calls, zero_span_rollouts=counts.get(0, 0),
                zero_token_generations=sum(not e['generated_token_ids'] for e in records))


def teacher_events(trace, result):
    """HF batch1 output includes its input; do not pretend it is response-only."""
    require(trace['schema'] == 'opsd-ablation-privileged-teacher-autoregressive-v1'
            and trace['teacher_forced'] is False and trace['original_training_objective'] is False
            and trace['return_objects_modified'] is False and trace['generation_options_modified'] is False
            and not trace['logging_errors'], 'Autonomous teacher trace scope/integrity differs')
    calls = []
    for event in trace['events']:
        kind = event['kind']
        if kind != 'native_generate':
            require(kind in ('tokenize_context', 'encode_before_observation_truncation', 'decode_actual_selected_ids'),
                    'Unexpected native teacher trace event')
            continue
        require(type(event['call']) is int and event['call'] == len(calls),
                'Teacher native generate call sequence incomplete')
        require(finite_number(event['started_unix']) and finite_number(event['ended_unix'])
                and event['ended_unix'] >= event['started_unix'], 'Teacher native timing incomplete')
        inputs = event['args'][0] if event.get('args') else event.get('kwargs', {}).get('inputs')
        raw = event['raw_output_ids']
        require(isinstance(inputs, list) and len(inputs) == 1 and ints(inputs[0])
                and isinstance(raw, list) and len(raw) == 1 and ints(raw[0])
                and raw[0][:len(inputs[0])] == inputs[0],
                'Teacher full native input+new output IDs missing/misaligned')
        require(isinstance(event['kwargs'], dict), 'Teacher native generation kwargs missing')
        calls.append((inputs[0], raw[0]))
    require(len(calls) == len(result['policy_spans']) <= 5,
            'Teacher actual generation calls do not cover authored spans')
    return dict(calls=len(calls), new_token_ids=sum(len(raw)-len(inp) for inp, raw in calls),
                generated_ids_scope='HF native full input+new output; no posthoc retokenization')


def http_payload(records, url, timeout, singleton=False):
    """Validate the actual immutable native requests.post payload and response.

    The frozen vLLM call batches all searches in one request, timeout300;
    frozen HF SearchAgent requests one query, timeout120. No network is called.
    """
    require(isinstance(url, str) and url, 'Configured retrieval URL missing')
    query_count = 0
    for record in records:
        args = record.get('positional_args'); kwargs = record.get('keyword_args')
        require(isinstance(args, list) and args == [url] and isinstance(kwargs, dict)
                and set(kwargs) == {'json', 'timeout'}
                and type(kwargs['timeout']) is int and kwargs['timeout'] == timeout,
                'Native HTTP URL/keyword arguments/timeout missing or changed')
        body = kwargs['json']
        require(isinstance(body, dict) and set(body) == {'queries', 'topk', 'return_scores'}
                and isinstance(body['queries'], list) and len(body['queries']) > 0
                and all(isinstance(query, str) for query in body['queries'])
                and type(body['topk']) is int and body['topk'] == 3 and body['return_scores'] is True,
                'Native HTTP queries/topk3/score request missing or changed')
        queries = body['queries']
        require(not singleton or len(queries) == 1, 'HF teacher native HTTP must retain one query per request')
        status = record.get('status_code')
        require(type(status) is int and 200 <= status < 300, 'Native HTTP response is missing a successful2xx status')
        text = record.get('response_body_text'); saved = record.get('response_json')
        require(isinstance(text, str) and isinstance(saved, dict), 'Native HTTP complete body/parsed JSON missing')
        def reject_constant(_):
            raise ValueError('Native HTTP response body has a non-JSON numeric constant')
        parsed = json.loads(text, parse_constant=reject_constant)
        require(parsed == saved and canonical(parsed) == canonical(saved),
                'Native HTTP body text differs from the retained parsed response JSON')
        result = saved.get('result')
        require(isinstance(result, list) and len(result) == len(queries)
                and all(isinstance(passages, list) and all(isinstance(passage, dict) for passage in passages) for passages in result),
                'Native HTTP query/result batch coverage or passage shape differs')
        query_count += len(queries)
    return dict(requests=len(records), queries=query_count, payload_and_body_verified=True,
                query_count_not_inferred_from_rollout_search_calls=True)


def controller_reader(native, config, cfg):
    """Construct only an inert reader; never call Controller init/gate/spawn/wait."""
    obj = object.__new__(native['controller'].Controller)
    # Original requests bind original fileproof metadata as well as path/SHA.
    # Recreate that proof using its native function, not our compact inventory ref.
    obj.config = config; obj.config_ref = native['checkpoint'].checked_file(cfg); obj.run_id = config['run_id']
    obj.owner = dict(uid=config['owner']['uid'])
    return obj


def exit_evidence(inventory, reader, value, phase, count):
    reader.verify_exit(value, phase, count)  # Unchanged original recorded-wait verifier.
    terminal, tp = inventory.json(value['actual_terminal'])
    evidence = value['child_exit_evidence']
    require(terminal['variant'] == reader.config['variant']
            and len({(e['identity']['pid'], e['identity']['starttick']) for e in evidence}) == count
            and all(type(e['identity']['starttick']) is int and e['identity']['starttick'] >= 0
                    and type(e['identity']['pid']) is int and e['identity']['pid'] > 0
                    and not e['identity'].get('identity_uncertain', False) for e in evidence),
            'Recorded native child process identity is uncertain/aliased')
    for e in evidence:
        if e.get('log'):
            inventory.file(e['log'])
    return tp


def scoped(inventory, config, ref):
    require(Path(ref['path']).resolve().is_relative_to(Path(config['paths']['output_root']).resolve()),
            'Debug input escapes this group output root')
    return inventory.file(ref)


def sampling_debug(inventory, native, reader, manifest, sample):
    cfg = reader.config; identity = manifest['sampling_model_identity']
    require(sample['schema'] == 'opsd-online-sampling-complete-v1' and sample['accepted'] is True
            and sample['run_id'] == reader.run_id and sample['sampling_model_identity'] == identity
            and same(sample['rollouts'], manifest['rollouts']) and same(sample['pairs'], manifest['pairs']),
            'Sampling receipt differs from actual immutable outer inputs')
    terminal = exit_evidence(inventory, reader, sample['exit_proof'], 'sampling-running', 8)
    waves = sample['exit_proof']['waves']
    require(len(waves) == 2, 'Two actual four-worker waves required')
    for wave in waves:
        exit_evidence(inventory, reader, wave, 'sampling-running', 4)
    flattened = [e for wave in waves for e in wave['child_exit_evidence']]
    require(flattened == sample['exit_proof']['child_exit_evidence'], 'Logical worker wave/terminal identity differs')
    all_rollouts, rp = inventory.rows(manifest['rollouts']); scoped(inventory, cfg, rp)
    questions, qp = inventory.rows(manifest['questions']); scoped(inventory, cfg, qp)
    require(len(questions) == 512 and len({q['question_id'] for q in questions}) == 512,
            'Original full512 question cohort required')
    require(len(all_rollouts) == 2560 and len(sample['workers']) == 8
            and len(sample['private_http_traces']) == 8, 'Full2560/eight debug worker inventories required')
    worker_rows = []; compact = []; calls = http_requests = raw_ids = 0
    for rank, ref in enumerate(sample['workers']):
        scoped(inventory, cfg, ref); worker, wp = inventory.json(ref)
        actual = flattened[rank]['identity']
        require(worker['schema'] == 'opsd-online-sampling-worker-complete-v1' and worker['accepted'] is True
                and worker['rank'] == rank and worker['run_id'] == reader.run_id and worker['config_sha256'] == reader.config_ref['sha256']
                and worker['variant'] == cfg['variant'] and worker['uid'] == cfg['owner']['uid']
                and worker['pid'] == actual['pid'] and worker['rows'] == 320
                and worker['precision'] == cfg['precision'] and worker['sampling_model_identity'] == identity['sha256']
                and worker['actual_llm_load']['actual_dtype'] == 'torch.'+cfg['sampling']['dtype'],
                'Logical worker raw model/process/precision/coverage differs')
        for name in ('rollouts', 'generation_trace', 'http_trace'):
            scoped(inventory, cfg, worker[name])
        data, _ = inventory.rows(worker['rollouts']); traces, _ = inventory.rows(worker['generation_trace'])
        expected = {q['question_id']: q for q in questions[rank::8]}
        require(len(expected) == 64 and len(data) == 320
                and len({(r['question_id'], r['sibling']) for r in data}) == 320
                and {(r['question_id'], r['sibling']) for r in data} == {(qid, s) for qid in expected for s in range(5)}
                and all(r['question'] == expected[r['question_id']]['question']
                        and r['answers'] == expected[r['question_id']]['answers'] for r in data),
                'Logical64-question/five-sibling actual rollout shard incomplete or foreign')
        measured = generation_trace(traces, data, rank, identity['sha256'])
        require(same(worker['http_trace'], sample['private_http_traces'][rank]), 'Parent HTTP reference differs')
        native['sampling'].verify_http_trace(worker['http_trace'], rank, identity['sha256'], worker['http_requests'])
        http, hp = inventory.rows(worker['http_trace'])
        http_payload(http, cfg['paths']['retriever_url'], 300)
        require(all(type(r.get('generation_call')) is int and 0 <= r['generation_call'] < measured['calls'] for r in http),
                'Native HTTP retrieval does not bind an actual completed generation call')
        worker_rows.extend(data); calls += measured['rows']; raw_ids += measured['generated_token_ids']; http_requests += len(http)
        compact.append(dict(rank=rank, worker=plain_ref(wp), generation_trace=plain_ref(worker['generation_trace']),
                            http_trace=plain_ref(hp), generated_turns=measured['rows'], http_requests=len(http)))
    require(worker_rows == all_rollouts, 'Merged rollout rows differ from eight actual retained shards')
    return dict(workers=8, generated_turns=calls, raw_generated_token_ids=raw_ids, http_requests=http_requests,
                worker_inventory_sha256=canonical(compact), terminal=plain_ref(terminal))


def teacher_debug(inventory, native, reader, manifest, index, parent_ref):
    cfg = reader.config; scoped(inventory, cfg, parent_ref)
    parent, pp = inventory.json(parent_ref); pairs, _ = inventory.rows(manifest['pairs'])
    selection = native['teacher_diag'].select_cases(pairs, index)
    saved, sp = inventory.json(parent['selection']); scoped(inventory, cfg, sp)
    require(saved == selection and parent['schema'] == 'opsd-ablation-teacher-diagnostic-parent-v1'
            and parent['accepted'] is True and parent['run_id'] == reader.run_id and parent['variant'] == cfg['variant']
            and parent['outer_index'] == index and parent['config_sha256'] == reader.config_ref['sha256']
            and same(parent['pairs'], manifest['pairs']) and parent['count'] == selection['count']
            and parent['diagnostic_teacher_autoregressive'] is True and parent['original_training_objective'] is False
            and parent['diagnostic_precision_common_bf16'] is True and parent['training_or_validation_denominator_contribution'] == 0,
            'Teacher diagnostic parent/input/count scope differs')
    if selection['not_applicable']:
        require(parent['not_applicable'] is True and parent['request'] is None and parent['child'] is None
                and parent['exit_proof'] is None and parent['count'] == 0,
                'No-success reference NA must not invent a teacher child')
        return dict(parent=plain_ref(pp), cases=0, not_applicable=True, generated_turns=0,
                    raw_generated_token_ids=0, http_requests=0)
    require(parent['not_applicable'] is False, 'Successful B cannot be labeled teacher NA')
    request, reqp = inventory.json(parent['request']); child, cp = inventory.json(parent['child'])
    for ref in (reqp, cp): scoped(inventory, cfg, ref)
    exit_evidence(inventory, reader, parent['exit_proof'], 'sampling-running', 1)
    # Reuse the exact original B/D/source/strict EM/HTTP/actual child checks.
    reader.verify_teacher_child(child, request, parent['request'], selection, index, parent['exit_proof'])
    generated_turns = requests = raw_ids = 0
    for ref in child['cases']:
        scoped(inventory, cfg, ref); case, _ = inventory.json(ref)
        measured = teacher_events(case['trace'], case['result']); generated_turns += measured['calls']; raw_ids += measured['new_token_ids']
        scoped(inventory, cfg, case['http']); http, _ = inventory.rows(case['http']); requests += len(http)
        http_payload(http, cfg['paths']['retriever_url'], 120, singleton=True)
        require(len(http) == case['result']['search_calls'], 'Teacher executed search/HTTP count differs')
        require(all(type(r.get('generation_call')) is int and 0 <= r['generation_call'] < measured['calls'] for r in http),
                'Teacher HTTP trace refers to no actual generated action')
    return dict(parent=plain_ref(pp), child=plain_ref(cp), cases=selection['count'], not_applicable=False,
                generated_turns=generated_turns, raw_generated_token_ids=raw_ids, http_requests=requests)


def outer_debug(inventory, native, config, cfg, entry, previous, updates, exposure):
    reader = controller_reader(native, config, cfg); index = entry['outer_index']
    refs = ('outer_manifest', 'sampling_complete', 'outer_complete', 'end_checkpoint', 'trainer_exit', 'teacher_parent')
    for name in refs: scoped(inventory, config, entry[name])
    manifest, mp = inventory.json(entry['outer_manifest']); sample, _ = inventory.json(entry['sampling_complete'])
    complete, cp = inventory.json(entry['outer_complete']); checkpoint, kp = inventory.json(entry['end_checkpoint'])
    require(manifest['run_id'] == config['run_id'] and manifest['config_sha256'] == cfg['sha256']
            and manifest['outer_index'] == index and manifest['previous_complete_checkpoint'] == previous
            and manifest['outer_old_adapter'] == (previous or config['common_initial_adapter_manifest'])
            and manifest['cumulative_optimizer_updates_before'] == updates
            and manifest['runtime_identity'] == config['runtime_manifest']['sha256']
            and manifest['source_model_identity'] == config['source_model_manifest']['sha256']
            and manifest['common_initial_adapter_manifest'] == config['common_initial_adapter_manifest']
            and manifest['contract'] == native['checkpoint'].science_contract(config),
            'Completed outer raw/config/runtime/cumulative sampling identity differs')
    identity = manifest['sampling_model_identity']
    prior = inventory.json(previous)[0] if previous else None
    require(identity['checkpoint'] == previous and identity['optimizer_updates'] == updates
            and identity['source_model_identity'] == config['source_model_manifest']['sha256']
            and identity['adapter_tensor_sha256'] == (prior['all_rank_adapter_sha256'][0] if prior else config['common_initial_adapter_tensor_sha256']),
            'Sampling did not bind the actual preceding raw checkpoint')
    require(complete['schema'] == 'opsd-online-outer-complete-v1' and complete['accepted'] is True
            and complete['run_id'] == config['run_id'] and complete['outer_index'] == index
            and same(complete['checkpoint'], entry['end_checkpoint']) and complete['P'] == manifest['P']
            and type(complete['K']) is int and 0 <= complete['K'] <= complete['P'] <= 512
            and complete['optimizer_updates_delta'] == math.ceil(complete['K']/32)
            and complete['optimizer_updates'] == updates+complete['optimizer_updates_delta']
            and complete['cumulative_pair_exposure'] == exposure+complete['K'],
            'Input outer is incomplete or optimizer/exposure chain differs')
    require(checkpoint['schema'] == 'opsd-online-checkpoint-v1' and checkpoint['accepted'] is True
            and checkpoint['run_id'] == config['run_id'] and checkpoint['config_sha256'] == cfg['sha256']
            and checkpoint['optimizer_updates'] == complete['optimizer_updates']
            and checkpoint['cursor']['outer_index'] == index
            and checkpoint['cursor']['next_pair_cursor'] == checkpoint['cursor']['usable_pair_count'] == complete['K']
            and same(checkpoint['training_input']['outer_manifest'], entry['outer_manifest'])
            and checkpoint['sampling_model_identity'] == identity
            and checkpoint['runtime_identity'] == config['runtime_manifest']['sha256']
            and checkpoint['source_model_identity'] == config['source_model_manifest']['sha256']
            and checkpoint['contract'] == native['checkpoint'].science_contract(config)
            and checkpoint['common_initial_adapter'] == config['common_initial_adapter_manifest']
            and checkpoint['outer_old_adapter'] == manifest['outer_old_adapter']
            and checkpoint['cumulative_pair_exposure'] == complete['cumulative_pair_exposure'],
            'Debug references are not a complete committed raw outer state')
    require(len(checkpoint['all_rank_adapter_sha256']) == 4
            and len(set(checkpoint['all_rank_adapter_sha256'])) == 1
            and isinstance(checkpoint['all_rank_adapter_sha256'][0], str) and len(checkpoint['all_rank_adapter_sha256'][0]) == 64,
            'Raw checkpoint lacks its actual four-rank adapter identity declaration')
    trainer, _ = inventory.json(entry['trainer_exit']); exit_evidence(inventory, reader, trainer, 'training-running', 1)
    sampling = sampling_debug(inventory, native, reader, manifest, sample)
    teacher = teacher_debug(inventory, native, reader, manifest, index, entry['teacher_parent'])
    require('torch' not in sys.modules, 'Retention audit imported Torch')
    return dict(variant=config['variant'], outer_index=index, run_id=config['run_id'], config=plain_ref(cfg),
                outer_manifest=plain_ref(mp), outer_complete=plain_ref(cp), end_checkpoint=plain_ref(kp),
                sampling_complete=plain_ref(entry['sampling_complete']), sampling=sampling, teacher=teacher), entry['end_checkpoint'], complete['optimizer_updates'], complete['cumulative_pair_exposure']


def verify_source(inventory, config):
    ref = config['helper_source_manifest']
    require(ref['sha256'] == ORIGINAL_SHA, 'Only original immutable46-file helper permitted')
    manifest, _ = inventory.json(ref); require(len(manifest['files']) == 46, 'Original helper inventory count differs')
    for name, item in manifest['files'].items():
        inventory.file(dict(path=str(Path(config['paths']['helper_root'])/name), **item))
    require(config['runtime_manifest']['sha256'] == RUNTIME_SHA, 'Original frozen305 runtime identity differs')
    runtime, _ = inventory.json(config['runtime_manifest']); files = runtime.get('files', runtime)
    require(len(files) == 305, 'Frozen305 runtime inventory differs')
    root = Path(config['paths']['runtime']).resolve()
    for name, value in files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and (root/name).resolve().is_relative_to(root), 'Runtime source path escaped original root')
        inventory.file(dict(path=str(root/name), **value) if isinstance(value, dict) else dict(path=str(root/name), sha256=value))
    native = bind_helper(config['paths']['helper_root'], ref, ('checkpoint', 'sampling', 'controller', 'teacher_diag', 'gates'))
    publication, _ = inventory.json(config['gates']['publication'])
    native['gates'].publication_admission(config, native['gates'].scoped(config, publication))
    return native


def published_reader(inventory, spec):
    manifest, mp = inventory.json(spec['source_manifest']); publication, pp = inventory.json(spec['publication'])
    root = Path(spec['helper_root']).resolve()
    require(root/'audit.py' == Path(__file__).resolve()
            and publication['accepted'] is True and publication['branch'] == 'search-e1'
            and isinstance(publication['source_commit'], str) and len(publication['source_commit']) == 40
            and publication['source_commit'] == publication['remote_readback_sha']
            and publication['source_manifest_sha256'] == mp['sha256'], 'Independent retention reader source is not published/bound')
    require({name for name in manifest['files'] if name.endswith('.py')}
            == {str(p.relative_to(root)) for p in root.rglob('*.py')}, 'Retention reader Python inventory differs')
    for name, item in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe retention reader source path')
        inventory.file(dict(path=str(root/name), **item))
    return dict(source_manifest=plain_ref(mp), publication=plain_ref(pp), source_commit=publication['source_commit'])


def audit(request):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'Retention audit requires empty CUDA visibility and no Torch import')
    require(request['schema'] == REQUEST and request['mode'] in ('prefix', 'final'), 'Wrong explicit retention request')
    n = request['currentN']; require(type(n) is int and 1 <= n <= 100, 'Explicit currentN1..100 required')
    final = request['mode'] == 'final'; require(not final or n == 100, 'Final retention requires100 completed outers')
    require(set(request['configs']) == set(VARIANTS), 'Four exact immutable config refs required')
    expected = [(v, o) for v in VARIANTS for o in range(25)][:n]
    require([(r['variant'], r['outer_index']) for r in request['outers']] == expected,
            'Explicit input is not the serial complete outer prefix')
    require(request['study_manifest']['sha256'] == STUDY_SHA
            and all(request['configs'][v]['sha256'] == CONFIG_SHAS[v] for v in VARIANTS),
            'Only the original explicit study/four immutable configs may be audited')
    inventory = Inventory(); source = published_reader(inventory, request['retention_source'])
    study, studyref = inventory.json(request['study_manifest'])
    require(study['study_base'] == STUDY_ROOT and [g['variant'] for g in study['groups']] == list(VARIANTS)
            and all(same(g['config'], request['configs'][g['variant']]) for g in study['groups']), 'Original study/config identities differ')
    configs = {}; natives = {}
    for v, ref in request['configs'].items():
        cfg, proof = inventory.json(ref); configs[v] = (cfg, proof)
        require(cfg['schema'] == 'opsd-signal-weight-config-v1' and cfg['variant'] == v
                and cfg['study_id'] == study['study_id'] and cfg['paths']['study_base'] == STUDY_ROOT
                and cfg['owner']['uid'] == os.getuid() == 1001 and cfg['scientific']['outer_iterations'] == 25
                and cfg['scientific']['global_batch'] == 32 and cfg['scientific']['micro_batch'] == 1
                and cfg['scientific']['world_size'] == 4 and cfg['scientific']['gradient_accumulation_full_batch_per_rank'] == 8
                and cfg['owner']['physical_gpus'] == [4, 5, 6, 7],
                'Authorized config/run/owner/four-card25 contract differs')
        natives[v] = verify_source(inventory, cfg)
    for name in ('source_model_manifest', 'runtime_manifest', 'common_initial_adapter_manifest', 'common_cohort_manifest', 'validation_identity', 'train_identity'):
        require(len({configs[v][0][name]['sha256'] for v in VARIANTS}) == 1, 'Four-arm common identity changed: '+name)
        for v in VARIANTS:
            inventory.file(configs[v][0][name])
    state = {v: (None, 0, 0) for v in VARIANTS}; compact = []
    for entry in request['outers']:
        v = entry['variant']; cfg, ref = configs[v]
        row, cp, updates, exposure = outer_debug(inventory, natives[v], cfg, ref, entry, *state[v])
        state[v] = (cp, updates, exposure); compact.append(row)
    coverage_ref = None
    if request.get('accepted_coverage'):
        coverage, coverage_ref = inventory.json(request['accepted_coverage'])
        require(final and coverage['schema'] == 'opsd-signal-final-coverage-v1' and coverage['accepted'] is True
                and coverage['coverage_accepted'] is True and coverage['complete_outers'] == 100
                and coverage['formal_native512_slots'] == 4608 and same(coverage['study_manifest'], studyref)
                and coverage['final_checkpoints'] == {v: state[v][0] for v in VARIANTS}
                and {(r['variant'], r['outer_index'], r['checkpoint']['sha256']) for r in coverage['outer_rows']}
                    == {(r['variant'], r['outer_index'], r['end_checkpoint']['sha256']) for r in compact},
                'Accepted final coverage belongs to another raw study/outer chain')
    end = inventory.finish(); require('torch' not in sys.modules, 'Retention audit imported Torch')
    workers = sum(r['sampling']['workers'] for r in compact)
    require(workers == 8*n and (not final or workers == 800), 'Debug worker coverage incomplete')
    return dict(schema=SCHEMA, accepted=True, full_study_complete=final, mode=request['mode'], currentN=n,
                complete_outers=n, sampling_workers=workers, teacher_parents=n,
                teacher_cases=sum(r['teacher']['cases'] for r in compact),
                teacher_not_applicable_parents=sum(r['teacher']['not_applicable'] for r in compact),
                configs={v: plain_ref(configs[v][1]) for v in VARIANTS},
                original_helper_source_manifest_sha256=ORIGINAL_SHA, frozen_runtime_manifest_sha256=RUNTIME_SHA,
                study_id=study['study_id'], study_manifest=plain_ref(studyref), retention_source=source,
                accepted_coverage=plain_ref(coverage_ref) if coverage_ref else None,
                per_outer=compact, referenced_inventory=end,
                torch_imported=False, no_model_gpu_http_signals_or_rng_restore=True,
                no_raw_content_in_stdout=True, sampled_full_vocabulary_not_audited_here=True,
                scope='Fresh retained native generated IDs/HTTP and autonomous teacher cases; no generation, numeric optimizer readback, online token-loss or causal attribution claims')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--request', required=True); p.add_argument('--request-sha256', required=True)
    a = p.parse_args()
    try:
        request, ref = read(dict(path=a.request, sha256=a.request_sha256)); result = audit(request); result['request'] = plain_ref(ref)
    except Exception as error:
        result = dict(schema=SCHEMA, accepted=False, full_study_complete=False, error_type=type(error).__name__,
                      error=str(error), torch_imported='torch' in sys.modules)
    print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0 if result['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
