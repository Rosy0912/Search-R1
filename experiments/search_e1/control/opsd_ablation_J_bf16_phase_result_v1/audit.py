#!/usr/bin/env python3
"""Closed J BF16512 phase only: stored primary slots, native QA and paired flips."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import sys
import time
import types

BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
EVAL = BASE / 'eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851'
PHASE_SHA = '221ed29ca109e2b4e1a4babd500c192f724615e0f624359e929e839381d9ebd6'
HELPER_MANIFEST_SHA = '2e55c809d91bd3b4654d0ec0c4b0532e3ee8c413aac69a19cfd8f28bf042c842'
SUMMARY_SHA = '609e2e3379d2a26d3af0f9ddad522dabf97df0bbbe63cf7a23d0937d52f48877'
RUNTIME_SHA = '3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc'
DATA_SHA = '75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76'
QA_SHA = '9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Reader:
    def __init__(self):
        self.proofs = {}

    def read(self, path, expected=None):
        path = Path(path)
        require(path.is_absolute() and '..' not in path.parts
                and all(not p.is_symlink() for p in (path, *path.parents)), 'Canonical nonsymlink input required')
        def stamp(value):
            return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
                        mode=value.st_mode, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)
        before = stamp(path.stat())
        require(stat.S_ISREG(before['mode']), 'Regular input required')
        with path.open('rb') as handle:
            require(stamp(os.fstat(handle.fileno())) == before, 'Opened input changed')
            raw = handle.read()
            require(stamp(os.fstat(handle.fileno())) == before, 'Input changed while reading')
        require(stamp(path.stat()) == before, 'Input path changed while reading')
        digest = hashlib.sha256(raw).hexdigest()
        require(expected is None or digest == expected, 'Pinned input SHA differs: ' + str(path))
        proof = dict(path=str(path), sha256=digest, **before)
        if str(path) in self.proofs:
            require(proof == self.proofs[str(path)], 'Repeated input identity changed')
        self.proofs[str(path)] = proof
        return raw, proof

    def json(self, path, expected=None):
        raw, proof = self.read(path, expected)
        return json.loads(raw), proof

    def bound(self, ref):
        value, proof = self.json(ref['path'], ref['sha256'])
        require('bytes' not in ref or ref['bytes'] == proof['bytes'], 'Declared JSON bytes differ')
        return value, proof

    def rows(self, ref):
        raw, proof = self.read(ref['path'], ref['sha256'])
        require(raw.endswith(b'\n') and ('bytes' not in ref or ref['bytes'] == proof['bytes']), 'Partial/size-mismatched JSONL')
        return [json.loads(line) for line in raw.splitlines()], proof


def selected_functions(raw, names, namespace, label):
    tree = ast.parse(raw)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require(len(nodes) == len(names) and {node.name for node in nodes} == set(names)
            and not any(node.decorator_list for node in nodes), 'Exact frozen undecorated functions required')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), label, 'exec'), namespace)


def load_module(name, raw, path):
    module = types.ModuleType(name); module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def validate_rows(rows, data, version, run):
    rows.sort(key=lambda row: row['validation_index'])
    require(len(rows) == 512 and [row['validation_index'] for row in rows] == list(range(512)), 'Formal512 omitted/duplicate slots')
    require(len({row['question_id'] for row in rows}) == 512, 'Question IDs duplicate')
    for row, source in zip(rows, data):
        index = row['validation_index']
        require(row['version'] == version and row['run_id'] == run and row['primary'] is True
                and row['status'] == 'complete' and row['rank'] == index % 8 and row['slot'] == index // 8,
                'Formal primary/version/run/rank/slot differs')
        require(row['question_id'] == source['question_id'] and row['dataset'] == source['data_source']
                and row['row_semantic_sha256'] == object_sha(source), 'Original heldout row identity differs')
        require(type(row['protocol_em']) is int and row['protocol_em'] in (0, 1)
                and type(row['protocol_valid_final_action']) is bool
                and row['failure_class'] == (None if row['protocol_em'] else 'final_invalid'
                    if not row['protocol_valid_final_action'] else 'valid_gold_mismatch')
                and (not row['protocol_em'] or row['protocol_valid_final_action']), 'Strict score/failure axis differs')
        require(type(row['search_calls']) is int and 0 <= row['search_calls'] <= 4
                and isinstance(row['termination'], str) and row['termination'], 'Search/termination differs')
    return rows


def flips(left, right):
    require(len(left) == len(right) and len(left) in (256, 512), 'Paired denominator differs')
    require(all((a['validation_index'], a['question_id'], a['dataset'], a['row_semantic_sha256'])
                == (b['validation_index'], b['question_id'], b['dataset'], b['row_semantic_sha256'])
                for a, b in zip(left, right)), 'Paired question identity/order differs')
    cells = Counter((a['protocol_em'], b['protocol_em']) for a, b in zip(left, right))
    lost = [b for a, b in zip(left, right) if a['protocol_em'] and not b['protocol_em']]
    return dict(questions=len(left), left_hits=sum(row['protocol_em'] for row in left),
        right_hits=sum(row['protocol_em'] for row in right), new_hits=cells[0, 1], lost_hits=cells[1, 0],
        preserved_hits=cells[1, 1], still_wrong=cells[0, 0], net_hits=cells[0, 1] - cells[1, 0],
        net_percentage_points=100 * (cells[0, 1] - cells[1, 0]) / len(left),
        lost_final_invalid=sum(row['failure_class'] == 'final_invalid' for row in lost),
        lost_valid_gold_mismatch=sum(row['failure_class'] == 'valid_gold_mismatch' for row in lost),
        lost_no_executed_search=sum(row['search_calls'] == 0 for row in lost),
        bootstrap_pending_full_study_summary=True)


def saved(path, value):
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'config-sha256', 'phase', 'phase-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require(os.getuid() == 1005 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'Original HKUST UID and CPU-only environment required')
    require(args.phase_sha256 == PHASE_SHA, 'This pass binds only the actually closed J BF16 phase')
    output = Path(args.output)
    require(output.is_absolute() and '..' not in output.parts and not output.exists()
            and output.resolve().is_relative_to(BASE / 'analysis')
            and all(not path.is_symlink() for path in output.parents), 'New independent HKUST output required')
    reader = Reader(); config, config_ref = reader.json(args.config, args.config_sha256)
    require(config['schema'] == 'opsd-ablation-native512-eval-config-v1' and config['variant'] == 'J'
            and config['evaluation_role'] == 'group' and config['owner']['uid'] == 1005
            and config['scientific']['versions'] == ['raw10_bf16', 'raw10_fp16'], 'Original J two-version config required')
    stage = Path(config['paths']['output_root']) / 'raw10_bf16'
    require(Path(args.phase) == stage / 'phase-complete.json', 'Phase is outside bound original J BF16 output')
    phase, phase_ref = reader.json(args.phase, PHASE_SHA)
    require(phase['accepted'] is True and phase['run_id'] == config['run_id'] and phase['version'] == 'raw10_bf16'
            and phase['actual_worker_count'] == phase['expected_worker_count'] == 8
            and len(phase['child_exit_evidence']) == 8, 'Accepted original J BF16 eight-worker phase absent')
    release, release_ref = reader.json(stage / 'release.json')
    terminal, terminal_ref = reader.json(stage / 'actual-terminal.json')
    require({key: value for key, value in release.items() if key not in ('version', 'run_id')} == phase['release']
            and release['version'] == 'raw10_bf16' and release['run_id'] == config['run_id']
            and release['accepted'] is True
            and release['all_owned_release']['accepted'] is True
            and terminal['all_actual_wait_zero'] is True and terminal['child_exit_evidence'] == phase['child_exit_evidence']
            and terminal['run_id'] == config['run_id'] and terminal['version'] == 'raw10_bf16',
            'Actual stored phase wait0/full-release chain differs')
    ticket, ticket_ref = reader.json(stage / 'launch-ticket.json')
    require(ticket['accepted'] is True and ticket['run_id'] == config['run_id']
            and ticket['config_sha256'] == config_ref['sha256'] and ticket['version'] == 'raw10_bf16'
            and ticket['worker_count'] == 8 and ticket['logical_rank_map'] == list(range(8))
            and ticket['output'] == str(stage) and ticket['model'] is None, 'Original ordinary BF16 launch ticket differs')
    train, train_ref = reader.bound(config['training']['config'])
    require(train['variant'] == 'J' and train['run_id'] == config['training']['run_id']
            and train['study_id'] == config['study_id'] and train['scientific']['outer_iterations'] == 10,
            'J raw10 training identity differs')
    raw_checkpoint, checkpoint_ref = reader.bound(config['training']['raw10_checkpoint'])
    require(raw_checkpoint['accepted'] is True and raw_checkpoint['run_id'] == train['run_id']
            and raw_checkpoint['config_sha256'] == train_ref['sha256']
            and raw_checkpoint['cursor']['outer_index'] == 9
            and raw_checkpoint['cursor']['next_pair_cursor'] == raw_checkpoint['cursor']['usable_pair_count']
            and raw_checkpoint['complete_rank_count'] == 8 and raw_checkpoint['equal_optimizer_steps'] is True
            and len(raw_checkpoint['all_rank_adapter_sha256']) == 8
            and len(set(raw_checkpoint['all_rank_adapter_sha256'])) == 1, 'Not the actual complete raw tenth-outer adapter')
    digest = raw_checkpoint['all_rank_adapter_sha256'][0]
    require(raw_checkpoint['source_model_identity'] == train['source_model_manifest']['sha256']
            and raw_checkpoint['runtime_identity'] == train['runtime_manifest']['sha256'], 'Raw10 source/runtime differs')
    raw_data, data_ref = reader.read(train['validation_identity']['path'], DATA_SHA)
    require(train['validation_identity']['sha256'] == DATA_SHA, 'Wrong fixed512 SHA')
    data = [json.loads(line) for line in raw_data.splitlines()]
    require(len(data) == 512 and Counter(row['data_source'] for row in data) == Counter(nq=256, hotpotqa=256), 'Original512 cohort differs')
    inventory, inventory_ref = reader.json(config['helper_source_manifest']['path'], HELPER_MANIFEST_SHA)
    require(config['helper_source_manifest']['sha256'] == HELPER_MANIFEST_SHA, 'Wrong published original72e evaluator')
    helper = Path(config['paths']['helper_root'])
    summary_raw, summary_ref = reader.read(helper / 'summary.py', SUMMARY_SHA)
    require(inventory['files']['summary.py']['sha256'] == SUMMARY_SHA, 'Frozen pure counts/bootstrap source differs')
    namespace = dict(require=require, Counter=Counter, random=random, object_sha=object_sha)
    selected_functions(summary_raw, ('counts', 'aggregate'), namespace, str(helper / 'summary.py'))
    counts = namespace['counts']
    runtime, runtime_ref = reader.json(train['runtime_manifest']['path'], RUNTIME_SHA)
    require(train['runtime_manifest']['sha256'] == RUNTIME_SHA, 'Wrong305 runtime manifest')
    files = runtime.get('files', runtime); require(len(files) == 305, 'Scientific runtime has changed')
    science_root = Path(train['paths']['runtime'])
    def scientific_bytes(name):
        item = files[name]; sha = item if isinstance(item, str) else item['sha256']
        return reader.read(science_root / name, sha)
    init_raw, init_ref = scientific_bytes('upstream/Search-R1/search_r1/__init__.py')
    package = load_module('search_r1', init_raw, init_ref['path'])
    package.__path__ = [str(Path(init_ref['path']).parent)]
    qa_raw, qa_ref = scientific_bytes('upstream/Search-R1/search_r1/qa_protocol.py')
    require(qa_ref['sha256'] == QA_SHA, 'Wrong original strict final-action QA')
    qa = load_module('search_r1.qa_protocol', qa_raw, qa_ref['path'])
    core_raw, core_ref = scientific_bytes('search_e1/core.py')
    core = load_module('_phase_bound_original_core', core_raw, core_ref['path'])
    worker_raw, worker_source_ref = reader.read(helper / 'worker.py', inventory['files']['worker.py']['sha256'])
    worker_ns = dict(require=require)
    selected_functions(worker_raw, ('protocol',), worker_ns, str(helper / 'worker.py'))
    namespace.update(science=lambda _: (core, qa, dict(qa=qa_ref, core=core_ref)),
        checked_json=reader.bound, worker=types.SimpleNamespace(protocol=worker_ns['protocol']))
    records, worker_details = [], []
    for rank in range(8):
        worker, worker_ref = reader.json(stage / ('worker-%d' % rank) / 'complete.json')
        evidence = phase['child_exit_evidence'][rank]; identity = evidence['identity']
        require(type(evidence['actual_wait_returncode']) is int and evidence['actual_wait_returncode'] == 0
                and identity['uid'] == 1005 and identity['run_id'] == config['run_id']
                and type(identity['starttick']) is int and identity['starttick'] > 0
                and not identity.get('capture_status'), 'Original worker actual wait/UID/run/starttick differs')
        require(worker['accepted'] is True and worker['status'] == 'COMPLETE' and worker['primary_rows'] == 64
                and worker['rank'] == rank and worker['version'] == 'raw10_bf16' and worker['run_id'] == config['run_id']
                and worker['variant'] == 'J' and worker['study_id'] == config['study_id']
                and worker['config']['path'] == config_ref['path'] and worker['config']['sha256'] == config_ref['sha256']
                and all(worker['process'][key] == identity[key] for key in ('pid', 'uid', 'starttick')),
                'Original worker64/phase/config/process differs')
        require(worker['source']['manifest']['sha256'] == train['source_model_manifest']['sha256']
                and worker['science']['manifest']['sha256'] == RUNTIME_SHA
                and worker['physical_gpu'] == rank and worker['gpu_uuid'] == config['owner']['gpu_uuids'][rank]
                and worker['rank_seed'] == 42 and worker['privileged_reference'] is False and worker['no_training'] is True,
                'Original model/scorer/GPU/ordinary scientific scope differs')
        require(worker['model_kind'] == 'BF16_base_FP32_unmerged_LoRA'
                and worker['adapter']['manifest']['sha256'] == checkpoint_ref['sha256']
                and worker['actual_adapter']['tensor_sha256'] == worker['actual_loaded_adapter']['tensor_sha256'] == digest
                and worker['actual_adapter']['all_finite'] is True and worker['actual_loaded_adapter']['all_finite'] is True
                and worker['actual_adapter']['tensors'] == 504 and worker['actual_adapter']['dtype'] == 'torch.float32'
                and worker['actual_adapter']['changed_tensors_vs_common_initial'] > 0,
                'Actual finite FP32 raw10 adapter load differs')
        start, start_ref = reader.json(stage / ('child-start-%d.json' % rank))
        child, child_ref = reader.json(stage / ('child-%d.json' % identity['pid']))
        require(start['rank'] == rank and start['version'] == 'raw10_bf16' and start['identity'] == identity
                and child['identity'] == identity and child['actual_wait_returncode'] == 0
                and child['run_id'] == config['run_id'] and child['version'] == 'raw10_bf16', 'Actual start/exit proof differs')
        expected_command = [config['paths']['python'], '-B', str(helper / 'worker.py'),
            '--config', config_ref['path'], '--config-sha256', config_ref['sha256'], '--ticket', ticket_ref['path'],
            '--ticket-sha256', ticket_ref['sha256'], '--version', 'raw10_bf16', '--rank', str(rank), '--output', str(stage)]
        require(start['command'] == expected_command and identity['argv'] == expected_command, 'Actual original worker argv differs')
        rows, result_ref = reader.rows(worker['results']); require(len(rows) == 64, 'Worker formal64 rows absent')
        require(all(row['rank'] == rank for row in rows), 'Results moved between ranks')
        records.extend(rows)
        worker_details.append(dict(rank=rank, complete=worker_ref, results=result_ref,
            start=start_ref, actual_exit=child_ref, actual_identity=identity,
            actual_adapter=worker['actual_adapter'], actual_loaded_adapter=worker['actual_loaded_adapter'],
            actual_generation_config=worker['actual_generation_config']))
    records = validate_rows(records, data, 'raw10_bf16', config['run_id'])
    restricted = json.loads(json.dumps(config)); restricted['scientific']['versions'] = ['raw10_bf16']
    aggregate = namespace['aggregate'](records, data, restricted, verify_traces=True)
    require(aggregate['primary_slots'] == 512 and aggregate['accepted'] is True, 'J trace/QA aggregate not accepted')
    j_counts = aggregate['models']['raw10_bf16']
    greedy = {key: config['scientific'][key] for key in ('temperature', 'rank_seed', 'max_context', 'max_new_tokens',
        'turns', 'max_search_calls', 'topk', 'observation_tokens', 'repetition_penalty', 'eos_token_id', 'attention_backend')}
    expected_identity = dict(source_model_identity=train['source_model_manifest']['sha256'], runtime_identity=RUNTIME_SHA,
        common_initial_adapter_tensor_sha256=train['common_initial_adapter_tensor_sha256'], validation_sha256=DATA_SHA,
        greedy_contract=greedy)
    models = {'J': j_counts}; paired = {}; baseline_proofs = {}
    for label, directory, version, expected_hits in (
            ('bf16_zero', EVAL / 'zero' / 'runs', 'bf16_zero', 239),
            ('C1', EVAL / 'C1' / 'runs', 'raw10_bf16', 236)):
        original, original_ref = reader.json(directory / 'summary.json')
        closed, closed_ref = reader.json(directory / 'controller-complete.json')
        require(original['schema'] == 'opsd-ablation-native512-summary-v1' and original['accepted'] is True
                and closed['accepted'] is True and closed['run_id'] == original['run_id']
                and closed['summary']['sha256'] == original_ref['sha256']
                and original['identities'] == expected_identity and original['study_id'] == config['study_id'],
                'Already closed baseline does not bind exact same source/data/greedy protocol')
        all_rows, records_ref = reader.rows(original['private_primary_records'])
        selected = validate_rows([row for row in all_rows if row['version'] == version], data, version, original['run_id'])
        reproduced = dict(overall=counts(selected), datasets={dataset: counts([row for row in selected if row['dataset'] == dataset])
            for dataset in ('nq', 'hotpotqa')})
        require(json.loads(json.dumps(reproduced)) == original['models'][version]
                and reproduced['overall']['strict_hits'] == expected_hits, 'Closed baseline counts changed')
        models[label] = reproduced
        paired[label + '->J'] = {stratum: flips(
            selected if stratum == 'overall' else [row for row in selected if row['dataset'] == stratum],
            records if stratum == 'overall' else [row for row in records if row['dataset'] == stratum])
            for stratum in ('overall', 'nq', 'hotpotqa')}
        baseline_proofs[label] = dict(summary=original_ref, actual_controller_complete=closed_ref, records=records_ref)
    require('torch' not in sys.modules, 'This source-only scoring pass imported Torch')
    output.mkdir(parents=True, exist_ok=False)
    details = saved(output / 'J-bf16-phase-details.private.json', dict(accepted=True, config=config_ref, phase=phase_ref,
        actual_terminal=terminal_ref, release=release_ref, ticket=ticket_ref, training_config=train_ref,
        raw10_checkpoint=checkpoint_ref, data=data_ref, runtime=runtime_ref, helper_manifest=inventory_ref,
        frozen_summary_source=summary_ref, worker_protocol_source=worker_source_ref, qa_source=qa_ref, core_source=core_ref,
        workers=worker_details, baselines=baseline_proofs, selected_read_proofs=list(reader.proofs.values())))
    public = saved(output / 'J-bf16-phase-publication.json', dict(schema='opsd-ablation-J-bf16-phase-only-readonly-v1',
        accepted=True, observed_unix=time.time(), models=models, paired=paired, primary_slots_J=512,
        actual_worker_count=8, actual_worker_wait_returncodes=[0] * 8, native_strict_cpu_rescoring_questions=512,
        native_strict_cpu_rescoring_passed=True, all_primary_slots_unique_and_heldout_SHA_matched=True,
        private_details_sha256=details['sha256'], phase_sha256=phase_ref['sha256'], config_sha256=config_ref['sha256'],
        frozen_native_counts_source_sha256=summary_ref['sha256'], frozen_QA_source_sha256=qa_ref['sha256'],
        whole_J_controller_completion_claim=False, full_eleven_model_completion_claim=False,
        FP16_phase_excluded=True, Mac_raw_payloads=False, processing_host='HKUST1',
        scope='Only accepted raw10_bf16 J phase, ordinary same512. FP16/whole J controller may still be running.',
        limitations=['Direct paired counts only; the final full-study entry supplies 10000/seed42 bootstrap after all versions complete.',
            'Frozen native QA re-scores stored J traces; no generation, HTTP, GPU or model/tensor reload.',
            'Small identity manifests and original actual worker loads are bound; source model weights are not rehashed here.',
            'No train/validation gains are inferred from training loss or differing pair/Adam exposure budgets.']))
    complete_ref = saved(output / 'complete.json', dict(accepted=True, schema='opsd-ablation-J-bf16-phase-only-complete-v1',
        publication=public, private_details=details, phase_sha256=phase_ref['sha256'], formal_J_predictions=512,
        whole_J_controller_completion_claim=False, Mac_raw_payloads=False))
    print(json.dumps(dict(accepted=True, complete=complete_ref, J=j_counts,
                          formal_J_predictions=512, whole_J_controller_completion_claim=False)))


if __name__ == '__main__':
    main()
