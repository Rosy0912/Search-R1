"""Offline CPU replay of c5bf real-question evidence; never a live eval launcher.

Externally supplied pins are the trust boundary. Content replay does not issue
eligibility, signature, lifecycle, resource-release or improvement acceptance.
"""
import argparse
import ast
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import importlib.abc
import importlib.util
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys
import traceback
import zipfile
import zlib

import numpy as np

TRAIN = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
EVAL = 'c5bf4c8db33562a259fa227a940dc69543901b98'
ROOT = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006'
REPO = ROOT + '/repo-final200-eval-c5bf4c8db335'
BASE = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/assets/qwen2.5-3b-instruct-aa8e725'
HF200 = ROOT + '/runs/formal-3304/checkpoints/actor/global_step_200'
SOURCE_INVENTORY = 'fd8fa25d8321a4cd11b9faa5d8fdbd5f1692172a9e51ea5eba447c3565b8cc0d'
COMMON = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
OFFICIAL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
OFFICIAL_MANIFEST = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
ROLES = ('official_base', 'actual_hf200')
COUNTS = dict(nq=3610, triviaqa=11313, popqa=14267, hotpotqa=7405,
              **{'2wikimultihopqa': 12576}, musique=2417, bamboogle=125)
SCORING_KEYS = {'responses', 'prompts', 'attention_mask', 'info_mask', 'position_ids'}
RAW_KEYS = {'prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids'}
MAX_BATCH_RAW = 512 * 1024**2
MAX_MEMBER = 256 * 1024**2
MAX_JSON = 1024 * 1024**2
FINAL_MODULE_SHA = '2d927c8bb87b55ea140d3d08931528c5b5fcf604902ccf67902c9830f1b4fd52'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, 'Invalid integer: ' + label)
    return value


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def loads(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def same(a, b, label):
    require(encoded(a) == encoded(b), label)


def valid_sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Full SHA required')
    return value


def reference(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Physical original reference required')
    valid_sha(value['sha256'])
    require(type(value['path']) is str and PurePosixPath(value['path']).is_absolute() and
            '..' not in PurePosixPath(value['path']).parts, 'Absolute safe original path required')
    return value


def identity(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class Originals:
    def __init__(self, uid):
        integer(uid, 'original owner')
        self.uid, self.records = uid, {}

    @contextmanager
    def opened(self, path, sha, limit=MAX_MEMBER, *, allow_empty=False):
        valid_sha(sha); path = Path(path)
        require(path.is_absolute() and path.resolve(strict=True) == path, 'Physical non-symlink path required')
        before = path.lstat()
        require(type(allow_empty) is bool and stat.S_ISREG(before.st_mode) and before.st_uid == self.uid and
                before.st_nlink == 1 and (0 <= before.st_size if allow_empty else 0 < before.st_size) and
                before.st_size <= limit, 'Owned bounded single-link regular original required')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            require(identity(os.fstat(stream.fileno())) == identity(before), 'Original replaced before open')
            def digest():
                stream.seek(0); h = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    h.update(chunk)
                return h.hexdigest()
            require(digest() == sha, 'Original full SHA differs')
            stream.seek(0)
            try:
                yield stream
            finally:
                require(digest() == sha and identity(os.fstat(stream.fileno())) == identity(before) and
                        identity(path.lstat()) == identity(before), 'Original changed during consumption')
            record = dict(path=str(path), sha256=sha, bytes=before.st_size, uid=before.st_uid,
                device=before.st_dev, inode=before.st_ino, nlink=before.st_nlink,
                mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns)
            if str(path) in self.records:
                same(record, self.records[str(path)], 'Previously read original changed')
            else:
                require(not any(r['device'] == record['device'] and r['inode'] == record['inode']
                    for r in self.records.values()), 'Distinct original paths alias one physical inode')
            self.records[str(path)] = record

    def json(self, ref):
        reference(ref)
        with self.opened(ref['path'], ref['sha256'], MAX_JSON) as stream:
            return loads(stream.read())

    def recheck(self):
        for record in list(self.records.values()):
            with self.opened(record['path'], record['sha256'], max(MAX_MEMBER, record['bytes']),
                             allow_empty=record['bytes'] == 0):
                pass


def member(directory, name):
    require(type(name) is str and PurePosixPath(name).name == name and name not in ('.', '..'), 'Unsafe member name')
    return str(Path(directory) / name)


def npz_arrays(reader, path, sha):
    with reader.opened(path, sha) as stream:
        with zipfile.ZipFile(stream) as z:
            names = z.namelist()
            require(len(names) == len(set(names)) and set(names) == {k + '.npy' for k in SCORING_KEYS}, 'Unexpected/duplicate NPZ members')
            require(sum(e.file_size for e in z.infolist()) <= MAX_BATCH_RAW, 'NPZ exceeds expanded bound')
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}


def full_turn_rows(reader, path, entry):
    total = integer(entry['raw_bytes'], 'raw bytes', 1)
    require(total <= MAX_BATCH_RAW, 'Expanded raw bound exceeded')
    valid_sha(entry['raw_sha256']); rows, pending = [], b''
    decompressor, h, size = zlib.decompressobj(16 + zlib.MAX_WBITS), hashlib.sha256(), 0
    with reader.opened(path, entry['sha256']) as stream:
        require(os.fstat(stream.fileno()).st_size == integer(entry['bytes'], 'gzip bytes', 1), 'Compressed size differs')
        for chunk in iter(lambda: stream.read(64 * 1024), b''):
            remaining = chunk
            while remaining:
                out = decompressor.decompress(remaining, 1024**2)
                remaining = decompressor.unconsumed_tail
                size += len(out); require(size <= total, 'Raw size exceeds exact declared bound')
                h.update(out); pending += out
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    require(line and len(rows) < entry['count'], 'Empty/extra raw question line')
                    rows.append(loads(line))
                require(len(pending) <= MAX_BATCH_RAW, 'Raw line exceeds bound')
                require(not decompressor.unused_data, 'Extra gzip member or trailing data')
                if not remaining:
                    break
        require(decompressor.eof and not decompressor.unused_data and not pending and
                size == total and h.hexdigest() == entry['raw_sha256'] and len(rows) == entry['count'],
                'Incomplete gzip/raw SHA/line cardinality')
    return rows


def vector(value, label):
    require(type(value) is list and all(type(x) is int and x >= 0 for x in value), 'Invalid vector: ' + label)
    return np.asarray(value, dtype=np.int64)


def invalid_literal(generation_raw):
    tree = ast.parse(generation_raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMGenerationManager')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_predictions')
    candidates = [n for n in ast.walk(fn) if isinstance(n, ast.JoinedStr) and n.values and
                  all(isinstance(v, ast.Constant) and type(v.value) is str for v in n.values) and
                  ''.join(v.value for v in n.values).startswith('\nMy previous action is invalid.')]
    require(len(candidates) == 1, 'Fixed invalid-action source literal missing/ambiguous')
    return ''.join(v.value for v in candidates[0].values)


def extra_raw_checks(records, rows, tools, arrays, tokenizer, role, invalid_text):
    n = len(records)
    require(n == len(rows) and 0 < n <= 64 and set(arrays) == SCORING_KEYS, 'Real batch/tensor keys differ')
    for key, value in arrays.items():
        require(value.ndim == 2 and len(value) == n and value.dtype.kind in 'iu' and
                not (value < 0).any(), 'Raw scoring shape/type differs: ' + key)
    padding_calls = []
    live = list(range(n))
    for step, shared in enumerate(tools['turns']):
        require(type(shared['step']) is int and shared['step'] == step, 'Shared step differs')
        active = shared['active_indices']
        require(type(active) is list and all(type(i) is int and 0 <= i < n for i in active) and
                active == sorted(set(active)) and active == live and active, 'Active rows/padding identity differs')
        padding_calls.append(dict(real_rows=len(active), rows=8 * ((len(active) + 7) // 8)))
        next_live = []
        for i in active:
            turn = rows[i]['turns'][step]
            flag = integer(turn['done'], 'done'); require(flag in (0, 1), 'Binary done required')
            if not flag:
                next_live.append(i)
        live = next_live
    pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
    for i, row in enumerate(rows):
        require(row['model_role'] == role, 'Mixed model role in raw batch')
        same(row['input_record'], records[i], 'Full raw identity differs')
        meta = row['generation_meta']
        same(meta, dict(eos_token_id=eos, pad_token_id=pad, recompute_log_prob=False, do_sample=False, validate=True), 'Initial generation metadata/gold differs')
        final = row['final']
        require(set(final) == RAW_KEYS | {'info_mask', 'responses_with_info_mask'}, 'Final raw tensor keys differ')
        for key in SCORING_KEYS:
            require(np.array_equal(vector(final[key], key), arrays[key][i]), 'Raw final/NPZ join differs: ' + key)
        prompts, responses = vector(final['prompts'], 'prompts'), vector(final['responses'], 'responses')
        ids = np.concatenate([prompts, responses]); attention = vector(final['attention_mask'], 'attention')
        info = vector(final['info_mask'], 'info'); position = vector(final['position_ids'], 'position')
        require(np.array_equal(vector(final['input_ids'], 'input_ids'), ids) and
                np.array_equal(attention, ids != pad) and np.array_equal(position, (attention.cumsum() - 1) * attention) and
                np.isin(info, [0, 1]).all() and np.all(info <= attention), 'Final tensor/mask/position rule differs')
        counters = dict(valid_action_stats=0, valid_search_stats=0)
        for turn in row['turns']:
            context, raw = turn['generation_context'], turn['raw_generation']
            same(turn['raw_generation_meta'], {}, 'Frozen no-logprob generation output metadata differs')
            require(set(context) == {'input_ids', 'attention_mask', 'position_ids'} and set(raw) == RAW_KEYS,
                    'Raw generation/context keys differ')
            ctx = {k: vector(v, k) for k, v in context.items()}
            rv = {k: vector(v, k) for k, v in raw.items()}
            response = rv['responses']; w = len(response)
            require(w == 500 and np.array_equal(rv['prompts'], ctx['input_ids']) and
                    np.array_equal(rv['input_ids'], np.concatenate([ctx['input_ids'], response])), 'Raw generation input/prompts differ')
            eos_flags = (response == eos).astype(np.int64)
            eos_mask = (eos_flags.cumsum() - eos_flags) == 0
            require(np.array_equal(rv['attention_mask'], np.concatenate([ctx['attention_mask'], eos_mask])) and
                    np.array_equal(rv['position_ids'], np.concatenate([ctx['position_ids'], ctx['position_ids'][-1] + np.arange(1, w + 1)])),
                    'Raw vLLM attention/position rule differs (including PAD)')
            for key in ('done', 'valid_action', 'search'):
                require(type(turn[key]) is int and turn[key] in (0, 1), 'Typed binary environment flag required')
            counters['valid_action_stats'] += turn['valid_action']; counters['valid_search_stats'] += turn['search']
            if turn['valid_action'] == 0:
                require(turn['observation_text'] == invalid_text, 'Invalid action fixed environment literal differs')
        fm = row['final_meta']
        require(set(fm) == {'turns_stats', 'active_mask', 'valid_action_stats', 'valid_search_stats'},
                'Frozen final output metadata keys differ')
        require(type(fm['turns_stats']) is int and fm['turns_stats'] == len(row['turns']) and
                type(fm['active_mask']) is bool and fm['active_mask'] == (not bool(row['turns'][-1]['done'])), 'Final active/turn stats differ')
        for key, expected in counters.items():
            require(type(fm[key]) is int and fm[key] == expected, 'Final environment stats differ')
    return padding_calls


def join_calls(expected, calls):
    require(len(expected) == len(calls), 'Missing/extra actual GPU generation call')
    for index, (wanted, actual) in enumerate(zip(expected, calls), 1):
        same(actual, dict(sequence=index, rows=wanted['rows'], do_sample=False, validate=True), 'Actual padded call mapping differs')
    return dict(actual_calls=len(calls), real_active_generation_rows=sum(x['real_rows'] for x in expected),
        actual_gpu_dispatch_rows=sum(x['rows'] for x in expected),
        discarded_padding_output_rows=sum(x['rows'] - x['real_rows'] for x in expected),
        discarded_padding_outputs_archived=False, discarded_padding_outputs_replayed=False)


class PinnedRepoImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Normal imports compiled from unchanged, physically verified source bytes."""
    def __init__(self, repo, pins, reader):
        self.repo, self.pins, self.reader, self.loaded = Path(repo), pins, reader, {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in ('search_r1', 'verl'):
            return None
        relative = fullname.replace('.', '/')
        for filename, package in ((relative + '/__init__.py', True), (relative + '.py', False)):
            if filename in self.pins:
                spec = importlib.util.spec_from_file_location(fullname, self.repo / filename, loader=self,
                    submodule_search_locations=[str(self.repo / relative)] if package else None)
                return spec
        if (self.repo / relative).is_dir():
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = [str(self.repo / relative)]
            return spec
        raise ImportError('Unpinned evaluation repo import: ' + fullname)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = Path(module.__spec__.origin)
        relative = path.relative_to(self.repo).as_posix()
        with self.reader.opened(str(path), self.pins[relative]['sha256'],
                                allow_empty=self.pins[relative]['bytes'] == 0) as stream:
            raw = stream.read()
        require(len(raw) == self.pins[relative]['bytes'], 'Imported source length differs')
        self.loaded[module.__name__] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)


def load_helpers(reader, inventory_ref):
    require(inventory_ref['sha256'] == SOURCE_INVENTORY, 'Actual c5bf source inventory required, not old211')
    inventory = reader.json(inventory_ref)
    require(inventory['source_commit'] == EVAL and inventory['remote_repository'] == REPO and
            type(inventory['member_count']) is int and inventory['member_count'] == 857 and len(inventory['files']) == 857,
            'Wrong source inventory/repository/cardinality')
    repo = Path(REPO)
    require(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == EVAL and
            subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True) == '', 'Clean c5bf source required')
    require(not any(n.split('.')[0] in ('search_r1', 'verl') for n in sys.modules), 'Evaluation package was pre-imported before source pinning')
    for name, record in inventory['files'].items():
        require(type(name) is str and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts, 'Unsafe source inventory path')
        valid_sha(record['sha256']); integer(record['bytes'], 'source bytes')
    importer = PinnedRepoImporter(repo, inventory['files'], reader)
    sys.meta_path.insert(0, importer)
    final = importlib.import_module('search_r1.baselines.final200_evaluation')
    require(importer.loaded[final.__name__]['sha256'] == FINAL_MODULE_SHA, 'Final200 source differs')
    helpers = dict(final=final, legacy=final.legacy,
        runtime=importlib.import_module('search_r1.baselines.corrected_runtime_capture'),
        qa=importlib.import_module('search_r1.qa_protocol'),
        em=importlib.import_module('verl.utils.reward_score.qa_em'),
        invalid_text=None, importer=importer)
    record = inventory['files']['search_r1/llm_agent/generation.py']
    with reader.opened(str(repo / 'search_r1/llm_agent/generation.py'), record['sha256']) as stream:
        helpers['invalid_text'] = invalid_literal(stream.read())
    return helpers


def validate_workers(manifest, allocation, helpers):
    prov, guard = manifest['provenance'], manifest['zero_update_guard']
    require(type(guard['attempted_update_dispatches']) is int and guard['attempted_update_dispatches'] == 0 and
            'actor_rollout_wg.update_actor' in guard['guarded_dispatches'] and guard['original_dispatches'] and
            guard['guard_source_sha256'] == prov['source_files']['search_r1/baselines/corrected_evaluation.py'], 'Actual dispatch guard differs')
    audit = guard['corrected_runtime']
    require(type(audit['rejected_calls']) is int and audit['rejected_calls'] == 0, 'Typed zero rejected generation calls required')
    helpers['runtime'].validate_completed_runtime(audit, prov)
    rows = audit['worker_audit']['rows']; by_rank, pids, uuids = {}, set(), set()
    require(type(allocation['job_id']) is str and re.fullmatch('[1-9][0-9]*', allocation['job_id']) and allocation['job_id'] != '3304', 'Own eval job required')
    require(type(allocation['step_id']) is str and re.fullmatch('(?:[0-9]+|batch|extern)', allocation['step_id']), 'Actual evaluation step required')
    require(allocation['node'] == 'h100-10' and type(allocation['uid']) is int and allocation['uid'] == 0, 'Same-node owned allocation required')
    require(type(allocation['gpu_uuids']) is list and len(set(allocation['gpu_uuids'])) == len(allocation['gpu_uuids']) == 8, 'Eight allocation UUIDs required')
    spec = audit['worker_audit']['spec']
    require(spec['job_id'] == allocation['job_id'] == prov['eval_job_id'] and spec['eval_repo'] == REPO and
            type(spec['uid']) is int and spec['uid'] == 0, 'Actual recorded eval scope differs')
    for row in rows:
        rank = integer(row['rank'], 'rank'); integer(row['pid'], 'worker PID', 1)
        require(rank < 8 and rank not in by_rank and row['pid'] not in pids and type(row['world_size']) is int and
                row['world_size'] == 8 and type(row['uid']) is int and row['uid'] == 0, 'Eight distinct typed worker identities required')
        process = row['process']
        require(type(process['uid']) is int and process['uid'] == 0 and process['uids'] == [0, 0, 0, 0] and
                all(type(u) is int for u in process['uids']), 'Historical worker UID differs')
        step_pattern = r'(?:^|/)job[_-]' + re.escape(allocation['job_id']) + r'(?:\.scope)?/step[_-]' + re.escape(allocation['step_id']) + r'(?:\.scope)?(?:/|$)'
        require(re.search(step_pattern, process['cgroup'], re.M), 'Historical job/step not in same cgroup path')
        require(type(row['gpu_uuid']) is str and row['gpu_uuid'] in allocation['gpu_uuids'] and row['gpu_uuid'] not in uuids, 'Worker GPU UUID/topology differs')
        by_rank[rank] = process; pids.add(row['pid']); uuids.add(row['gpu_uuid'])
    require(set(by_rank) == set(range(8)), 'Worker rank cardinality differs')
    zero = manifest['actual_eval_worker_zero_optimizer']
    for phase_key, phase in (('start', 'install'), ('finish', 'finish')):
        require(len(zero[phase_key]) == 8, 'Actual worker optimizer guard missing')
        seen = set()
        for row in zero[phase_key]:
            rank = integer(row['rank'], 'zero rank'); require(rank in by_rank and rank not in seen, 'Zero guard rank differs'); seen.add(rank)
            require(type(row['world_size']) is int and row['world_size'] == 8 and row['phase'] == phase and
                    row['scope'] == 'INDEPENDENT_EVAL_WORKER_ZERO_STEPS_NOT_TRAINING_ADAM_RESTORE' and
                    row['source_sha256'] == FINAL_MODULE_SHA, 'Zero guard phase/source differs')
            for key in ('attempted_optimizer_steps', 'attempted_scheduler_steps'):
                require(type(row[key]) is int and row[key] == 0, 'Actual optimizer/scheduler attempted step')
            for key in ('pid', 'process_start_ticks', 'uid', 'uids', 'cgroup'):
                same(row['process'][key], by_rank[rank][key], 'Zero guard historical process join differs')
            same(row['before'], row['after'], 'Eval optimizer/scheduler clocks changed')
            clock = row['before']
            integer(clock['state_entries'], 'optimizer state entries')
            require(type(clock['steps']) is list and len(clock['steps']) <= clock['state_entries'] and
                    all(type(s) is int and s >= 0 for s in clock['steps']) and type(clock['scheduler']) is dict,
                    'Typed optimizer/scheduler clock required')
            before = zero['start'][next(i for i, r in enumerate(zero['start']) if r['rank'] == rank)]['before']
            same(row['before'], before, 'Start/finish eval clocks differ')
    return audit['generation_calls']


def check_header(manifest, stage, role, input_ref):
    require(type(manifest['version']) is int and manifest['version'] == 1 and manifest['scope'] == 'FULL_FINAL200_TOOL_TURNS_AND_NATIVE_SCORING_VIEW' and
            type(manifest['count']) is int and manifest['count'] == 51713 and manifest['source_counts'] == COUNTS and
            manifest['lifecycle_scope'] == 'EVALUATION_OUTPUT_COMPLETE_NOT_SCHEDULER_TERMINAL_OR_RESOURCE_RELEASE', 'Completed full real-question output required')
    same(manifest['input_manifest_sha256'], input_ref['sha256'], 'Role cohort input differs')
    prov = manifest['provenance']
    for key, value in stage.items():
        same(prov.get(key), value, 'Runtime provenance differs from actual stage: ' + key)
    require(stage['kind'] == 'REVIEWED_FINAL200_PAIRED_EVAL_STAGE' and stage['model_role'] == role and
            stage['training_commit'] == TRAIN and stage['training_job_id'] == '3304' and
            type(stage['actual_outer_updates']) is int and stage['actual_outer_updates'] == 200 and
            stage['eval_commit'] == EVAL and stage['common_manifest_sha256'] == COMMON and
            stage['official_base_revision'] == OFFICIAL_REVISION and stage['official_base_manifest_sha256'] == OFFICIAL_MANIFEST and
            stage['model_directory'] == {'official_base': BASE, 'actual_hf200': HF200}[role], 'Wrong source/model/stored200 role')
    require(stage['generation_budget_sha256'] == hashlib.sha256(encoded(stage['generation_budget'])).hexdigest(), 'Budget encoding differs')
    budget = stage['generation_budget']
    required = {'data.val_batch_size':64, 'data.max_start_length':2048, 'data.max_prompt_length':4096,
        'data.max_response_length':500, 'data.max_obs_length':500, 'max_turns':4, 'do_search':True,
        'force_final_answer':False, 'retriever.topk':3, 'baselines.seed':42, 'trainer.n_gpus_per_node':8,
        'trainer.nnodes':1, 'actor_rollout_ref.rollout.n':1, 'actor_rollout_ref.rollout.temperature':1,
        'actor_rollout_ref.rollout.top_p':1, 'actor_rollout_ref.rollout.top_k':-1,
        'actor_rollout_ref.rollout.tensor_model_parallel_size':1,
        'native_validation_sampling':dict(best_of=1,do_sample=False,min_p=0.0,n=1,temperature=0.0,top_k=-1,top_p=1.0)}
    for key, expected in required.items():
        same(budget.get(key), expected, 'Fixed final200 budget differs: ' + key)
    require(type(budget.get('retriever.url')) is str and re.fullmatch(r'http://127\.0\.0\.1:[0-9]+/retrieve',budget['retriever.url']), 'Loopback-only retrieval budget required')
    require(stage['input_manifest_sha256'] == input_ref['sha256'], 'Stage ordered cohort differs')
    return prov


class RoleReplay:
    def __init__(self, reader, helpers, role, entry, records, input_ref):
        self.reader, self.helpers, self.role, self.entry, self.records = reader, helpers, role, entry, records
        self.directory = Path(entry['directory'])
        self.manifest = reader.json(dict(path=str(self.directory / 'manifest.json'), sha256=entry['manifest_sha256']))
        self.stage = reader.json(entry['stage'])
        self.prov = check_header(self.manifest, self.stage, role, input_ref)
        require(self.prov['reviewed_stage_path'] == entry['stage']['path'] and self.prov['reviewed_stage_sha256'] == entry['stage']['sha256'], 'Actual published stage pin differs')
        require(self.prov['input_manifest_path'] == input_ref['path'], 'Actual input source path differs')
        self.calls = validate_workers(self.manifest, entry['allocation'], helpers)
        self.expected_calls, self.scores, self.rewards = [], [], []
        self.padding, self.members = None, set()

    def walk(self, tokenizer):
        m, cursor = self.manifest, 0
        require(len(m['files']) == len(m['full_turn_files']) == 809, 'Expected809 real batches required')
        h = self.helpers
        for batch, (scoring, full) in enumerate(zip(m['files'], m['full_turn_files'])):
            n = min(64, 51713 - cursor)
            for entry in (scoring, full):
                require(type(entry['start']) is int and entry['start'] == cursor and type(entry['count']) is int and entry['count'] == n and n > 0, 'Batch order/last1 cardinality differs')
            stem = 'batch-' + str(batch).zfill(6)
            names = {'tensors': stem + '.npz', 'rows': stem + '.json', 'path': stem + '.full-turns.jsonl.gz', 'tool_batches': stem + '.tool-batches.json'}
            for key, name in names.items():
                entry = scoring if key in ('tensors', 'rows') else full
                require(entry[key] == name and name not in self.members, 'Duplicate/misnamed paired batch member'); self.members.add(name)
            arrays = npz_arrays(self.reader, member(self.directory, scoring['tensors']), scoring['tensors_sha256'])
            saved = self.reader.json(dict(path=member(self.directory, scoring['rows']), sha256=scoring['rows_sha256']))
            rows = full_turn_rows(self.reader, member(self.directory, full['path']), full)
            tools = self.reader.json(dict(path=member(self.directory, full['tool_batches']), sha256=full['tool_batches_sha256']))
            wanted = self.records[cursor:cursor + n]
            require(len(saved) == n, 'Scoring row count differs')
            same([r['input_record'] for r in saved], wanted, 'Scoring identities differ')
            self.expected_calls.extend(extra_raw_checks(wanted, rows, tools, arrays, tokenizer, self.role, h['invalid_text']))
            h['final'].validate_raw_batch(wanted, rows, tools, tokenizer)
            h['legacy'].validate_arrays(arrays)
            scores = h['legacy'].score_arrays(arrays, wanted, tokenizer)
            same(scores, [r['metrics'] for r in saved], 'Saved scoring metrics differ from CPU replay')
            w = arrays['responses'].shape[1]
            for i, (score, source, row) in enumerate(zip(scores, saved, rows)):
                mask = arrays['attention_mask'][i, -w:].astype(bool) & arrays['info_mask'][i, -w:].astype(bool)
                spans = h['qa'].decode_model_spans(tokenizer, arrays['responses'][i].tolist(), mask.tolist())
                same(spans, source['model_spans'], 'Saved model spans differ from raw masks')
                reward = h['em'].compute_score_generated_em(spans, dict(target=wanted[i]['identity']['targets']))
                require(type(source['reward_score']) in (int, float) and source['reward_score'] == reward, 'Saved native reward differs')
                self.scores.append(score); self.rewards.append(reward)
                yield dict(question_uid=wanted[i]['question_uid'], question_id=wanted[i]['question_id'],
                    ordinal=cursor + i, source=wanted[i]['identity']['source'], role=self.role,
                    answer=h['qa'].extract_final_answer(spans), metrics=score, reward=reward,
                    env_done=not row['final_meta']['active_mask'], strict_terminal_answer=bool(score['answer_format']),
                    strict_protocol_format=bool(score['format_rate']), batch=batch, row=i)
            cursor += n
        require(cursor == 51713 and set(p.name for p in self.directory.iterdir()) == self.members | {'manifest.json'}, 'Full output membership differs')
        self.padding = join_calls(self.expected_calls, self.calls)
        actual = h['legacy'].aggregate_with_rewards(self.scores, self.rewards)
        same(actual, self.manifest['online_metrics'], 'Full CPU replay differs from original online aggregate')


def verify_model(reader, provenance):
    directory = Path(provenance['model_directory']); pins = provenance['model_files']
    if provenance['model_role'] == 'official_base':
        require(pins.get('VERIFIED-MODEL-MANIFEST.json') == OFFICIAL_MANIFEST, 'Official pinned model manifest missing')
    require(set(p.name for p in directory.iterdir()) == set(pins) and {'config.json', 'tokenizer.json', 'tokenizer_config.json'} <= set(pins), 'Actual complete model membership differs')
    require(any(n.endswith('.safetensors') for n in pins), 'Actual model weights missing')
    for name, sha in pins.items():
        with reader.opened(member(directory, name), sha, 32 * 1024**3):
            pass


def write_once(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY); os.fsync(fd); os.close(fd)


def run(request, reader, helpers, output):
    require(request['kind'] == 'EXTERNALLY_PINNED_PAIRED_FINAL200_CPU_REPLAY_INPUTS' and request['source_commit'] == EVAL and
            type(request['roles']) is dict and set(request['roles']) == set(ROLES), 'Actual externally qualified paired inputs required')
    # Read and bind upstream qualification; this consumer does not replace it.
    reader.json(request['root_qualification']); reader.json(request['independent_qualification'])
    reference(request['input_manifest']); reference(request['common_manifest'])
    require(request['common_manifest']['sha256'] == COMMON, 'Common manifest differs'); reader.json(request['common_manifest'])
    inputs = helpers['final'].load_inputs(request['input_manifest']['path'], request['input_manifest']['sha256'])
    # Add every original data/input file to the physical ledger, then replay them.
    reader.json(request['input_manifest'])
    for item in inputs['inputs']:
        with reader.opened(item['path'], item['sha256'], 32 * 1024**3):
            pass
    roles = [RoleReplay(reader, helpers, role, request['roles'][role], inputs['records'], request['input_manifest']) for role in ROLES]
    same(roles[0].stage['generation_budget'], roles[1].stage['generation_budget'], 'Paired native tool/sampling budgets differ')
    for role in roles:
        for name, sha in role.prov['source_files'].items():
            require(name in helpers['importer'].pins and helpers['importer'].pins[name]['sha256'] == sha, 'Role source inventory/import pin differs')
            with reader.opened(str(Path(REPO) / name), sha):
                pass
        verify_model(reader, role.prov)
    from transformers import AutoTokenizer
    tokenizers = [AutoTokenizer.from_pretrained(r.prov['model_directory'], local_files_only=True, trust_remote_code=False) for r in roles]
    def semantics(tokenizer):
        return dict(vocab=tokenizer.get_vocab(), chat_template=tokenizer.chat_template, eos=tokenizer.eos_token_id,
                    pad=tokenizer.pad_token_id, specials={k:str(v) for k,v in tokenizer.special_tokens_map.items()})
    same(semantics(tokenizers[0]), semantics(tokenizers[1]), 'Official paired tokenizer semantics differ')
    out = Path(output)
    require(out.is_absolute() and out.parent.resolve(strict=True) == out.parent and
            out.parent.is_dir() and out.parent.stat().st_uid == reader.uid,
            'Fresh output requires an owned physical parent')
    require(not any(out.is_relative_to(path) for path in [Path(REPO), Path(BASE), Path(HF200)] +
            [role.directory for role in roles]), 'Derived output must not modify an original evidence/model/source directory')
    out.mkdir(parents=False, exist_ok=False)
    fd = os.open(out.parent, os.O_RDONLY); os.fsync(fd); os.close(fd)
    counts, effects, done = {}, {}, {role:0 for role in ROLES}
    try:
        with (out / 'paired-derived-question-scores.jsonl').open('xb') as stream:
            generators = [role.walk(tok) for role, tok in zip(roles, tokenizers)]
            for index, pair in enumerate(itertools.zip_longest(*generators)):
                a, b = pair; require(a is not None and b is not None and index < 51713, 'Paired row drop/overrun')
                for key in ('question_uid', 'question_id', 'ordinal', 'source'):
                    same(a[key], b[key], 'Paired real-question identity differs')
                require(a['ordinal'] == index, 'Paired ordinal differs')
                source = a['source']; counts[source] = counts.get(source, 0) + 1
                for row in pair:
                    done[row['role']] += int(row['env_done'])
                delta = {key:b['metrics'][key]-a['metrics'][key] for key in ('em','f1','answer_format','format_rate','generated_tokens','search_calls')}
                for key, value in delta.items():
                    effects.setdefault(source, {}).setdefault(key, 0)
                    effects[source][key] += value
                stream.write(encoded(dict(identity={k:a[k] for k in ('question_uid','question_id','ordinal','source')}, official_base=a, actual_hf200=b, paired_delta=delta)) + b'\n')
            stream.flush(); os.fsync(stream.fileno())
        require(counts == COUNTS and len(roles[0].scores) == len(roles[1].scores) == 51713, 'Full paired seven-source coverage differs')
        import torch
        require(not torch.cuda.is_initialized(), 'CPU replay initialized CUDA')
        reader.recheck()
        result = dict(version=1, scope='CPU_RAW_CONTENT_REPLAY_NOT_PREREQUISITES_OR_GPU_OR_LIFECYCLE_OR_IMPROVEMENT_ACCEPTANCE',
            source_commit=EVAL, real_questions=51713, source_counts=counts,
            role_metrics={r.role:helpers['legacy'].aggregate_with_rewards(r.scores,r.rewards) for r in roles},
            environment_done_counts=done, per_source_paired_metric_sums=effects,
            padding={r.role:r.padding for r in roles}, verified_originals=reader.records,
            source_imports=helpers['importer'].loaded,
            generation_budget=roles[0].stage['generation_budget'],
            guard_scope='RECORDED_INDEPENDENT_EVAL_ZERO_STEPS_NOT_TRAINING_ADAM_RESTORE',
            original_checkpoint_before_after_qualification=request.get('original_checkpoint_before_after_qualification'),
            original_rng_scope='ORIGINAL_CHECKPOINT_PRESERVATION_EXTERNAL_EVAL_OWN_RNG_CONSUMPTION_ALLOWED',
            lifecycle_qualification='NOT_PERFORMED_OR_UPGRADED', gpu_executed=False)
        write_once(out / 'content-replay-report.json', encoded(result))
        return result
    except BaseException as error:
        write_once(out / 'failure.original.json', encoded(dict(scope='FAILED_OR_INCOMPLETE_CPU_CONTENT_REPLAY',
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), completion_claim=False)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True); parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--self-sha256', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    require(os.getuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10' and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Owned native offline CPU context/CUDA empty required')
    reader = Originals(0)
    with reader.opened(str(Path(__file__).resolve()), args.self_sha256):
        pass
    request = reader.json(dict(path=args.request, sha256=args.request_sha256))
    helpers = load_helpers(reader, request['source_inventory'])
    result = run(request, reader, helpers, args.output)
    print(encoded(dict(scope=result['scope'], output=str(Path(args.output) / 'content-replay-report.json'))).decode())


if __name__ == '__main__':
    main()
