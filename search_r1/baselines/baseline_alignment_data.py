"""Bind shared prepared data and replay the baseline's actual question order.

Seed equality alone is insufficient: the baseline's RandomSampler consumes the
global CPU RNG after DataLoader draws its iterator seed. Capture the baseline
CPU RNG immediately before ``iter(train_dataloader)`` or export its actual
ordered indices. This module never regenerates the NQ/Hotpot split.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from .holdout import file_hash, plain


def object_hash(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def semantic_rows(frame):
    """The question identity, prompt and references all participate in equality."""
    rows = []
    seen = set()
    for row in frame.to_dict('records'):
        row = plain(row)
        index = row['extra_info']['index']
        if not isinstance(index, str) or index in seen:
            raise ValueError('Shared baseline data requires unique original string question IDs')
        seen.add(index)
        rows.append(dict(question_id=index, data_source=row['data_source'],
                         prompt=row['prompt'], target=row['reward_model']['ground_truth']['target']))
    return rows


def dataset_identity(files):
    files = [files] if isinstance(files, (str, Path)) else list(files)
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    rows = semantic_rows(frame)
    return dict(files=[dict(path=str(Path(p).resolve()), sha256=file_hash(p)) for p in files],
                rows=len(rows), semantic_sha256=object_hash(rows),
                question_ids=[r['question_id'] for r in rows],
                prompt_sha256s=[object_hash(r['prompt']) for r in rows])


def prepared_manifest(train_files, validation_files, test_files, stage_files=None,
                      stage_train_overlap_policy='disjoint'):
    """Import immutable prepared splits, preserving their existing row order."""
    splits = dict(train=dataset_identity(train_files), validation=dataset_identity(validation_files),
                  test=dataset_identity(test_files))
    if set(splits['train']['question_ids']) & set(splits['validation']['question_ids']):
        raise ValueError('Shared training/validation IDs overlap')
    result = dict(schema='baseline-aligned-data-v1', preparation='import-existing-shared-prepared-files',
                  splits=splits)
    if stage_files is not None:
        if stage_train_overlap_policy not in ('disjoint', 'preserve-common-baseline-train'):
            raise ValueError('Unknown explicit shared stage/train policy')
        result['splits']['stage'] = dataset_identity(stage_files)
        result['stage_train_overlap_policy'] = stage_train_overlap_policy
        verify_stage_files(result, stage_files, train_files, validation_files, test_files)
    return result


def verify_prepared_manifest(manifest, train_files, validation_files, test_files):
    actual = prepared_manifest(train_files, validation_files, test_files)
    if manifest.get('schema') != actual['schema']:
        raise ValueError('Unknown shared data manifest')
    for split in actual['splits']:
        expected = manifest['splits'][split]
        observed = actual['splits'][split]
        # Relocated files are allowed only when both byte and semantic proofs agree.
        for key in ('rows', 'semantic_sha256', 'question_ids', 'prompt_sha256s'):
            if observed[key] != expected[key]:
                raise ValueError('Shared baseline data differs: ' + split + '/' + key)
        if [f['sha256'] for f in observed['files']] != [f['sha256'] for f in expected['files']]:
            raise ValueError('Shared prepared file bytes differ: ' + split)
    return actual


def verify_stage_files(manifest, stage_files, train_files, dev_files, final_files):
    """Bind an explicitly chosen stage split without silently removing train rows.

    A stage overlapping the common baseline train is permitted only by the
    recorded ``preserve-common-baseline-train`` policy. Such a stage is reported
    as training-derived, never as an independent holdout. Dev/final overlap is
    always rejected by both original question ID and full shared prompt.
    """
    verify_prepared_manifest(manifest, train_files, dev_files, final_files)
    actual = dataset_identity(stage_files)
    expected = manifest.get('splits', {}).get('stage')
    if expected is None:
        raise ValueError('Shared manifest has no explicit stage dataset identity')
    for key in ('rows', 'semantic_sha256', 'question_ids', 'prompt_sha256s'):
        if actual[key] != expected[key]:
            raise ValueError('Shared stage dataset differs: ' + key)
    if [f['sha256'] for f in actual['files']] != [f['sha256'] for f in expected['files']]:
        raise ValueError('Shared stage file bytes differ')
    def overlap(other):
        return dict(question_ids=sorted(set(actual['question_ids']) & set(other['question_ids'])),
                    prompt_sha256s=sorted(set(actual['prompt_sha256s']) & set(other['prompt_sha256s'])))
    overlaps = {name: overlap(manifest['splits'][name]) for name in ('train', 'validation', 'test')}
    for name in ('validation', 'test'):
        if any(overlaps[name].values()):
            raise ValueError('Stage dataset overlaps shared dev/final: ' + name)
    policy = manifest.get('stage_train_overlap_policy')
    if policy not in ('disjoint', 'preserve-common-baseline-train'):
        raise ValueError('Explicit shared stage/train policy missing')
    training_derived = any(overlaps['train'].values())
    if training_derived and policy != 'preserve-common-baseline-train':
        raise ValueError('Stage dataset overlaps shared train under disjoint policy')
    return dict(stage=actual, train_rows_preserved=manifest['splits']['train']['rows'],
                train_semantic_sha256=manifest['splits']['train']['semantic_sha256'],
                overlap=overlaps, training_derived=training_derived,
                independent_holdout=not training_derived, stage_train_overlap_policy=policy)


def order_from_rng(question_ids, batch_size, rng_state, epochs=1, shuffle=True):
    """Replay real RandomSampler/DataLoader consumption without touching live RNG.

    ``rng_state`` must be the baseline CPU RNG *just before its first iterator*;
    a seed42 state before model initialization cannot establish equal order.
    Export baseline per-epoch indices instead if other code consumes CPU RNG
    between epochs. This function models uninterrupted loader iteration only.
    """
    if batch_size <= 0 or epochs <= 0 or len(set(question_ids)) != len(question_ids):
        raise ValueError('Invalid shared order dimensions or duplicate IDs')
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng_state)
        loader = torch.utils.data.DataLoader(list(range(len(question_ids))), batch_size=batch_size,
                                           shuffle=shuffle, num_workers=0, drop_last=True)
        orders = [[int(i) for batch in loader for i in batch.tolist()] for _ in range(epochs)]
    result = order_manifest(question_ids, batch_size, orders)
    result.update(source='baseline-pre-iterator-cpu-rng', torch_version=torch.__version__,
                  initial_rng_sha256=hashlib.sha256(bytes(rng_state.tolist())).hexdigest(),
                  shuffle=bool(shuffle))
    result['order_sha256'] = object_hash({k: v for k, v in result.items() if k != 'order_sha256'})
    return result


def order_manifest(question_ids, batch_size, epoch_indices):
    """Bind actual baseline loader indices, including its drop-last decision."""
    question_ids = list(question_ids)
    if type(batch_size) is not int or batch_size <= 0 or len(set(question_ids)) != len(question_ids):
        raise ValueError('Invalid shared order dimensions or duplicate IDs')
    expected_count = len(question_ids) // batch_size * batch_size
    orders = []
    for indices in epoch_indices:
        indices = list(indices)
        if (len(indices) != expected_count or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < len(question_ids) for i in indices)):
            raise ValueError('Baseline epoch must contain each emitted question once and match drop_last')
        orders.append(indices)
    if not orders or not expected_count:
        raise ValueError('Shared order requires at least one complete baseline batch')
    result = dict(schema='baseline-aligned-order-v1', source='baseline-observed-indices',
                  batch_size=batch_size, drop_last=True, num_workers=0,
                  question_ids=question_ids, epochs=orders)
    result['order_sha256'] = object_hash(result)
    return result


def verify_order_manifest(manifest, question_ids, batch_size):
    body = {k: v for k, v in manifest.items() if k != 'order_sha256'}
    if object_hash(body) != manifest.get('order_sha256'):
        raise ValueError('Shared baseline order manifest modified')
    if manifest['question_ids'] != list(question_ids) or manifest['batch_size'] != batch_size:
        raise ValueError('Shared order dataset or batch size differs')
    rebuilt = order_manifest(question_ids, batch_size, manifest['epochs'])
    for key in ('schema', 'batch_size', 'drop_last', 'num_workers', 'question_ids', 'epochs'):
        if rebuilt[key] != manifest[key]:
            raise ValueError('Shared order schema or loader settings differ')
    return manifest


class RecordedOrderSampler(torch.utils.data.Sampler):
    """Replay pinned baseline order with an explicit recoverable question cursor."""
    def __init__(self, question_ids, batch_size, manifest):
        self.manifest = verify_order_manifest(manifest, question_ids, batch_size)
        self.epoch = self.cursor = 0

    def __iter__(self):
        if self.epoch >= len(self.manifest['epochs']):
            raise ValueError('No recorded baseline order for requested epoch')
        order = self.manifest['epochs'][self.epoch]
        while self.cursor < len(order):
            index = order[self.cursor]
            self.cursor += 1
            yield index
        self.epoch += 1
        self.cursor = 0

    def __len__(self):
        return len(self.manifest['epochs'][0])

    def state_dict(self):
        return dict(order_sha256=self.manifest['order_sha256'], epoch=self.epoch, cursor=self.cursor)

    def load_state_dict(self, state):
        epoch, cursor = state['epoch'], state['cursor']
        if state['order_sha256'] != self.manifest['order_sha256']:
            raise ValueError('Checkpoint baseline order differs')
        if (type(epoch) is not int or type(cursor) is not int
                or not 0 <= epoch <= len(self.manifest['epochs'])
                or not 0 <= cursor <= len(self)
                or (epoch == len(self.manifest['epochs']) and cursor != 0)):
            raise ValueError('Invalid recorded order checkpoint cursor')
        self.epoch, self.cursor = epoch, cursor


def validate_emitted_batch(manifest, epoch, batch_index, question_ids):
    """Audit actual trainer emission before rollout; never infer it from a seed."""
    size = manifest['batch_size']
    indices = manifest['epochs'][epoch][batch_index * size:(batch_index + 1) * size]
    expected = [manifest['question_ids'][i] for i in indices]
    if len(expected) != size or expected != list(question_ids):
        raise ValueError('Actual question batch differs from pinned baseline order')
    return dict(epoch=epoch, batch_index=batch_index, question_ids=expected,
                batch_sha256=object_hash(expected))
